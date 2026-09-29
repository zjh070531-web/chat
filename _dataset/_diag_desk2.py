# -*- coding: utf-8 -*-
"""诊断2: 低阈值检出所有人 + 脚点可视化 + 多桌线 RANSAC 观察几何"""
import sys, os, math
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\chatting\_dataset")
import cv2
import numpy as np
import torch
import sample_pairs as SP

CONF, MIN_FS = 0.15, 0.30
img_path = sys.argv[1]
im0 = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
img_h, img_w = im0.shape[:2]
print("image", im0.shape)

agent, det_model = SP.load_models(1280)
img = SP.letterbox(im0, 1280, stride=SP.STRIDE, auto=True)[0]
img = np.ascontiguousarray(img.transpose((2, 0, 1))[::-1])
x = torch.from_numpy(img).to(SP.DEVICE) / 255.0
x = x[None]
with torch.inference_mode():
    with SP.AMP:
        out = det_model(x, augment=False, scales=SP.V.SCALES)[0]
        body_dets = SP.V.non_max_suppression(out, CONF, SP.V.IOU_THRES,
                                             classes=[0], num_offsets=SP.V.NUM_OFFSETS)
        part_dets = SP.V.non_max_suppression(out, CONF, SP.V.IOU_THRES,
                                             classes=list(range(1, 1 + SP.V.NUM_OFFSETS // 2)),
                                             num_offsets=SP.V.NUM_OFFSETS)
    bboxes, points, scores = SP.V.post_process_batch(
        x, [], [[(img_h, img_w)]], body_dets, part_dets, SP.V.NUM_OFFSETS, SP.V.MATCH_IOU_THRES)

cands = []
for bbox, point, score in zip(bboxes, points, scores):
    f_score, f_bbox = point[0][2], point[0][3:]
    hx1, hy1, hx2, hy2 = f_bbox
    if f_score < MIN_FS or hx2 - hx1 <= 0 or hy2 - hy1 <= 0:
        continue
    body = None
    if bbox is not None and len(bbox):
        body = [int(v) for v in bbox]
    cands.append({'head': [int(hx1), int(hy1), int(hx2), int(hy2)], 'body': body})
print(f"低阈值头数: {len(cands)}")

heads = []
crops, tensors = [], []
for hb in cands:
    hx1, hy1, hx2, hy2 = hb['head']
    pcx, pcy = (hx1 + hx2) / 2.0, (hy1 + hy2) / 2.0
    hs = max(hx2 - hx1, hy2 - hy1)
    ex = ey = (0.5 - SP.V.EDGES_SCALE) * hs
    nx1 = max(0, int(pcx - ex)); nx2 = min(img_w - 1, int(pcx + ex))
    ny1 = max(0, int(pcy - ey)); ny2 = min(img_h - 1, int(pcy + ey))
    crop = im0[ny1:ny2, nx1:nx2]
    rgb = cv2.cvtColor(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR),
                       cv2.COLOR_BGR2RGB)
    tensors.append(torch.from_numpy(rgb.transpose(2, 0, 1)).float().div_(255.0))
    crops.append([nx1, ny1, nx2, ny2])
batch = torch.stack(tensors).to(SP.DEVICE)
batch = (batch - SP.MEAN) / SP.STD
with torch.inference_mode():
    with SP.AMP:
        fisher = agent.net(batch)
        pd_m = SP.V.batch_torch_A_to_R(fisher.float())
rot_mats = pd_m.detach().cpu().numpy()
for k, rb in enumerate(crops):
    rm = np.transpose(rot_mats[k])
    ang = SP.V.Rotation.from_matrix(rm).as_euler("xyz", degrees=True)
    roll, pitch, yaw = list(map(SP.V.limit_angle, [ang[2], ang[0] - 180, ang[1]]))
    heads.append({'cx': (rb[0] + rb[2]) / 2.0, 'cy': (rb[1] + rb[3]) / 2.0,
                  'bbox': rb, 'body': cands[k]['body'],
                  'yaw': yaw, 'pitch': pitch, 'roll': roll})

vis = im0.copy()
feet = []
for k, h in enumerate(heads):
    x1, y1, x2, y2 = h['bbox']
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 160, 0), 2)
    cv2.putText(vis, str(k), (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 160, 0), 2)
    if h['body']:
        bx1, by1, bx2, by2 = h['body']
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 200, 0), 1)
        f = ((bx1 + bx2) / 2.0, float(by2))
        feet.append(f)
        cv2.circle(vis, (int(f[0]), int(f[1])), 6, (0, 0, 255), -1)
        print(f"head{k}: 头=({h['cx']:.0f},{h['cy']:.0f}) 头高={y2-y1} 脚=({f[0]:.0f},{f[1]:.0f}) yaw={h['yaw']:.1f}")

# 人工目测辅助: 把所有脚点两两连线中, 找出"共线≥3点"的直线方向 (穷举)
ref_h = sorted([h['bbox'][3] - h['bbox'][1] for h in heads])[len(heads) // 2]
n = len(feet)
print(f"\n中位头高={ref_h}  穷举找共线≥3的脚点组合:")
seen = set()
for i in range(n):
    for j in range(i + 1, n):
        p, q = feet[i], feet[j]
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        if L < 3.0 * ref_h:
            continue
        dx, dy = q[0] - p[0], q[1] - p[1]
        inl = [k for k in range(n)
               if abs((feet[k][0] - p[0]) * dy - (feet[k][1] - p[1]) * dx) / L <= 1.0 * ref_h]
        if len(inl) >= 3:
            key = tuple(sorted(inl))
            if key in seen:
                continue
            seen.add(key)
            a = math.degrees(math.atan2(dy, dx))
            print(f"  线 {key}: 方向角={a:.1f}° ({len(inl)}点)")

os.makedirs(r"D:\chatting\outputs", exist_ok=True)
cv2.imencode('.png', vis)[1].tofile(r"D:\chatting\outputs\_desk_geom.png")
print("saved: D:/chatting/outputs/_desk_geom.png")
