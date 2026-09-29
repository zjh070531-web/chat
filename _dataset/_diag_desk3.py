# -*- coding: utf-8 -*-
"""诊断3: 单帧脚点 → 拟合桌线 → 逐对判定同桌/邻座 (验证桌子坐标系几何)"""
import sys, os, math
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\chatting\_dataset")
import cv2
import numpy as np
import torch
import sample_pairs as SP
import video as V

# 单帧只有 ~9 个脚点, 临时放宽"累积点数"门槛(真实运行靠跨帧累积)
V.DESK_MIN_LINE_PTS = 3
V.DESK_MIN_FEET = 3

img_path = sys.argv[1]
im0 = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
img_h, img_w = im0.shape[:2]
print("image", im0.shape)

CONF, MIN_FS = 0.15, 0.30
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
    body = [int(v) for v in bbox] if (bbox is not None and len(bbox)) else None
    cands.append({'head': [int(hx1), int(hy1), int(hx2), int(hy2)], 'body': body})

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
print(f"检出人头: {len(heads)}")

# ---- 拟合桌线 ----
hs_sorted = sorted(h['bbox'][3] - h['bbox'][1] for h in heads)
ref_h = hs_sorted[len(hs_sorted) // 2]
feet = [V.feet_point(h) for h in heads]
feet = [f for f in feet if f is not None]
lines = V.fit_desk_lines(feet, ref_h)
print(f"中位头高={ref_h}  拟合出 {len(lines)} 条桌线:")
for k, (mx, my, dx, dy, n, ang) in enumerate(lines):
    print(f"  线{k}: 过({mx:.0f},{my:.0f}) 方向角={ang:.1f}° 支持点={n}")

desk = V.DeskFrame()
desk.lines = lines

# ---- 逐对判定 ----
print("\n逐对结果 (同桌=✓):")
for i in range(len(heads)):
    for j in range(i + 1, len(heads)):
        a, b = heads[i], heads[j]
        h_a = a['bbox'][3] - a['bbox'][1]
        h_b = b['bbox'][3] - b['bbox'][1]
        ref = (h_a + h_b) * 0.5
        if max(h_a, h_b) / min(h_a, h_b) > V.HEAD_SIZE_RATIO_MAX:
            continue
        ga, gb = V.feet_point(a), V.feet_point(b)
        if ga is None or gb is None:
            continue
        same = desk.same_desk(ga, gb, ref)
        dist = math.hypot(gb[0] - ga[0], gb[1] - ga[1])
        if same or dist <= 4.0 * ref:      # 只显示通过的和"距离较近但被否"的
            print(f"  ({i},{j}) 脚距={dist:.0f} ({dist/ref:.1f}×头高) "
                  f"头=({a['cx']:.0f},{a['cy']:.0f})↔({b['cx']:.0f},{b['cy']:.0f}) "
                  f"→ {'✓同桌邻座' if same else '✗不同桌'}")

# ---- 可视化桌线 ----
vis = im0.copy()
L0 = max(img_w, img_h) * 0.8
for k, h in enumerate(heads):
    x1, y1, x2, y2 = h['bbox']
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 160, 0), 2)
    cv2.putText(vis, str(k), (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 160, 0), 2)
    f = V.feet_point(h)
    if f:
        cv2.circle(vis, (int(f[0]), int(f[1])), 6, (0, 0, 255), -1)
for k, (mx, my, dx, dy, n, ang) in enumerate(lines):
    cv2.line(vis, (int(mx - dx * L0), int(my - dy * L0)),
             (int(mx + dx * L0), int(my + dy * L0)), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, f"desk{k} {ang:.0f}deg ({n}pts)", (int(mx), int(my)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imencode('.png', vis)[1].tofile(r"D:\chatting\outputs\_desk_lines.png")
print("saved: D:/chatting/outputs/_desk_lines.png")
