# -*- coding: utf-8 -*-
"""诊断4: depth 模式(同深度=同桌) 逐对判定, 与旧版 image 模式对照"""
import sys, os, math
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\chatting\_dataset")
import cv2
import numpy as np
import torch
import sample_pairs as SP
import video as V

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
for k, h in enumerate(heads):
    print(f"  head{k}: ({h['cx']:.0f},{h['cy']:.0f}) 头高={h['bbox'][3]-h['bbox'][1]} yaw={h['yaw']:.1f}")

print("\n=== 逐对: 旧版(image) vs 新版(depth) ===")
for i in range(len(heads)):
    for j in range(i + 1, len(heads)):
        a, b = heads[i], heads[j]
        h_a = a['bbox'][3] - a['bbox'][1]
        h_b = b['bbox'][3] - b['bbox'][1]
        ref = (h_a + h_b) * 0.5
        szr = max(h_a, h_b) / min(h_a, h_b)
        dx = abs(a['cx'] - b['cx']); dy = abs(a['cy'] - b['cy'])
        d = math.hypot(dx, dy)
        # 旧版
        ok_img = (dy <= V.ROW_Y_RATIO * ref and dx <= V.NEAR_DX_RATIO * ref
                  and dx >= dy * V.NEIGHBOR_DX_RATIO and szr <= V.HEAD_SIZE_RATIO_MAX)
        # 新版
        ok_dep = (szr <= V.SEAT_DEPTH_RATIO_MAX and d <= V.SEAT_DIST_RATIO * ref)
        if ok_img or ok_dep or d <= 4.0 * ref:
            print(f"  ({i},{j}) 头高{a['bbox'][3]-a['bbox'][1]}/{b['bbox'][3]-b['bbox'][1]}"
                  f" 比={szr:.2f} 距={d:.0f}({d/ref:.1f}×头高) dy={dy:.0f} "
                  f"→ 旧版{'✓' if ok_img else '✗'}  新版{'✓' if ok_dep else '✗'}")

print("\n=== 通过新版座位约束的对, 再查视线 ===")
for i in range(len(heads)):
    for j in range(i + 1, len(heads)):
        a, b = heads[i], heads[j]
        h_a = a['bbox'][3] - a['bbox'][1]
        h_b = b['bbox'][3] - b['bbox'][1]
        ref = (h_a + h_b) * 0.5
        szr = max(h_a, h_b) / min(h_a, h_b)
        d = math.hypot(a['cx'] - b['cx'], a['cy'] - b['cy'])
        if not (szr <= V.SEAT_DEPTH_RATIO_MAX and d <= V.SEAT_DIST_RATIO * ref):
            continue
        ya, yb = V.YAW_SIGN * a['yaw'], V.YAW_SIGN * b['yaw']
        turned = abs(ya) >= V.MIN_TURN_YAW and abs(yb) >= V.MIN_TURN_YAW
        a_left = a['cx'] < b['cx']
        a_t = (ya > 0) if a_left else (ya < 0)      # 已按实测符号修正
        b_t = (yb < 0) if a_left else (yb > 0)
        gaze = abs(ya - yb)
        if gaze > 180:
            gaze = 360 - gaze
        ok = turned and a_t and b_t and gaze >= V.GAZE_OBTUSE_THRESH
        print(f"  ({i},{j}) 头=({a['cx']:.0f},{a['cy']:.0f})↔({b['cx']:.0f},{b['cy']:.0f}) "
              f"转头{turned} 互朝{a_t and b_t} 夹角{gaze:.1f}° → {'✓交头接耳' if ok else '✗视线不足'}")
