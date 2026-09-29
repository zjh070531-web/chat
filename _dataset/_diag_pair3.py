# -*- coding: utf-8 -*-
"""诊断3: 跳过预过滤, 对原始检出的每个头跑头姿, 再做 7 步配对判定"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\chatting\_dataset")
import cv2
import numpy as np
import torch
import sample_pairs as SP

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
        body_dets = SP.V.non_max_suppression(out, SP.V.CONF_THRES, SP.V.IOU_THRES,
                                             classes=[0], num_offsets=SP.V.NUM_OFFSETS)
        part_dets = SP.V.non_max_suppression(out, SP.V.CONF_THRES, SP.V.IOU_THRES,
                                             classes=list(range(1, 1 + SP.V.NUM_OFFSETS // 2)),
                                             num_offsets=SP.V.NUM_OFFSETS)
    bboxes, points, scores = SP.V.post_process_batch(
        x, [], [[(img_h, img_w)]], body_dets, part_dets, SP.V.NUM_OFFSETS, SP.V.MATCH_IOU_THRES)

# 跳过预过滤, 只要头框有效就送头姿
head_boxes = []
for bbox, point, score in zip(bboxes, points, scores):
    f_score, f_bbox = point[0][2], point[0][3:]
    hx1, hy1, hx2, hy2 = f_bbox
    if hx2 - hx1 <= 0 or hy2 - hy1 <= 0:
        continue
    head_boxes.append([int(hx1), int(hy1), int(hx2), int(hy2)])
print(f"送头姿的头数: {len(head_boxes)}")

heads = []
crops, tensors = [], []
for hx1, hy1, hx2, hy2 in head_boxes:
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
    heads.append({"cx": (rb[0] + rb[2]) / 2.0, "cy": (rb[1] + rb[3]) / 2.0,
                  "bbox": rb, "yaw": yaw, "pitch": pitch, "roll": roll})
    print(f"head{k}: cx={heads[-1]['cx']:.0f} cy={heads[-1]['cy']:.0f} "
          f"头高={rb[3]-rb[1]} yaw={yaw:.1f} pitch={pitch:.1f} roll={roll:.1f}")

V = SP.V
n = len(heads)
for i in range(n):
    for j in range(i + 1, n):
        a, b = heads[i], heads[j]
        h_a = a["bbox"][3] - a["bbox"][1]
        h_b = b["bbox"][3] - b["bbox"][1]
        ref = (h_a + h_b) * 0.5
        dx = abs(a["cx"] - b["cx"]); dy = abs(a["cy"] - b["cy"])
        szr = max(h_a, h_b) / min(h_a, h_b)
        ya = V.YAW_SIGN * a["yaw"]; yb = V.YAW_SIGN * b["yaw"]
        a_on_left = a["cx"] < b["cx"]
        a_toward = (ya < 0) if a_on_left else (ya > 0)
        b_toward = (yb > 0) if a_on_left else (yb < 0)
        gaze = abs(ya - yb)
        if gaze > 180.0:
            gaze = 360.0 - gaze
        turned = abs(ya) >= V.MIN_TURN_YAW and abs(yb) >= V.MIN_TURN_YAW
        ok_seat = dy <= V.ROW_Y_RATIO * ref
        ok_depth = szr <= V.HEAD_SIZE_RATIO_MAX
        ok_near = dx <= V.NEAR_DX_RATIO * ref
        ok_lr = dx >= dy * V.NEIGHBOR_DX_RATIO
        ok_turn = turned
        ok_toward = a_toward and b_toward
        ok_gaze = gaze >= V.GAZE_OBTUSE_THRESH
        print(f"\n=== 配对 ({i},{j}) ===")
        print(f"  ①同排    dy={dy:.0f} ≤ {V.ROW_Y_RATIO * ref:.0f}   {'✓' if ok_seat else '✗ 前后排'}")
        print(f"  ②同深度  头高比={szr:.2f} ≤ {V.HEAD_SIZE_RATIO_MAX}   {'✓' if ok_depth else '✗ 前后排'}")
        print(f"  ③邻座    dx={dx:.0f} ≤ {V.NEAR_DX_RATIO * ref:.0f}   {'✓' if ok_near else '✗ 隔太远'}")
        print(f"  ④左右    dx={dx:.0f} ≥ dy×{V.NEIGHBOR_DX_RATIO}={dy * V.NEIGHBOR_DX_RATIO:.0f}   {'✓' if ok_lr else '✗ 偏前后'}")
        print(f"  ⑤转头    |ya|={abs(ya):.0f}° |yb|={abs(yb):.0f}° ≥ {V.MIN_TURN_YAW}°   {'✓' if ok_turn else '✗ 有人没转头'}")
        print(f"  ⑥互朝    a→b={a_toward} b→a={b_toward}   {'✓' if ok_toward else '✗ 背对背/同向'}")
        print(f"  ⑦夹角    {gaze:.1f}° ≥ {V.GAZE_OBTUSE_THRESH}°   {'✓' if ok_gaze else '✗ 不够钝'}")
        passed = ok_seat and ok_depth and ok_near and ok_lr and ok_turn and ok_toward and ok_gaze
        print(f"  => {'✓ 会判为交头接耳' if passed else '✗ 不算(卡在上面 ✗ 的那步)'}")
