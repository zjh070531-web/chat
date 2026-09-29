# -*- coding: utf-8 -*-
"""裁剪图专用: 低置信度检出全部人头 → 取头最大的两个(即"画面中两人")
→ 打印座席(depth) / 转头 / 互朝 / 夹角 每一步判定。
用于 4K 原图裁出来的小截图(小截图直接跑 video.py 因尺度不同检不出人)。
"""
import sys, os, math
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
CONF, MIN_FS = 0.15, 0.30          # 小截图要放宽, 否则人检不出来
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

head_boxes = []
for bbox, point, score in zip(bboxes, points, scores):
    fs, fb = point[0][2], point[0][3:]
    hx1, hy1, hx2, hy2 = fb
    if fs < MIN_FS or hx2 - hx1 <= 0 or hy2 - hy1 <= 0:
        continue
    head_boxes.append([int(hx1), int(hy1), int(hx2), int(hy2)])
head_boxes.sort(key=lambda b: -(b[3] - b[1]))
print(f"检出人头数: {len(head_boxes)}  → 取头最大的两个作主对")
for k, b in enumerate(head_boxes[:4]):
    print(f"   #{k} 框={b} 头高={b[3]-b[1]}")
head_boxes = head_boxes[:2]
if len(head_boxes) < 2:
    print("!! 不足两人, 无法配对")
    sys.exit(0)

crops, tensors, heads = [], [], []
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
batch = ((torch.stack(tensors).to(SP.DEVICE) - SP.MEAN) / SP.STD)
with torch.inference_mode():
    with SP.AMP:
        pd_m = SP.V.batch_torch_A_to_R(agent.net(batch).float())
rot_mats = pd_m.detach().cpu().numpy()

for k, rb in enumerate(crops):
    rm = np.transpose(rot_mats[k])
    ang = SP.V.Rotation.from_matrix(rm).as_euler("xyz", degrees=True)
    roll, pitch, yaw = list(map(SP.V.limit_angle, [ang[2], ang[0] - 180, ang[1]]))
    ya = SP.V.YAW_SIGN * yaw
    fwd = rm @ np.array([0.0, 0.0, 1.0])
    fwd = np.array([fwd[0], -fwd[1], -fwd[2]])
    heads.append({"cx": (rb[0] + rb[2]) / 2.0, "cy": (rb[1] + rb[3]) / 2.0,
                  "bbox": rb, "ya": ya, "yaw": yaw, "pitch": pitch})
    print(f"P{k}: 中心=({heads[-1]['cx']:.0f},{heads[-1]['cy']:.0f}) 头高={rb[3]-rb[1]} "
          f"yaw={yaw:.1f} → ya={ya:.1f} ({'朝图像右' if ya > 0 else '朝图像左'})  pitch={pitch:.1f}")

V = SP.V
a, b = heads
h_a, h_b = a["bbox"][3] - a["bbox"][1], b["bbox"][3] - b["bbox"][1]
ref = (h_a + h_b) * 0.5
szr = max(h_a, h_b) / min(h_a, h_b)
dist = math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])
lim_dist = V.SEAT_DIST_RATIO * ref
ya, yb = a["ya"], b["ya"]
a_on_left = a["cx"] < b["cx"]
a_toward = (ya > 0) if a_on_left else (ya < 0)
b_toward = (yb < 0) if a_on_left else (yb > 0)
gaze = abs(ya - yb)
if gaze > 180.0:
    gaze = 360.0 - gaze
ok_seat = szr <= V.SEAT_DEPTH_RATIO_MAX
ok_near = dist <= lim_dist
ok_turn = abs(ya) >= V.MIN_TURN_YAW and abs(yb) >= V.MIN_TURN_YAW
ok_toward = a_toward and b_toward
ok_gaze = gaze >= V.GAZE_OBTUSE_THRESH

print("\n=== P0-P1 判定 (SEAT_MODE=%s) ===" % V.SEAT_MODE)
print(f"  ①同深度=同桌  头高 {h_a}/{h_b} 比={szr:.2f} ≤ {V.SEAT_DEPTH_RATIO_MAX}   {'✓' if ok_seat else '✗ 前后排/隔桌'}")
print(f"  ②邻座距离     中心距={dist:.0f}px ≤ {lim_dist:.0f}px (= {V.SEAT_DIST_RATIO}×平均头高{ref:.0f})   {'✓' if ok_near else '✗ 隔太远'}")
print(f"  ③转头        |ya|={abs(ya):.0f}° |yb|={abs(yb):.0f}° ≥ {V.MIN_TURN_YAW}°   {'✓' if ok_turn else '✗ 有人没转头'}")
print(f"  ④互朝对方     左={'P0' if a_on_left else 'P1'}  a→b={a_toward} b→a={b_toward}   {'✓' if ok_toward else '✗ 背对背/同向'}")
print(f"  ⑤视线夹角     {gaze:.1f}° ≥ {V.GAZE_OBTUSE_THRESH}°   {'✓' if ok_gaze else '✗ 不够钝 (差 %.1f°)' % (V.GAZE_OBTUSE_THRESH - gaze)}")
print(f"  => {'✓ 会判为交头接耳' if (ok_seat and ok_near and ok_turn and ok_toward and ok_gaze) else '✗ 不算(卡在上面 ✗ 的步骤)'}")
