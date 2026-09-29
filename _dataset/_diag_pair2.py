# -*- coding: utf-8 -*-
"""诊断2: 打印 BPJDet 原始检测 + 每条预过滤的通过情况"""
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

print(f"原始检测组数: {len(bboxes)}")
for k, (bbox, point, score) in enumerate(zip(bboxes, points, scores)):
    f_score, f_bbox = point[0][2], point[0][3:]
    fails = []
    if f_score < SP.V.HEAD_CONF_THRESH: fails.append(f"头置信度{f_score:.2f}<{SP.V.HEAD_CONF_THRESH}")
    if score < SP.V.BODY_CONF_THRESH: fails.append(f"身置信度{score:.2f}<{SP.V.BODY_CONF_THRESH}")
    if bbox is None or len(bbox) == 0:
        fails.append("无身体框"); print(f"[{k}] {'; '.join(fails)}"); continue
    bx1, by1, bx2, by2 = bbox
    bw, bh = bx2 - bx1, by2 - by1
    hx1, hy1, hx2, hy2 = f_bbox
    hw, hh = hx2 - hx1, hy2 - hy1
    if bh < SP.V.MIN_BODY_HEIGHT: fails.append(f"身高{bh:.0f}<{SP.V.MIN_BODY_HEIGHT}")
    if bw < SP.V.MIN_BODY_WIDTH: fails.append(f"身宽{bw:.0f}<{SP.V.MIN_BODY_WIDTH}")
    hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
    if hcy > by1 + bh * 0.50: fails.append("头不在身体上半部")
    if hcx < bx1 - bw * 0.25 or hcx > bx2 + bw * 0.25: fails.append("头偏离身体")
    if not (SP.V.BH_RATIO_MIN <= bh / max(hh,1) <= SP.V.BH_RATIO_MAX):
        fails.append(f"身头比{bh/max(hh,1):.1f}不在[{SP.V.BH_RATIO_MIN},{SP.V.BH_RATIO_MAX}]")
    if (hy2 - by1) / bh > 0.40: fails.append(f"头贴身顶超限{(hy2-by1)/bh:.2f}>0.40")
    hr = hw / img_w
    if not (SP.V.HEAD_MIN_RATIO <= hr <= SP.V.HEAD_MAX_RATIO):
        fails.append(f"头宽占比{hr:.3f}不在[{SP.V.HEAD_MIN_RATIO},{SP.V.HEAD_MAX_RATIO}]")
    if not (0.5 <= hw / max(hh,1) <= 1.8): fails.append(f"头长宽比{hw/max(hh,1):.2f}超限")
    tag = "PASS" if not fails else "FAIL: " + "; ".join(fails)
    print(f"[{k}] head=({hx1:.0f},{hy1:.0f},{hx2:.0f},{hy2:.0f}) 头{hw:.0f}x{hh:.0f} "
          f"body=({bx1:.0f},{by1:.0f},{bx2:.0f},{by2:.0f}) 身{bw:.0f}x{bh:.0f} fs={f_score:.2f} bs={score:.2f} => {tag}")
