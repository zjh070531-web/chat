# -*- coding: utf-8 -*-
"""探测: 对极小截图试多种放大倍数, 看哪个尺度能检出人; 同时强化 OCR 读 CHAT/HUD 文字"""
import sys, os, re
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\chatting\_dataset")
import cv2
import numpy as np
import torch
import sample_pairs as SP

IMGS = []
for a in sys.argv[1:]:
    if a.startswith("@"):
        IMGS += [ln.strip() for ln in open(a[1:], encoding="utf-8") if ln.strip()]
    else:
        IMGS.append(a)

V = SP.V
agent, det_model = SP.load_models(1280)
from rapidocr_onnxruntime import RapidOCR
OCR = RapidOCR()


def dets_at(im, conf, min_fs=0.20):
    h, w = im.shape[:2]
    img = SP.letterbox(im, 1280, stride=SP.STRIDE, auto=True)[0]
    img = np.ascontiguousarray(img.transpose((2, 0, 1))[::-1])
    x = torch.from_numpy(img).to(SP.DEVICE) / 255.0
    x = x[None]
    with torch.inference_mode():
        with SP.AMP:
            out = det_model(x, augment=False, scales=V.SCALES)[0]
            bd = V.non_max_suppression(out, conf, V.IOU_THRES, classes=[0],
                                       num_offsets=V.NUM_OFFSETS)
            pd = V.non_max_suppression(out, conf, V.IOU_THRES,
                                       classes=list(range(1, 1 + V.NUM_OFFSETS // 2)),
                                       num_offsets=V.NUM_OFFSETS)
        bboxes, points, scores = V.post_process_batch(
            x, [], [[(h, w)]], bd, pd, V.NUM_OFFSETS, V.MATCH_IOU_THRES)
    r = []
    for bb, pt, sc in zip(bboxes, points, scores):
        fs, fb = pt[0][2], pt[0][3:]
        hx1, hy1, hx2, hy2 = fb
        if fs < min_fs or hx2 - hx1 <= 0 or hy2 - hy1 <= 0:
            continue
        r.append(([int(hx1), int(hy1), int(hx2), int(hy2)], float(fs)))
    return r


for p in IMGS:
    im0 = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    if im0 is None:
        print("!!! 读不到", p)
        continue
    h, w = im0.shape[:2]
    print(f"\n{'='*72}\n{os.path.basename(p)[-16:]}  {w}x{h}")
    # --- OCR 多尺度 ---
    for s in (3, 5, 8):
        big = cv2.resize(im0, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        res, _ = OCR(big)
        if res:
            txt = " | ".join(str(r[1]) for r in res)
            print(f"  OCR x{s}: {txt}")
            break
    # --- 多尺度检测 ---
    for s in (1.0, 2.0, 3.0, 4.0, 6.0):
        img = im0 if s == 1.0 else cv2.resize(im0, None, fx=s, fy=s,
                                               interpolation=cv2.INTER_CUBIC)
        d = dets_at(img, 0.10)
        hs = [f"{b[3]-b[1]}px/{c:.2f}" for b, c in d]
        print(f"  检测 x{s}: {len(d)} 头  {hs}")
