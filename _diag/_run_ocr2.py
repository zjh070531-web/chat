# -*- coding: utf-8 -*-
"""OCR 读取截图上的文字(用于模型无法直接看图时读取界面标注)"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

engine = RapidOCR()


def run(img, tag):
    res, _ = engine(img)
    print(f"--- {tag}  (boxes={0 if not res else len(res)}) ---")
    if not res:
        print("   [无文字]")
        return
    items = []
    for rec in res:
        box, text = rec[0], str(rec[1])
        try:
            score = float(rec[2])
        except (TypeError, ValueError):
            score = -1.0
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append((min(ys), min(xs), text, score))
    items.sort()
    for y, x, text, score in items:
        print(f"   y={int(y):4d} x={int(x):4d} | {text}   ({score:.2f})")


for p in sys.argv[1:]:
    im = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        print("!!! 读不到", p)
        continue
    print("=" * 70)
    print(os.path.basename(p), im.shape)
    run(im, "原图")
    run(cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), "放大2x")
