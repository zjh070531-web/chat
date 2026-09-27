# -*- coding: utf-8 -*-
"""临时脚本: 用 rapidocr 读取两张截图的文字内容"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

engine = RapidOCR()

SHOTS = [r"D:\chatting\_shot1.png", r"D:\chatting\_shot2.png"]


def run(img, tag):
    res, elapse = engine(img)
    print(f"--- {tag}  (boxes={0 if not res else len(res)}) ---")
    if not res:
        print("   [无文字]")
        return
    # 按 y 排序, 同一行合并
    items = []
    for rec in res:
        box = rec[0]
        text = str(rec[1])
        try:
            score = float(rec[2])
        except (TypeError, ValueError):
            score = -1.0
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append((min(ys), min(xs), text, score))
    items.sort()
    lines = []
    for y, x, text, score in items:
        placed = False
        for L in lines:
            if abs(L[0] - y) < 12:
                L[1].append((x, text, score))
                placed = True
                break
        if not placed:
            lines.append([y, [(x, text, score)]])
    for y, seg in lines:
        seg.sort()
        joined = "  ".join(f"{t}({s:.2f})" for _, t, s in seg)
        print(f"   y={int(y):4d} | {joined}")


for p in SHOTS:
    im = cv2.imread(p)
    print("=" * 70)
    print(os.path.basename(p), im.shape)
    # 1) 原图
    run(im, "原图")
    # 2) 放大 3 倍
    big = cv2.resize(im, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    run(big, "放大3x")
    # 3) 反色 + 放大 3 倍 (应对深底浅字)
    inv = 255 - im
    invb = cv2.resize(inv, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    run(invb, "反色+放大3x")
