# -*- coding: utf-8 -*-
"""导出 my_dataset 两类代表帧的对照拼图, 供人眼核对数据形态"""
import sys, os, random
sys.stdout.reconfigure(encoding="utf-8")
import cv2
import numpy as np

random.seed(7)
OUT = r"D:\chatting\_diag\dataset_preview.png"
CELL_W, CELL_H = 260, 200
N = 6


def pick_frames(sub, n):
    d = os.path.join(r"D:\jepa-main\my_dataset", sub)
    fs = sorted(f for f in os.listdir(d) if f.endswith(".mp4"))
    random.shuffle(fs)
    out = []
    for f in fs:
        if len(out) >= n:
            break
        p = os.path.join(d, f)
        cap = cv2.VideoCapture(p)
        cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if cnt < 5:
            cap.release(); continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, cnt // 2)
        ok, img = cap.read()
        cap.release()
        if ok:
            out.append((f, img, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if False else img.shape[1], img.shape[0])))
    return out


def cell(img, label, w, h):
    scale = min(w / img.shape[1], h / img.shape[0])
    nw, nh = max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h + 22, w, 3), np.uint8)
    y0 = 22 + (h - nh) // 2
    x0 = (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    cv2.putText(canvas, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return canvas


rows = []
for sub in ["chatting", "not_chatting"]:
    items = pick_frames(sub, N)
    cs = [cell(im, f"{sub[:3]} {f} {im.shape[1]}x{im.shape[0]}", CELL_W, CELL_H)
          for f, im, _ in items]
    while len(cs) < N:
        cs.append(np.zeros((CELL_H + 22, CELL_W, 3), np.uint8))
    rows.append(np.hstack(cs))

grid = np.vstack(rows)
cv2.putText(grid, "my_dataset preview  (top: chatting=1 / bottom: not_chatting=0)",
            (6, grid.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
cv2.imwrite(OUT, grid)
print("已保存:", OUT, grid.shape)
print("chatting 行:", [f for f, _, _ in pick_frames("chatting", N)])
print("not_chatting 行:", [f for f, _, _ in pick_frames("not_chatting", N)])
