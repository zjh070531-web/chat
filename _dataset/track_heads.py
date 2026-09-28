# -*- coding: utf-8 -*-
r"""track_heads.py —— 给每个 clip 追踪两人头的位置, 让标注页的红框"跟着人走"

背景: 采样时只在"采样帧"(clip 第 0 帧)跑过一次检测, 存了两人头心 ax/ay/bx/by。
      clip 有 3 秒, 人会动 → 静止红框会"漂"。本脚本用归一化互相关模板匹配,
      从第 0 帧的头块出发, 逐帧追踪两个头, 把每帧归一化位置写成 track.json。

产物: _dataset/track.json  {clip: {n: 帧数, t: [[ax,ay,bx,by], ...每帧...]}}
      (坐标均为相对裁剪框的 0~1 比例, 标注页按播放进度取对应帧)
"""
import csv
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

BASE = r"D:\chatting\_dataset"

# ---- 追踪参数 ----
CORR_THRESH = 0.35    # 模板相关度阈值: 低于它认为没匹配上(保持原位, 不跳变)
SMOOTH = 0.6          # 位置 EMA 平滑系数: 越大越跟手、越小越稳 (0.6 = 平滑度较强)
TPL_UPDATE = 0.25     # 模板自适应更新比例: 每帧把模板向新匹配到的头块靠拢一点,
                      #   让模板跟着人的转头/侧脸走, 避免"第0帧的模板用3秒就过时"


def main():
    csvp = os.path.join(BASE, "candidates.csv")
    rows = list(csv.DictReader(open(csvp, encoding="utf-8-sig")))
    out = {}
    N = len(rows)
    for k, r in enumerate(rows, 1):
        clip = str(r["clip"]).replace("\\", "/")
        mp4 = os.path.join(BASE, clip.replace("/", os.sep))
        if not os.path.exists(mp4):
            continue
        try:
            W = float(r["x2"]) - float(r["x1"])
            H = float(r["y2"]) - float(r["y1"])
            hh = float(r["head_h"])
        except (TypeError, ValueError):
            continue
        if W <= 0 or H <= 0 or hh <= 0:
            continue
        ax = (float(r["ax"]) - float(r["x1"])) / W
        ay = (float(r["ay"]) - float(r["y1"])) / H
        bx = (float(r["bx"]) - float(r["x1"])) / W
        by = (float(r["by"]) - float(r["y1"])) / H

        cap = cv2.VideoCapture(mp4)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if len(frames) < 2:
            continue

        Hf, Wf = frames[0].shape[:2]
        r0 = max(8, int(hh / W * Wf * 0.55))          # 头块半边长(裁剪图里约 0.55×头高)
        centers = [(ax * Wf, ay * Hf), (bx * Wf, by * Hf)]
        tmpl = [None, None]
        for c in range(2):
            cx, cy = int(centers[c][0]), int(centers[c][1])
            x0, y0 = max(0, cx - r0), max(0, cy - r0)
            x1, y1 = min(Wf, cx + r0), min(Hf, cy + r0)
            if x1 > x0 and y1 > y0:
                tmpl[c] = frames[0][y0:y1, x0:x1]
            centers[c] = (cx, cy)

        win = r0 + 20
        tracks = []
        for f in frames:
            line = []
            for c in range(2):
                cx, cy = centers[c]
                if tmpl[c] is not None and tmpl[c].size > 0:
                    th, tw = tmpl[c].shape[:2]
                    x0 = max(0, int(cx) - win)
                    y0 = max(0, int(cy) - win)
                    x1 = min(Wf - tw, int(cx) + win)
                    y1 = min(Hf - th, int(cy) + win)
                    rx, ry = cx, cy                        # 默认: 保持原位
                    if x1 - x0 >= tw and y1 - y0 >= th:
                        roi = f[y0:y1, x0:x1]
                        res = cv2.matchTemplate(roi, tmpl[c], cv2.TM_CCOEFF_NORMED)
                        _, mx, _, loc = cv2.minMaxLoc(res)
                        if mx >= CORR_THRESH:
                            rx = x0 + loc[0] + tw / 2.0
                            ry = y0 + loc[1] + th / 2.0
                    # 时序平滑: 新位置 = 匹配位置与上一帧位置的 EMA
                    cx = SMOOTH * rx + (1.0 - SMOOTH) * cx
                    cy = SMOOTH * ry + (1.0 - SMOOTH) * cy
                    # 自适应模板: 取当前(平滑后)位置的头块, 向模板混入一点
                    px0 = max(0, int(cx) - r0)
                    py0 = max(0, int(cy) - r0)
                    px1 = min(Wf, int(cx) + r0)
                    py1 = min(Hf, int(cy) + r0)
                    if px1 - px0 == tw and py1 - py0 == th:
                        patch = f[py0:py1, px0:px1]
                        tmpl[c] = cv2.addWeighted(tmpl[c], 1.0 - TPL_UPDATE,
                                                  patch, TPL_UPDATE, 0)
                    centers[c] = (cx, cy)
                line += [centers[c][0] / Wf, centers[c][1] / Hf]
            tracks.append(line)

        out[clip] = {"n": len(frames), "t": tracks}
        if k % 200 == 0:
            print("  [%d/%d]" % (k, N), flush=True)

    dst = os.path.join(BASE, "track.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print("完成: %d 个 clip 已追踪 -> %s" % (len(out), dst))


if __name__ == "__main__":
    main()
