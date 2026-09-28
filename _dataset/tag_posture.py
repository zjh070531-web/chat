# -*- coding: utf-8 -*-
r"""tag_posture.py —— 给现有候选补上"坐/站"身头比标签, 并重排

对每条候选的采样帧重新检测, 找到被点名的两个头(A=ax/ay, B=bx/by),
计算身头比 ra/rb, 判定 posture(0=都坐/1=一坐一站/2=都站),
写回 candidates.csv 并按 (互相朝向, 坐站, 夹角) 重排:
坐坐的排前, 一坐一站的排后。
"""
import csv
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\chatting\_dataset")

import cv2
import numpy as np
import sample_pairs as S

try:
    from utils.augmentations import letterbox
except ImportError:
    from utils.general import letterbox

BASE = r"D:\chatting\_dataset"
IMGSZ = 1280


def main():
    csvp = os.path.join(BASE, "candidates.csv")
    rows = list(csv.DictReader(open(csvp, encoding="utf-8-sig")))
    agent, det_model = S.load_models(IMGSZ)
    N = len(rows)
    done = 0
    for k, r in enumerate(rows, 1):
        # 已有 posture 就跳过(幂等)
        if r.get("posture") not in (None, ""):
            done += 1
            continue
        src = r["src"]
        try:
            fi = int(r["frame"])
            ax, ay = float(r["ax"]), float(r["ay"])
            bx, by = float(r["bx"]), float(r["by"])
        except (TypeError, ValueError):
            r["ra"], r["rb"], r["posture"] = "", "", ""
            continue
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            r["ra"], r["rb"], r["posture"] = "", "", ""
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, im0 = cap.read()
        cap.release()
        if not ok:
            r["ra"], r["rb"], r["posture"] = "", "", ""
            continue
        img = letterbox(im0, IMGSZ, stride=S.STRIDE, auto=True)[0]
        img = np.ascontiguousarray(img.transpose((2, 0, 1))[::-1])
        heads = S.detect_heads(img, im0, agent, det_model)
        if len(heads) < 2:
            r["ra"], r["rb"], r["posture"] = "", "", ""
            continue

        def ratio(h):
            bh = h["body"][3] - h["body"][1]
            hh = h["bbox"][3] - h["bbox"][1]
            return bh / hh if hh > 0 else 0.0

        def near(hx, hy):
            return min(heads, key=lambda h: (h["cx"] - hx) ** 2 + (h["cy"] - hy) ** 2)

        ha, hb = near(ax, ay), near(bx, by)
        ra, rb = ratio(ha), ratio(hb)
        posture = (1 if ra >= S.SIT_STAND_RATIO else 0) + (1 if rb >= S.SIT_STAND_RATIO else 0)
        r["ra"], r["rb"], r["posture"] = "%.2f" % ra, "%.2f" % rb, str(posture)
        if k % 200 == 0:
            print("  [%d/%d]" % (k, N), flush=True)

    # 重排: 互相朝向优先, 组内坐坐(0) > 一坐一站(1) > 都站(2), 再按夹角降序
    def key(r):
        toward = 0 if r.get("toward") == "1" else 1
        try:
            posture = int(r.get("posture") or 0)
        except (TypeError, ValueError):
            posture = 0
        try:
            gaze = float(r.get("gaze") or 0)
        except (TypeError, ValueError):
            gaze = 0
        return (toward, posture, -gaze)

    rows.sort(key=key)
    with open(csvp, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    c = Counter(r.get("posture") for r in rows)
    print("完成: 共 %d 条, 补标 %d 条" % (N, N - done))
    print("posture 分布: 坐坐(0)=%d / 一坐一站(1)=%d / 都站(2)=%d / 未知=%d"
          % (c.get("0", 0), c.get("1", 0), c.get("2", 0), c.get("", 0)))


if __name__ == "__main__":
    main()
