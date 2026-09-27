# -*- coding: utf-8 -*-
"""诊断: 各机位视频里 BPJDet 能检出多少人头、能凑出多少邻座配对"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\chatting\_dataset")

import cv2
import sample_pairs as S
import video as V

VIDEOS = sys.argv[1:] or [
    r"D:\video\2\主考室\主考室1\cutting\segment_000.mp4",
]

agent, det_model = S.load_models()

for vpath in VIDEOS:
    cap = cv2.VideoCapture(vpath)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"\n===== {os.path.basename(vpath)}  {w}x{h} {fps:.0f}fps {n}帧 "
          f"({n / max(fps, 1):.0f}s)")

    ds = V.LoadImages(vpath, img_size=S.IMGSZ_CHK, stride=S.STRIDE, auto=True)
    step = max(1, int(round(10.0 * fps)))     # 每 10 秒看一次, 取更多样本
    stats = []
    for fi, (path, img, im0, _) in enumerate(ds):
        if fi % step:
            continue
        heads = S.detect_heads(img, im0, agent, det_model)
        pairs = S.seat_pairs(heads, im0.shape[1])
        labs = [S.prelabel(g, t) for _, _, g, t in pairs]
        stats.append((fi, len(heads), len(pairs), labs))
        print(f"  f{fi:5d}({fi/max(fps,1):5.1f}s) heads={len(heads):3d} "
              f"seat_pairs={len(pairs)} pre={labs}")
    if stats:
        import numpy as np
        hh = np.array([s[1] for s in stats])
        pp = np.array([s[2] for s in stats])
        n0 = sum(sum(1 for x in s[3] if x == 0) for s in stats)
        n1 = sum(sum(1 for x in s[3] if x == 1) for s in stats)
        n2 = sum(sum(1 for x in s[3] if x == 2) for s in stats)
        print(f"  --- 采样 {len(stats)} 次: 平均 heads={hh.mean():.1f} "
              f"平均 pairs={pp.mean():.2f} | 累计 正常{n0} 强可疑{n1} 待定{n2}")
