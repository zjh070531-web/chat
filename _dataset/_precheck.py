# -*- coding: utf-8 -*-
"""只读预检 v2: 顺序读(不 seek)新文件夹 HEVC 视频, 确认能否检出人 + 真·邻座"""
import sys
import cv2
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, r"D:\chatting\_dataset")
import sample_pairs as SP

V = r"D:\video\1\教室前中-左\教室前中_左_802监控_802监控_20260830170102\教室前中_左_802监控_20260830152959_20260830155245_3127338.mp4"

agent, det_model = SP.load_models(1280)
SP.load_pose_model()

cap = cv2.VideoCapture(V)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("fps", fps, "nframes", n, "时长", round(n / fps / 60, 1), "min")

step = int(fps * 5)          # 每 5 秒检一帧
tot_heads = tot_pairs = 0
samples = 0
max_hh = 0
fi = 0
while True:                   # 顺序读, 不 seek
    ok, im0 = cap.read()
    if not ok:
        break
    if fi % step == 0 and fi <= n - 120:
        img_h, img_w = im0.shape[:2]
        img = SP.letterbox(im0, 1280, stride=SP.STRIDE, auto=True)[0]
        img = np.ascontiguousarray(img.transpose((2, 0, 1))[::-1])
        heads = SP.detect_heads(img, im0, agent, det_model)
        samples += 1
        tot_heads += len(heads)
        npairs = 0
        if len(heads) >= 2:
            pairs = list(SP.seat_pairs(heads, img_w))
            npairs = len(pairs)
            tot_pairs += npairs
            for (i, j, gaze, toward) in pairs:
                a, b = heads[i], heads[j]
                hh = ((a["bbox"][3] - a["bbox"][1]) + (b["bbox"][3] - b["bbox"][1])) / 2.0
                max_hh = max(max_hh, hh)
        if fi % (step * 6) == 0:
            print(f"  {fi/fps:6.1f}s: {len(heads):2d} 人, 邻座对 {npairs}")
    fi += 1

cap.release()
print(f"\n平均 {tot_heads/max(1,samples):.1f} 人/帧, 邻座对 {tot_pairs/max(1,samples):.1f}/帧, 最大邻座头高 {max_hh:.0f}px (采样 {samples} 帧)")
