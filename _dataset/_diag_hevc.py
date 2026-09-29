# -*- coding: utf-8 -*-
"""诊断: HEVC 视频在 cv2 下顺序解码(不 seek)是否正常、能否检出人"""
import os, sys
import cv2
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

V = r"D:\video\1\教室前中-左\教室前中_左_802监控_802监控_20260830170102\教室前中_左_802监控_20260830152959_20260830155245_3127338.mp4"

cap = cv2.VideoCapture(V)
print("fourcc", hex(int(cap.get(cv2.CAP_PROP_FOURCC))), "w", cap.get(3), "h", cap.get(4), "n", cap.get(7))

# 顺序读前 1200 帧(60 秒), 不 seek
ok, f = cap.read()
print("frame0 ok", ok, "shape", None if f is None else f.shape, "mean", None if f is None else round(float(f.mean()), 1))
means = []
for i in range(200):
    ok, f = cap.read()
    if not ok:
        break
    if i % 20 == 0:
        means.append((i + 1, round(float(f.mean()), 1), round(float(f.std()), 1)))
print("seq 采样 mean/std:", means)
cap.release()

# 用 ffmpeg 解一帧出来看
import subprocess, tempfile, os
out = r"D:\chatting\_dataset\_hevc_frame0.png"
subprocess.run([
    r"D:\ffmpeg\ffmpeg-2026-09-07-git-ecc7eb519e-essentials_build\bin\ffmpeg.EXE",
    "-y", "-ss", "20", "-i", V, "-frames:v", "1", out], capture_output=True)
print("ffmpeg frame @20s saved:", os.path.exists(out), os.path.getsize(out) if os.path.exists(out) else 0)
