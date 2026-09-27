# -*- coding: utf-8 -*-
"""扫描各机位: 抽样若干帧, 统计 BPJDet 平均能检出多少人头/配对 -> 挑可用素材"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\chatting\_dataset")

import cv2
import numpy as np
import torch
import sample_pairs as S
import video as V

try:
    from utils.augmentations import letterbox
except ImportError:
    from utils.general import letterbox

N_SAMPLE = 8
IMGSZ_LIST = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["960"])]


def detect_frame(im0, agent, det_model, imgsz):
    img = letterbox(im0, imgsz, stride=S.STRIDE, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    return S.detect_heads(img, im0, agent, det_model)


def gather_videos():
    out = []
    d1 = r"D:\video\1\1\right\cutting_video2"
    if os.path.isdir(d1):
        fs = sorted(f for f in os.listdir(d1) if f.endswith(".mp4"))
        for f in fs[:2]:
            out.append(("用户场景/" + f, os.path.join(d1, f)))
    base = r"D:\video\2"
    for root, _dirs, files in os.walk(base):
        vids = sorted(f for f in files if f.lower().endswith(".mp4"))
        if vids and "cutting" not in root:
            rel = os.path.relpath(root, base)
            out.append((rel + "/" + vids[0], os.path.join(root, vids[0])))
    return out


agent, det_model = S.load_models()
videos = gather_videos()
print(f"\n扫描 {len(videos)} 个机位, 每机位抽 {N_SAMPLE} 帧, "
      f"imgsz={IMGSZ_LIST}\n" + "=" * 96)

for tag, path in videos:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"{tag:48s} [打不开]")
        continue
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if n < 10:
        cap.release()
        continue
    idxs = np.linspace(int(n * 0.05), int(n * 0.95), N_SAMPLE).astype(int)
    res = {}
    for imgsz in IMGSZ_LIST:
        heads_n, pair_n = [], []
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, im0 = cap.read()
            if not ok:
                continue
            hs = detect_frame(im0, agent, det_model, imgsz)
            heads_n.append(len(hs))
            pair_n.append(len(S.seat_pairs(hs, im0.shape[1])))
        res[imgsz] = (np.mean(heads_n) if heads_n else 0,
                      np.sum(pair_n) if pair_n else 0)
    cap.release()
    parts = [f"imgsz{im}: {res[im][0]:5.1f}人/{int(res[im][1])}对"
             for im in IMGSZ_LIST]
    print(f"{tag:48s} {w}x{h} | " + " | ".join(parts))
