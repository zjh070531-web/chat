# -*- coding: utf-8 -*-
"""在你实际跑的测试视频上, 对比'整幅画面'vs'两人区域裁剪'两种口径的 JEPA 输出"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import cv2
import torch

JEPA_ROOT = r"D:\jepa-main"
sys.path.insert(0, JEPA_ROOT)
os.chdir(JEPA_ROOT)
import src.models.vision_transformer as vit
import src.models.attentive_pooler as ap

DEV = "cuda"
FR, STEP, NSEG, RES = 16, 4, 8, 224
SHORT = int(RES * 256 / 224)
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

enc = vit.vit_large(patch_size=16, tubelet_size=2, img_size=224, num_frames=FR, uniform_power=True)
ck = torch.load(r"D:\jepa-main\pretrained_weights\vitl16.pth.tar", map_location="cpu", weights_only=False)
enc.load_state_dict({k[len("module.backbone."):] if k.startswith("module.backbone.") else k: v
                     for k, v in ck["target_encoder"].items()}, strict=True)
del ck
clf = ap.AttentiveClassifier(embed_dim=1024, num_heads=16, depth=1, num_classes=2, dropout=0.1)
h = torch.load(r"D:\jepa-main\pretrained_weights\video_classification_frozen\ssv2-16x2x3\jepa-best.pth.tar",
               map_location="cpu", weights_only=False)
clf.load_state_dict(h["classifier"])
enc.eval().half().to(DEV); clf.eval().half().to(DEV)


def _t(frames):
    x = np.stack(frames).astype(np.float32) / 255.0
    x = (x[..., ::-1] - MEAN) / STD
    return torch.from_numpy(x).permute(3, 0, 1, 2).unsqueeze(0).half().to(DEV)


def whole_official(f):
    hh, ww = f.shape[:2]
    if hh < ww: nh, nw = SHORT, int(round(ww * SHORT / hh))
    else: nw, nh = SHORT, int(round(hh * SHORT / ww))
    f = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_LINEAR)
    nh, nw = f.shape[:2]
    y0, x0 = (nh - RES) // 2, (nw - RES) // 2
    return f[y0:y0 + RES, x0:x0 + RES]


def region_v1(f, rx=0.45, ry=0.45, cx=0.5, cy=0.55):
    """模拟 v1: 截取画面中部的'两人并排区域', 直接拉伸到 224"""
    hh, ww = f.shape[:2]
    x1, x2 = int(ww * (cx - rx / 2)), int(ww * (cx + rx / 2))
    y1, y2 = int(hh * (cy - ry / 2)), int(hh * (cy + ry / 2))
    return cv2.resize(f[y1:y2, x1:x2], (RES, RES), interpolation=cv2.INTER_LINEAR)


def whole_stretch(f):
    return cv2.resize(f, (RES, RES), interpolation=cv2.INTER_LINEAR)


def sample(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    need = STEP * (FR - 1) + 1
    segs = []
    for s in range(NSEG):
        start = int(s * n / NSEG)
        if start + need >= n: start = max(0, n - need - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        buf = []
        while len(buf) < need:
            ok, f = cap.read()
            if not ok: break
            buf.append(f)
        if len(buf) == need:
            segs.append([buf[i * STEP] for i in range(FR)])
    cap.release()
    return segs, n


@torch.no_grad()
def pred(fr, fn):
    p = torch.softmax(clf(enc(_t([fn(f) for f in fr]))).float(), 1)[0].cpu().numpy()
    return float(p[1])


BASE = r"D:\video\1\1\right\cutting_video2"
names = ["segment_000.mp4", "segment_003.mp4", "segment_007.mp4"]
MODES = [("整幅画面(官方预处理)", whole_official),
         ("整幅画面(直接拉伸)", whole_stretch),
         ("两人区域裁剪(v1口径)", region_v1)]

for nm in names:
    path = os.path.join(BASE, nm)
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    segs, n = sample(path)
    print(f"\n== {nm}  {w}x{hh} fps={fps:.0f} 帧数={n} 抽到{len(segs)}段")
    for tag, fn in MODES:
        ps = [pred(s, fn) for s in segs]
        print(f"   {tag:22s} 段P: {[round(x,3) for x in ps]}")
        print(f"   {'':22s} 均值={np.mean(ps):.3f}  判chat段数={sum(1 for x in ps if x>=0.5)}/{len(ps)}")
