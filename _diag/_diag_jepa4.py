# -*- coding: utf-8 -*-
"""补充: 裁剪区域越小/越窄, JEPA 的 P(chat) 如何变化 (复现 v1 的 720p 输入口径)"""
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
SHORT = 256
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


@torch.no_grad()
def p_of(fr, fn):
    p = torch.softmax(clf(enc(_t([fn(f) for f in fr]))).float(), 1)[0].cpu().numpy()
    return float(p[1])


def whole(f):
    hh, ww = f.shape[:2]
    if hh < ww: nh, nw = SHORT, int(round(ww * SHORT / hh))
    else: nw, nh = SHORT, int(round(hh * SHORT / ww))
    f = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_LINEAR)
    nh, nw = f.shape[:2]
    return f[(nh - RES) // 2:(nh - RES) // 2 + RES, (nw - RES) // 2:(nw - RES) // 2 + RES]


def crop_stretch(rx, ry, cx, cy):
    def fn(f):
        hh, ww = f.shape[:2]
        x1, x2 = int(ww * (cx - rx / 2)), int(ww * (cx + rx / 2))
        y1, y2 = int(hh * (cy - ry / 2)), int(hh * (cy + ry / 2))
        return cv2.resize(f[y1:y2, x1:x2], (RES, RES), interpolation=cv2.INTER_LINEAR)
    return fn


def crop_official(rx, ry, cx, cy):
    """先裁剪, 再按官方短边缩放+中心裁剪"""
    def fn(f):
        hh, ww = f.shape[:2]
        x1, x2 = int(ww * (cx - rx / 2)), int(ww * (cx + rx / 2))
        y1, y2 = int(hh * (cy - ry / 2)), int(hh * (cy + ry / 2))
        r = f[y1:y2, x1:x2]
        h2, w2 = r.shape[:2]
        if h2 < w2: nh, nw = SHORT, int(round(w2 * SHORT / h2))
        else: nw, nh = SHORT, int(round(h2 * SHORT / w2))
        r = cv2.resize(r, (max(nw, 1), max(nh, 1)), interpolation=cv2.INTER_LINEAR)
        h2, w2 = r.shape[:2]
        if h2 < RES or w2 < RES:
            r = cv2.resize(r, (max(w2, RES), max(h2, RES)), interpolation=cv2.INTER_LINEAR)
            h2, w2 = r.shape[:2]
        return r[(h2 - RES) // 2:(h2 - RES) // 2 + RES, (w2 - RES) // 2:(w2 - RES) // 2 + RES]
    return fn


path = r"D:\video\1\1\right\cutting_video2\segment_007.mp4"
cap = cv2.VideoCapture(path)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
need = STEP * (FR - 1) + 1
segs_4k, segs_720 = [], []
for s in range(NSEG):
    st = int(s * n / NSEG)
    if st + need >= n: st = max(0, n - need - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, st)
    buf = []
    while len(buf) < need:
        ok, f = cap.read()
        if not ok: break
        buf.append(f)
    if len(buf) == need:
        fr = [buf[i * STEP] for i in range(FR)]
        segs_4k.append(fr)
        # 复现 v1: 先整体降采样到 720p 再裁
        segs_720.append([cv2.resize(x, (1280, 720), interpolation=cv2.INTER_LINEAR) for x in fr])
cap.release()
print(f"segment_007 抽到 {len(segs_4k)} 段; 4K 原帧 + 模拟 v1 的 720p 缓冲\n")

CASES = [
    ("整幅画面(官方)", whole, None),
    ("区域45%x45% 拉伸", None, (0.45, 0.45, 0.50, 0.55)),
    ("区域35%x40% 拉伸", None, (0.35, 0.40, 0.50, 0.58)),
    ("区域25%x35% 拉伸", None, (0.25, 0.35, 0.50, 0.60)),
    ("区域18%x28% 拉伸", None, (0.18, 0.28, 0.50, 0.60)),
    ("区域12%x20% 拉伸", None, (0.12, 0.20, 0.50, 0.60)),
    ("两人窄条35%x18% 拉伸", None, (0.35, 0.18, 0.50, 0.55)),
]

for tag, fn0, box in CASES:
    fn = fn0 if fn0 else crop_stretch(*box)
    r4 = [p_of(s, fn) for s in segs_4k]
    r7 = [p_of(s, fn) for s in segs_720]
    print(f"{tag:22s} 4K输入: 均值{np.mean(r4):.3f} 判chat {sum(1 for x in r4 if x>=0.5)}/{len(r4)}"
          f"   | 720p输入(v1): 均值{np.mean(r7):.3f} 判chat {sum(1 for x in r7 if x>=0.5)}/{len(r7)}")

print()
for tag, box in [("区域25%x35%", (0.25, 0.35, 0.50, 0.60)),
                 ("区域18%x28%", (0.18, 0.28, 0.50, 0.60))]:
    fn = crop_official(*box)
    r7 = [p_of(s, fn) for s in segs_720]
    print(f"{tag:22s} 720p+官方缩放: 均值{np.mean(r7):.3f} 判chat {sum(1 for x in r7 if x>=0.5)}/{len(r7)}")
