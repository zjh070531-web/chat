# -*- coding: utf-8 -*-
"""对照实验: 三种预处理口径下, JEPA 分类头在真实 val 数据上的准确率"""
import sys, os, random
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
SHORT = int(RES * 256 / 224)          # 官方: 短边 256
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

# ---------------- 模型 ----------------
enc = vit.vit_large(patch_size=16, tubelet_size=2, img_size=224,
                    num_frames=FR, uniform_power=True)
ck = torch.load(r"D:\jepa-main\pretrained_weights\vitl16.pth.tar",
                map_location="cpu", weights_only=False)
sd = {k[len("module.backbone."):] if k.startswith("module.backbone.") else k: v
      for k, v in ck["target_encoder"].items()}
enc.load_state_dict(sd, strict=True)
del ck, sd

clf = ap.AttentiveClassifier(embed_dim=1024, num_heads=16, depth=1,
                             num_classes=2, dropout=0.1)
h = torch.load(r"D:\jepa-main\pretrained_weights\video_classification_frozen"
               r"\ssv2-16x2x3\jepa-best.pth.tar", map_location="cpu",
               weights_only=False)
clf.load_state_dict(h["classifier"])
enc.eval().half().to(DEV)
clf.eval().half().to(DEV)
print("[ok] 模型就绪 (主干+分类头, 严格加载)")


def _to_tensor(frames):
    x = np.stack(frames).astype(np.float32) / 255.0
    x = x[..., ::-1]                          # BGR -> RGB
    x = (x - MEAN) / STD
    return torch.from_numpy(x).permute(3, 0, 1, 2).unsqueeze(0).half().to(DEV)


def _short_center(f, box=None):
    """短边缩放 SHORT -> 中心裁剪 RES。box=(x1,y1,x2,y2) 先裁区域。"""
    if box is not None:
        x1, y1, x2, y2 = box
        f = f[y1:y2, x1:x2]
    h, w = f.shape[:2]
    if h < w:
        nh, nw = SHORT, int(round(w * SHORT / h))
    else:
        nw, nh = SHORT, int(round(h * SHORT / w))
    f = cv2.resize(f, (max(nw, 1), max(nh, 1)), interpolation=cv2.INTER_LINEAR)
    nh, nw = f.shape[:2]
    if nh < RES or nw < RES:
        f = cv2.resize(f, (max(nw, RES), max(nh, RES)),
                       interpolation=cv2.INTER_LINEAR)
        nh, nw = f.shape[:2]
    y0, x0 = (nh - RES) // 2, (nw - RES) // 2
    return f[y0:y0 + RES, x0:x0 + RES]


def prep_official(fr):
    return [_short_center(f) for f in fr]


def prep_stretch(fr):
    return [cv2.resize(f, (RES, RES), interpolation=cv2.INTER_LINEAR) for f in fr]


def prep_region(fr):
    """模拟 v1 的'两人区域裁剪': 取中心 60%x50% 的宽扁区域(再按官方缩放)"""
    out = []
    for f in fr:
        h, w = f.shape[:2]
        box = (int(w * 0.20), int(h * 0.25), int(w * 0.80), int(h * 0.75))
        out.append(_short_center(f, box))
    return out


def sample(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    need = STEP * (FR - 1) + 1
    if n < need:
        cap.release()
        return []
    segs = []
    for s in range(NSEG):
        start = int(s * n / NSEG)
        if start + need >= n:
            start = max(0, n - need - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        buf = []
        while len(buf) < need:
            ok, f = cap.read()
            if not ok:
                break
            buf.append(f)
        if len(buf) < need:
            continue
        segs.append([buf[i * STEP] for i in range(FR)])
    cap.release()
    return segs


@torch.no_grad()
def predict(fr, prep):
    t = _to_tensor(prep(fr))
    p = torch.softmax(clf(enc(t)).float(), 1)[0].cpu().numpy()
    return float(p[1])


# ---------------- 取样本 ----------------
val = []
with open(r"D:\jepa-main\my_dataset\val.csv", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        path, lab = line.rsplit(" ", 1)
        val.append((path.strip(), int(lab)))

pos = [v for v in val if v[1] == 1][:6]
neg = [v for v in val if v[1] == 0][:6]
random.seed(0)
random.shuffle(pos); random.shuffle(neg)
samples = [(p, 1) for p, _ in [(x[0], x[1]) for x in pos]] + \
          [(p, 0) for p, _ in [(x[0], x[1]) for x in neg]]
print(f"测试样本: {len(samples)} 个 (6 chat + 6 not_chat), 每视频 {NSEG} 段\n")

PREPS = [("官方(短边256+中心裁剪224)", prep_official),
         ("v1拉伸(区域直接resize224)", prep_stretch),
         ("模拟两人区域(短边缩放+中心裁剪)", prep_region)]

stats = {name: {"ok": 0, "n": 0, "ps": []} for name, _ in PREPS}

for vi, (path, lab) in enumerate(samples):
    segs = sample(path)
    if not segs:
        print(f"  [跳过] {os.path.basename(path)} 帧数不足")
        continue
    print(f"[{vi+1}/{len(samples)}] {os.path.basename(path)}  标签={lab}  段数={len(segs)}")
    for name, prep in PREPS:
        ps = [predict(s, prep) for s in segs]
        m = float(np.mean(ps))
        pred = int(m >= 0.5)
        st = stats[name]
        st["n"] += 1
        st["ok"] += int(pred == lab)
        st["ps"].extend(ps)
        print(f"    {name:32s} 段均值P={m:.3f} 预测={pred} "
              f"{'✓' if pred == lab else '✗'}  (段范围 {min(ps):.3f}~{max(ps):.3f})")

print("\n" + "=" * 72)
for name, _ in PREPS:
    st = stats[name]
    ps = np.array(st["ps"])
    acc = 100.0 * st["ok"] / max(st["n"], 1)
    print(f"{name:32s} 准确率 {st['ok']}/{st['n']} = {acc:.1f}%   "
          f"P(chat) 全域 {ps.min():.3f}~{ps.max():.3f} 均值 {ps.mean():.3f}")
