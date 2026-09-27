# -*- coding: utf-8 -*-
"""JEPA 二次判端口径诊断: 权重是否真加载 + 输出是否恒定"""
import sys, os, io
sys.stdout.reconfigure(encoding="utf-8")
import torch, numpy as np

JEPA_ROOT = r"D:\jepa-main"
sys.path.insert(0, JEPA_ROOT)
os.chdir(JEPA_ROOT)

JEPA_BACKBONE = r"D:\jepa-main\pretrained_weights\vitl16.pth.tar"
JEPA_HEAD = r"D:\jepa-main\pretrained_weights\video_classification_frozen\ssv2-16x2x3\jepa-best.pth.tar"
FRAMES = 16

import src.models.vision_transformer as vit
import src.models.attentive_pooler as ap

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, torch.__version__)


def cmp_sd(tag, model, ckpt_sd, strip_prefixes):
    sd = dict(ckpt_sd)
    for pre in strip_prefixes:
        sd = {k[len(pre):] if k.startswith(pre) else k: v for k, v in sd.items()}
    mk = set(model.state_dict().keys())
    ck = set(sd.keys())
    miss = sorted(mk - ck)
    unexp = sorted(ck - mk)
    print(f"\n[{tag}] model_keys={len(mk)} ckpt_keys={len(ck)} matched={len(mk & ck)} "
          f"missing={len(miss)} unexpected={len(unexp)}")
    if miss:
        print("  missing 样例:", miss[:5])
    if unexp:
        print("  unexpected 样例:", unexp[:5])
    for k in sorted(mk & ck)[:3]:
        a, b = model.state_dict()[k], sd[k]
        same = a.shape == b.shape
        print(f"  ok {k} model{a.shape} ckpt{tuple(b.shape)} {'一致' if same else '★形状不符★'}")
    return sd, miss, unexp


# ---------- 1. encoder ----------
enc = vit.vit_large(patch_size=16, tubelet_size=2, img_size=224,
                    num_frames=FRAMES, uniform_power=True)
ck = torch.load(JEPA_BACKBONE, map_location="cpu", weights_only=False)
print("\nbackbone ckpt 顶层键:", list(ck.keys())[:10])
raw = ck["target_encoder"]
print("target_encoder 键样例:", list(raw.keys())[:4])

# 试多种前缀写法, 看哪种匹配率最高
for tag, pres in [("module.backbone.", ("module.backbone.",)),
                  ("module.", ("module.",)),
                  ("backbone.", ("backbone.",)),
                  ("原样", ())]:
    sd = dict(raw)
    for pre in pres:
        sd = {k[len(pre):] if k.startswith(pre) else k: v for k, v in sd.items()}
    inter = len(set(enc.state_dict()) & set(sd))
    print(f"  前缀[{tag}] 匹配 {inter}/{len(enc.state_dict())}")

sd, miss, unexp = cmp_sd("backbone", enc, raw,
                         [p for p in ("module.backbone.",) ])
enc.load_state_dict(sd, strict=False)

# ---------- 2. classifier ----------
clf = ap.AttentiveClassifier(embed_dim=1024, num_heads=16, depth=1,
                             num_classes=2, dropout=0.1)
h = torch.load(JEPA_HEAD, map_location="cpu", weights_only=False)
print("\nhead ckpt 顶层键:", list(h.keys())[:10])
csd_raw = h["classifier"]
print("classifier 键样例:", list(csd_raw.keys())[:6])
csd, cmiss, cunexp = cmp_sd("classifier", clf, csd_raw, ("module.",))
clf.load_state_dict(csd)

# ---------- 3. 前向: 输出是否随输入变化 ----------
enc.eval().to(DEV)
clf.eval().to(DEV)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

torch.manual_seed(0)
print("\n--- 前向测试 (随机输入 x3) ---")
with torch.no_grad():
    for t in range(3):
        x = torch.rand(1, 3, FRAMES, 224, 224, device=DEV)
        x = (x - torch.tensor(MEAN, device=DEV).view(1, 3, 1, 1, 1)) / torch.tensor(STD, device=DEV).view(1, 3, 1, 1, 1)
        feat = enc(x)
        print("  encoder 输出类型:", type(feat).__name__,
              "shape:", tuple(feat.shape) if torch.is_tensor(feat) else [tuple(f.shape) for f in feat])
        logits = clf(feat)
        p = torch.softmax(logits.float(), 1)[0]
        print(f"  logits={logits.float().cpu().numpy().round(4)}  P(chat)={float(p[1]):.4f}")

print("\n--- 输入敏感度: 同一输入 vs 轻微扰动 ---")
torch.manual_seed(1)
x0 = torch.rand(1, 3, FRAMES, 224, 224, device=DEV)
with torch.no_grad():
    l0 = clf(enc(x0)).float().cpu().numpy()[0]
    l1 = clf(enc(x0 + 0.01)).float().cpu().numpy()[0]
    l2 = clf(enc(x0 * 0 + 1)).float().cpu().numpy()[0]
print(" logits 原输入      :", l0.round(4))
print(" logits +0.01 扰动  :", l1.round(4))
print(" logits 全 1 常量   :", l2.round(4))
