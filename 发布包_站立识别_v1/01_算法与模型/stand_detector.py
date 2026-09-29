#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
站立识别算法 (Standing Posture Recognition)
===========================================

输入 : 视频文件（或摄像头 ``0``）
输出 : 渲染后的视频（骨架 + 人体框 + 站立/非站立标签 + 统计 HUD）

算法流程
--------
1. 用本目录下的 ``yolo26s-pose.pt`` 做人体检测 + 17 点 COCO 姿态估计，并用 ByteTrack
   给每个人分配稳定的 track id（用于时序平滑）。
2. 对每个人计算 6 个可解释的几何特征，线性归一化到 [0,1]：

   ==================== ============================================== ======
   特征                  站立时应有的表现                               权重
   ==================== ============================================== ======
   torso_verticality    躯干向量与"竖直向上"的夹角余弦（越接近 1 越直）     3.0
   leg_drop_ratio       |髋-踝垂直距离| / 躯干长（站立≈1.6，坐姿≈0.8）       3.0
   knee_angle           膝盖夹角（站立≈170°，坐姿≈90°）                  2.0
   body_extension       |头-踝垂直距离| / 躯干长（站立≈3.1，坐姿≈2.0）      2.0
   bbox_aspect          人体框高/宽（站立≈2.5-3.5，坐姿≈1.3-1.8）         1.5
   hip_height_ratio     |髋-踝垂直距离| / 人体框高（站立≈0.5）              1.0
   ==================== ============================================== ======

3. 加权平均得到"站立置信度" score ∈ [0,1]（仅对可见特征做归一化，缺失特征不投票，
   同时给出 coverage 覆盖率；覆盖不足时标记为"未知"或"低置信"）。
4. 规则判定：躯干近水平 + 人体框偏宽 → 躺卧；score ≥ 阈值 → 站立；否则 → 坐/蹲。
5. 时序平滑：对每个 track id 的 score 做滑动窗口均值 + EMA，并用滞回阈值
   (enter=0.60 / exit=0.45, 连续 N 帧确认) 抑制抖动，记录每人"站立时间段"。

用法示例
--------
python stand_detector.py --source input.mp4 --output output/result.mp4
python stand_detector.py --source input.mp4 --show          # 边处理边预览
python stand_detector.py --source 0                         # 摄像头实时
python stand_detector.py --source input.mp4 --csv out.csv --json out.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import temporal
import video_io
import perspective

# Windows 控制台（GBK 等）下避免中文打印抛 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

# --------------------------------------------------------------------------- #
# 常量定义
# --------------------------------------------------------------------------- #

KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

# COCO 17 点骨架连线
SKELETON = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6), (0, 1), (0, 2), (1, 3), (2, 4),
)
TORSO_EDGES = {(5, 11), (6, 12), (11, 12), (5, 6)}
LEG_EDGES = {(11, 13), (13, 15), (12, 14), (14, 16)}
FACE_IDX = (0, 1, 2, 3, 4)

L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

STANDING, SITTING, LYING, UNKNOWN = "STANDING", "SITTING", "LYING", "UNKNOWN"
LABEL_CN = {STANDING: "站立", SITTING: "坐/蹲", LYING: "躺卧", UNKNOWN: "未知"}
LABEL_EN = {
    STANDING: "STANDING",
    SITTING: "SITTING / CROUCH",
    LYING: "LYING",
    UNKNOWN: "UNKNOWN",
}
# BGR 颜色
LABEL_COLOR = {
    STANDING: (80, 220, 60),
    SITTING: (60, 170, 255),
    LYING: (80, 80, 235),
    UNKNOWN: (175, 175, 175),
}

# 特征权重与归一化区间 (low, high)：特征值 <= low 记 0 分, >= high 记 1 分
FEATURE_WEIGHTS = OrderedDict((
    ("torso_verticality", 3.0),
    ("leg_drop_ratio", 3.0),
    ("knee_angle", 2.0),
    ("body_extension", 2.0),
    ("bbox_aspect", 1.5),
    ("hip_height_ratio", 1.0),
))
FEATURE_RANGE = {
    "torso_verticality": (0.55, 0.93),   # cos(躯干与竖直方向夹角)
    "leg_drop_ratio": (0.95, 1.35),      # 髋->踝垂直距离 / 躯干长
    "knee_angle": (120.0, 160.0),        # 膝角(度)
    "body_extension": (2.00, 2.70),      # 头->踝垂直距离 / 躯干长
    "bbox_aspect": (1.30, 2.10),         # 人体框 高/宽
    "hip_height_ratio": (0.30, 0.50),    # 髋->踝垂直距离 / 框高
}
TOTAL_WEIGHT = float(sum(FEATURE_WEIGHTS.values()))

MIN_COVERAGE = 0.30        # 低于此覆盖率 -> 未知
LOW_COVERAGE = 0.50        # 低于此覆盖率 -> 结果标记为"低置信(推定)"
# 腿部证据（髋-踝跨度 / 膝角 / 全身伸展比）是否存在。仅凭"躯干竖直"无法区分
# "站立"与"坐在桌后"，因此默认在缺少腿部证据时给出"未知"。
LEG_EVIDENCE_FEATURES = ("leg_drop_ratio", "knee_angle", "body_extension")

WINDOW_TITLE = "Standing Recognition - yolo26s-pose"


# --------------------------------------------------------------------------- #
# 1. 姿态几何特征 + 站立判定
# --------------------------------------------------------------------------- #

def _ramp(value: float, low: float, high: float) -> float:
    """把 value 线性映射到 [0,1]：<=low → 0，>=high → 1。"""
    if value is None or not math.isfinite(float(value)):
        return 0.0
    if high == low:
        return 1.0 if value >= high else 0.0
    return float(min(1.0, max(0.0, (value - low) / (high - low))))


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
    """b 为顶点，返回角 ABC 的度数（直线时 ≈180）。"""
    v1 = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    v2 = np.asarray(c, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _midpoint(kpts: np.ndarray, i: int, j: int, kconf: float) -> Tuple[Optional[np.ndarray], float]:
    """左右对称点的中点；只有一侧可见时退化为该侧点。"""
    ci, cj = float(kpts[i, 2]), float(kpts[j, 2])
    vi, vj = ci >= kconf, cj >= kconf
    if vi and vj:
        return (kpts[i, :2].astype(np.float32) + kpts[j, :2].astype(np.float32)) / 2.0, min(ci, cj)
    if vi:
        return kpts[i, :2].astype(np.float32), ci
    if vj:
        return kpts[j, :2].astype(np.float32), cj
    return None, 0.0


@dataclass
class PostureResult:
    label: str
    score: float                 # 站立置信度 0~1
    coverage: float              # 特征覆盖率 0~1
    leg_evidence: bool = False   # 是否有关键的腿部证据
    raw_score: float = 0.0       # 未做透视补偿的分数（用于"原本是否已达标"判定）
    persp: Tuple[float, float] = (1.0, 1.0)   # 实际施加的 (k_leg, k_asp)
    features: Dict[str, float] = field(default_factory=dict)
    points: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def low_confidence(self) -> bool:
        return self.coverage < LOW_COVERAGE


def classify_pose(
    keypoints: np.ndarray,
    box: Sequence[float],
    kconf: float = 0.35,
    stand_threshold: float = 0.60,
    require_leg_evidence: bool = True,
    persp: Optional[Tuple[float, float, float]] = None,
) -> PostureResult:
    """由 17 个 COCO 关键点判定"站立 / 坐蹲 / 躺卧 / 未知"。

    Parameters
    ----------
    keypoints : (17, 3) 数组, 每行为 (x, y, confidence)
    box       : 人体检测框 (x1, y1, x2, y2)
    kconf     : 关键点可见性阈值
    require_leg_evidence : True 时，若看不到腿（髋/膝/踝被遮挡）则判为"未知"，
                           避免把"坐在桌后只露上半身"的人误判为站立。
    persp     : (k_leg, k_ext, k_asp) 透视补偿系数；k>=1，只放大被俯视透视压扁的
                纵向特征（髋-踝跨度、全身伸展比、框高宽比），不改变视角不变特征。
    """
    k = np.asarray(keypoints, dtype=np.float32).reshape(-1, 3)
    x1, y1, x2, y2 = (float(v) for v in box)
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)

    pts: Dict[str, np.ndarray] = {}
    feats: Dict[str, float] = {}

    sh_mid, _ = _midpoint(k, L_SHOULDER, R_SHOULDER, kconf)
    hip_mid, _ = _midpoint(k, L_HIP, R_HIP, kconf)
    knee_mid, _ = _midpoint(k, L_KNEE, R_KNEE, kconf)
    ank_mid, _ = _midpoint(k, L_ANKLE, R_ANKLE, kconf)

    head: Optional[np.ndarray] = None
    for i in FACE_IDX:
        if float(k[i, 2]) >= kconf:
            p = k[i, :2].astype(np.float32)
            if head is None or p[1] < head[1]:
                head = p
    if head is None:
        head = sh_mid

    for name, p in (("shoulder", sh_mid), ("hip", hip_mid),
                    ("knee", knee_mid), ("ankle", ank_mid), ("head", head)):
        if p is not None:
            pts[name] = np.asarray(p, dtype=np.float32)

    # --- 特征 1: 躯干竖直度 ------------------------------------------------
    if sh_mid is not None and hip_mid is not None:
        v = sh_mid - hip_mid                       # 图像坐标 y 向下，向上 => v_y < 0
        torso_len = float(np.linalg.norm(v))
        if torso_len > 1.0:
            feats["torso_verticality"] = float(-v[1] / torso_len)
            feats["torso_angle_deg"] = float(
                math.degrees(math.acos(max(-1.0, min(1.0, feats["torso_verticality"]))))
            )
            feats["torso_len"] = torso_len

    torso_len = feats.get("torso_len")

    # --- 特征 2: 髋-踝垂直跨度 / 躯干长 -------------------------------------
    if torso_len and ank_mid is not None and hip_mid is not None:
        feats["leg_drop_ratio"] = float((ank_mid[1] - hip_mid[1]) / torso_len)

    # --- 特征 3: 膝角 ------------------------------------------------------
    angles: List[float] = []
    for hip_i, knee_i, ank_i in ((L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE)):
        if min(float(k[hip_i, 2]), float(k[knee_i, 2]), float(k[ank_i, 2])) >= kconf:
            a = _angle_deg(k[hip_i, :2], k[knee_i, :2], k[ank_i, :2])
            if a is not None:
                angles.append(a)
    if angles:
        feats["knee_angle"] = float(sum(angles) / len(angles))

    # --- 特征 4: 全身伸展比 (头->踝)/躯干长 ---------------------------------
    if torso_len and ank_mid is not None and head is not None:
        feats["body_extension"] = float((ank_mid[1] - head[1]) / torso_len)

    # --- 特征 5: 人体框高宽比 ----------------------------------------------
    feats["bbox_aspect"] = float(bh / bw)
    feats["bbox_w"], feats["bbox_h"] = bw, bh

    # --- 特征 6: 髋部离地比 (髋->踝)/框高 -----------------------------------
    if ank_mid is not None and hip_mid is not None:
        feats["hip_height_ratio"] = float((ank_mid[1] - hip_mid[1]) / bh)

    # --- 透视补偿（k>=1，只放大纵向压缩特征）-------------------------------
    k_leg = k_ext = k_asp = 1.0
    if persp is not None:
        k_leg, k_ext, k_asp = (float(v) for v in persp)
        feats = perspective.apply_perspective(feats, k_leg, k_ext, k_asp)

    # --- 加权打分 ----------------------------------------------------------
    num, den = 0.0, 0.0
    for name, weight in FEATURE_WEIGHTS.items():
        v = feats.get(name)
        if v is None:
            continue
        lo, hi = FEATURE_RANGE[name]
        num += weight * _ramp(v, lo, hi)
        den += weight
    coverage = den / TOTAL_WEIGHT
    score = (num / den) if den > 0 else 0.0
    raw_score = score
    if persp is not None:
        # 未补偿分数（供"原本是否已达标"判定，保证不挤掉现有标注）
        rn, rd = 0.0, 0.0
        for name, weight in FEATURE_WEIGHTS.items():
            cur = feats.get(name)
            if cur is None:
                continue
            v = (cur / (k_leg if name == "leg_drop_ratio" else
                        k_ext if name == "body_extension" else
                        k_asp if name == "bbox_aspect" else 1.0))
            lo, hi = FEATURE_RANGE[name]
            rn += weight * _ramp(v, lo, hi)
            rd += weight
        raw_score = (rn / rd) if rd > 0 else 0.0

    # --- 规则判定 ----------------------------------------------------------
    aspect = feats["bbox_aspect"]
    torso_cos = feats.get("torso_verticality")
    leg_evidence = any(name in feats for name in LEG_EVIDENCE_FEATURES)
    if coverage < MIN_COVERAGE or (require_leg_evidence and not leg_evidence):
        label = UNKNOWN
    elif (torso_cos is not None and torso_cos < 0.30 and aspect < 1.60) or (
            torso_cos is None and aspect < 0.95):
        label = LYING
    elif score >= stand_threshold:
        label = STANDING
    else:
        label = SITTING

    return PostureResult(label=label, score=float(score), coverage=float(coverage),
                         leg_evidence=leg_evidence, raw_score=float(raw_score),
                         persp=(float(k_leg), float(k_asp)),
                         features=feats, points=pts)


# --------------------------------------------------------------------------- #
# 2. 时序平滑 + 状态机（每个 track 一个实例）
# --------------------------------------------------------------------------- #

class PostureSmoother:
    """滑动窗口均值 + EMA + 滞回阈值，抑制逐帧抖动并记录稳定状态。"""

    def __init__(self, window: int = 15, min_confirm: int = 3, min_release: int = 5,
                 stand_threshold: float = 0.60, release_threshold: float = 0.45,
                 ema_alpha: float = 0.35):
        self.window = max(1, int(window))
        self.min_confirm = max(1, int(min_confirm))
        self.min_release = max(1, int(min_release))
        self.stand_threshold = float(stand_threshold)
        self.release_threshold = float(release_threshold)
        self.ema_alpha = float(ema_alpha)
        self.buf: deque = deque(maxlen=self.window)
        self.ema: Optional[float] = None
        self.value: Optional[float] = None      # 融合后的平滑值（供时序状态机作为观测）
        self.smooth: Optional[float] = None
        self.is_standing = False
        self._above = 0
        self._below = 0

    def update(self, score: float) -> Tuple[bool, Optional[str]]:
        """返回 (当前是否站立, 事件 'on'/'off'/None)。"""
        self.buf.append(float(score))
        self.ema = score if self.ema is None else (
            self.ema_alpha * score + (1.0 - self.ema_alpha) * self.ema
        )
        smooth = float(np.mean(self.buf))
        value = 0.5 * smooth + 0.5 * float(self.ema)   # 均值与 EMA 各占一半，兼顾稳定与响应
        self.smooth = smooth
        self.value = value

        event: Optional[str] = None
        if not self.is_standing:
            if value >= self.stand_threshold:
                self._above += 1
                self._below = 0
                if self._above >= self.min_confirm:
                    self.is_standing = True
                    self._above = 0
                    event = "on"
            else:
                self._above = 0
        else:
            if value < self.release_threshold:
                self._below += 1
                self._above = 0
                if self._below >= self.min_release:
                    self.is_standing = False
                    self._below = 0
                    event = "off"
            else:
                self._below = 0
        return self.is_standing, event

    def force_off(self) -> bool:
        """强制退出站立状态（用于"长时间没有有效证据"的情况）。返回是否发生变化。"""
        self._above = self._below = 0
        if self.is_standing:
            self.is_standing = False
            return True
        return False


@dataclass
class TrackRecord:
    """单个跟踪目标的统计信息。"""

    tid: int
    first_t: float
    last_t: float
    n_frames: int = 0
    n_standing: int = 0
    s2_frames: int = 0
    held_frames: int = 0
    state_frames: List[int] = field(default_factory=lambda: [0, 0, 0])
    best_score: float = 0.0
    last_label: str = UNKNOWN
    intervals: List[List[float]] = field(default_factory=list)
    fps: float = 20.0

    def summary(self) -> dict:
        intervals = [
            {"start": round(a, 3), "end": round(b, 3), "duration": round(b - a, 3)}
            for a, b in self.intervals if b > a
        ]
        total = sum(i["duration"] for i in intervals)
        fps = max(1e-6, self.fps)
        return {
            "track_id": self.tid,
            "first_seen": round(self.first_t, 3),
            "last_seen": round(self.last_t, 3),
            "frames": self.n_frames,
            "standing_frames": self.n_standing,
            "s2_frames": self.s2_frames,
            "held_frames": self.held_frames,
            "state_seconds": {name: round(self.state_frames[i] / fps, 2)
                              for i, name in enumerate(("S0_非站立", "S1_过渡", "S2_站立"))},
            "standing_ratio": round(self.n_standing / self.n_frames, 4) if self.n_frames else 0.0,
            "standing_time": round(total, 3),
            "max_standing_score": round(self.best_score, 4),
            "final_label": LABEL_EN[self.last_label],
            "standing_intervals": intervals,
        }


# --------------------------------------------------------------------------- #
# 3. 中文文字绘制（OpenCV 不支持 CJK，用 PIL 生成小图再混合）
# --------------------------------------------------------------------------- #

class TextPainter:
    FONT_CANDIDATES = (
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\Deng.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    )

    def __init__(self, font_path: Optional[str] = None, cache_size: int = 2048):
        from PIL import Image, ImageDraw, ImageFont

        self._Image, self._ImageDraw, self._ImageFont = Image, ImageDraw, ImageFont
        self.font_path = font_path
        if self.font_path is None:
            for cand in self.FONT_CANDIDATES:
                if os.path.exists(cand):
                    self.font_path = cand
                    break
        self._fonts: Dict[int, object] = {}
        self._cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._measure_cache: Dict[tuple, Tuple[int, int]] = {}
        self._cache_size = cache_size

    @property
    def available(self) -> bool:
        return self.font_path is not None

    def _font(self, size: int):
        size = max(8, int(size))
        font = self._fonts.get(size)
        if font is None:
            try:
                font = self._ImageFont.truetype(self.font_path, size)
            except Exception:
                font = self._ImageFont.load_default()
            self._fonts[size] = font
        return font

    def measure(self, text: str, size: int) -> Tuple[int, int]:
        key = (text, int(size))
        cached = self._measure_cache.get(key)
        if cached is not None:
            return cached
        if not self.available:
            w, h = 0, 0
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size / 30.0, 2)
            result = (int(w), int(h))
            self._measure_cache[key] = result
            return result
        font = self._font(size)
        try:
            l, t, r, b = font.getbbox(text)
        except Exception:
            l, t, r, b = (0, 0, size * len(text), size)
        result = (int(max(1, r - l)), int(max(1, b - t)))
        self._measure_cache[key] = result
        return result

    def tile(self, text: str, size: int, color: Sequence[int], stroke: int = 1) -> np.ndarray:
        """生成 BGRA 文字小图（带黑色描边），带缓存。"""
        key = (text, int(size), tuple(int(c) for c in color), int(stroke))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        b, g, r = int(color[0]), int(color[1]), int(color[2])
        sw = max(0, int(stroke))
        if self.available:
            font = self._font(size)
            try:
                l, t, rr, bb = font.getbbox(text)
            except Exception:
                l, t, rr, bb = (0, 0, size * len(text), size)
            pad = sw + 2
            w, h = int(rr - l) + pad * 2, int(bb - t) + pad * 2
            img = self._Image.new("RGBA", (max(1, w), max(1, h)), (0, 0, 0, 0))
            draw = self._ImageDraw.Draw(img)
            draw.text((pad - l, pad - t), text, font=font, fill=(r, g, b, 255),
                      stroke_width=sw, stroke_fill=(0, 0, 0, 235) if sw else None)
        else:
            scale = size / 30.0
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, sw + 1))
            pad = sw + 2
            img = self._Image.new("RGBA", (tw + pad * 2, th + base + pad * 2), (0, 0, 0, 0))
            draw = self._ImageDraw.Draw(img)
            draw.text((pad, pad), text, fill=(r, g, b, 255))

        arr = np.asarray(img)                       # RGBA
        bgra = np.ascontiguousarray(arr[:, :, [2, 1, 0, 3]])
        self._cache[key] = bgra
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return bgra

    def paste(self, img: np.ndarray, text: str, org: Sequence[int], size: int = 22,
              color: Sequence[int] = (255, 255, 255), stroke: int = 1,
              anchor: str = "lt") -> Tuple[int, int]:
        """把文字混合到 BGR 图像上，返回实际绘制区域左上角坐标。"""
        tile = self.tile(text, size, color, stroke)
        th, tw = tile.shape[:2]
        x, y = int(org[0]), int(org[1])
        if anchor[0] == "c":
            x -= tw // 2
        elif anchor[0] == "r":
            x -= tw
        if anchor[1] == "b":
            y -= th
        elif anchor[1] == "c":
            y -= th // 2

        H, W = img.shape[:2]
        xs, ys = max(0, -x), max(0, -y)
        xe = tw - max(0, (x + tw) - W)
        ye = th - max(0, (y + th) - H)
        if xe <= xs or ye <= ys:
            return x, y

        sub = tile[ys:ye, xs:xe]
        x0, y0 = max(0, x), max(0, y)
        roi = img[y0:y0 + (ye - ys), x0:x0 + (xe - xs)]
        alpha = sub[:, :, 3:4].astype(np.float32) / 255.0
        blended = roi.astype(np.float32) * (1.0 - alpha) + sub[:, :, :3].astype(np.float32) * alpha
        roi[:] = blended.astype(np.uint8)
        return x, y


# --------------------------------------------------------------------------- #
# 4. 渲染
# --------------------------------------------------------------------------- #

def draw_skeleton(img: np.ndarray, kpts: np.ndarray, kconf: float, color: Sequence[int],
                  lw: int = 2) -> None:
    color = tuple(int(c) for c in color)
    k = np.asarray(kpts, dtype=np.float32)
    for i, j in SKELETON:
        if float(k[i, 2]) < kconf or float(k[j, 2]) < kconf:
            continue
        p1 = (int(k[i, 0]), int(k[i, 1]))
        p2 = (int(k[j, 0]), int(k[j, 1]))
        if (i, j) in TORSO_EDGES:
            w, c = lw + 2, color
        elif (i, j) in LEG_EDGES:
            w, c = lw + 1, color
        else:
            w, c = lw, tuple(int(v * 0.85) for v in color)
        cv2.line(img, p1, p2, (0, 0, 0), w + 2, cv2.LINE_AA)
        cv2.line(img, p1, p2, c, w, cv2.LINE_AA)

    for idx in range(k.shape[0]):
        conf = float(k[idx, 2])
        if conf < kconf:
            continue
        p = (int(k[idx, 0]), int(k[idx, 1]))
        radius = lw + 2 if conf >= 0.7 else lw
        cv2.circle(img, p, radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, p, radius, color, -1, cv2.LINE_AA)


def draw_box(img: np.ndarray, box: Sequence[float], color: Sequence[int], lw: int = 2,
             dashed: bool = False) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    color = tuple(int(c) for c in color)
    if not dashed:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), lw + 2, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, lw, cv2.LINE_AA)
        return
    seg = max(10, lw * 6)
    for x in range(x1, x2, seg * 2):
        cv2.line(img, (x, y1), (min(x + seg, x2), y1), color, lw, cv2.LINE_AA)
        cv2.line(img, (x, y2), (min(x + seg, x2), y2), color, lw, cv2.LINE_AA)
    for y in range(y1, y2, seg * 2):
        cv2.line(img, (x1, y), (x1, min(y + seg, y2)), color, lw, cv2.LINE_AA)
        cv2.line(img, (x2, y), (x2, min(y + seg, y2)), color, lw, cv2.LINE_AA)


def _alpha_rect(img: np.ndarray, rect: Sequence[int], color: Sequence[int], alpha: float = 0.55,
                radius: int = 6) -> None:
    x1, y1, x2, y2 = (int(v) for v in rect)
    H, W = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.empty_like(roi)
    overlay[:] = np.array(color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, roi)


def draw_person(img: np.ndarray, painter: TextPainter, box: Sequence[float],
                kpts: np.ndarray, result: PostureResult, is_standing: bool,
                smooth_value: float, state: int, kconf: float,
                args: argparse.Namespace) -> None:
    """绘制单人：框 + 骨架 + 标签 + 置信度条（标签不含编号）。"""
    color = LABEL_COLOR[result.label]
    if state == temporal.S2:
        # 时序状态判定为"稳定站立(S2)"时统一用站立色，保证视觉连续
        color = LABEL_COLOR[STANDING]

    x1, y1, x2, y2 = (float(v) for v in box)
    if args.skeleton:
        draw_skeleton(img, kpts, kconf, color, lw=max(1, args.line_width - 1))

    # 躯干主轴（可视化"竖直度"）
    hip = result.points.get("hip")
    shoulder = result.points.get("shoulder")
    if args.skeleton and hip is not None and shoulder is not None:
        v = shoulder - hip
        n = float(np.linalg.norm(v))
        if n > 1e-3:
            v = v / n
            p1 = hip + v * n * 0.7
            p2 = shoulder + v * n * 1.1
            cv2.arrowedLine(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                            (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)

    draw_box(img, box, color, lw=args.line_width, dashed=result.low_confidence)

    # 标签条（不含编号）：S2 状态统一按"站立"显示，避免绿框配"未知/坐蹲"字样
    bucket = round(min(1.0, max(0.0, smooth_value if state == temporal.S2 else result.score))
                   / 0.05) * 0.05
    if state == temporal.S2:
        label = f"Standing {bucket:.2f}"
    else:
        label = f"{LABEL_CN[result.label]} {bucket:.2f}"
        if result.low_confidence:
            label += " ?"
        elif is_standing != (result.label == STANDING):
            label += " ~"
    size = max(14, args.text_size)
    tw, th = painter.measure(label, size)
    pad_x, pad_y = 8, 5
    chip_w, chip_h = tw + pad_x * 2, th + pad_y * 2
    cx = int(x1)
    cy = int(y1 - chip_h - 8)
    if cy < 0:
        cy = int(y1 + 6)
    _alpha_rect(img, (cx, cy, cx + chip_w, cy + chip_h), color, alpha=0.78)
    text_color = (255, 255, 255) if sum(color) < 480 else (16, 16, 16)
    painter.paste(img, label, (cx + pad_x, cy + pad_y), size=size, color=text_color, stroke=0)



def draw_hud(img: np.ndarray, painter: TextPainter, lines: Sequence[Tuple[str, Tuple[int, int, int]]],
             legend: bool = True, pad: int = 12, size: int = 20) -> None:
    """左上角半透明信息面板 + 左下角颜色图例。"""
    widths, heights = [], []
    for text, _ in lines:
        w, h = painter.measure(text, size)
        widths.append(w)
        heights.append(h + 6)
    panel_w = max(widths) + pad * 2 if widths else 0
    panel_h = sum(heights) + pad * 2
    _alpha_rect(img, (10, 10, 10 + panel_w, 10 + panel_h), (25, 25, 25), alpha=0.6)
    cv2.line(img, (10, 10), (10 + panel_w, 10), (0, 200, 255), 3, cv2.LINE_AA)
    y = 10 + pad
    for (text, color), h in zip(lines, heights):
        painter.paste(img, text, (10 + pad, y), size=size, color=color, stroke=1)
        y += h

    if legend:
        items = [(LABEL_CN[k], LABEL_COLOR[k]) for k in (STANDING, SITTING, LYING, UNKNOWN)]
        box_w, box_h, gap = 16, 16, 10
        sizes = [painter.measure(t, size - 4)[0] for t, _ in items]
        total = sum(24 + s + 24 for s in sizes) + gap * (len(items) - 1)
        H, W = img.shape[:2]
        x = 12
        y = H - 24 - box_h
        _alpha_rect(img, (x - 6, y - 8, x + total, y + box_h + 8), (25, 25, 25), alpha=0.55)
        for (text, color), _ in zip(items, sizes):
            cv2.rectangle(img, (x, y), (x + box_w, y + box_h), color, -1)
            cv2.rectangle(img, (x, y), (x + box_w, y + box_h), (0, 0, 0), 1)
            painter.paste(img, text, (x + box_w + 6, y + box_h // 2), size=size - 4,
                          color=(240, 240, 240), stroke=1, anchor="lc")
            x += box_w + 6 + painter.measure(text, size - 4)[0] + 24


# --------------------------------------------------------------------------- #
# 5. 视频读写
# --------------------------------------------------------------------------- #

def _precision_kwargs(use_half: bool) -> dict:
    """精度参数：ultralytics ≥8.4 用 quantize=16 表示 FP16，旧版本用 half=True。"""
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
        if "quantize" in DEFAULT_CFG_DICT:
            return {"quantize": 16 if use_half else None}
    except Exception:
        pass
    return {"half": bool(use_half)}


@dataclass
class BridgeEntry:
    """一条轨迹的时序状态：Viterbi 解码器 + 末次观测缓存（用于漏检补框）。"""

    tid: int
    viterbi: "temporal.TrackViterbi"
    smoother: PostureSmoother
    raw_smoother: PostureSmoother   # 未做透视补偿的分数平滑（仅用于"原本是否达标"）
    rec: TrackRecord
    state: int = temporal.S0
    prev_state: int = temporal.S0
    last_idx: int = -1
    last_t: float = 0.0
    last_box: Optional[np.ndarray] = None
    last_kpts: Optional[np.ndarray] = None
    last_result: Optional[PostureResult] = None
    closed: bool = False
    neutral_run: int = 0        # 连续"检测到但无腿部证据"的帧数
    det_count: int = 0          # 该轨迹累计的真实检测帧数（用于冷启动/补检准入）
    next_recheck: int = 0       # 下次允许补检的帧号（失败后退避）
    recheck_hits: int = 0       # 补检成功次数
    last_confirmed_idx: int = -1   # 最近一次"真实检测/补检命中"的帧号
    was_standing: bool = False     # 上一帧是否按站立输出（用于时间段边缘判定）


@dataclass
class Candidate:
    """本帧一个待绘制/待计数的人（用于帧级去重与统一输出）。"""

    tid: int
    box: np.ndarray
    kpts: np.ndarray
    result: PostureResult
    state: int
    smooth: float
    held: int
    priority: int          # 2=真实检测/补检命中，1=沿用（补框）
    is_standing: bool      # 仅 --no-track 时直接给定；有追踪时由 _apply_state 判定
    entry: Optional["BridgeEntry"] = None
    confirmed_idx: int = -1
    first_idx: int = 0
    was_passing: bool = False   # 不做补偿时该目标是否已达标注门槛（去重保护用）


def kpt_out_ratio(kpts: np.ndarray, width: int, height: int, kconf: float) -> float:
    """可见关键点中越出画面的比例（识别被画面切断的近距目标）。"""
    if kpts is None or len(kpts) == 0:
        return 1.0
    vis = np.asarray(kpts)[:, 2] >= kconf
    if not vis.any():
        return 1.0
    x, y = np.asarray(kpts)[vis, 0], np.asarray(kpts)[vis, 1]
    out = (x < 0) | (x > width - 1) | (y < 0) | (y > height - 1)
    return float(out.mean())


def dedup_candidates(cands: List["Candidate"], iou_thr: float
                     ) -> Tuple[List["Candidate"], List["Candidate"]]:
    """帧级去重（标注层 NMS）：按优先级贪心抑制重叠框。

    返回 (保留, 抑制)。优先级：真实检测/补检命中(2) > 沿用帧(1)；
    同级按平滑分数降序、最近确认帧更新者优先、track id 更小者（更早出现）优先。
    ``priority < 1`` 的候选直接判为不展示。
    """
    kept: List[Candidate] = []
    hidden: List[Candidate] = [c for c in cands if c.priority < 1]
    eligible = [c for c in cands if c.priority >= 1]
    for c in sorted(eligible, key=lambda x: (-x.priority, not x.was_passing, -x.smooth,
                                             -x.confirmed_idx, x.tid)):
        if any(temporal.iou(c.box, k.box) > iou_thr for k in kept):
            hidden.append(c)
        else:
            kept.append(c)
    return kept, hidden


def _build_tracker(tracker_yaml: str, device: str):
    """构造 ultralytics BYTETracker（批量推理时由本程序逐帧驱动）。"""
    from ultralytics.trackers import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace, YAML
    from ultralytics.utils.checks import check_yaml
    cfg = IterableSimpleNamespace(**YAML.load(check_yaml(tracker_yaml)))
    try:
        import torch
        cfg.device = torch.device("cpu" if device == "cpu" else f"cuda:{device}")
    except Exception:
        cfg.device = None
    tracker_cls = BYTETracker
    return tracker_cls(cfg)


def _apply_tracker(tracker, result, frame: np.ndarray):
    """把一帧检测结果送进跟踪器，并把 track id 写回 Results（与 ultralytics 内部逻辑一致）。

    跟踪器返回 [x1, y1, x2, y2, track_id, score, cls, 检测索引]，
    据此先按索引切片 Results（同时切 boxes/keypoints），再写回带 id 的 boxes。
    """
    import torch

    det = result.boxes.cpu().numpy()
    tracks = tracker.update(det, frame)
    if tracks is None or len(tracks) == 0:
        result.update(boxes=torch.zeros((0, 7), dtype=torch.float32))
        return result
    keep = tracks[:, -1].astype(int)
    result = result[keep]
    result.update(boxes=torch.as_tensor(tracks[:, :-1]))
    return result


def _lost_track_boxes(tracker) -> Dict[int, np.ndarray]:
    """取跟踪器里"丢失但仍存活"轨迹的卡尔曼预测框（漏检补框用）。

    ByteTrack 每帧已对 lost 轨迹做过一步预测，这里只读取当前状态，不重复 predict()。
    """
    out: Dict[int, np.ndarray] = {}
    if tracker is None:
        return out
    for st in getattr(tracker, "lost_stracks", []) or []:
        box = temporal.kalman_box(st, advance=False)
        if box is not None:
            out[int(st.track_id)] = box
    return out


# --------------------------------------------------------------------------- #
# 6. 主流程
# --------------------------------------------------------------------------- #

def build_argparser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="站立识别：输入视频 → 输出带骨架与站立判定的渲染视频",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g_in = p.add_argument_group("输入 / 输出")
    g_in.add_argument("--source", "-s", required=True,
                      help="输入视频路径；摄像头填 0 / 1 ...")
    g_in.add_argument("--output", "-o", default=None,
                      help="输出视频路径，默认 output/<视频名>_standing.mp4")
    g_in.add_argument("--model", "-m", default=str(here / "yolo26s-pose.pt"),
                      help="YOLO pose 权重路径")
    g_in.add_argument("--show", action="store_true", help="同时弹出窗口实时预览（按 q 退出）")
    g_in.add_argument("--start-time", type=float, default=0.0,
                      help="从视频第几秒开始处理（长视频分段用）")
    g_in.add_argument("--end-time", type=float, default=0.0,
                      help="处理到第几秒结束（0=到结尾）")
    g_in.add_argument("--out-width", type=int, default=0,
                      help="输出宽度（如 1280）。0=保持原分辨率；4K 监控视频建议缩小以加速")
    g_in.add_argument("--io", choices=["auto", "ffmpeg", "opencv"], default="auto",
                      help="读帧方式：auto=ffmpeg 管道(NVDEC)→失败回退 OpenCV")
    g_in.add_argument("--seek", choices=["hybrid", "accurate", "fast"], default="hybrid",
                      help="起始跳转方式：hybrid=先快跳再精确裁剪(推荐)；accurate=顺序解码到目标点；"
                           "fast=仅关键帧对齐(开 GOP 素材会解出错帧)")

    g_m = p.add_argument_group("模型推理")
    g_m.add_argument("--device", default="auto", help="auto / 0 / cpu")
    g_m.add_argument("--imgsz", type=int, default=640, help="推理分辨率")
    g_m.add_argument("--conf", type=float, default=0.35, help="人体检测置信度阈值")
    g_m.add_argument("--iou", type=float, default=0.5, help="NMS IoU 阈值")
    g_m.add_argument("--max-det", type=int, default=30, help="每帧最多人数")
    g_m.add_argument("--half", dest="half", action="store_true", default=None,
                     help="使用 FP16（GPU 上默认开启）")
    g_m.add_argument("--no-half", dest="half", action="store_false", help="关闭 FP16")
    g_m.add_argument("--tracker", default="bytetrack.yaml", help="跟踪器配置")
    g_m.add_argument("--no-track", action="store_true", help="关闭跟踪（无时序平滑）")
    g_m.add_argument("--kconf", type=float, default=0.35, help="关键点可见性阈值")
    g_m.add_argument("--batch", type=int, default=8,
                     help="批量推理帧数（>1 时手工驱动跟踪器；1=逐帧，作为回归基准）")

    g_a = p.add_argument_group("判定算法")
    g_a.add_argument("--standing-threshold", type=float, default=0.60, help="进入'站立'的置信度阈值")
    g_a.add_argument("--release-threshold", type=float, default=0.45, help="退出'站立'的置信度阈值")
    g_a.add_argument("--standing-min-score", type=float, default=0.80,
                     help="标注精度门槛：平滑置信度低于该值的站立估计不标注/不计数（默认 0.80）")
    g_a.add_argument("--smooth-window", type=int, default=15, help="时序平滑窗口(帧)")
    g_a.add_argument("--min-confirm", type=int, default=3, help="连续多少帧达标才确认'站立'")
    g_a.add_argument("--min-release", type=int, default=5, help="连续多少帧低于阈值才取消'站立'")
    g_a.add_argument("--ema-alpha", type=float, default=0.35, help="EMA 平滑系数")
    g_a.add_argument("--torso-only", choices=["unknown", "judge"], default="unknown",
                     help="看不到腿时如何处理：unknown=判为未知(推荐，避免把坐在桌后的人误判为站立)；"
                          "judge=仅凭躯干与框形判定")
    g_a.add_argument("--track-ttl", type=int, default=90, help="跟踪丢失多少帧后结算该目标")
    g_a.add_argument("--evidence-grace", type=int, default=12,
                     help="[已废弃] 由时序状态机替代：无有效证据的帧改为中立观测")
    g_a.add_argument("--bridge-frames", type=int, default=38,
                     help="漏检补框/轨迹保活/注销窗口（帧，默认 38）")
    g_a.add_argument("--viterbi-window", type=int, default=0,
                     help="Viterbi 滑窗帧数（0=与补框窗口相同）")
    g_a.add_argument("--viterbi-stay", type=float, default=0.92,
                     help="每帧保持原状态的概率，越大越不容易跳变（防抖强度）")
    g_a.add_argument("--viterbi-sigma", type=float, default=0.15,
                     help="发射模型标准差，越小越紧跟分数、越大越平滑")
    g_a.add_argument("--recheck", dest="recheck", action="store_true", default=True,
                     help="漏检时对预测区域做定向补检（拿真实关键点，推荐）")
    g_a.add_argument("--no-recheck", dest="recheck", action="store_false",
                     help="关闭定向补检（退回卡尔曼预测框画外推框）")
    g_a.add_argument("--recheck-pad", type=float, default=0.25,
                     help="补检裁剪框的向外扩张比例")
    g_a.add_argument("--recheck-min-det", type=int, default=2,
                     help="轨迹累计真实检测帧数达到该值才允许补检（防单帧误检被放大）")
    g_a.add_argument("--relink-frames", type=int, default=10,
                     help="ID 重连继承状态的帧窗口（默认 10 帧）")
    g_a.add_argument("--recheck-imgsz", type=int, default=480,
                     help="补检裁剪的推理分辨率（人在裁剪里很大，可小于整帧）")
    g_a.add_argument("--recheck-per-frame", type=int, default=2,
                     help="每帧最多补检几条轨迹（控制开销）")
    g_a.add_argument("--recheck-every", type=int, default=2,
                     help="补检失败后隔几帧再试（退避）")
    g_a.add_argument("--draw-grace", type=int, default=4,
                     help="无新确认时仍沿用最后一次确认框的最大帧数（≈0.15s，肉眼不可辨）")
    g_a.add_argument("--dedup-iou", type=float, default=0.60,
                     help="标注层去重阈值：同帧 IoU 超过该值的框只保留优先级最高的一个")
    g_a.add_argument("--zombie-iou", type=float, default=0.15,
                     help="僵尸轨迹阻断阈值：漏检轨迹的补检命中若与本帧已确认框 IoU 超过该值，"
                          "视为同一人的重复轨迹、不予确认（比显示去重更严格，因为只作用于未检测到的轨迹）")
    g_a.add_argument("--near-ratio", type=float, default=0.50,
                     help="框高/画面高超过该值视为离摄像头很近，触发裁剪放大重检")
    g_a.add_argument("--near-kpt-out", type=float, default=0.30,
                     help="关键点越界比例超过该值同样触发近距重检")
    g_a.add_argument("--near-imgsz", type=int, default=640,
                     help="近距重检的裁剪推理分辨率")
    g_a.add_argument("--near-per-frame", type=int, default=2,
                     help="每帧最多对几个近距目标做裁剪放大重检")
    g_a.add_argument("--persp-calib", default=None,
                     help="透视补偿曲线文件：存在则加载，不存在则计算后保存")
    g_a.add_argument("--no-persp", dest="persp", action="store_false", default=True,
                     help="关闭透视补偿（逐位复现未补偿的行为）")
    g_a.add_argument("--persp-frames", type=int, default=400,
                     help="自动标定预扫帧数（只检测、不跟踪不渲染）")
    g_a.add_argument("--persp-y0", type=float, default=450.0,
                     help="补偿生效的最小框底 y（像素）：小于该值的行不补偿")
    g_a.add_argument("--persp-max-k", type=float, default=2.0,
                     help="补偿系数上限")

    g_o = p.add_argument_group("渲染 / 编码")
    g_o.add_argument("--annotate", choices=["standing", "all"], default="standing",
                     help="渲染范围：standing=只标注站立的人（默认）；all=标注所有人")
    g_o.add_argument("--no-skeleton", dest="skeleton", action="store_false", help="不画骨架")
    g_o.add_argument("--no-hud", dest="hud", action="store_false", help="不画信息面板")
    g_o.add_argument("--line-width", type=int, default=2, help="线条宽度")
    g_o.add_argument("--text-size", type=int, default=22, help="标签字号")
    g_o.add_argument("--hud-refresh", type=int, default=1,
                     help="HUD 文字刷新间隔(帧)；1=每帧刷新（人数实时准确）")
    g_o.add_argument("--font", default=None, help="中文字体路径（默认自动查找微软雅黑）")
    g_o.add_argument("--encoder", choices=["auto", "nvenc", "x264", "mp4v"], default="auto",
                     help="输出编码器：auto=NVENC→x264→mp4v；nvenc=强制硬件编码")
    g_o.add_argument("--crf", type=int, default=20, help="H.264 画质(越小越清晰)")
    g_o.add_argument("--preset", default="p4", help="NVENC preset(p1~p7)")
    g_o.add_argument("--x264-preset", default="veryfast",
                     help="libx264 preset（veryfast 默认；medium 为旧版默认，仅作基线对比）")
    g_o.add_argument("--audio", choices=["keep", "drop"], default="keep", help="是否保留原音轨")
    g_o.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=None,
                     help="手动旋转帧（默认自动读取视频元数据）")

    g_x = p.add_argument_group("调试 / 结果导出")
    g_x.add_argument("--max-frames", type=int, default=0, help="仅处理前 N 帧（0=全部）")
    g_x.add_argument("--csv", default=None, help="导出逐帧逐人判定结果 CSV")
    g_x.add_argument("--json", dest="json_path", default=None, help="导出汇总 JSON")
    g_x.add_argument("--debug-timing", action="store_true", help="打印各阶段耗时统计")
    g_x.add_argument("--quiet", action="store_true", help="不打印进度")
    return p


def _resolve_device(device: str) -> Tuple[str, bool]:
    import torch
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    use_half = device != "cpu" and torch.cuda.is_available()
    return device, use_half


def run(args: argparse.Namespace) -> dict:
    from ultralytics import YOLO

    source = args.source
    is_camera = str(source).isdigit()
    src_path: Optional[Path] = None if is_camera else Path(source)
    if src_path is not None and not src_path.exists():
        raise FileNotFoundError(f"找不到输入视频: {src_path}")

    if not Path(args.model).exists():
        raise FileNotFoundError(f"找不到模型权重: {args.model}")

    device, use_half = _resolve_device(args.device)
    if args.half is not None:
        use_half = bool(args.half) and device != "cpu"

    model = YOLO(args.model)

    # ---- 元数据（ffmpeg 探测，失败回退 OpenCV） ----
    if is_camera:
        cam = cv2.VideoCapture(int(source))
        if not cam.isOpened():
            raise RuntimeError(f"无法打开摄像头: {source}")
        cam_fps = cam.get(cv2.CAP_PROP_FPS)
        info = video_io.VideoInfo(path=None,
                                  fps=float(cam_fps) if cam_fps and cam_fps > 0.1 else 30.0,
                                  width=int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                  height=int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cam.release()
    else:
        info = video_io.probe_video(src_path)
    fps = info.fps if info.fps and info.fps > 0.1 else 30.0
    width, height = info.width, info.height
    if not width or not height:
        raise RuntimeError(f"无法读取视频尺寸: {source}")
    total_frames = 0 if is_camera else info.frame_count

    # 分段：本段帧数 = (end - start) * fps
    seg_start = max(0.0, float(args.start_time))
    seg_end = float(args.end_time) if args.end_time and args.end_time > 0 else None
    if is_camera:
        segment_frames = args.max_frames
    else:
        avail = max(0.0, (info.duration - seg_start)) if info.duration else None
        seg_len = (seg_end - seg_start) if seg_end else avail
        if seg_len is None:
            segment_frames = total_frames
        else:
            segment_frames = max(0, int(round(seg_len * fps)))
        if total_frames:
            segment_frames = min(segment_frames, total_frames) if segment_frames else total_frames
        if args.max_frames > 0:
            segment_frames = min(segment_frames, args.max_frames) if segment_frames else args.max_frames

    rotation = args.rotate if args.rotate is not None else (0 if is_camera else info.rotation)
    if rotation in (90, 270):
        width, height = height, width

    # 输出分辨率（可选降采样，4K 监控视频强烈建议使用）
    out_w, out_h = width, height
    if args.out_width and 0 < args.out_width < width:
        scale = args.out_width / float(width)
        out_w = max(2, int(round(width * scale)) // 2 * 2)
        out_h = max(2, int(round(height * scale)) // 2 * 2)
    # 输出路径
    if args.output:
        out_path = Path(args.output)
    else:
        stem = "camera" if is_camera else src_path.stem
        out_path = Path("output") / f"{stem}_standing.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    audio_path = None
    if args.audio == "keep" and not is_camera and str(out_path).lower().endswith(".mp4"):
        if info.has_audio:
            audio_path = str(src_path)
        else:
            print("[info] 源视频无音轨，输出为纯视频。")

    precision = _precision_kwargs(use_half)
    require_leg_evidence = args.torso_only != "judge"

    # 时序参数（全部以帧为单位）
    bridge_frames = max(1, int(args.bridge_frames))
    viterbi_window = int(args.viterbi_window) if args.viterbi_window > 0 else bridge_frames
    relink_frames = max(1, int(args.relink_frames))

    painter = TextPainter(args.font)
    if not painter.available:
        print("[warn] 未找到中文字体，界面文字可能显示为方块；可用 --font 指定字体文件。")

    csv_file = csv_writer = None
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w", newline="", encoding="utf-8-sig")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "time_s", "track_id", "label", "is_standing",
                             "score", "coverage", "torso_angle_deg", "leg_drop_ratio",
                             "knee_angle", "bbox_aspect", "x1", "y1", "x2", "y2", "held",
                             "smooth_score", "s2_state", "dup", "persp_k_leg", "persp_k_asp"])

    tracks: Dict[int, TrackRecord] = {}
    bridge: Dict[int, BridgeEntry] = {}
    expired: List[BridgeEntry] = []
    temp_id = 0
    global_stats = {"frames": 0, "frames_with_standing": 0, "max_simultaneous": 0,
                    "total_person_frames": 0, "standing_person_frames": 0,
                    "s2_person_frames": 0, "held_person_frames": 0,
                    "recheck_hits": 0, "recheck_tries": 0,
                    "dup_person_frames": 0, "zombie_blocked": 0,
                    "near_triggered": 0, "near_replaced": 0}
    state_hist = [0, 0, 0]
    label_hist = {STANDING: 0, SITTING: 0, LYING: 0, UNKNOWN: 0}
    hud_cache: Dict[str, object] = {"lines": None, "next": 0}
    # 逐阶段耗时（不含首帧预热，见 warmup_s）
    phase = {"read": 0.0, "infer": 0.0, "track": 0.0, "recheck": 0.0, "draw": 0.0,
             "encode": 0.0}
    warmup_s = 0.0
    recheck_ms = 0.0
    _t = time.perf_counter

    sink = video_io.open_video_sink(out_path, (out_w, out_h), fps, encoder=args.encoder,
                                    audio_path=audio_path, cq=args.crf, preset=args.preset,
                                    x264_crf=args.crf, x264_preset=args.x264_preset)
    print(f"[info] 输入 : {source}  {width}x{height} @ {fps:.2f}fps  ({info.codec or '?'})"
          + (f"  本次处理 {segment_frames} 帧"
             + (f"（{seg_start:.1f}s 起）" if seg_start else "")
             if segment_frames else "  (实时流)"))
    print(f"[info] 设备 : {device}{' (FP16)' if use_half else ''}   模型: {Path(args.model).name}"
          f"   批量 {max(1, args.batch)}")
    print(f"[info] 输出 : {out_path}  [{sink.mode}]"
          + f"  {out_w}x{out_h}"
          + (f"  旋转 {rotation}°" if rotation else ""))

    source_reader = video_io.open_frame_source(
        info, (out_w, out_h), start_time=seg_start, end_time=seg_end, io_mode=args.io,
        seek=args.seek, rotation=rotation, camera=int(source) if is_camera else None)

    batch_size = max(1, int(args.batch))
    tracker = None
    if batch_size > 1 and not args.no_track:
        tracker = _build_tracker(args.tracker, device)

    print(f"[info] 时序 : Viterbi 三状态(S0/S1/S2) 滑窗 {viterbi_window} 帧, "
          f"stay={args.viterbi_stay} σ={args.viterbi_sigma}；"
          f"漏检补框窗口 {bridge_frames} 帧；补检 {'开' if args.recheck else '关'}")

    predict_kw = dict(conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=device,
                      verbose=False, max_det=args.max_det, **precision)

    def _calibrate_perspective() -> perspective.PerspectiveCalib:
        """预扫前 N 帧，用与俯角无关的"明确站立"样本标定透视压缩曲线。"""
        samples: List[perspective.Sample] = []
        if args.persp_frames <= 0:
            return perspective.identity_calib(args.persp_y0, status="disabled")
        # 跨整段采样：片段较长时只解码关键帧（快且覆盖全片），短片则逐帧
        seg_len = segment_frames or info.frame_count or 0
        # 短片（≤3 分钟）逐帧解码后跨段抽样；长片只解关键帧（跨全片且极快）
        use_key = bool(seg_len) and seg_len > 3600
        stride = 1 if use_key else max(1, int(math.ceil(seg_len / max(1, args.persp_frames))))
        try:
            reader = video_io.open_frame_source(
                info, (out_w, out_h), start_time=seg_start, end_time=seg_end,
                io_mode=args.io, seek=args.seek, rotation=rotation,
                camera=int(source) if is_camera else None, skip_nonkey=use_key,
                log=lambda *a: None)
        except Exception as exc:
            print(f"[warn] 透视标定无法打开视频源({exc})，跳过补偿。")
            return perspective.identity_calib(args.persp_y0, status="open_failed")
        pending: List[np.ndarray] = []
        scanned = 0          # 解码帧数
        processed = 0        # 参与标定的帧数
        try:
            while processed < args.persp_frames:
                frame = reader.read()
                if frame is None:
                    break
                scanned += 1
                if (scanned - 1) % stride:
                    continue
                pending.append(frame)
                processed += 1
                if len(pending) < batch_size:
                    continue
                for res in model.predict(pending, **predict_kw):
                    samples.extend(_collect_calib_samples(res))
                pending = []
            if pending:
                for res in model.predict(pending, **predict_kw):
                    samples.extend(_collect_calib_samples(res))
        finally:
            reader.close()
        calib = perspective.fit_perspective(samples, y0=args.persp_y0, max_k=args.persp_max_k)
        calib.meta.update({"scanned_frames": scanned, "processed_frames": processed,
                           "decode": ("keyframe" if use_key else "all")})
        return calib

    def _collect_calib_samples(res) -> List[perspective.Sample]:
        """从一帧的检测里挑出"明确站立"样本：覆盖度满、膝角≥160°、躯干≤30°。"""
        out: List[perspective.Sample] = []
        if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
            return out
        boxes = res.boxes.xyxy.detach().cpu().numpy()
        kpts = res.keypoints.data.detach().cpu().numpy()
        for i in range(len(boxes)):
            r = classify_pose(kpts[i], boxes[i], kconf=args.kconf,
                              stand_threshold=args.standing_threshold,
                              require_leg_evidence=require_leg_evidence)
            f = r.features
            if (r.coverage >= 0.999 and f.get("knee_angle", 0.0) >= 160.0
                    and f.get("torso_angle_deg", 99.0) <= 30.0
                    and "leg_drop_ratio" in f and "body_extension" in f):
                out.append(perspective.Sample(y=float(boxes[i][3]),
                                              leg=float(f["leg_drop_ratio"]),
                                              ext=float(f["body_extension"]),
                                              asp=float(f["bbox_aspect"])))
        return out

    # ---- 透视补偿标定：优先加载已有曲线，否则用前 N 帧预扫自动标定 ----
    calib_t0 = time.time()
    calib: Optional[perspective.PerspectiveCalib] = None
    if not args.persp:
        calib = perspective.identity_calib(args.persp_y0, status="off")
    elif args.persp_calib and Path(args.persp_calib).exists():
        try:
            calib = perspective.PerspectiveCalib.load(args.persp_calib)
            meta = calib.meta or {}
            if meta.get("width") and int(meta["width"]) != out_w:
                print(f"[warn] 标定文件分辨率({meta['width']}x{meta.get('height')})与当前输出"
                      f"({out_w}x{out_h})不一致，请确认是同一机位。")
            print(f"[info] 透视补偿：加载 {args.persp_calib}（样本 {calib.samples}，"
                  f"状态 {calib.status}）")
        except Exception as exc:
            print(f"[warn] 标定文件读取失败({exc})，改为自动标定。")
            calib = None

    if calib is None:
        calib = _calibrate_perspective()
        if args.persp_calib:
            try:
                calib.meta.update({"source": str(source), "width": out_w, "height": out_h,
                                   "fps": round(fps, 3)})
                calib.save(args.persp_calib)
                print(f"[info] 透视补偿：曲线已保存到 {args.persp_calib}")
            except Exception as exc:
                print(f"[warn] 标定文件保存失败: {exc}")
    calib_s = time.time() - calib_t0
    print(f"[info] 透视补偿：y0={calib.y0:g} 样本 {calib.samples} 状态 {calib.status}；"
          + (f"k_leg={[round(v, 2) for v in calib.k_leg]}（用时 {calib_s:.1f}s）"
             if calib.status == "ok" else f"（用时 {calib_s:.1f}s）"))

    def _persp_for(y2: float) -> Optional[Tuple[float, float, float]]:
        if calib is None or calib.status != "ok":
            return None
        k = calib.factors(y2)
        return None if k == (1.0, 1.0, 1.0) else k

    def _new_entry(tid: int, t_now: float) -> BridgeEntry:
        rec = tracks.get(tid)
        if rec is None:
            rec = TrackRecord(tid=tid, first_t=t_now, last_t=t_now, fps=fps)
            tracks[tid] = rec
        return BridgeEntry(
            tid=tid,
            viterbi=temporal.TrackViterbi(args.standing_threshold, args.release_threshold,
                                          sigma=args.viterbi_sigma, stay=args.viterbi_stay,
                                          window=viterbi_window),
            smoother=PostureSmoother(args.smooth_window, args.min_confirm, args.min_release,
                                     args.standing_threshold, args.release_threshold,
                                     args.ema_alpha),
            raw_smoother=PostureSmoother(args.smooth_window, args.min_confirm, args.min_release,
                                         args.standing_threshold, args.release_threshold,
                                         args.ema_alpha),
            rec=rec)

    def _apply_state(entry: BridgeEntry, t_now: float, held: bool,
                     smooth_value: Optional[float] = None) -> bool:
        """累计统计与站立时间段。

        返回"是否按站立输出"——需要同时满足：时序状态为 S2，且平滑置信度 ≥
        ``--standing-min-score``（精度门槛，低于门槛的站立估计不标注/不计数）。
        """
        rec = entry.rec
        is_s2 = entry.state == temporal.S2
        is_standing = bool(is_s2 and smooth_value is not None
                           and smooth_value >= args.standing_min_score)
        rec.last_t = t_now
        rec.n_frames += 1
        rec.state_frames[entry.state] += 1
        rec.s2_frames += int(is_s2)
        if held:
            rec.held_frames += 1
        if is_standing:
            rec.n_standing += 1
        if is_standing and not entry.was_standing:
            rec.intervals.append([t_now, t_now])
        elif not is_standing and entry.was_standing:
            if rec.intervals and rec.intervals[-1][1] <= rec.intervals[-1][0]:
                rec.intervals[-1][1] = t_now
        entry.was_standing = is_standing
        entry.prev_state = entry.state
        return is_standing

    def _finalize(entry: BridgeEntry) -> None:
        """补框窗口超时：收尾站立时间段（到末次真实观测）并注销解码器。"""
        rec = entry.rec
        if rec.intervals and rec.intervals[-1][1] <= rec.intervals[-1][0]:
            rec.intervals[-1][1] = entry.last_t
        entry.closed = True
        expired.append(entry)
        bridge.pop(entry.tid, None)

    def _try_inherit(entry: BridgeEntry, box: np.ndarray, idx: int) -> bool:
        """ID 重连继承：新轨迹与刚消失轨迹框重叠时继承其解码器状态。"""
        best, best_iou = None, 0.5
        for other in list(bridge.values()) + expired:
            if other is entry or other.last_box is None:
                continue
            if idx - other.last_idx > relink_frames:
                continue
            v = temporal.iou(box, other.last_box)
            if v > best_iou:
                best, best_iou = other, v
        if best is None:
            return False
        entry.viterbi.clone_from(best.viterbi)
        entry.state = entry.prev_state = best.state
        if best.last_kpts is not None:
            entry.last_kpts = best.last_kpts.copy()
        return True

    def _crop_detect(frame: np.ndarray, ref_box: np.ndarray, pad: float, imgsz: int,
                     iou_min: float):
        """在参考框附近裁剪后重新检测，返回 (box, kpts, result) 或 None。

        用于两处：① 漏检时的定向补检（拿真实观测，替代外推框）；
        ② 离摄像头过近/被画面切断的目标做裁剪放大重检（提升关键点质量）。
        """
        if ref_box is None or frame is None:
            return None
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in ref_box)
        pad = max(0.02, float(pad))
        px, py = pad * (x2 - x1), pad * (y2 - y1)
        cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
        cx2, cy2 = min(W, int(x2 + px)), min(H, int(y2 + py))
        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            return None
        crop = frame[cy1:cy2, cx1:cx2]
        t0 = _t()
        kw = dict(predict_kw)
        kw["imgsz"] = int(imgsz)
        res = model.predict(crop, **kw)[0]
        nonlocal recheck_ms
        recheck_ms += _t() - t0
        if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
            return None
        boxes = res.boxes.xyxy.detach().cpu().numpy()
        kpts = res.keypoints.data.detach().cpu().numpy()
        best, best_iou = None, float(iou_min)
        for i in range(len(boxes)):
            full_box = boxes[i] + np.array([cx1, cy1, cx1, cy1], dtype=np.float32)
            v = temporal.iou(full_box, ref_box)
            if v > best_iou:
                best, best_iou = i, v
        if best is None:
            return None
        box = (boxes[best] + np.array([cx1, cy1, cx1, cy1], dtype=np.float32))
        k = kpts[best].copy()
        k[:, 0] += cx1
        k[:, 1] += cy1
        result = classify_pose(k, box, kconf=args.kconf,
                               stand_threshold=args.standing_threshold,
                               require_leg_evidence=require_leg_evidence,
                               persp=_persp_for(float(box[3])))
        return box, k, result

    def _recheck(entry: BridgeEntry, frame: np.ndarray):
        """漏检轨迹的定向补检（沿用既有参数）。"""
        return _crop_detect(frame, entry.last_box, args.recheck_pad, args.recheck_imgsz, 0.10)

    def _write_csv_row(c: Candidate, idx: int, t_now: float, is_standing: bool,
                       dup: int) -> None:
        if csv_writer is None:
            return
        f = c.result.features
        csv_writer.writerow([
            idx, round(t_now, 3), c.tid, LABEL_EN[c.result.label], int(is_standing),
            round(c.result.score, 4), round(c.result.coverage, 4),
            round(f.get("torso_angle_deg", float("nan")), 1) if "torso_angle_deg" in f else "",
            round(f["leg_drop_ratio"], 3) if "leg_drop_ratio" in f else "",
            round(f["knee_angle"], 1) if "knee_angle" in f else "",
            round(f["bbox_aspect"], 3),
            *[round(float(v), 1) for v in c.box],
            c.held, round(float(c.smooth), 4), int(c.state == temporal.S2), dup,
            round(float(c.result.persp[0]), 3), round(float(c.result.persp[1]), 3),
        ])

    def _close_interval(entry: Optional[BridgeEntry], t_now: float) -> None:
        if entry is None or not entry.was_standing:
            return
        rec = entry.rec
        if rec.intervals and rec.intervals[-1][1] <= rec.intervals[-1][0]:
            rec.intervals[-1][1] = t_now
        entry.was_standing = False

    def _emit_candidates(cands: List[Candidate], t_now: float) -> Tuple[int, int, int]:
        """帧级去重（标注层 NMS）→ 统一计数 / 写 CSV / 绘制。

        优先级：真实检测与补检命中(2) > 沿用帧(1)；priority<1 表示"不参与展示"。
        与已保留框 IoU 超过 ``--dedup-iou`` 的候选被抑制：不绘制、不计数，
        但仍在 CSV 里保留并标 ``dup=1``（保证数据可追溯）。
        """
        kept, hidden = dedup_candidates(cands, args.dedup_iou)

        for c in hidden:                       # 未展示：保留数据、不计数、结束时间段
            _close_interval(c.entry, t_now)
            global_stats["dup_person_frames"] += 1
            _write_csv_row(c, idx, t_now, False, 1)

        n_p = n_s = n_u = 0
        for c in kept:
            if c.entry is not None:
                is_standing = _apply_state(c.entry, t_now, held=bool(c.held),
                                           smooth_value=c.smooth)
                c.entry.rec.best_score = max(c.entry.rec.best_score, c.result.score)
                c.entry.rec.last_label = c.result.label
            else:
                is_standing = bool(c.is_standing)
            n_p += 1
            n_s += int(is_standing)
            n_u += int(c.result.label == UNKNOWN)
            label_hist[c.result.label] += 1
            state_hist[c.state] += 1
            global_stats["s2_person_frames"] += int(c.state == temporal.S2)
            global_stats["total_person_frames"] += 1
            global_stats["standing_person_frames"] += int(is_standing)
            _write_csv_row(c, idx, t_now, is_standing, 0)
            if args.annotate == "all" or is_standing:
                draw_person(canvas, painter, c.box, c.kpts, c.result, is_standing,
                            c.smooth, c.state, args.kconf, args)
        return n_p, n_s, n_u

    def _frame_stream():
        """产出 (frame, results, 读帧耗时, 推理耗时, 跟踪耗时)。

        batch=1 时逐帧调用 model.track（与旧行为一致）；batch>1 时批量推理，
        再由本程序逐帧驱动 BYTETracker，保证积分顺序与逐帧模式一致。
        """
        pending: List[np.ndarray] = []

        def flush(frames: List[np.ndarray], d_read: float):
            t_i = _t()
            results = model.predict(frames, **predict_kw)
            d_infer = (_t() - t_i) / max(1, len(frames))
            for fr, res in zip(frames, results):
                if tracker is not None:
                    t_t = _t()
                    res = _apply_tracker(tracker, res, fr)
                    d_track = _t() - t_t
                else:
                    d_track = 0.0
                yield fr, res, d_read, d_infer, d_track

        while True:
            t_r = _t()
            frame = source_reader.read()
            d_read = _t() - t_r
            if frame is None:
                if pending:
                    frames, pending = pending, []
                    yield from flush(frames, d_read)
                return
            if rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if batch_size <= 1:
                t_i = _t()
                if args.no_track:
                    results = model.predict(frame, **predict_kw)
                else:
                    results = model.track(frame, persist=True, tracker=args.tracker, **predict_kw)
                yield frame, results[0], d_read, _t() - t_i, 0.0
                continue
            pending.append(frame)
            if len(pending) >= batch_size:
                frames, pending = pending, []
                yield from flush(frames, d_read)

    idx = 0
    t_start = time.time()
    t_start_perf = _t()
    try:
        for frame, res, d_read, d_infer, d_track in _frame_stream():
            t_person = _t()
            d_phase: Dict[str, float] = {"read": d_read, "infer": d_infer, "track": d_track}
            recheck_before = recheck_ms
            # 标注画在副本上：原帧保持干净，供"定向补检"裁剪使用。
            # （否则已绘制的骨架/框会被补检当成图像内容，检测结果反过来受绘制模式影响）
            canvas = frame.copy() if args.recheck else frame

            t_now = seg_start + idx / fps
            elapsed = time.time() - t_start

            n_people = 0
            n_standing = 0
            n_unknown = 0
            seen: set = set()
            frame_cands: List[Candidate] = []      # 本帧候选（去重后统一输出）
            confirmed_boxes: List[np.ndarray] = []  # 本帧真实检测到的框（用于僵尸阻断）
            near_used = 0
            ids: List[Optional[int]] = []
            if res.boxes is not None and len(res.boxes) > 0:
                boxes = res.boxes.xyxy.detach().cpu().numpy()
                if getattr(res.boxes, "id", None) is not None:
                    ids = [int(v) for v in res.boxes.id.detach().cpu().numpy().tolist()]
                else:
                    ids = [None] * len(boxes)
                kpts_all = (res.keypoints.data.detach().cpu().numpy()
                            if res.keypoints is not None else None)
            else:
                boxes, kpts_all = np.zeros((0, 4), dtype=np.float32), None

            for i in range(len(boxes)):
                kpts = (kpts_all[i] if kpts_all is not None and i < len(kpts_all)
                        else np.zeros((17, 3), dtype=np.float32))
                result = classify_pose(kpts, boxes[i], kconf=args.kconf,
                                       stand_threshold=args.standing_threshold,
                                       require_leg_evidence=require_leg_evidence,
                                       persp=_persp_for(float(boxes[i][3])))
                # ---- 近距/被画面切断目标：裁剪放大重检，提升关键点质量 ----
                Hf, Wf = frame.shape[:2]
                h_ratio = float(boxes[i][3] - boxes[i][1]) / max(1, Hf)
                near_trigger = (h_ratio >= args.near_ratio
                                or kpt_out_ratio(kpts, Wf, Hf, args.kconf) >= args.near_kpt_out)
                if near_trigger and near_used < max(0, int(args.near_per_frame)):
                    near_used += 1
                    global_stats["near_triggered"] += 1
                    refined = _crop_detect(frame, boxes[i], 0.15, args.near_imgsz, 0.30)
                    if refined is not None and refined[2].coverage > result.coverage:
                        boxes[i], kpts, result = refined
                        global_stats["near_replaced"] += 1
                tid = ids[i]
                if tid is None:
                    temp_id -= 1
                    tid = temp_id
                    has_track = False
                else:
                    has_track = not args.no_track

                # 证据不足（看不到腿/关键点太少）→ 中立观测，交给时域状态机推断
                evidence_ok = result.coverage >= MIN_COVERAGE and (
                    result.leg_evidence or not require_leg_evidence
                )
                held = 0
                is_standing = False          # 有追踪时由 _apply_state 在去重后判定
                if not has_track:
                    # --no-track：逐帧判定，不做时序（同样受精度门槛约束）
                    entry = None
                    is_standing = (result.label == STANDING
                                   and result.score >= args.standing_min_score)
                    state = temporal.S2 if is_standing else temporal.S0
                    smooth_value = result.score
                else:
                    entry = bridge.get(tid)
                    if entry is None:
                        entry = _new_entry(tid, t_now)
                        _try_inherit(entry, boxes[i], idx)
                        bridge[tid] = entry
                    seen.add(tid)
                    entry.det_count += 1
                    entry.last_idx = idx
                    entry.last_confirmed_idx = idx
                    entry.last_t = t_now
                    entry.last_box = boxes[i].copy()
                    entry.last_result = result
                    if kpts is not None:
                        entry.last_kpts = kpts.copy()
                    if evidence_ok:
                        entry.smoother.update(result.score)
                        entry.raw_smoother.update(result.raw_score)
                        obs = entry.smoother.value
                        entry.neutral_run = 0
                    else:
                        # 看不到腿等无有效证据：先按中立观测保持状态，超过宽限期后
                        # 施加持续向下的证据，避免"坐着但腿被挡住"的人被无限期判为站立
                        entry.neutral_run += 1
                        obs = (None if entry.neutral_run <= max(0, args.evidence_grace)
                               else args.release_threshold - 0.05)
                    entry.state = entry.viterbi.update(obs)
                    state = entry.state
                    smooth_value = (entry.smoother.value if entry.smoother.value is not None
                                    else result.score)
                    confirmed_boxes.append(boxes[i])

                frame_cands.append(Candidate(
                    tid=tid, box=boxes[i].copy(), kpts=kpts, result=result, state=state,
                    smooth=float(smooth_value), held=held, priority=2,
                    is_standing=bool(is_standing), entry=entry,
                    was_passing=bool(
                        (entry.raw_smoother.value if entry is not None
                         and entry.raw_smoother.value is not None else result.raw_score)
                        >= args.standing_min_score)))

            # ---- 漏检补框：本帧没有检测到、但仍在桥接窗口内的轨迹 ----
            if not args.no_track and bridge:
                lost_boxes = _lost_track_boxes(tracker)
                # 补检预算：只补"站立且漏检"的轨迹，每帧最多 N 条，失败后退避
                recheck_budget = max(0, int(args.recheck_per_frame)) if args.recheck else 0
                candidates = []
                for tid, entry in list(bridge.items()):
                    if tid in seen:
                        continue
                    if idx - entry.last_idx > bridge_frames:
                        _finalize(entry)
                        continue
                    if (recheck_budget > 0 and entry.state == temporal.S2
                            and entry.det_count >= max(1, args.recheck_min_det)
                            and idx - entry.last_confirmed_idx > max(1, args.draw_grace)
                            and idx >= entry.next_recheck):
                        candidates.append(entry)
                candidates.sort(key=lambda e: e.last_idx, reverse=True)
                recheck_ok = {e.tid for e in candidates[:recheck_budget]}
                for e in candidates:
                    if e.tid not in recheck_ok:
                        e.next_recheck = idx + max(1, int(args.recheck_every))
                for e in candidates[:recheck_budget]:
                    global_stats["recheck_tries"] += 1
                for tid, entry in list(bridge.items()):
                    if tid in seen or idx - entry.last_idx > bridge_frames:
                        continue
                    # ---- 定向补检：拿真实框/关键点，避免画外推"幽灵框" ----
                    rechecked = None
                    if tid in recheck_ok:
                        rechecked = _recheck(entry, frame)
                        # 僵尸阻断：补检命中框若与"本帧已确认的其它轨迹"重合，说明是同一人的
                        # 重复轨迹，不算作该轨迹的确认（否则失效 ID 会被反复复活）
                        if rechecked is not None and any(
                                temporal.iou(rechecked[0], b) > args.zombie_iou
                                for b in confirmed_boxes):
                            global_stats["zombie_blocked"] += 1
                            rechecked = None
                    pred = lost_boxes.get(tid)
                    if rechecked is not None:
                        box, kpts_r, result_r = rechecked
                        entry.last_box = box
                        entry.last_kpts = kpts_r
                        entry.last_result = result_r
                        entry.last_idx = idx          # 补检成功视为真实观测，重置保留计时
                        entry.last_confirmed_idx = idx
                        entry.det_count += 1
                        entry.recheck_hits += 1
                        entry.next_recheck = idx + max(1, int(args.recheck_every))
                        global_stats["recheck_hits"] += 1
                        evidence_ok_r = result_r.coverage >= MIN_COVERAGE and (
                            result_r.leg_evidence or not require_leg_evidence)
                        if evidence_ok_r:
                            entry.smoother.update(result_r.score)
                            entry.raw_smoother.update(result_r.raw_score)
                            entry.neutral_run = 0
                            obs_r = entry.smoother.value
                        else:
                            entry.neutral_run += 1
                            obs_r = (None if entry.neutral_run <= max(0, args.evidence_grace)
                                     else args.release_threshold - 0.05)
                        entry.state = entry.viterbi.update(obs_r)
                        smooth_r = (entry.smoother.value if entry.smoother.value is not None
                                    else result_r.score)
                        frame_cands.append(Candidate(
                            tid=tid, box=box.copy(), kpts=kpts_r, result=result_r,
                            state=entry.state, smooth=float(smooth_r), held=0, priority=2,
                            is_standing=False, entry=entry,
                            was_passing=bool(
                                (entry.raw_smoother.value
                                 if entry.raw_smoother.value is not None else result_r.raw_score)
                                >= args.standing_min_score)))
                        confirmed_boxes.append(box)
                        continue
                    # ---- 补检未命中：保留状态但不画外推框（避免"滞留"幽灵框）----
                    if tid in recheck_ok:
                        entry.next_recheck = idx + max(1, int(args.recheck_every))
                    entry.state = entry.viterbi.update(None)     # 中立观测
                    smooth_held = (entry.smoother.value if entry.smoother.value is not None
                                   else None)
                    box = pred if pred is not None else entry.last_box
                    if box is None:
                        continue
                    # 沿用帧仅在"最近一次确认还很新"时参与绘制（否则只记数据不画）
                    in_grace = (not args.recheck) or (
                        entry.last_box is not None
                        and idx - entry.last_confirmed_idx <= max(1, args.draw_grace))
                    kpts_draw = entry.last_kpts
                    if kpts_draw is None:
                        kpts_draw = np.zeros((17, 3), dtype=np.float32)
                    frame_cands.append(Candidate(
                        tid=tid, box=box.copy(), kpts=kpts_draw,
                        result=entry.last_result or classify_pose(
                            kpts_draw, box, args.kconf, args.standing_threshold,
                            require_leg_evidence, _persp_for(float(box[3]))),
                        state=entry.state,
                        smooth=float(smooth_held) if smooth_held is not None else 0.0,
                        held=1, priority=1 if in_grace else 0, is_standing=False, entry=entry))

            # ---- 帧级去重后统一计数 / 写 CSV / 绘制 ----
            n_people, n_standing, n_unknown = _emit_candidates(frame_cands, t_now)

            if n_standing > 0:
                global_stats["frames_with_standing"] += 1
            global_stats["max_simultaneous"] = max(global_stats["max_simultaneous"], n_standing)

            # 清理过期的重连候选
            if expired:
                expired[:] = [e for e in expired if idx - e.last_idx <= relink_frames]

            if args.hud:
                if hud_cache["lines"] is None or idx >= int(hud_cache["next"]):
                    proc_fps = int(round((idx + 1) / elapsed)) if elapsed > 0 else 0
                    span = segment_frames or total_frames
                    frame_info = f"帧 {idx + 1}" + (f"/{span}" if span else "")
                    hud_cache["lines"] = [
                        (f"{frame_info}   {proc_fps} FPS   {t_now:.1f}s", (0, 214, 255)),
                        (f"Standing {n_standing}" if args.annotate == "standing"
                         else f"Standing {n_standing}   非站立 "
                              f"{n_people - n_standing - n_unknown}   未知 {n_unknown}",
                         LABEL_COLOR[STANDING] if n_standing else (235, 235, 235)),
                        *([("只标注站立的人（绿色框）", (150, 150, 150))]
                          if args.annotate == "standing" else []),
                        (f"阈值 {args.standing_threshold:.2f}/{args.release_threshold:.2f}  "
                         f"平滑 {args.smooth_window}帧", (170, 170, 170)),
                    ]
                    hud_cache["next"] = idx + max(1, args.hud_refresh)
                draw_hud(canvas, painter, hud_cache["lines"],
                         legend=(args.annotate == "all"),
                         size=max(14, args.text_size - 2))

            d_phase["recheck"] = recheck_ms - recheck_before
            d_phase["draw"] = _t() - t_person - d_phase["recheck"]
            t_write = _t()
            sink.write(canvas)
            d_phase["encode"] = _t() - t_write

            if args.show:
                cv2.imshow(WINDOW_TITLE, canvas)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    print("[info] 用户中断预览。")
                    break

            idx += 1
            if idx == 1:
                warmup_s = _t() - t_start_perf      # 首批包含推理预热，单独统计
            elif idx > batch_size:
                for _k, _v in d_phase.items():
                    phase[_k] += _v
            if (args.max_frames and idx >= args.max_frames) or (
                    segment_frames and idx >= segment_frames):
                break
            if not args.quiet and (idx % 30 == 0 or (total_frames and idx == total_frames)):
                msg = f"\r[进度] {idx}" + (f"/{total_frames}" if total_frames else "")
                steady = (_t() - t_start_perf - warmup_s)
                if steady > 0 and idx > 1:
                    msg += f"  {(idx - 1) / steady:.1f} FPS"
                sys.stdout.write(msg + "   ")
                sys.stdout.flush()
    finally:
        source_reader.close()
        sink.close()
        if csv_file is not None:
            csv_file.close()
        if args.show:
            cv2.destroyAllWindows()
        if not args.quiet:
            sys.stdout.write("\n")

    global_stats["frames"] = idx
    # 视频结束：仍在桥接窗口内的轨迹按"末次真实观测"收尾
    for entry in list(bridge.values()):
        _finalize(entry)
    # 收尾：把视频结束时仍处于"站立"的时间段闭合
    for rec in tracks.values():
        if rec.intervals and rec.intervals[-1][1] <= rec.intervals[-1][0]:
            rec.intervals[-1][1] = rec.last_t
    elapsed = time.time() - t_start
    processed = max(1, idx)
    if args.debug_timing:
        n_profiled = max(1, processed - batch_size)
        total = sum(phase.values()) or 1.0
        print(f"[timing] 首批预热(模型/跟踪器/CUDA 初始化, 批量 {batch_size}): "
              f"{warmup_s * 1000:.0f} ms")
        print(f"[timing] 各阶段平均耗时 (ms/帧, 不含预热):")
        for name, value in phase.items():
            print(f"    {name:8s} {value / n_profiled * 1000:7.2f} ms   {value / total * 100:5.1f}%")

    summary = {
        "source": str(source),
        "output": str(out_path),
        "video": {
            "width": width, "height": height, "fps": round(fps, 3),
            "source_codec": info.codec, "source_duration_s": round(info.duration, 3),
            "output_width": out_w, "output_height": out_h,
            "frames_read": idx, "duration_s": round(idx / fps, 3) if fps else 0,
            "rotation_applied": rotation,
            "start_time": seg_start, "end_time": args.end_time or None,
            "seek_mode": args.seek,
        },
        "io": {
            "read_mode": source_reader.kind,
            "hwaccel": getattr(source_reader, "hwaccel", None),
            "decode_attempts": getattr(source_reader, "attempts", []),
            "decode_errors": int(getattr(source_reader, "decode_errors", 0)),
            "frames_short": max(0, int(segment_frames) - idx) if (segment_frames and not is_camera) else 0,
            "encoder": getattr(sink, "encoder", sink.kind),
            "sink_mode": sink.mode,
            "batch": batch_size,
            "x264_preset": args.x264_preset,
        },
        "model": {"path": str(args.model), "imgsz": args.imgsz, "device": device,
                  "half": bool(use_half), "conf": args.conf, "kconf": args.kconf},
        "algorithm": {
            "feature_weights": dict(FEATURE_WEIGHTS),
            "feature_ranges": {k: list(v) for k, v in FEATURE_RANGE.items()},
            "standing_threshold": args.standing_threshold,
            "release_threshold": args.release_threshold,
            "standing_min_score": args.standing_min_score,
            "smooth_window": args.smooth_window,
            "min_confirm": args.min_confirm,
            "min_release": args.min_release,
            "temporal_smoothing": not args.no_track,
            "temporal_model": "viterbi3" if not args.no_track else "none",
            "bridge_frames": bridge_frames,
            "viterbi_window": viterbi_window, "viterbi_stay": args.viterbi_stay,
            "viterbi_sigma": args.viterbi_sigma,
            "deprecated": ["evidence_grace", "min_confirm", "min_release", "track_ttl"],
            "recheck": {"enabled": bool(args.recheck), "imgsz": args.recheck_imgsz,
                        "pad": args.recheck_pad, "per_frame": args.recheck_per_frame,
                        "every": args.recheck_every, "min_det": args.recheck_min_det,
                        "draw_grace": args.draw_grace, "relink_frames": relink_frames,
                        "hits": global_stats.get("recheck_hits", 0),
                        "tries": global_stats.get("recheck_tries", 0)},
            "dedup": {"iou": args.dedup_iou,
                      "zombie_iou": args.zombie_iou,
                      "suppressed_frames": global_stats.get("dup_person_frames", 0),
                      "zombie_blocked": global_stats.get("zombie_blocked", 0)},
            "near": {"ratio": args.near_ratio, "kpt_out": args.near_kpt_out,
                     "imgsz": args.near_imgsz, "per_frame": args.near_per_frame,
                     "triggered": global_stats.get("near_triggered", 0),
                     "replaced": global_stats.get("near_replaced", 0)},
            "perspective": {"mode": ("off" if not args.persp else
                                     ("loaded" if (args.persp_calib and
                                                   Path(args.persp_calib).exists()) else "auto")),
                            "y0": calib.y0, "max_k": args.persp_max_k,
                            "frames_used": calib.meta.get("scanned_frames", 0),
                            "samples": calib.samples, "status": calib.status,
                            "points": calib.to_dict()["points"]},
        },
        "performance": {"processing_fps": round(processed / elapsed, 2) if elapsed else None,
                        "steady_state_fps": (round((processed - 1) / (elapsed - warmup_s), 2)
                                             if processed > 1 and elapsed > warmup_s else None),
                        "warmup_s": round(warmup_s, 2),
                        "elapsed_s": round(elapsed, 2),
                        "calib_s": round(calib_s, 2),
                        "stages_ms": {k: round(v / max(1, processed - batch_size) * 1000, 2)
                                      for k, v in phase.items()}},
        "totals": {
            **global_stats,
            "state_histogram": {name: state_hist[i] for i, name in
                                enumerate(("S0_非站立", "S1_过渡", "S2_站立"))},
            "standing_person_frame_ratio": round(
                global_stats["standing_person_frames"] / max(1, global_stats["total_person_frames"]), 4),
            "label_histogram": {LABEL_EN[k]: v for k, v in label_hist.items()},
            "unique_tracks": len(tracks),
        },
        "tracks": [rec.summary() for rec in sorted(tracks.values(), key=lambda r: r.first_t)
                   if rec.n_frames >= 3],
    }

    if args.json_path:
        jp = Path(args.json_path)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[info] 汇总 JSON: {jp}")
    if csv_file is not None:
        print(f"[info] 逐帧 CSV : {args.csv}")

    fps_txt = summary["performance"]["steady_state_fps"]
    print(f"[done] 已渲染 {idx} 帧 → {out_path}  "
          f"(站立帧占比 {summary['totals']['standing_person_frame_ratio'] * 100:.1f}%, "
          + (f"稳态 {fps_txt:.1f} FPS, " if fps_txt else "")
          + f"总耗时 {elapsed:.1f}s)")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("\n[info] 已中断。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
