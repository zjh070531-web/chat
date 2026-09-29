# -*- coding: utf-8 -*-
"""时序决策层：轨迹级三状态 Viterbi 解码 + 帧级补框辅助。

设计（对应监控场景的"防跳变 + 防漏检"需求）
-------------------------------------------
* 隐状态：``S0`` 非站立 / ``S1`` 起立·坐下过渡 / ``S2`` 稳定站立。
* 转移只允许 ``S0↔S1``、``S1↔S2``；**``S0↔S2`` 概率为 0**，单帧从"非站立"跳到
  "站立"在数学上不可能，必须经过过渡态，从根本上消除单帧误判造成的频闪。
* 发射模型由既有的两个阈值推导，保持"0.60 进站立 / 0.45 退出"的调参语义。
* 中立观测（补框帧、无腿部证据帧）不参与发射，状态完全由转移矩阵决定——
  对应"预测框置信度给中立值，交给时域逻辑推断"的要求。
* 每个轨迹维护长度为 W 的 FIFO 观测窗口，每帧在窗口内做全局最优路径求解，
  只提交最新帧的状态（因果滑窗，实时与离线表现一致）。

本模块只依赖 numpy，便于单元测试。
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

S0, S1, S2 = 0, 1, 2
STATE_NAMES = {S0: "S0_非站立", S1: "S1_过渡", S2: "S2_站立"}
NEG_INF = -1e9


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """两个 (x1, y1, x2, y2) 框的 IoU。"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / max(1e-6, area))


class TrackViterbi:
    """单条轨迹的三状态 Viterbi 解码器（因果滑窗）。"""

    def __init__(self, stand_threshold: float = 0.60, release_threshold: float = 0.45,
                 sigma: float = 0.15, stay: float = 0.92, window: int = 30,
                 min_valid_obs: int = 2):
        self.window = max(3, int(window))
        self.stay = float(min(0.999, max(0.5, stay)))
        self.sigma = float(max(0.02, sigma))
        # 发射均值：以现有阈值为状态翻转临界点
        mu1 = 0.5 * (stand_threshold + release_threshold)
        mu2 = 2.0 * stand_threshold - mu1
        mu0 = 2.0 * release_threshold - mu1
        self.means = (mu0, mu1, mu2)
        # 转移矩阵（对数）：S0↔S1、S1↔S2 允许；S0↔S2 禁止
        switch = 1.0 - self.stay
        log = math.log
        self.log_trans = [
            [log(self.stay), log(switch), NEG_INF],                 # 从 S0
            [log(switch / 2), log(self.stay), log(switch / 2)],     # 从 S1
            [NEG_INF, log(switch), log(self.stay)],                 # 从 S2
        ]
        self.obs: deque = deque(maxlen=self.window)
        self.state = S0
        self.last_valid_obs: Optional[float] = None
        self.min_valid_obs = max(1, int(min_valid_obs))   # 轨迹冷启动防护
        self.valid_obs_count = 0

    # -- 发射 --------------------------------------------------------------
    def _emission(self, obs: Optional[float]) -> List[float]:
        if obs is None:
            return [0.0, 0.0, 0.0]        # 中立观测：三态等似然
        v = float(obs)
        return [-0.5 * ((v - mu) / self.sigma) ** 2 for mu in self.means]

    # -- 解码 --------------------------------------------------------------
    def update(self, obs: Optional[float]) -> int:
        """写入本帧观测（None=中立/缺失），返回本帧状态 S0/S1/S2。"""
        self.obs.append(None if obs is None else float(obs))
        if obs is not None:
            self.last_valid_obs = float(obs)
            self.valid_obs_count += 1
        # 窗口起点先验：延续上一帧已提交的状态（避免窗口滑走后"忘记历史"而漂移），
        # 切换到其它状态需要付出与一次转移相当的对数代价。
        log_switch = math.log(1.0 - self.stay)
        logp = [0.0 if s == self.state else log_switch for s in range(3)]
        for o in self.obs:
            emit = self._emission(o)
            new = [NEG_INF] * 3
            for s_to in range(3):
                for s_from in range(3):
                    t = self.log_trans[s_from][s_to]
                    if t <= NEG_INF:
                        continue
                    val = logp[s_from] + t + emit[s_to]
                    if val > new[s_to]:
                        new[s_to] = val
            logp = new
        self.state = int(max(range(3), key=lambda s: logp[s]))
        # 冷启动防护：实测证据不足 min_valid_obs 帧时不允许进入 S2（单帧高分不得成为站立）
        if self.state == S2 and self.valid_obs_count < self.min_valid_obs:
            self.state = S1
        return self.state

    # -- 继承（ID 重连） ----------------------------------------------------
    def clone_from(self, other: "TrackViterbi") -> None:
        """继承另一解码器的窗口与状态（用于轨道 ID 切换时保持判断连续）。"""
        self.obs = deque(other.obs, maxlen=self.window)
        self.state = other.state
        self.last_valid_obs = other.last_valid_obs
        self.valid_obs_count = other.valid_obs_count


def kalman_box(strack, advance: bool = False) -> Optional[np.ndarray]:
    """从跟踪器丢失轨迹中取卡尔曼预测框（无可用轨迹时返回 None）。

    ``advance=False`` 时只读取当前预测状态（ByteTrack 每帧已对 lost 轨迹做过一步预测），
    避免重复 predict() 造成外推过头。
    """
    if strack is None:
        return None
    try:
        if advance:
            strack.predict()
        box = np.asarray(strack.xyxy, dtype=np.float32).reshape(4)
        if np.all(np.isfinite(box)) and box[2] > box[0] and box[3] > box[1]:
            return box
    except Exception:
        return None
    return None


def translate_kpts(kpts: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """把缓存的关键点按框中心位移平移（补框帧绘制用）。"""
    out = np.array(kpts, dtype=np.float32, copy=True)
    out[:, 0] += dx
    out[:, 1] += dy
    return out
