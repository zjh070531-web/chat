# -*- coding: utf-8 -*-
"""透视压缩标定与补偿曲线。

固定机位俯视场景下，越靠画面下方（越靠近摄像头）的目标，纵向几何被透视压得越扁：
实测"明确站立"样本中，框底 y 从 300 增到 720 时，髋-踝跨度/躯干长由 1.47 降到 1.03、
框高/宽由 2.77 降到 2.12，站立分数由 0.97 掉到 0.66。本模块按"框底 y"标定一条
压缩曲线 k(y)，用于把这些被压低的特征乘回参考量纲。

结构性保证
----------
* ``k >= 1.0``：补偿只放大特征，而打分是单调不减的加权平均 → 分数只会升高，
  因此已经能过门槛的目标不会因补偿而被取消。
* 只有 ``y >= y0`` 的行参与补偿，``y < y0`` 恒为 1.0 → 上半画面目标逐位不受影响。
* 曲线对 y 强制单调不减并做滑动平均，避免出现"越靠下补偿反而越小"的台阶。
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

DEFAULT_BINS: Tuple[Tuple[float, float], ...] = (
    (0.0, 300.0), (300.0, 450.0), (450.0, 550.0),
    (550.0, 620.0), (620.0, 680.0), (680.0, 1e9),
)
FEATURES = ("leg_drop_ratio", "body_extension", "bbox_aspect")


@dataclass
class Sample:
    """一个"明确站立"样本：框底 y 与三项受压缩特征。"""

    y: float
    leg: float
    ext: float
    asp: float


@dataclass
class PerspectiveCalib:
    """分段线性补偿曲线（断点 y 单调递增，k 单调不减）。"""

    ys: List[float]
    k_leg: List[float]
    k_ext: List[float]
    k_asp: List[float]
    y0: float = 450.0
    samples: int = 0
    status: str = "ok"
    meta: Dict = field(default_factory=dict)

    def factors(self, y2: float) -> Tuple[float, float, float]:
        """返回该框底位置应施加的 (k_leg, k_ext, k_asp)；y2 < y0 时恒为 1。"""
        if self.status != "ok" or y2 < self.y0 or not self.ys:
            return 1.0, 1.0, 1.0
        y = float(y2)
        if y <= self.ys[0]:
            i, t = 0, 0.0
        elif y >= self.ys[-1]:
            i, t = len(self.ys) - 2, 1.0
        else:
            i = max(i for i, v in enumerate(self.ys) if v <= y)
            t = (y - self.ys[i]) / max(1e-6, self.ys[i + 1] - self.ys[i])
        i = max(0, min(i, len(self.ys) - 2))

        def lerp(seq: List[float]) -> float:
            return seq[i] * (1 - t) + seq[i + 1] * t

        return lerp(self.k_leg), lerp(self.k_ext), lerp(self.k_asp)

    def to_dict(self) -> Dict:
        return {
            "y0": self.y0, "samples": self.samples, "status": self.status,
            "meta": self.meta,
            "points": [{"y": y, "k_leg": kl, "k_ext": ke, "k_asp": ka}
                       for y, kl, ke, ka in zip(self.ys, self.k_leg, self.k_ext, self.k_asp)],
        }

    @staticmethod
    def from_dict(d: Dict) -> "PerspectiveCalib":
        return PerspectiveCalib(
            ys=[p["y"] for p in d["points"]],
            k_leg=[p["k_leg"] for p in d["points"]],
            k_ext=[p["k_ext"] for p in d["points"]],
            k_asp=[p["k_asp"] for p in d["points"]],
            y0=float(d.get("y0", 450.0)), samples=int(d.get("samples", 0)),
            status=d.get("status", "ok"), meta=d.get("meta", {}),
        )

    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(path) -> "PerspectiveCalib":
        return PerspectiveCalib.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def identity_calib(y0: float = 450.0, status: str = "off") -> PerspectiveCalib:
    """不补偿的标定（k 恒为 1）。"""
    ys = [0.0, 1000.0]
    return PerspectiveCalib(ys=ys, k_leg=[1.0, 1.0], k_ext=[1.0, 1.0], k_asp=[1.0, 1.0],
                            y0=y0, samples=0, status=status)


def _median(values: Sequence[float], default: float = 1.0) -> float:
    return float(statistics.median(values)) if values else default


def fit_perspective(samples: Sequence[Sample], y0: float = 450.0, max_k: float = 2.0,
                    windows: int = 20, min_total: int = 200) -> PerspectiveCalib:
    """由"明确站立"样本拟合分段线性补偿曲线（滑动窗口中位数 + 单调化 + 截断）。

    用滑动窗（按 y 排序后每窗约 1/windows 的样本）取中位数，避免稀疏区域分箱不稳；
    参考值取最上方窗口的中位数（压缩最轻），``k=clamp(ref/median, 1, max_k)``，
    再强制单调不减。``y < y0`` 在查询时恒返回 1.0。
    """
    if len(samples) < min_total:
        c = identity_calib(y0, status="insufficient_data")
        c.samples = len(samples)
        return c
    order = sorted(samples, key=lambda s: s.y)
    n = len(order)
    win = max(40, n // max(1, windows))
    step = max(1, win // 2)
    ys: List[float] = []
    med_leg: List[float] = []
    med_ext: List[float] = []
    med_asp: List[float] = []
    start = 0
    while start < n:
        chunk = order[start:start + win]
        ys.append(float(statistics.median([s.y for s in chunk])))
        med_leg.append(_median([s.leg for s in chunk]))
        med_ext.append(_median([s.ext for s in chunk]))
        med_asp.append(_median([s.asp for s in chunk]))
        if start + win >= n:
            break
        start += step
    ref_leg = med_leg[0] if med_leg[0] > 0 else 1.0
    ref_ext = med_ext[0] if med_ext[0] > 0 else 1.0
    ref_asp = med_asp[0] if med_asp[0] > 0 else 1.0

    def curve(med: List[float], ref: float) -> List[float]:
        k = [min(max_k, max(1.0, ref / m)) if m > 0 else 1.0 for m in med]
        for i in range(1, len(k)):
            k[i] = max(k[i], k[i - 1])
        k[0] = 1.0
        return k

    return PerspectiveCalib(ys=ys, k_leg=curve(med_leg, ref_leg),
                            k_ext=curve(med_ext, ref_ext), k_asp=curve(med_asp, ref_asp),
                            y0=y0, samples=n, status="ok")


def apply_perspective(features: Dict[str, float], k_leg: float, k_ext: float,
                      k_asp: float) -> Dict[str, float]:
    """返回补偿后的特征副本（仅放大纵向压缩特征，k>=1）。"""
    out = dict(features)
    if "leg_drop_ratio" in out:
        out["leg_drop_ratio"] = out["leg_drop_ratio"] * k_leg
    if "body_extension" in out:
        out["body_extension"] = out["body_extension"] * k_ext
    if "bbox_aspect" in out:
        out["bbox_aspect"] = out["bbox_aspect"] * k_asp
    return out
