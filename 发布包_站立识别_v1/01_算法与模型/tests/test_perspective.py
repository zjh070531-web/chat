# -*- coding: utf-8 -*-
"""透视补偿（标定器 + 应用器）单元测试。

运行:  python tests/test_perspective.py   或   pytest tests/test_perspective.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import perspective  # noqa: E402
from stand_detector import classify_pose  # noqa: E402


def _standing_kpts(y2: float, leg_comp: float = 1.0, torso: float = 130.0,
                   width: float = 140.0):
    """构造"站立"骨架：躯干竖直、膝近伸直。

    ``leg_comp`` 复现俯视透视：下半身（髋→踝）纵向被压缩，而躯干基本不变——
    实测近区 leg_drop_ratio 1.03（远区 1.42）、bbox_aspect 2.12（远区 2.77）。
    """
    cx = 600.0
    hip_y = y2 - 1.55 * torso * leg_comp
    knee_y = y2 - 0.95 * torso * leg_comp
    sh_y = hip_y - torso
    head_y = sh_y - 0.35 * torso
    k = np.zeros((17, 3), dtype=np.float32)
    joints = {
        0: (cx, head_y), 1: (cx - 12, head_y - 8), 2: (cx + 12, head_y - 8),
        5: (cx - width * 0.32, sh_y), 6: (cx + width * 0.32, sh_y),
        7: (cx - width * 0.40, sh_y + 0.5 * torso), 8: (cx + width * 0.40, sh_y + 0.5 * torso),
        9: (cx - width * 0.42, sh_y + torso), 10: (cx + width * 0.42, sh_y + torso),
        11: (cx - width * 0.25, hip_y), 12: (cx + width * 0.25, hip_y),
        13: (cx - width * 0.20, knee_y), 14: (cx + width * 0.20, knee_y),
        15: (cx - width * 0.20, y2), 16: (cx + width * 0.20, y2),
    }
    for i, (x, y) in joints.items():
        k[i] = (x, y, 0.95)
    box = (cx - width * 0.5, head_y - 10, cx + width * 0.5, y2)
    return k, box


def test_calibrator_recovers_known_compression():
    """合成"近区被压缩约 1.8 倍"的数据 → 估计系数与真值误差 ≤0.15。"""
    random.seed(0)
    samples = []
    for _ in range(2000):
        y = random.uniform(0, 720)
        comp = 1.0 if y < 450 else (1.0 + 0.9 * (y - 450) / 270)
        samples.append(perspective.Sample(y=y, leg=1.45 / comp + random.gauss(0, 0.04),
                                          ext=2.9 / comp + random.gauss(0, 0.04),
                                          asp=2.7 / comp + random.gauss(0, 0.04)))
    calib = perspective.fit_perspective(samples, y0=450, max_k=2.0)
    assert calib.status == "ok"
    k_leg, k_ext, k_asp = calib.factors(700)
    true_k = 1.0 + 0.9 * (700 - 450) / 270
    assert abs(k_leg - true_k) <= 0.15, (k_leg, true_k)
    assert abs(k_asp - true_k) <= 0.15, (k_asp, true_k)
    assert calib.factors(100) == (1.0, 1.0, 1.0)
    assert calib.factors(449) == (1.0, 1.0, 1.0)


def test_calibrator_monotone_and_clamped():
    random.seed(1)
    samples = []
    for _ in range(1200):
        y = random.uniform(0, 720)
        samples.append(perspective.Sample(y=y, leg=1.4 - 0.0006 * y,
                                          ext=2.8 - 0.001 * y, asp=2.7 - 0.0008 * y))
    calib = perspective.fit_perspective(samples, y0=450, max_k=1.5)
    assert all(1.0 <= v <= 1.5 for v in calib.k_leg + calib.k_ext + calib.k_asp)
    for seq in (calib.k_leg, calib.k_ext, calib.k_asp):
        assert all(b >= a - 1e-9 for a, b in zip(seq, seq[1:])), seq
    assert calib.k_leg[0] == 1.0


def test_calibrator_insufficient_samples_falls_back():
    calib = perspective.fit_perspective([perspective.Sample(700, 1.0, 2.0, 2.0)] * 10)
    assert calib.status == "insufficient_data"
    assert calib.factors(700) == (1.0, 1.0, 1.0)


def test_compensation_is_monotone_and_scoped():
    """结构性保证：补偿只增不减分数；raw_score 保持未补偿值。"""
    kpts, box = _standing_kpts(y2=700.0, leg_comp=0.65)
    raw = classify_pose(kpts, box, persp=None)
    comp = classify_pose(kpts, box, persp=(1.7, 1.7, 1.7))
    assert comp.score >= raw.score - 1e-9, (raw.score, comp.score)
    assert abs(comp.raw_score - raw.score) < 1e-9, (comp.raw_score, raw.score)
    k2, b2 = _standing_kpts(y2=300.0, leg_comp=1.0)
    s1 = classify_pose(k2, b2, persp=None).score
    s2 = classify_pose(k2, b2, persp=(1.0, 1.0, 1.0)).score
    assert abs(s1 - s2) < 1e-12


def test_near_zone_score_crosses_gate_after_compensation():
    """近区被压扁的样本补偿后应越过 0.80 门槛并判为站立。"""
    kpts, box = _standing_kpts(y2=700.0, leg_comp=0.65)
    raw = classify_pose(kpts, box, persp=None)
    assert raw.score < 0.80, raw.score
    calib = perspective.PerspectiveCalib(
        ys=[0.0, 450.0, 550.0, 620.0, 680.0, 1000.0],
        k_leg=[1.0, 1.0, 1.2, 1.2, 1.35, 1.45],
        k_ext=[1.0, 1.0, 1.2, 1.2, 1.35, 1.45],
        k_asp=[1.0, 1.0, 1.2, 1.2, 1.35, 1.45], y0=450.0, samples=500, status="ok")
    k = calib.factors(float(box[3]))
    comp = classify_pose(kpts, box, persp=k)
    assert comp.score >= 0.80, (comp.score, k)
    assert comp.label == "STANDING", comp.label
    assert comp.persp[0] > 1.0


def test_apply_perspective_only_scales_vertical_features():
    feats = {"torso_verticality": 0.99, "knee_angle": 175.0, "leg_drop_ratio": 1.0,
             "body_extension": 2.0, "bbox_aspect": 2.1, "hip_height_ratio": 0.5}
    out = perspective.apply_perspective(feats, 1.5, 1.4, 1.3)
    assert abs(out["leg_drop_ratio"] - 1.5) < 1e-9
    assert abs(out["body_extension"] - 2.8) < 1e-9
    assert abs(out["bbox_aspect"] - 2.73) < 1e-9
    assert out["torso_verticality"] == 0.99 and out["knee_angle"] == 175.0
    assert feats["leg_drop_ratio"] == 1.0


def _run_all() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:                        # pragma: no cover
            failed += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
