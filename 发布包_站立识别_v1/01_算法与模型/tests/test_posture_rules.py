# -*- coding: utf-8 -*-
"""站立判定规则的单元测试（合成骨架，不依赖 GPU/模型）。

运行:  python tests/test_posture_rules.py
      pytest tests/test_posture_rules.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stand_detector import (  # noqa: E402
    LYING, SITTING, STANDING, UNKNOWN, PostureSmoother, classify_pose,
)

# COCO 关键点索引
NOSE = 0
L_EYE, R_EYE, L_EAR, R_EAR = 1, 2, 3, 4
L_SH, R_SH = 5, 6
L_EL, R_EL = 7, 8
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12
L_KN, R_KN = 13, 14
L_AN, R_AN = 15, 16


def make_kpts(joints: dict) -> np.ndarray:
    """joints: {索引: (x, y)}，未给出的点置信度为 0。"""
    k = np.zeros((17, 3), dtype=np.float32)
    for i, (x, y) in joints.items():
        k[i] = (x, y, 0.95)
    return k


def full_body(hip_y: float, sh_y: float, nose_y: float, knee_y: float, ank_y: float,
              cx: float = 500.0, knee_dx: float = 0.0, ank_dx: float = 0.0,
              sh_dx: float = 55.0, hip_dx: float = 45.0) -> dict:
    """构造一个左右对称的完整骨架（图像坐标，y 向下）。"""
    return {
        NOSE: (cx, nose_y), L_EYE: (cx - 20, nose_y - 12), R_EYE: (cx + 20, nose_y - 12),
        L_SH: (cx - sh_dx, sh_y), R_SH: (cx + sh_dx, sh_y),
        L_EL: (cx - sh_dx - 25, sh_y + 120), R_EL: (cx + sh_dx + 25, sh_y + 120),
        L_WR: (cx - sh_dx - 25, sh_y + 240), R_WR: (cx + sh_dx + 25, sh_y + 240),
        L_HIP: (cx - hip_dx, hip_y), R_HIP: (cx + hip_dx, hip_y),
        L_KN: (cx - 20 + knee_dx, knee_y), R_KN: (cx + 20 + knee_dx, knee_y),
        L_AN: (cx - 20 + ank_dx, ank_y), R_AN: (cx + 20 + ank_dx, ank_y),
    }


# --- 场景 1: 正立站立（躯干 290px，髋->踝 490px，膝接近直线）--------------
STAND_POSE = make_kpts(full_body(hip_y=470, sh_y=180, nose_y=60, knee_y=720, ank_y=960))
STAND_BOX = (400, 40, 600, 1000)

# --- 场景 2: 坐在椅子上（大腿水平、小腿垂直、膝角约 90°）------------------
SIT_POSE = make_kpts(full_body(hip_y=740, sh_y=450, nose_y=380, knee_y=745, ank_y=960,
                               knee_dx=250, ank_dx=260))
SIT_BOX = (460, 370, 790, 985)

# --- 场景 3: 躺在地面（躯干近水平、人体框扁宽）----------------------------
LIE_POSE = make_kpts({
    NOSE: (120, 860), L_EYE: (140, 855), R_EYE: (150, 865),
    L_SH: (200, 870), R_SH: (210, 880),
    L_EL: (330, 880), R_EL: (340, 890),
    L_WR: (470, 890), R_WR: (480, 900),
    L_HIP: (620, 880), R_HIP: (630, 890),
    L_KN: (820, 890), R_KN: (830, 900),
    L_AN: (980, 900), R_AN: (990, 910),
})
LIE_BOX = (60, 840, 1010, 920)

# --- 场景 4: 只拍到上半身（腿部被遮挡）------------------------------------
UPPER_ONLY = make_kpts({
    NOSE: (500, 60), L_EYE: (480, 45), R_EYE: (520, 45),
    L_SH: (445, 180), R_SH: (555, 180),
    L_EL: (420, 300), R_EL: (580, 300),
    L_WR: (415, 420), R_WR: (585, 420),
})
UPPER_BOX = (400, 40, 600, 480)


def test_standing():
    r = classify_pose(STAND_POSE, STAND_BOX)
    assert r.label == STANDING, f"站立被误判为 {r.label}, score={r.score:.3f}"
    assert r.score > 0.85, r.score
    assert r.coverage > 0.95, r.coverage
    assert not r.low_confidence


def test_sitting():
    r = classify_pose(SIT_POSE, SIT_BOX)
    assert r.label == SITTING, f"坐姿被误判为 {r.label}, score={r.score:.3f}"
    assert r.score < 0.60, r.score
    assert r.features["knee_angle"] < 130, r.features["knee_angle"]


def test_lying():
    r = classify_pose(LIE_POSE, LIE_BOX)
    assert r.label == LYING, f"躺卧被误判为 {r.label}, score={r.score:.3f}"


# --- 场景 5: 肩/髋可见，但腿被桌子等遮挡 -----------------------------------
HIP_VISIBLE = make_kpts({
    NOSE: (500, 60), L_EYE: (480, 45), R_EYE: (520, 45),
    L_SH: (445, 180), R_SH: (555, 180),
    L_EL: (420, 300), R_EL: (580, 300),
    L_WR: (415, 420), R_WR: (585, 420),
    L_HIP: (455, 470), R_HIP: (545, 470),
})
HIP_VISIBLE_BOX = (400, 40, 600, 520)


def test_upper_body_only_is_unknown():
    """连髋部都不可见时无法判断姿态，应给出'未知'而不是猜一个结果。"""
    r = classify_pose(UPPER_ONLY, UPPER_BOX)
    assert r.label == UNKNOWN, r.label
    assert r.coverage < 0.30, r.coverage


def test_hip_visible_legs_occluded_is_unknown_by_default():
    """默认策略：看不到腿时判为'未知'，避免把坐在桌后的人误判为站立。"""
    r = classify_pose(HIP_VISIBLE, HIP_VISIBLE_BOX)
    assert r.label == UNKNOWN, r.label
    assert not r.leg_evidence
    assert r.score > 0.6, "站立分数仍应保留，供用户自行判断"


def test_legs_occluded_with_judge_policy():
    """--torso-only judge 时按躯干与框形判定（旧行为），并标记低置信。"""
    r = classify_pose(HIP_VISIBLE, HIP_VISIBLE_BOX, require_leg_evidence=False)
    assert r.label == STANDING, r.label
    assert r.low_confidence, "腿部缺失时应标记为低置信"


def test_no_keypoints_is_unknown():
    r = classify_pose(np.zeros((17, 3), dtype=np.float32), (0, 0, 100, 400))
    assert r.label == UNKNOWN, r.label
    assert r.coverage < 0.30, r.coverage


def test_arms_out_still_standing():
    """张开双臂使人体框变宽，也不应误判为躺卧。"""
    wide = dict(full_body(hip_y=470, sh_y=180, nose_y=60, knee_y=720, ank_y=960))
    wide[L_WR] = (120, 180)
    wide[R_WR] = (880, 180)
    r = classify_pose(make_kpts(wide), (100, 40, 900, 1000))
    assert r.label == STANDING, r.label


def test_smoother_hysteresis():
    s = PostureSmoother(window=9, min_confirm=3, min_release=5,
                        stand_threshold=0.6, release_threshold=0.45)
    events = []
    standing = False
    for _ in range(3):
        standing, ev = s.update(0.85)
        events.append(ev)
    assert standing and events.count("on") == 1, events
    standing, _ = s.update(0.20)
    assert standing, "单帧噪声不应立刻取消站立状态"
    for _ in range(8):
        standing, _ = s.update(0.10)
    assert not standing, "持续低分后应退出站立状态"


def test_smoother_rejects_single_frame_spike():
    s = PostureSmoother(window=9, min_confirm=4, min_release=3)
    standing = False
    for i in range(12):
        standing, _ = s.update(0.9 if i == 5 else 0.2)
    assert not standing, "孤立的高分不应触发站立判定"


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
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
