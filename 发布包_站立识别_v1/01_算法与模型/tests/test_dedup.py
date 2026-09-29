# -*- coding: utf-8 -*-
"""标注层去重（NMS）与近距判据测试。

运行:  python tests/test_dedup.py   或   pytest tests/test_dedup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stand_detector import Candidate, dedup_candidates, kpt_out_ratio  # noqa: E402
from stand_detector import PostureResult, UNKNOWN  # noqa: E402


def make_cand(tid: int, box, priority: int, smooth: float, confirmed: int = 0) -> Candidate:
    result = PostureResult(label=UNKNOWN, score=smooth, coverage=1.0, leg_evidence=True)
    return Candidate(tid=tid, box=np.asarray(box, dtype=np.float32),
                     kpts=np.zeros((17, 3), dtype=np.float32), result=result,
                     state=2, smooth=smooth, held=0 if priority >= 2 else 1,
                     priority=priority, is_standing=True,
                     confirmed_idx=confirmed, first_idx=tid)


def test_dedup_keeps_highest_priority():
    """同一位置三个框：只保留优先级最高的一个（真实检测 > 补检命中 > 沿用）。"""
    cands = [
        make_cand(1, (100, 100, 200, 400), priority=1, smooth=0.95),   # 沿用
        make_cand(2, (102, 101, 201, 402), priority=2, smooth=0.90),   # 真实检测
        make_cand(3, (99, 99, 199, 398), priority=2, smooth=0.93),     # 补检命中(同级高分)
    ]
    kept, hidden = dedup_candidates(cands, 0.60)
    assert len(kept) == 1, f"应只保留 1 个，实际 {len(kept)}"
    assert kept[0].tid == 3, f"应保留同级中分数最高者(id=3)，实际 id={kept[0].tid}"
    assert {c.tid for c in hidden} == {1, 2}


def test_dedup_keeps_distinct_people():
    """相邻两人框重叠但不超阈值时应各自保留。"""
    cands = [
        make_cand(1, (100, 100, 260, 500), priority=2, smooth=0.9),
        make_cand(2, (240, 110, 400, 510), priority=2, smooth=0.9),
    ]
    kept, hidden = dedup_candidates(cands, 0.60)
    assert len(kept) == 2 and not hidden


def test_priority_zero_is_hidden():
    """priority<1（超出沿用窗口）的候选不参与展示。"""
    cands = [make_cand(1, (100, 100, 200, 400), priority=0, smooth=0.9)]
    kept, hidden = dedup_candidates(cands, 0.60)
    assert not kept and len(hidden) == 1


def test_dedup_threshold_boundary():
    """IoU 刚好低于阈值保留、超过阈值抑制。"""
    a = (0, 0, 100, 100)
    b = (40, 0, 140, 100)      # IoU = 60/140 ≈ 0.43
    c = (10, 0, 110, 100)      # IoU = 90/110 ≈ 0.82
    kept, hidden = dedup_candidates([make_cand(1, a, 2, 0.9), make_cand(2, b, 2, 0.9)], 0.60)
    assert len(kept) == 2, "低重叠不应被抑制"
    kept, hidden = dedup_candidates([make_cand(1, a, 2, 0.9), make_cand(2, c, 2, 0.9)], 0.60)
    assert len(kept) == 1 and len(hidden) == 1, "高重叠应被抑制"


def test_kpt_out_ratio():
    """近距判据：关键点越界比例。"""
    W, H, kconf = 1280, 720, 0.35
    k = np.zeros((17, 3), dtype=np.float32)
    k[:, 2] = 0.9
    k[:, 0] = 640.0
    k[:, 1] = 360.0
    assert kpt_out_ratio(k, W, H, kconf) == 0.0
    k[0, 0] = -10.0            # 1/17 越界
    assert abs(kpt_out_ratio(k, W, H, kconf) - 1 / 17) < 1e-6
    k[:6, 0] = -10.0           # 6/17 ≈ 0.35 ≥ 0.30 → 触发
    assert kpt_out_ratio(k, W, H, kconf) >= 0.30
    k2 = np.zeros((17, 3), dtype=np.float32)    # 全部不可见
    assert kpt_out_ratio(k2, W, H, kconf) == 1.0


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
