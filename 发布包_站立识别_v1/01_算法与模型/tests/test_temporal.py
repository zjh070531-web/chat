# -*- coding: utf-8 -*-
"""时序决策层测试：三状态 Viterbi 防抖、漏检桥接、状态继承。

运行:  python tests/test_temporal.py   或   pytest tests/test_temporal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temporal import S0, S1, S2, TrackViterbi, iou, translate_kpts  # noqa: E402

STAY = 0.92
SIGMA = 0.15      # 参数扫描确定：σ=0.15 时防抖与响应速度同时达标
WIN = 30


def make(window: int = WIN) -> TrackViterbi:
    return TrackViterbi(0.60, 0.45, sigma=SIGMA, stay=STAY, window=window)


def run(v: TrackViterbi, seq) -> list:
    return [v.update(o) for o in seq]


def test_single_frame_spike_is_rejected():
    """单帧高分尖峰不得进入 S2（S0→S2 被禁止，必须经过 S1）。"""
    seq = [0.10] * 20 + [0.95] + [0.10] * 20
    states = run(make(), seq)
    assert S2 not in states, f"单帧尖峰触发了站立：{states}"


def test_cold_start_single_frame_cannot_be_standing():
    """冷启动防护：轨迹第一帧即使高分也不能直接判为 S2。"""
    v = make()
    assert v.update(0.95) != S2, "首帧高分直接进入 S2（冷启动无转移代价）"
    assert v.update(0.95) == S2, "第二帧应当可以进入 S2"


def test_two_frame_spike_is_rejected():
    """两帧高分也不应直接跳到 S2（转移代价 + 发射证据不足）。"""
    seq = [0.10] * 20 + [0.95, 0.95] + [0.10] * 20
    states = run(make(), seq)
    assert states.count(S2) <= 1, f"两帧尖峰产生了 {states.count(S2)} 帧站立"


def test_real_stand_up_is_detected_quickly():
    """真实起立：经过 S1，S2 应在 3 帧内到达，站立帧数误差 ≤2。"""
    seq = [0.15] * 15 + [0.90] * 25
    states = run(make(), seq)
    assert S1 in states, "未经过过渡态"
    first_s2 = states.index(S2) if S2 in states else -1
    assert first_s2 > 0, "未检出站立"
    assert first_s2 - 15 <= 3, f"进入站立用了 {first_s2 - 15} 帧"
    assert abs(states.count(S2) - 25) <= 2, f"站立帧数 {states.count(S2)} 与真实 25 相差过大"


def test_bridge_holds_state_during_dropout():
    """漏检 30 帧（中立观测）期间状态保持 S2，不被误释放。"""
    seq = [0.15] * 15 + [0.90] * 20 + [None] * 30
    states = run(make(), seq)
    assert states[-1] == S2, "漏检期间状态掉出站立"
    assert all(s == S2 for s in states[-30:]), "漏检期间状态不稳定"


def test_real_sit_down_exits_quickly():
    """真实坐下（持续低分）：应经 S1 退出 S2，5 帧内到 S0。"""
    seq = [0.90] * 25 + [0.12] * 30
    states = run(make(), seq)
    tail = states[-20:]
    assert tail[-1] == S0, f"未回到 S0：{tail}"
    assert S2 not in tail, "退出后仍残留站立状态"


def test_forbidden_transition_matrix():
    v = make()
    assert v.log_trans[S0][S2] < -1e8 and v.log_trans[S2][S0] < -1e8
    assert v.log_trans[S0][S1] > -1e8 and v.log_trans[S1][S2] > -1e8


def test_clone_from_inherits_state():
    """ID 重连继承：新轨迹直接延续旧轨迹的站立状态。"""
    old = make()
    run(old, [0.15] * 15 + [0.90] * 20)
    assert old.state == S2
    new = make()
    new.clone_from(old)
    assert new.state == S2
    assert new.update(0.9) == S2, "继承后立即掉出站立"
    assert len(new.obs) == len(old.obs)


def test_window_bounds_memory():
    v = make(window=10)
    for _ in range(500):
        v.update(0.9)
    assert len(v.obs) <= 10, len(v.obs)


def test_iou_and_translate_helpers():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    k = np.zeros((17, 3), np.float32)
    k[:, 0] = 5.0
    out = translate_kpts(k, 3.0, -2.0)
    assert out[0, 0] == 8.0 and out[0, 1] == -2.0
    assert k[0, 0] == 5.0, "不得修改原始关键点"


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
