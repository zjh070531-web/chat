# -*- coding: utf-8 -*-
"""视频 I/O 层测试：探测、管道解码、跳转一致性、回退链、批量端到端。

运行:  python tests/test_video_io.py    或   pytest tests/test_video_io.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import video_io  # noqa: E402

W, H, FPS, N = 320, 240, 10.0, 30


def make_video(path: Path, n: int = N) -> None:
    """生成灰度递增 + 移动方块的合成视频，便于按帧校验内容。"""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(n):
        frame = np.full((H, W, 3), (i * 6) % 220 + 16, dtype=np.uint8)
        x = 10 + (i * 7) % (W - 40)
        cv2.rectangle(frame, (x, 60), (x + 30, 100), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def _tmp_video() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="stand_io_"))
    path = tmp / "synthetic.mp4"
    make_video(path)
    return path


def test_probe_video():
    path = _tmp_video()
    info = video_io.probe_video(path)
    assert info.width == W and info.height == H, (info.width, info.height)
    assert abs(info.fps - FPS) < 0.5, info.fps
    assert info.frame_count == N, info.frame_count
    assert abs(info.duration - N / FPS) < 0.3, info.duration


def test_ffmpeg_source_reads_all_frames():
    path = _tmp_video()
    info = video_io.probe_video(path)
    src = video_io.FfmpegFrameSource(info, (W, H))
    frames = []
    while True:
        f = src.read()
        if f is None:
            break
        frames.append(f)
        assert f.flags.writeable, "帧数组必须可写（后续要在上面绘制）"
    src.close()
    assert len(frames) == N, len(frames)
    assert frames[0].shape == (H, W, 3)
    # 亮度应随帧号递增（合成视频特征）
    means = [float(f.mean()) for f in frames[:8]]
    assert means[0] < means[-1], means


def test_seek_modes_agree():
    """快速跳转前的混合跳转必须与顺序解码得到同一帧（t_v 素材上曾出现坏帧）。"""
    path = _tmp_video()
    info = video_io.probe_video(path)
    target = 2.0                      # 第 20 帧
    ref = video_io.FfmpegFrameSource(info, (W, H))
    seq = [ref.read() for _ in range(int(target * FPS) + 1)]
    ref.close()
    expect = seq[-1]

    for seek in ("accurate", "hybrid"):
        src = video_io.FfmpegFrameSource(info, (W, H), start_time=target, seek=seek)
        got = src.read()
        src.close()
        diff = float(np.abs(got.astype(int) - expect.astype(int)).mean())
        assert diff < 4.0, f"{seek} 跳转帧与顺序解码差异过大: {diff}"


def test_fallback_when_ffmpeg_missing(monkeypatch=None):
    path = _tmp_video()
    info = video_io.probe_video(path)
    original = video_io.ffmpeg_exe
    video_io.ffmpeg_exe = lambda: None      # 模拟没有 ffmpeg
    try:
        src = video_io.open_frame_source(info, (W, H), io_mode="auto")
        assert src.kind == "opencv", src.kind
        assert src.read() is not None
        src.close()
        try:
            video_io.open_frame_source(info, (W, H), io_mode="ffmpeg")
            raise AssertionError("io_mode=ffmpeg 时应当直接报错而不是回退")
        except RuntimeError:
            pass
    finally:
        video_io.ffmpeg_exe = original


def test_encoder_selection_and_fallback():
    path = _tmp_video()
    if video_io.ffmpeg_exe():
        caps = video_io.ffmpeg_capabilities()["encoders"]
        sink = video_io.open_video_sink(path.parent / "out_nv.mp4", (W, H), FPS,
                                        encoder="auto")
        assert sink.kind == "ffmpeg"
        assert sink.encoder in ("h264_nvenc", "libx264")
        if "h264_nvenc" in caps:
            assert sink.encoder == "h264_nvenc"
        sink.close()
    sink2 = video_io.open_video_sink(path.parent / "out_cv.mp4", (W, H), FPS, encoder="mp4v")
    assert sink2.kind == "opencv"
    sink2.close()


def test_encoder_pick_rejects_unavailable():
    try:
        video_io.FfmpegVideoSink._pick_encoder("nvenc", {"libx264"}, 23, "p4", "veryfast", 20)
        raise AssertionError("nvenc 不可用时应当报错")
    except RuntimeError:
        pass
    codec, params = video_io.FfmpegVideoSink._pick_encoder("auto", {"libx264"}, 23, "p4",
                                                            "veryfast", 20)
    assert codec == "libx264" and "-crf" in params


def test_cli_batch_end_to_end():
    """整机烟测：批量推理跑通，输出帧数一致，JSON 含新增字段。"""
    path = _tmp_video()
    out = path.parent / "batch_out.mp4"
    js = path.parent / "batch_out.json"
    cmd = [sys.executable, str(ROOT / "stand_detector.py"), "--source", str(path),
           "--output", str(out), "--json", str(js), "--batch", "4", "--quiet",
           "--max-frames", str(N)]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    assert res.returncode == 0, res.stderr[-800:]
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["video"]["frames_read"] == N, data["video"]["frames_read"]
    assert data["io"]["batch"] == 4
    assert "stages_ms" in data["performance"]
    assert set(data["performance"]["stages_ms"]) >= {"read", "infer", "track", "draw", "encode"}
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == N
    cap.release()


def test_cli_opencv_legacy_path():
    """回退路径（OpenCV + x264）必须仍能跑通。"""
    path = _tmp_video()
    out = path.parent / "legacy.mp4"
    cmd = [sys.executable, str(ROOT / "stand_detector.py"), "--source", str(path),
           "--output", str(out), "--io", "opencv", "--batch", "1", "--encoder", "x264",
           "--x264-preset", "medium", "--quiet", "--max-frames", str(N)]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    assert res.returncode == 0, res.stderr[-800:]
    assert out.exists()


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
        except Exception as exc:                    # pragma: no cover
            failed += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
