# -*- coding: utf-8 -*-
"""视频 I/O 层：ffmpeg 管道解码（NVDEC）+ 编码（NVENC），带 CPU / OpenCV 回退。

设计要点
--------
* 解码在 ffmpeg 子进程里完成，并**在管道内直接缩放到目标分辨率**，Python 只接收
  目标分辨率的 BGR 帧。4K→720p 时每帧数据量从 24 MB 降到 2.7 MB，这是本机 4K
  监控视频提速的关键（实测 32 ms/帧 → 10 ms/帧）。
* 解码可选 NVDEC（``hevc_cuvid``/``h264_cuvid``），把 CPU 让给绘制等其它工作。
* 跳转默认"精确跳转"（``-i src -ss N``，顺序解码后丢弃）：监控 HEVC 多为开 GOP，
  ``-ss`` 放在 ``-i`` 前的快速跳转会解出损坏帧（实测同一时间点人数 15 vs 1）。
* 编码优先 NVENC（实测 720p 281 FPS，libx264 medium 仅 28.7 FPS），失败回退
  libx264，再回退 OpenCV mp4v；写入在独立线程 + 有界队列进行，避免管道背压阻塞主循环。
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Full, Queue
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_CODEC_TO_CUVID = {
    "hevc": "hevc_cuvid",
    "h264": "h264_cuvid",
    "av1": "av1_cuvid",
    "mjpeg": "mjpeg_cuvid",
    "mpeg2video": "mpeg2_cuvid",
    "mpeg4": "mpeg4_cuvid",
    "vp8": "vp8_cuvid",
    "vp9": "vp9_cuvid",
}
_ERROR_HINTS = ("error", "corrupt", "invalid", "Could not find ref", "missing picture",
                "CABAC", "cu_qp_delta", "POC")


def ffmpeg_exe() -> Optional[str]:
    """返回可用 ffmpeg 可执行文件路径（可用环境变量 STAND_FFMPEG 覆盖）。"""
    env = os.environ.get("STAND_FFMPEG")
    if env and Path(env).exists():
        return env
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        return exe if exe and Path(exe).exists() else None
    except Exception:
        return None


_CAPS: Optional[Dict[str, set]] = None


def ffmpeg_capabilities(refresh: bool = False) -> Dict[str, set]:
    """解析 ffmpeg 支持的编码器/解码器/硬件加速（结果缓存）。"""
    global _CAPS
    if _CAPS is not None and not refresh:
        return _CAPS
    caps: Dict[str, set] = {"encoders": set(), "decoders": set(), "hwaccels": set()}
    exe = ffmpeg_exe()
    if exe:
        def run(flag: str, key: str) -> None:
            try:
                out = subprocess.run([exe, "-hide_banner", flag], capture_output=True,
                                     text=True, errors="ignore").stdout
                for line in out.splitlines():
                    parts = line.split()
                    if len(parts) >= 2 and parts[0] and parts[0][0] in "VASD" and len(parts[0]) <= 8:
                        caps[key].add(parts[1])
            except Exception:
                pass
        run("-encoders", "encoders")
        run("-decoders", "decoders")
        try:
            out = subprocess.run([exe, "-hide_banner", "-hwaccels"], capture_output=True,
                                 text=True, errors="ignore").stdout
            caps["hwaccels"] = {line.strip() for line in out.splitlines()
                                if line.strip() and not line.startswith("Hardware")}
        except Exception:
            pass
    _CAPS = caps
    return caps


@dataclass
class VideoInfo:
    path: Optional[Path] = None
    fps: float = 30.0
    width: int = 0
    height: int = 0
    duration: float = 0.0
    frame_count: int = 0
    codec: str = ""
    rotation: int = 0
    has_audio: bool = False
    audio_codec: str = ""


def probe_video(path) -> VideoInfo:
    """用 ffmpeg 解析视频元数据（失败时回退 OpenCV）。"""
    path = Path(path)
    info = VideoInfo(path=path)
    exe = ffmpeg_exe()
    if exe:
        try:
            err = subprocess.run([exe, "-hide_banner", "-i", str(path)], capture_output=True,
                                 text=True, errors="ignore").stderr
            m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", err)
            if m:
                info.duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
            vm = re.search(r"Stream #\d+:\d+.*?: Video: (\w+)", err)
            if vm:
                info.codec = vm.group(1).lower()
            sm = re.search(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})", err)
            if sm:
                info.width, info.height = int(sm.group(1)), int(sm.group(2))
            fm = re.search(r"(\d+(?:\.\d+)?)\s*fps", err)
            if fm and float(fm.group(1)) > 0.1:
                info.fps = float(fm.group(1))
            am = re.search(r"Stream #\d+:\d+.*?: Audio: ([\w_]+)", err)
            if am:
                info.has_audio = True
                info.audio_codec = am.group(1).lower()
            rm = re.search(r"rotate\s*:\s*(\d+)", err)
            if rm:
                info.rotation = int(rm.group(1)) % 360
            elif "displaymatrix" in err:
                mm = re.search(r"displaymatrix:\s*(-?\d+)", err)
                if mm:
                    info.rotation = (-int(mm.group(1))) % 360
        except Exception:
            pass
    if not info.width or not info.height:
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            info.width = info.width or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info.height = info.height or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            f = cap.get(cv2.CAP_PROP_FPS)
            if f and f > 0.1:
                info.fps = float(f)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not info.duration and n and info.fps:
                info.duration = n / info.fps
        cap.release()
    info.frame_count = int(round(info.duration * info.fps)) if info.duration and info.fps else 0
    return info


class FrameSource:
    """帧源接口：read() 返回 BGR ndarray，结束时返回 None。"""

    kind = "base"
    hwaccel: Optional[str] = None
    decode_errors = 0

    def __init__(self, size: Tuple[int, int], fps: float):
        self.width, self.height = int(size[0]), int(size[1])
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.frames_read = 0

    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class CvFrameSource(FrameSource):
    """OpenCV 解码（摄像头 / 兼容 / 回退路径）。"""

    kind = "opencv"

    def __init__(self, info: VideoInfo, size: Tuple[int, int], start_time: float = 0.0,
                 end_time: Optional[float] = None, camera: Optional[int] = None):
        super().__init__(size, info.fps)
        self.camera = camera
        self.info = info
        self.start_time = float(start_time)
        self.end_time = end_time
        self.cap = cv2.VideoCapture(camera if camera is not None else str(info.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {camera if camera is not None else info.path}")
        if camera is None and self.start_time > 0:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, self.start_time * 1000.0)

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        if self.camera is None and self.end_time is not None:
            if self.start_time + self.frames_read / self.fps >= self.end_time:
                return None
        self.frames_read += 1
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            interp = cv2.INTER_AREA if frame.shape[1] > self.width else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (self.width, self.height), interpolation=interp)
        return frame

    def close(self) -> None:
        if getattr(self, "cap", None) is not None:
            self.cap.release()
            self.cap = None


class FfmpegFrameSource(FrameSource):
    """ffmpeg 管道解码：NVDEC 解码 + 管道内缩放，Python 只收目标分辨率帧。"""

    kind = "ffmpeg"

    def __init__(self, info: VideoInfo, size: Tuple[int, int], start_time: float = 0.0,
                 end_time: Optional[float] = None, hwaccel: str = "auto",
                 seek: str = "hybrid", seek_preroll: float = 10.0, probe_frames: int = 5,
                 skip_nonkey: bool = False, log=print):
        super().__init__(size, info.fps)
        self.exe = ffmpeg_exe()
        if not self.exe:
            raise RuntimeError("未找到 ffmpeg 可执行文件")
        self.info = info
        self.start_time = float(start_time)
        self.end_time = end_time
        self.seek = seek
        self.seek_preroll = float(seek_preroll)
        self.skip_nonkey = bool(skip_nonkey)
        self.log = log
        self.frame_bytes = self.width * self.height * 3
        self.hwaccel: Optional[str] = None
        self.attempts: List[str] = []
        self._proc: Optional[subprocess.Popen] = None
        self._buf: deque = deque()
        self._stderr: deque = deque(maxlen=200)
        self._thread: Optional[threading.Thread] = None

        last_err: Optional[Exception] = None
        for mode in self._hw_candidates(hwaccel):
            try:
                self._spawn(mode)
                self._prefetch(probe_frames)
                self.hwaccel = mode
                break
            except Exception as exc:
                last_err = exc
                self.log(f"[warn] ffmpeg 解码方式 {mode or 'cpu'} 不可用：{exc}")
                self._kill()
        else:
            raise RuntimeError(f"ffmpeg 无法解码该视频：{last_err}")

    def _hw_candidates(self, hwaccel: str) -> List[Optional[str]]:
        if hwaccel == "cpu":
            return [None]
        cuvid = _CODEC_TO_CUVID.get((self.info.codec or "").lower())
        hw_ok = "cuda" in ffmpeg_capabilities()["hwaccels"]
        if hwaccel == "cuda" and not (cuvid and hw_ok):
            raise RuntimeError(f"源编码 {self.info.codec} 无可用 CUDA 解码器")
        if cuvid and hw_ok:
            return ["cuda", None]
        return [None]

    def _spawn(self, hw_mode: Optional[str]) -> None:
        cmd = [self.exe, "-hide_banner", "-nostdin", "-loglevel", "warning"]
        if self.skip_nonkey:          # 只解码关键帧：用于跨整段快速采样
            cmd += ["-skip_frame", "nokey"]
        if hw_mode == "cuda":
            cmd += ["-hwaccel", "cuda", "-c:v", _CODEC_TO_CUVID[(self.info.codec or "").lower()]]
        pre_trim = 0.0
        if self.seek == "fast" and self.start_time > 0:
            cmd += ["-ss", f"{self.start_time:.3f}"]
        elif self.seek == "hybrid" and self.start_time > 0:
            if self.start_time > self.seek_preroll:
                # 先快跳到目标点前 10s（关键帧），再在输入端精确裁剪，避免开 GOP 解出坏帧
                cmd += ["-ss", f"{self.start_time - self.seek_preroll:.3f}"]
                pre_trim = self.seek_preroll
            else:
                pre_trim = self.start_time      # 目标点在前 10s 内：从头顺序解码到目标点
        cmd += ["-i", str(self.info.path)]
        if pre_trim > 0:
            cmd += ["-ss", f"{pre_trim:.3f}"]
        elif self.seek == "accurate" and self.start_time > 0:
            cmd += ["-ss", f"{self.start_time:.3f}"]
        if self.end_time is not None and self.end_time > self.start_time:
            cmd += ["-t", f"{self.end_time - self.start_time:.3f}"]
        if self.info.width != self.width or self.info.height != self.height:
            cmd += ["-vf", f"scale={self.width}:{self.height}"]
        if self.skip_nonkey:
            # 否则 rawvideo 输出会按帧率把关键帧补齐成整段长度
            cmd += ["-fps_mode", "vfr"]
        cmd += ["-an", "-sn", "-dn", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        self._cmd = cmd
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      bufsize=self.frame_bytes * 2)
        self._thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._thread.start()
        self.attempts.append(hw_mode or "cpu")

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for raw in proc.stderr:
                line = raw.decode("utf-8", "ignore").strip()
                if not line:
                    continue
                self._stderr.append(line)
                if any(h in line for h in _ERROR_HINTS):
                    self.decode_errors += 1
        except Exception:
            pass

    def _read_frame(self) -> Optional[np.ndarray]:
        """直接读进可写 numpy 数组（readinto），避免只读缓冲与额外拷贝。"""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return None
        arr = np.empty(self.frame_bytes, dtype=np.uint8)
        view = memoryview(arr)
        got = 0
        while got < self.frame_bytes:
            n = proc.stdout.readinto(view[got:])
            if not n:
                return None
            got += n
        return arr.reshape(self.height, self.width, 3)

    def _prefetch(self, n: int) -> None:
        for _ in range(max(0, int(n))):
            frame = self._read_frame()
            if frame is None:
                break
            self._buf.append(frame)
        if not self._buf:
            self._kill()
            raise RuntimeError("ffmpeg 未解出任何帧")

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=3)
        except Exception:
            pass

    def read(self) -> Optional[np.ndarray]:
        if self._buf:
            self.frames_read += 1
            return self._buf.popleft()
        frame = self._read_frame()
        if frame is not None:
            self.frames_read += 1
        return frame

    def close(self) -> None:
        self._kill()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    @property
    def stderr_sample(self) -> List[str]:
        return list(self._stderr)


def open_frame_source(info: VideoInfo, size: Tuple[int, int], start_time: float = 0.0,
                      end_time: Optional[float] = None, io_mode: str = "auto",
                      seek: str = "hybrid", rotation: int = 0,
                      camera: Optional[int] = None, skip_nonkey: bool = False,
                      log=print) -> FrameSource:
    """按 io_mode 打开帧源，自动回退 ffmpeg(NVDEC) → ffmpeg(CPU) → OpenCV。"""
    if camera is not None or io_mode == "opencv":
        return CvFrameSource(info, size, start_time, end_time, camera=camera)
    if io_mode in ("auto", "ffmpeg"):
        if rotation in (90, 270):
            log("[warn] 源视频带旋转元数据，改用 OpenCV 解码路径。")
            return CvFrameSource(info, size, start_time, end_time, camera=camera)
        try:
            src = FfmpegFrameSource(info, size, start_time=start_time, end_time=end_time,
                                    hwaccel="auto", seek=seek, skip_nonkey=skip_nonkey,
                                    log=log)
            log(f"[info] 帧源 : ffmpeg 管道（解码 {src.hwaccel or 'cpu'}，seek={seek}，"
                f"{size[0]}x{size[1]}）")
            return src
        except Exception as exc:
            if io_mode == "ffmpeg":
                raise
            log(f"[warn] ffmpeg 帧源不可用（{exc}），回退 OpenCV。")
    return CvFrameSource(info, size, start_time, end_time, camera=camera)


class BaseVideoSink:
    """视频输出接口。"""

    kind = "base"
    frames_written = 0
    failed: Optional[str] = None

    def write(self, frame_bgr: np.ndarray) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    @property
    def mode(self) -> str:
        return self.kind


class FfmpegVideoSink(BaseVideoSink):
    """ffmpeg 编码（优先 NVENC）：单条命令完成编码 + 原音轨复用，写入走独立线程。"""

    kind = "ffmpeg"

    def __init__(self, path, size: Tuple[int, int], fps: float, encoder: str = "auto",
                 audio_path: Optional[str] = None, cq: int = 23, preset: str = "p4",
                 x264_preset: str = "veryfast", x264_crf: int = 20, queue_frames: int = 16,
                 log=print):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = int(size[0]) - int(size[0]) % 2
        self.height = int(size[1]) - int(size[1]) % 2
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.audio_path = str(audio_path) if audio_path else None
        self.log = log
        self.frames_written = 0
        self.failed: Optional[str] = None
        self.exe = ffmpeg_exe()
        if not self.exe:
            raise RuntimeError("未找到 ffmpeg 可执行文件")

        caps = ffmpeg_capabilities()["encoders"]
        vcodec, vparams = self._pick_encoder(encoder, caps, cq, preset, x264_preset, x264_crf)
        self.encoder = vcodec

        cmd = [self.exe, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{self.width}x{self.height}",
               "-r", f"{self.fps:.6f}", "-i", "pipe:0"]
        if self.audio_path:
            cmd += ["-i", self.audio_path]
        cmd += ["-map", "0:v:0"]
        if self.audio_path:
            cmd += ["-map", "1:a:0?"]
        cmd += ["-c:v", vcodec] + vparams + ["-pix_fmt", "yuv420p"]
        if self.audio_path:
            cmd += ["-c:a", "aac", "-b:a", "64k", "-shortest"]
        cmd += ["-movflags", "+faststart", str(self.path)]
        self._cmd = cmd
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.PIPE)
        self._queue: Queue = Queue(maxsize=max(2, int(queue_frames)))
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    @staticmethod
    def _pick_encoder(encoder: str, caps: set, cq: int, preset: str,
                      x264_preset: str, x264_crf: int) -> Tuple[str, List[str]]:
        if encoder in ("auto", "nvenc") and "h264_nvenc" in caps:
            return "h264_nvenc", ["-preset", preset, "-rc", "vbr", "-cq", str(cq), "-b:v", "0"]
        if encoder == "nvenc":
            raise RuntimeError("本机 ffmpeg 不支持 h264_nvenc")
        if encoder in ("auto", "x264") and "libx264" in caps:
            return "libx264", ["-preset", x264_preset, "-crf", str(x264_crf)]
        if encoder == "x264":
            raise RuntimeError("本机 ffmpeg 不支持 libx264")
        raise RuntimeError("没有可用的 ffmpeg 编码器")

    def _worker(self) -> None:
        proc = self._proc
        try:
            while True:
                item = self._queue.get()
                if item is None or proc.stdin is None:
                    break
                proc.stdin.write(item)
                self.frames_written += 1
        except Exception as exc:                       # pragma: no cover
            self.failed = str(exc)
        finally:
            try:
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.close()
            except Exception:
                pass

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = frame_bgr[:self.height, :self.width]
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))
        if self.failed:
            raise RuntimeError(f"ffmpeg 编码线程失败: {self.failed}")
        self._queue.put(np.ascontiguousarray(frame_bgr))   # 有界队列形成背压

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._queue.put(None, timeout=30)
        except Full:
            pass
        self._thread.join(timeout=60)
        proc, self._proc = self._proc, None
        try:
            proc.wait(timeout=120)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        if proc.returncode not in (0, None) and not self.failed:
            self.failed = f"ffmpeg 退出码 {proc.returncode}"

    @property
    def mode(self) -> str:
        return f"ffmpeg/{self.encoder}" + ("+audio" if self.audio_path else "")


class CvVideoSink(BaseVideoSink):
    """OpenCV mp4v 编码（最后回退，无音轨）。"""

    kind = "opencv"

    def __init__(self, path, size: Tuple[int, int], fps: float):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = int(size[0]) - int(size[0]) % 2
        self.height = int(size[1]) - int(size[1]) % 2
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.frames_written = 0
        self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                       self.fps, (self.width, self.height))
        if not self._writer.isOpened():
            raise RuntimeError(f"无法创建输出视频: {self.path}")

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = frame_bgr[:self.height, :self.width]
        self._writer.write(frame_bgr)
        self.frames_written += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    @property
    def mode(self) -> str:
        return "opencv/mp4v"


def open_video_sink(path, size: Tuple[int, int], fps: float, encoder: str = "auto",
                    audio_path: Optional[str] = None, cq: int = 23, preset: str = "p4",
                    x264_crf: int = 20, x264_preset: str = "veryfast",
                    log=print) -> BaseVideoSink:
    """按 encoder 打开输出：NVENC → libx264 → OpenCV mp4v。"""
    if encoder != "mp4v":
        try:
            sink = FfmpegVideoSink(path, size, fps, encoder=encoder, audio_path=audio_path,
                                   cq=cq, preset=preset, x264_crf=x264_crf,
                                   x264_preset=x264_preset, log=log)
            log(f"[info] 编码 : {sink.mode}")
            return sink
        except Exception as exc:
            if encoder in ("nvenc", "x264"):
                raise
            log(f"[warn] ffmpeg 编码不可用（{exc}），回退 OpenCV mp4v。")
    return CvVideoSink(path, size, fps)
