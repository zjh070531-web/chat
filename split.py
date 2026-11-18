"""
长视频分割脚本
功能：将一个几小时的长视频分割为多个约 3 分钟的片段
提供两种实现方式：
  1. ffmpeg 方式（推荐：无损 stream copy，速度快，需安装 ffmpeg 并加入 PATH）
  2. OpenCV 方式（备选：无需额外依赖，但需重新编码，速度较慢）
"""

import os
import subprocess
from math import ceil

import cv2
from tqdm import tqdm

# ====================== 方式一：ffmpeg（推荐） ======================
def get_video_duration_ffmpeg(video_path):
    """使用 ffprobe 获取视频时长（秒）"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe 执行失败：{result.stderr.strip()}")
    return float(result.stdout.strip())

def split_video_ffmpeg(input_path, output_dir, segment_duration=180):
    """
    使用 ffmpeg 无损分割视频（stream copy，不重新编码，速度快）
    注意：由于按关键帧切割，每段实际时长可能略长于设定值

    :param input_path: 输入视频路径
    :param output_dir: 输出目录
    :param segment_duration: 每段时长（秒），默认 180（3 分钟）
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"输入视频不存在：{input_path}")
    os.makedirs(output_dir, exist_ok=True)

    total = get_video_duration_ffmpeg(input_path)
    num = ceil(total / segment_duration)
    print(f"[ffmpeg] 总时长 {total:.1f}s ({total/3600:.2f}h)，分割为 {num} 段，每段约 {segment_duration}s")

    pbar = tqdm(total=num, desc="ffmpeg 分割", unit="段",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    for i in range(num):
        start = i * segment_duration
        out_path = os.path.join(output_dir, f"segment_{i:03d}.mp4")
        cmd = [
            "ffmpeg",
            "-ss", str(start),
            "-i", input_path,
            "-t", str(segment_duration),
            "-c:v", "copy",
            "-an",
            "-avoid_negative_ts", "make_zero",
            "-y", out_path,
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg 分割第 {i} 段失败：{result.stderr.strip()[-500:]}")
        pbar.update(1)
    pbar.close()
    print(f"[ffmpeg] 完成，共 {num} 段，输出目录：{output_dir}")

# ====================== 方式二：OpenCV（备选） ======================
def split_video_opencv(input_path, output_dir, segment_duration=180):
    """
    使用 OpenCV 逐帧分割视频（需重新编码，速度较慢，但兼容性好、按帧精确）

    :param input_path: 输入视频路径
    :param output_dir: 输出目录
    :param segment_duration: 每段时长（秒），默认 180（3 分钟）
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"输入视频不存在：{input_path}")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 25.0

    frames_per_segment = int(fps * segment_duration)
    num_segments = ceil(total_frames / frames_per_segment)
    print(f"[OpenCV] fps={fps:.2f}, 总帧数={total_frames}, "
          f"每段帧数={frames_per_segment}, 分割段数={num_segments}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    seg_idx = 0
    writer = None
    frame_in_seg = 0
    cur_seg_pbar = None

    pbar = tqdm(total=total_frames, desc="OpenCV 分割", unit="frame")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if writer is None or frame_in_seg >= frames_per_segment:
                if writer is not None:
                    writer.release()
                if cur_seg_pbar is not None:
                    cur_seg_pbar.close()
                seg_idx += 1
                frame_in_seg = 0
                out_path = os.path.join(output_dir, f"segment_{seg_idx - 1:03d}.mp4")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                cur_seg_pbar = tqdm(
                    total=frames_per_segment, desc=f"  段 {seg_idx - 1:03d}",
                    unit="frame", leave=False
                )
                pbar.set_postfix_str(out_path)

            writer.write(frame)
            frame_in_seg += 1
            pbar.update(1)
            if cur_seg_pbar is not None:
                cur_seg_pbar.update(1)
    finally:
        pbar.close()
        if cur_seg_pbar is not None:
            cur_seg_pbar.close()

    if writer is not None:
        writer.release()
    cap.release()
    print(f"[OpenCV] 完成，共 {seg_idx} 段，输出目录：{output_dir}")

# ====================== 主入口 ======================
if __name__ == "__main__":
    # ============ 输入 ============
    INPUT_VIDEO = r"D:\video\2\教室中后-右\教室中后_右_802监控_802监控_20260830161715\教室中后_右_802监控_20260830124530_20260830130901_834841.mp4" # 输入：长视频路径（mp4/avi/mkv 等）
    OUTPUT_DIR = r"D:\video\2\教室中后-右\cutting"     # 输出：分割片段保存目录
    SEGMENT_SECONDS = 180                                   # 每段时长（秒），3 分钟 = 180

    # ============ 选择分割方式 ============
    # True  -> 使用 ffmpeg（推荐，速度快，需安装 ffmpeg）
    # False -> 使用 OpenCV（备选，无需额外安装）
    USE_FFMPEG = True
    # ============ 执行 ============
    if USE_FFMPEG:
        split_video_ffmpeg(INPUT_VIDEO, OUTPUT_DIR, SEGMENT_SECONDS)
    else:
        split_video_opencv(INPUT_VIDEO, OUTPUT_DIR, SEGMENT_SECONDS)