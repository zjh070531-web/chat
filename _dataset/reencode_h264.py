# -*- coding: utf-8 -*-
r"""reencode_h264.py —— 把标注用 clip 批量重编码为 H.264

背景: sample_pairs.py 早期用 cv2.VideoWriter + "mp4v"(MPEG-4 Part 2) 写出,
      Chrome/Edge/Firefox 的 <video> 元素都不支持该编码 → 标注页视频全黑。
      本脚本把已有 clip 无损(仅重编码)转成 H.264 + yuv420p + faststart, 浏览器可播。

用法:
    python reencode_h264.py                        # 处理 D:\chatting\_dataset\clips
    python reencode_h264.py --dir D:\some\clips    # 指定目录
    python reencode_h264.py --force                # 连已是 H.264 的也重编
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")

FFMPEG_HINTS = [
    r"D:\ffmpeg\ffmpeg-2026-09-07-git-ecc7eb519e-essentials_build\bin\ffmpeg.EXE",
    r"D:\ffmpeg\bin\ffmpeg.exe",
    "ffmpeg",
]


def find_ffmpeg():
    ck = set()
    cands = list(FFMPEG_HINTS)
    try:
        import imageio_ffmpeg
        cands.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    for c in cands:
        if not c or c in ck:
            continue
        ck.add(c)
        p = c if os.path.isfile(c) else shutil.which(c)
        if not p:
            continue
        try:
            r = subprocess.run([p, "-hide_banner", "-encoders"], capture_output=True,
                               text=True, errors="replace", timeout=30)
            if "libx264" in (r.stdout or ""):
                return p
        except Exception:
            continue
    return None


def probe_codec(ff, path):
    """返回 ffmpeg 认到的视频编码名(如 h264 / mpeg4), 失败返回 ''"""
    try:
        r = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True,
                           text=True, errors="replace", timeout=60)
    except Exception:
        return ""
    for line in (r.stderr or "").splitlines():
        line = line.strip()
        if line.startswith("Stream #") and "Video:" in line:
            seg = line.split("Video:", 1)[1].strip()
            name = seg.split(",")[0].strip().split(" ")[0].strip("()")
            return name.lower()
    return ""


def reencode(ff, src, crf=20):
    """就地重编码为 H.264; 成功返回 True"""
    tmp = src + ".tmp.mp4"
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
           "-i", src,
           "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=180)
        if r.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
            err = (r.stderr or "")[:200]
            print("   [fail] %s  %s" % (os.path.basename(src), err))
            if os.path.exists(tmp):
                os.remove(tmp)
            return False
        os.replace(tmp, src)
        return True
    except Exception as e:
        print("   [fail] %s  %s" % (os.path.basename(src), e))
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"D:\chatting\_dataset\clips")
    ap.add_argument("--force", action="store_true", help="连已是 H.264 的也重编")
    ap.add_argument("--crf", type=int, default=20)
    a = ap.parse_args()

    ff = find_ffmpeg()
    if not ff:
        print("找不到带 libx264 的 ffmpeg, 无法转码")
        return 1
    print("ffmpeg:", ff)

    files = sorted(glob.glob(os.path.join(a.dir, "*.mp4")))
    files = [f for f in files if not f.endswith(".tmp.mp4")]
    print("待检查 clip:", len(files))
    if not files:
        return 0

    done = skip = fail = 0
    for i, f in enumerate(files, 1):
        codec = probe_codec(ff, f)
        if codec == "h264" and not a.force:
            skip += 1
            continue
        if reencode(ff, f, a.crf):
            done += 1
        else:
            fail += 1
        if i % 25 == 0 or i == len(files):
            print("  [%d/%d] 转码 %d  跳过 %d  失败 %d" % (i, len(files), done, skip, fail))
    print("完成: 转码 %d, 跳过(已是H.264) %d, 失败 %d" % (done, skip, fail))
    return 0


if __name__ == "__main__":
    sys.exit(main())
