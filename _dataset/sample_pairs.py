# -*- coding: utf-8 -*-
r"""
sample_pairs.py —— 为 JEPA 分类头重训采样"邻座两人配对"候选 clip

设计要点(解决旧数据集的捷径问题):
  1. 两类样本用**完全相同的裁剪规格**: 以两人头中心连线中点为心, 正方形边
     side = max(2.2*两人头中心距, 7.0*平均头高), 再统一缩放到 OUT_SIZE。
     -> 样本的取景尺度只由"人物自身大小"决定, 与画面来源无关
  2. 一律保持同一个预处理链: OUT_SIZE -> 短边256 -> 中心裁224 -> 归一化
     (与训练/推理共用, 见 common_prep)
  3. 预标注: 座位约束通过 + 双方互相朝向 + 视线夹角 >= PRE1_GAZE 记 1(强可疑);
     互相朝向且夹角 >= PRE2_GAZE 记 2(待定); 其余记 0(正常)
     -> 人工只需快速过一遍修正

用法:
    python sample_pairs.py                 # 用下面的默认配置
    python sample_pairs.py --dirs "D:\a;D:\b" --out D:\out --interval 3
"""

import argparse
import csv
import math
import os
import subprocess
import sys

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")   # 压掉 HEVC 长GOP 的刷屏警告
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import torch

PY_ROOT = r"D:\chatting\SemiUHPE-main"
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

import video as V   # 复用 BPJDet 后处理 / 判定常量

try:
    from utils.augmentations import letterbox
except ImportError:
    from utils.general import letterbox

# ---------------- 默认配置 ----------------
SRC_DIRS = [
    r"D:\video\1\1\right\cutting_video",
    r"D:\video\1\1\right\cutting_video2",
    r"D:\video\1\1\right\cutting_video3",
    r"D:\video\1\1\right\cutting_video4",
]
OUT_DIR = r"D:\chatting\_dataset"
IMGSZ_DET = 1920             # 检测分辨率(调高, 远景小头也能检出)
SAMPLE_INTERVAL = 3.0        # 每隔多少秒检一次
CLIP_DUR = 3.0               # 每个候选 clip 的时长(秒)
OUT_SIZE = 256               # 裁剪区统一缩放到的边长(与预处理一致)
MAX_PER_VIDEO = 90           # 每个源视频最多导出多少 clip
PRE1_GAZE = 100.0            # 预标 1 的视线夹角阈值(与 v1 一致)
PRE2_GAZE = 70.0             # 预标 2 的阈值
MAX_PER_CLASS = 400          # 每类上限(全局, 防止某一类爆量)

DEVICE = V.DEVICE
IMGSZ = V.IMGSZ
STRIDE = None
IMGSZ_CHK = None
AMP = torch.autocast(device_type="cuda", dtype=torch.float16)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


# ==============================================================================
# 模型加载 (与 video.py 的 __main__ 保持一致)
# ==============================================================================
def load_models(imgsz=IMGSZ_DET):
    global STRIDE, IMGSZ_CHK
    sys.argv = ["sample_pairs.py", "DAD-COCOHead-ResNet50-best", "--is_full_range",
                "--config", "settings/DAD3DHeads_COCOHead.yml",
                "--network", "resnet50", "--gpu_ids", "0"]
    config = V.get_config("test")
    config.device = DEVICE
    agent = V.get_agent(config)
    agent.load_ckpt(config.ckpt)
    agent.net.eval()
    agent.net.to(DEVICE)
    agent.net = agent.net.half()

    weights_path = os.path.join(PY_ROOT, "weights", "ch_head_l_1536_e150_best_mMR.pt")
    det_model = V.attempt_load(weights_path, map_location=DEVICE)
    STRIDE = int(det_model.stride.max())
    IMGSZ_CHK = V.check_img_size(imgsz, s=STRIDE)
    det_model.eval()
    det_model = det_model.half()
    print(f"[模型] BPJDet(imgsz={IMGSZ_CHK}) + SemiUHPE 已加载")
    return agent, det_model


# ==============================================================================
# 单帧检测 → heads (逐行对应 video.py 的实现)
# ==============================================================================
def detect_heads(img, im0, agent, det_model):
    img_h, img_w = im0.shape[:2]
    x = torch.from_numpy(img).to(DEVICE)
    x = x / 255.0
    if len(x.shape) == 3:
        x = x[None]
    with torch.inference_mode():
        with AMP:
            out = det_model(x, augment=False, scales=V.SCALES)[0]
            body_dets = V.non_max_suppression(out, V.CONF_THRES, V.IOU_THRES,
                                              classes=[0], num_offsets=V.NUM_OFFSETS)
            part_dets = V.non_max_suppression(out, V.CONF_THRES, V.IOU_THRES,
                                              classes=list(range(1, 1 + V.NUM_OFFSETS // 2)),
                                              num_offsets=V.NUM_OFFSETS)
        bboxes, points, scores = V.post_process_batch(
            x, [], [[(img_h, img_w)]], body_dets, part_dets, V.NUM_OFFSETS, V.MATCH_IOU_THRES)

    head_boxes = []
    for bbox, point, score in zip(bboxes, points, scores):
        f_score, f_bbox = point[0][2], point[0][3:]
        if f_score < V.HEAD_CONF_THRESH or score < V.BODY_CONF_THRESH:
            continue
        if bbox is None or len(bbox) == 0:
            continue
        bx1, by1, bx2, by2 = bbox
        body_w, body_h = bx2 - bx1, by2 - by1
        if body_h < V.MIN_BODY_HEIGHT or body_w < V.MIN_BODY_WIDTH:
            continue
        hx1, hy1, hx2, hy2 = f_bbox
        head_w, head_h = hx2 - hx1, hy2 - hy1
        if head_w <= 0 or head_h <= 0:
            continue
        head_cx, head_cy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
        if head_cy > by1 + body_h * 0.50:
            continue
        if head_cx < bx1 - body_w * 0.25 or head_cx > bx2 + body_w * 0.25:
            continue
        if not (V.BH_RATIO_MIN <= body_h / head_h <= V.BH_RATIO_MAX):
            continue
        if (hy2 - by1) / body_h > 0.40:
            continue
        head_ratio = head_w / img_w
        if not (V.HEAD_MIN_RATIO <= head_ratio <= V.HEAD_MAX_RATIO):
            continue
        if not (0.5 <= head_w / head_h <= 1.8):
            continue
        head_boxes.append({"head": [int(hx1), int(hy1), int(hx2), int(hy2)],
                           "body": [int(bx1), int(by1), int(bx2), int(by2)]})

    heads = []
    if head_boxes:
        crops, tensors = [], []
        for hb in head_boxes:
            hx1, hy1, hx2, hy2 = hb["head"]
            pcx, pcy = (hx1 + hx2) / 2.0, (hy1 + hy2) / 2.0
            hs = max(hx2 - hx1, hy2 - hy1)
            ex = ey = (0.5 - V.EDGES_SCALE) * hs
            nx1 = max(0, int(pcx - ex)); nx2 = min(img_w - 1, int(pcx + ex))
            ny1 = max(0, int(pcy - ey)); ny2 = min(img_h - 1, int(pcy + ey))
            crop = im0[ny1:ny2, nx1:nx2]
            rgb = cv2.cvtColor(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR),
                               cv2.COLOR_BGR2RGB)
            tensors.append(torch.from_numpy(rgb.transpose(2, 0, 1)).float().div_(255.0))
            crops.append([nx1, ny1, nx2, ny2, hb["body"]])
        batch = torch.stack(tensors).to(DEVICE, non_blocking=True)
        batch = (batch - MEAN) / STD
        with torch.inference_mode():
            with AMP:
                fisher = agent.net(batch)
                pd_m = V.batch_torch_A_to_R(fisher.float())
        rot_mats = pd_m.detach().cpu().numpy()
        for k, rb in enumerate(crops):
            rm = np.transpose(rot_mats[k])
            ang = V.Rotation.from_matrix(rm).as_euler("xyz", degrees=True)
            roll, pitch, yaw = list(map(V.limit_angle, [ang[2], ang[0] - 180, ang[1]]))
            heads.append({"cx": (rb[0] + rb[2]) / 2.0, "cy": (rb[1] + rb[3]) / 2.0,
                          "bbox": rb[:4], "body": rb[4],
                          "yaw": yaw, "pitch": pitch, "roll": roll})
    return heads


# ==============================================================================
# 放宽版配对: 只卡座位约束, 返回所有邻座配对 + 夹角 + 是否互相朝向
# ==============================================================================
def seat_pairs(heads, img_w):
    out = []
    n = len(heads)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = heads[i], heads[j]
            h_a = a["bbox"][3] - a["bbox"][1]
            h_b = b["bbox"][3] - b["bbox"][1]
            if h_a <= 0 or h_b <= 0:
                continue
            ref_head = (h_a + h_b) * 0.5
            dx = abs(a["cx"] - b["cx"])
            dy = abs(a["cy"] - b["cy"])
            if dy > V.ROW_Y_RATIO * ref_head:
                continue
            if dx > V.SEAT_DIST_RATIO * img_w:
                continue
            if dx < dy * V.NEIGHBOR_DX_RATIO:
                continue

            ya, yb = V.YAW_SIGN * a["yaw"], V.YAW_SIGN * b["yaw"]
            a_on_left = a["cx"] < b["cx"]
            a_toward = (ya < 0) if a_on_left else (ya > 0)
            b_toward = (yb > 0) if a_on_left else (yb < 0)
            gaze = abs(ya - yb)
            if gaze > 180.0:
                gaze = 360.0 - gaze
            turned = abs(ya) >= V.MIN_TURN_YAW and abs(yb) >= V.MIN_TURN_YAW
            toward = bool(a_toward and b_toward and turned)
            out.append((i, j, gaze, toward))
    return out


def prelabel(gaze, toward):
    """预标注: 1=强可疑(交头接耳候选), 2=待定, 0=正常"""
    if toward and gaze >= PRE1_GAZE:
        return 1
    if toward and gaze >= PRE2_GAZE:
        return 2
    return 0


# ==============================================================================
# 固定规格裁剪框
# ==============================================================================
def make_box(a, b, img_w, img_h):
    cx = (a["cx"] + b["cx"]) / 2.0
    cy = (a["cy"] + b["cy"]) / 2.0
    hh = ((a["bbox"][3] - a["bbox"][1]) + (b["bbox"][3] - b["bbox"][1])) / 2.0
    d = math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])
    side = max(2.2 * d, 7.0 * hh, 40.0)
    cy = cy + 0.18 * side                      # 稍微下移, 多纳入身体
    x1, y1 = cx - side / 2.0, cy - side / 2.0
    x2, y2 = x1 + side, y1 + side
    if x1 < 0:                                  # 平移回画面内(保持边长)
        x2 -= x1; x1 = 0.0
    if y1 < 0:
        y2 -= y1; y1 = 0.0
    if x2 > img_w:
        x1 -= (x2 - img_w); x2 = float(img_w)
    if y2 > img_h:
        y1 -= (y2 - img_h); y2 = float(img_h)
    x1 = max(0, x1); y1 = max(0, y1)
    return int(x1), int(y1), int(x2), int(y2)


# ==============================================================================
# H.264 编码 —— 浏览器 <video> 只认 H.264(yuv420p), 不认 OpenCV 默认的 mp4v(FMP4)
# ==============================================================================
FFMPEG_HINTS = [
    r"D:\ffmpeg\ffmpeg-2026-09-07-git-ecc7eb519e-essentials_build\bin\ffmpeg.EXE",
    r"D:\ffmpeg\bin\ffmpeg.exe",
    "ffmpeg",
]
FFMPEG = None
_FFMPEG_TRIED = False


def find_ffmpeg():
    """找到一个带 libx264 的 ffmpeg; 找不到返回 None"""
    import shutil
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
            r = subprocess.run([p, "-hide_banner", "-encoders"],
                               capture_output=True, text=True, errors="replace", timeout=30)
            if "libx264" in (r.stdout or ""):
                return p
        except Exception:
            continue
    return None


def encode_h264(frames, out_path, size, fps, crf=20):
    """把 BGR 帧序列编成 H.264 mp4; 返回写入帧数(全失败返回 0)"""
    global FFMPEG, _FFMPEG_TRIED
    if not frames:
        return 0
    if not _FFMPEG_TRIED:
        FFMPEG = find_ffmpeg()
        _FFMPEG_TRIED = True
    if FFMPEG:
        cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", "%dx%d" % (size, size), "-r", "%.6f" % float(fps), "-i", "-",
               "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            for f in frames:
                p.stdin.write(np.ascontiguousarray(f).tobytes())
            p.stdin.close()
            p.wait(timeout=180)
            if p.returncode == 0 and os.path.getsize(out_path) > 0:
                return len(frames)
            err = (p.stderr.read() or b"").decode("utf-8", "replace")[:200]
            print("  [warn] ffmpeg 退出码 %s: %s" % (p.returncode, err))
        except Exception as e:
            print("  [warn] ffmpeg 编码异常:", e)
    # 兜底: OpenCV avc1(openh264) → mp4v(不可播, 仅保证有文件)
    for tag in ("avc1", "mp4v"):
        try:
            w = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*tag), fps, (size, size))
            if not w.isOpened():
                continue
            for f in frames:
                w.write(f)
            w.release()
            if os.path.getsize(out_path) > 0:
                if tag == "mp4v":
                    print("  [warn] 无 ffmpeg, 回落 mp4v(浏览器播不了):", os.path.basename(out_path))
                return len(frames)
        except Exception:
            continue
    return 0


def write_clip(cap, start_frame, box, fps, out_path, size=OUT_SIZE):
    """从 start_frame 起连续读 CLIP_DUR 秒, 用固定 box 裁剪并缩放写出(H.264)"""
    x1, y1, x2, y2 = box
    if x2 - x1 < 8 or y2 - y1 < 8:
        return 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    need = max(1, int(round(CLIP_DUR * fps)))
    frames = []
    for _ in range(need):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA))
    if len(frames) < need * 0.6:                # 帧数明显不足 → 丢弃
        return 0
    got = encode_h264(frames, out_path, size, fps)
    if got < need * 0.6:
        try:
            os.remove(out_path)
        except OSError:
            pass
        return 0
    return got


def list_videos(dirs):
    exts = (".mp4", ".avi", ".mov", ".mkv", ".ts", ".flv", ".wmv")
    out = []
    for d in dirs:
        if os.path.isfile(d):
            out.append(d)
            continue
        for root, _dirs, files in os.walk(d):
            for n in files:
                if n.lower().endswith(exts):
                    out.append(os.path.join(root, n))
    out.sort()
    return out


def dump_csv(rows, out_dir):
    """把候选清单落盘(可随时写, 便于边采样边标注)"""
    p = os.path.join(out_dir, "candidates.csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=["clip", "src", "frame", "sec", "pre",
                                            "gaze", "toward", "dist", "head_h",
                                            "x1", "y1", "x2", "y2"])
        wr.writeheader()
        wr.writerows(rows)
    return p


def main():
    ap = argparse.ArgumentParser(description="采样邻座两人配对候选 clip")
    ap.add_argument("--dirs", default=";".join(SRC_DIRS), help="源目录, 分号分隔")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--interval", type=float, default=SAMPLE_INTERVAL)
    ap.add_argument("--max-per-video", type=int, default=MAX_PER_VIDEO)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个源视频(0=全部)")
    ap.add_argument("--imgsz", type=int, default=IMGSZ_DET, help="检测分辨率")
    ap.add_argument("--resume", action="store_true",
                    help="续采: 读已有 candidates.csv, 跳过已处理过的源视频")
    ap.add_argument("--only", default="", help="只处理路径含该子串的源视频")
    args = ap.parse_args()

    dirs = [d.strip() for d in args.dirs.split(";") if d.strip()]
    clips_dir = os.path.join(args.out, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    videos = list_videos(dirs)
    if args.only:
        videos = [v for v in videos if args.only in v]
    if args.limit:
        videos = videos[:args.limit]

    # ---- 续采: 恢复已有候选清单, 跳过已处理源视频 ----
    rows, counts, seq = [], {0: 0, 1: 0, 2: 0}, 0
    done_srcs = set()
    csv_path = os.path.join(args.out, "candidates.csv")
    if args.resume and os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            src = (r.get("src") or "").strip()
            if src:
                done_srcs.add(os.path.normcase(os.path.abspath(src)))
            try:
                counts[int(r.get("pre") or 0)] += 1
            except (TypeError, ValueError):
                pass
        for r in rows:                       # seq 接着最大值往后排, 避免重名覆盖
            try:
                seq = max(seq, int(str(r["clip"]).rsplit("_", 1)[-1].split(".")[0]))
            except (KeyError, ValueError, IndexError):
                pass
    print(f"[输入] {len(videos)} 个源视频 (已采过 {len(done_srcs)})")
    print(f"[输出] {clips_dir}")
    if done_srcs:
        print(f"[续采] 已有 {len(rows)} 条候选, seq 从 {seq} 继续")

    agent, det_model = load_models(args.imgsz)

    for vi, vpath in enumerate(videos, 1):
        if os.path.normcase(os.path.abspath(vpath)) in done_srcs:
            print(f"[{vi}/{len(videos)}] [跳过] 已采过: {os.path.basename(vpath)}")
            continue
        stem = os.path.splitext(os.path.basename(vpath))[0]
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"[{vi}] [跳过] 打不开: {vpath}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if not fps or fps != fps or fps <= 0:
            fps = 25.0
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(round(args.interval * fps)))
        clip_len = max(1, int(round(CLIP_DUR * fps)))
        made = 0
        recent = []                          # [(fi, cx, cy)] 相邻采样点去重
        print(f"\n[{vi}/{len(videos)}] {stem}  {fps:.0f}fps {nframes}帧 "
              f"每{step}帧检一次 imgsz={args.imgsz}")

        for fi in range(0, max(1, nframes - clip_len), step):
            if made >= args.max_per_video:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, im0 = cap.read()
            if not ok:
                continue
            img = letterbox(im0, args.imgsz, stride=STRIDE, auto=True)[0]
            img = np.ascontiguousarray(img.transpose((2, 0, 1))[::-1])
            img_h, img_w = im0.shape[:2]
            heads = detect_heads(img, im0, agent, det_model)
            if len(heads) < 2:
                continue
            # 优先采"强可疑"(正样本稀缺), 其次待定, 最后正常; 每时刻最多 2 对
            cand = list(seat_pairs(heads, img_w))
            cand.sort(key=lambda x: (0 if prelabel(x[2], x[3]) == 1
                                     else (1 if prelabel(x[2], x[3]) == 2 else 2), -x[2]))
            for (i, j, gaze, toward) in cand[:2]:
                if made >= args.max_per_video:
                    break
                pre = prelabel(gaze, toward)
                cap_cls = MAX_PER_CLASS if pre != 2 else max(60, MAX_PER_CLASS // 3)
                if counts[pre] >= cap_cls:
                    continue
                a, b = heads[i], heads[j]
                mcx, mcy = (a["cx"] + b["cx"]) / 2.0, (a["cy"] + b["cy"]) / 2.0
                hh = ((a["bbox"][3] - a["bbox"][1]) + (b["bbox"][3] - b["bbox"][1])) / 2.0
                if any(abs(fi - pf) < step * 1.5
                       and math.hypot(mcx - px, mcy - py) < 1.5 * hh
                       for pf, px, py in recent):
                    continue                     # 同一对刚采过, 跳过
                box = make_box(a, b, img_w, img_h)
                seq += 1
                cid = f"{stem}_f{fi:05d}_{seq:04d}"
                cpath = os.path.join(clips_dir, cid + ".mp4")
                if os.path.exists(cpath) or write_clip(cap, fi, box, fps, cpath) == 0:
                    continue
                d = math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])
                rows.append({
                    "clip": os.path.relpath(cpath, args.out), "src": vpath,
                    "frame": fi, "sec": round(fi / fps, 2), "pre": pre,
                    "gaze": round(gaze, 1), "toward": int(toward),
                    "dist": round(d, 1), "head_h": round(hh, 1),
                    "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                })
                recent.append((fi, mcx, mcy))
                counts[pre] += 1
                made += 1
        cap.release()
        print(f"    导出 {made} 个; 累计 正常{counts[0]} / 强可疑{counts[1]} / 待定{counts[2]}")
        dump_csv(rows, args.out)        # 每处理完一个视频就落盘, 方便边采边标

    csv_path = dump_csv(rows, args.out)
    print(f"\n完成: {len(rows)} 个候选 clip")
    print(f"  正态(预标0) {counts[0]} | 强可疑(预标1) {counts[1]} | 待定(预标2) {counts[2]}")
    print(f"  清单: {csv_path}")


if __name__ == "__main__":
    main()
