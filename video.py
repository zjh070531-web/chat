# -*- coding: utf-8 -*-
r"""
交头接耳检测 v1 (精简版, 从零手写)

判定逻辑只有三件事:
  ① 座位约束  - 只判同一排的左右邻座 (前后排的人直接排除)
  ② 视线夹角  - 两人头部朝向夹角达到钝角 = 相互对看 (= 交头接耳)
  ③ 一对一匹配 - 按夹角排序贪心, 一个人只能和一个人配对

已去掉的东西: yaw时序信任 / 8道关卡 / 加权评分 / 追踪补位 / 时序平滑 / 3D头模 / 多进程 / GIF
设计取舍: 宁可漏检, 不要误判; 所有可调参数都在下面【参数】区

运行: python D:\chatting\video.py
"""

import os
import sys
import glob
import math

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from pathlib import Path
from tqdm import tqdm

# ============ 路径: 代码/权重在 SemiUHPE-main 子目录 ============
PY_ROOT = r"D:\chatting\SemiUHPE-main"
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)
os.chdir(PY_ROOT)  # 让 weights/ 和 settings/ 等相对路径生效

from src.config import get_config
from src.agent import get_agent
from src.fisher.fisher_utils import batch_torch_A_to_R
from src.utils import limit_angle
from utils.general import check_img_size, scale_coords, non_max_suppression
from utils.datasets import LoadImages
from models.experimental import attempt_load


# ==============================================================================
# BPJDet 后处理: 把"身体框"和"身体上的头框"关联起来 (原样取自 SemiUHPE-main)
# ==============================================================================
def cal_inside_iou(bigBox, smallBox):  # body_box, part_box
    [Ax0, Ay0, Ax1, Ay1] = bigBox[0:4]
    [Bx0, By0, Bx1, By1] = smallBox[0:4]
    W = min(Ax1, Bx1) - max(Ax0, Bx0)
    H = min(Ay1, By1) - max(Ay0, By0)
    if W <= 0 or H <= 0:
        return 0
    areaB = (Bx1 - Bx0) * (By1 - By0)
    return (W * H) / areaB


def post_process_batch(imgs, paths, shapes, body_dets, part_dets, num_offsets, match_iou_thres):
    batch_bboxes, batch_points, batch_scores = [], [], []
    for si, (bdet, pdet) in enumerate(zip(body_dets, part_dets)):
        nbody, npart = bdet.shape[0], pdet.shape[0]
        if nbody:
            path, shape = Path(paths[si]) if len(paths) else '', shapes[si][0]
            scores = bdet[:, 4].cpu().numpy()
            bboxes = scale_coords(imgs[si].shape[1:], bdet[:, :4], shape).round().cpu().numpy()
            points = scale_coords(imgs[si].shape[1:], bdet[:, -num_offsets:], shape).cpu().numpy()
            points = points.reshape((nbody, -1, 2))
            points = np.concatenate((points, np.zeros((nbody, points.shape[1], 5))), axis=-1)
            if npart:
                pdet[:, :4] = scale_coords(imgs[si].shape[1:], pdet[:, :4].clone(), shape)
                pdet_slim = pdet[:, :6].cpu().numpy()
                for id, (x1, y1, x2, y2, conf, cls) in enumerate(pdet_slim):
                    p_xc, p_yc = np.mean((x1, x2)), np.mean((y1, y2))
                    part_pts = points[:, int(cls - 1)]
                    dist = np.linalg.norm(part_pts[:, :2] - np.array([[p_xc, p_yc]]), axis=-1)
                    pt_match = np.argmin(dist)
                    tmp_iou = cal_inside_iou(bboxes[pt_match], [x1, y1, x2, y2])
                    if conf > part_pts[pt_match][2] and tmp_iou > match_iou_thres:
                        part_pts[pt_match] = [p_xc, p_yc, conf, x1, y1, x2, y2]
            batch_bboxes.extend(bboxes)
            batch_points.extend(points)
            batch_scores.extend(scores)
    return batch_bboxes, batch_points, batch_scores


# ==============================================================================
# 【参数】★ 需要调节的都在这
# ==============================================================================
VIDEO_DIR = r"D:\video\1\1\right\cutting_video2"   # 输入视频目录
OUTPUT_DIR = r"D:\chatting\outputs"                # 输出目录
FRAME_SKIP = 3                                      # 每3帧处理1帧 (提速; 输出视频时长不变)

# ---- 检测 (防椅子靠背被当成人头) ----
HEAD_CONF_THRESH = 0.65      # ★人头置信度: 调高更严, 更少误检
BODY_CONF_THRESH = 0.62      # 身体置信度
MIN_BODY_HEIGHT = 50         # 最小身体高度(像素)
MIN_BODY_WIDTH = 30          # 最小身体宽度(像素)
BH_RATIO_MIN, BH_RATIO_MAX = 2.5, 8.0   # 身体高/头高 的合理范围
HEAD_MIN_RATIO, HEAD_MAX_RATIO = 0.015, 0.35   # 头宽占图宽的比例范围

# ---- ① 座位约束 ----
ROW_Y_RATIO = 1.2            # 同排判定: 两头中心高度差 ≤ 该倍数×平均头高 (前后排排除)
SEAT_DIST_RATIO = 0.35       # 邻座判定: 两头水平距离 ≤ 该比例×图宽
NEIGHBOR_DX_RATIO = 0.5      # 左右相邻: 水平距离 ≥ 垂直距离×该比例

# ---- ② 视线夹角 (钝角 = 相互对看) ★主要调节点 ----
GAZE_OBTUSE_THRESH = 100.0   # 夹角 ≥ 该值判为交头接耳 (慢慢试: 90~130)
MIN_TURN_YAW = 10.0          # 每人至少转头该角度 (没转头不可能在说话)

# ---- 模型设置 (勿改) ----
DEVICE = 'cuda:0'
IMGSZ = 960
NUM_OFFSETS = 2
SCALES = [1]
CONF_THRES = 0.50
IOU_THRES = 0.75
MATCH_IOU_THRES = 0.6
EDGES_SCALE = -0.05
YAW_SIGN = -1                # yaw 符号约定 (与 SemiUHPE 输出对接)
SAVE_HEIGHT = 720            # 输出视频高度


# ==============================================================================
# 配对判定: 座位约束 + 视线夹角钝角
# ==============================================================================
def judge_pairs(heads, img_w):
    """heads: [{cx,cy,bbox,yaw,...}]; 返回通过判定的配对列表 [(i, j, 夹角)]"""
    pairs = []
    n = len(heads)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = heads[i], heads[j]

            # ---------- ① 座位约束 ----------
            h_a = a['bbox'][3] - a['bbox'][1]
            h_b = b['bbox'][3] - b['bbox'][1]
            if h_a <= 0 or h_b <= 0:
                continue
            ref_head = (h_a + h_b) * 0.5
            dx = abs(a['cx'] - b['cx'])
            dy = abs(a['cy'] - b['cy'])
            if dy > ROW_Y_RATIO * ref_head:          # 不在同一排 (前后排)
                continue
            if dx > SEAT_DIST_RATIO * img_w:         # 距离太远, 不是邻座
                continue
            if dx < dy * NEIGHBOR_DX_RATIO:          # 偏前后而不是左右
                continue

            # ---------- ② 视线夹角 (钝角 = 相互对看) ----------
            ya, yb = YAW_SIGN * a['yaw'], YAW_SIGN * b['yaw']
            if abs(ya) < MIN_TURN_YAW or abs(yb) < MIN_TURN_YAW:
                continue                              # 有人没转头
            # 方向检查: 两人都得朝向对方 (否则背对背也是钝角)
            a_on_left = a['cx'] < b['cx']
            a_toward_b = (ya < 0) if a_on_left else (ya > 0)
            b_toward_a = (yb > 0) if a_on_left else (yb < 0)
            if not (a_toward_b and b_toward_a):
                continue
            gaze = abs(ya - yb)
            if gaze > 180.0:
                gaze = 360.0 - gaze
            if gaze < GAZE_OBTUSE_THRESH:            # 不是钝角, 不算相互对看
                continue

            pairs.append((i, j, gaze))
    return pairs


def one_to_one(pairs):
    """③ 一对一匹配: 夹角大的优先, 一人只能配一人"""
    pairs = sorted(pairs, key=lambda x: -x[2])
    used, keep = set(), []
    for i, j, g in pairs:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        keep.append((i, j, g))
    return keep


# ==============================================================================
# 主流程
# ==============================================================================
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    print(f"[输入] {VIDEO_DIR}\n[输出] {OUTPUT_DIR}\n[视频] {len(video_files)} 个")

    # ---------- 加载 SemiUHPE (头姿) ----------
    sys.argv = ['video.py', 'DAD-COCOHead-ResNet50-best', '--is_full_range',
                '--config', 'settings/DAD3DHeads_COCOHead.yml',
                '--network', 'resnet50', '--gpu_ids', '0']
    config = get_config('test')
    config.device = DEVICE
    agent = get_agent(config)
    agent.load_ckpt(config.ckpt)
    agent.net.eval()
    agent.net.to(DEVICE)
    agent.net = agent.net.half()
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)

    # ---------- 加载 BPJDet (头身检测) ----------
    weights_path = os.path.join(PY_ROOT, "weights", "ch_head_l_1536_e150_best_mMR.pt")
    det_model = attempt_load(weights_path, map_location=DEVICE)
    stride = int(det_model.stride.max())
    imgsz = check_img_size(IMGSZ, s=stride)
    det_model.eval()
    det_model = det_model.half()
    print(f"[模型] BPJDet imgsz={imgsz} + SemiUHPE(ResNet50) 已加载")

    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

    # ---------- 逐个视频处理 ----------
    for vi, video_path in enumerate(video_files):
        name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n===== [{vi + 1}/{len(video_files)}] {name} =====")
        dataset = LoadImages(video_path, img_size=imgsz, stride=stride, auto=True)
        fps = dataset.cap.get(cv2.CAP_PROP_FPS) or 25.0

        writer, out_path = None, os.path.join(OUTPUT_DIR, name + "_chat.mp4")
        for codec in ['avc1', 'mp4v', 'XVID']:
            w0 = int(dataset.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h0 = int(dataset.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*codec), fps,
                                     (int(SAVE_HEIGHT * w0 / h0), SAVE_HEIGHT))
            if writer.isOpened():
                break
        if writer is None or not writer.isOpened():
            print("!! 无法创建输出视频, 跳过"); continue

        total_chat = 0
        for fi, (path, img, im0, _) in enumerate(tqdm(dataset, desc=name)):
            if fi % FRAME_SKIP != 0:
                continue
            img_h, img_w = im0.shape[:2]

            # ===== 1. BPJDet 检测 (身体 + 人头) =====
            x = torch.from_numpy(img).to(DEVICE)
            x = x / 255.0
            if len(x.shape) == 3:
                x = x[None]
            with torch.inference_mode():
                with amp_ctx:
                    out = det_model(x, augment=False, scales=SCALES)[0]
                    body_dets = non_max_suppression(out, CONF_THRES, IOU_THRES,
                                                    classes=[0], num_offsets=NUM_OFFSETS)
                    part_dets = non_max_suppression(out, CONF_THRES, IOU_THRES,
                                                    classes=list(range(1, 1 + NUM_OFFSETS // 2)),
                                                    num_offsets=NUM_OFFSETS)
                bboxes, points, scores = post_process_batch(
                    x, [], [[(img_h, img_w)]], body_dets, part_dets, NUM_OFFSETS, MATCH_IOU_THRES)

            # ===== 2. 过滤: 只保留"像人"的头身对 (椅子靠背在这里被丢掉) =====
            head_boxes = []
            for ind, (bbox, point, score) in enumerate(zip(bboxes, points, scores)):
                f_score, f_bbox = point[0][2], point[0][3:]
                if f_score < HEAD_CONF_THRESH or score < BODY_CONF_THRESH:
                    continue
                if bbox is None or len(bbox) == 0:      # 必须头身成对
                    continue
                bx1, by1, bx2, by2 = bbox
                body_w, body_h = bx2 - bx1, by2 - by1
                if body_h < MIN_BODY_HEIGHT or body_w < MIN_BODY_WIDTH:
                    continue
                hx1, hy1, hx2, hy2 = f_bbox
                head_w, head_h = hx2 - hx1, hy2 - hy1
                if head_w <= 0 or head_h <= 0:
                    continue
                head_cx, head_cy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
                if head_cy > by1 + body_h * 0.50:                       # 头要在身体上半部
                    continue
                if head_cx < bx1 - body_w * 0.25 or head_cx > bx2 + body_w * 0.25:
                    continue
                if not (BH_RATIO_MIN <= body_h / head_h <= BH_RATIO_MAX):  # 身头比例
                    continue
                if (hy2 - by1) / body_h > 0.40:                         # 头要贴近身体顶部
                    continue
                head_ratio = head_w / img_w
                if not (HEAD_MIN_RATIO <= head_ratio <= HEAD_MAX_RATIO):   # 头尺寸
                    continue
                if not (0.5 <= head_w / head_h <= 1.8):                 # 头长宽比
                    continue
                head_boxes.append([int(hx1), int(hy1), int(hx2), int(hy2)])

            # ===== 3. SemiUHPE 头姿 (yaw/pitch/roll) =====
            heads = []
            if head_boxes:
                crops, tensors = [], []
                for (hx1, hy1, hx2, hy2) in head_boxes:
                    pcx, pcy = (hx1 + hx2) / 2.0, (hy1 + hy2) / 2.0
                    hs = max(hx2 - hx1, hy2 - hy1)
                    ex, ey = (0.5 - EDGES_SCALE) * hs, (0.5 - EDGES_SCALE) * hs
                    nx1 = max(0, int(pcx - ex)); nx2 = min(img_w - 1, int(pcx + ex))
                    ny1 = max(0, int(pcy - ey)); ny2 = min(img_h - 1, int(pcy + ey))
                    crop = im0[ny1:ny2, nx1:nx2]
                    rgb = cv2.cvtColor(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR),
                                       cv2.COLOR_BGR2RGB)
                    tensors.append(torch.from_numpy(rgb.transpose(2, 0, 1)).float().div_(255.0))
                    crops.append([nx1, ny1, nx2, ny2])
                batch = torch.stack(tensors).to(DEVICE, non_blocking=True)
                batch = (batch - mean) / std
                with torch.inference_mode():
                    with amp_ctx:
                        fisher = agent.net(batch)
                        pd_m = batch_torch_A_to_R(fisher.float())
                rot_mats = pd_m.detach().cpu().numpy()
                for k, rb in enumerate(crops):
                    rm = np.transpose(rot_mats[k])
                    ang = Rotation.from_matrix(rm).as_euler("xyz", degrees=True)
                    roll, pitch, yaw = list(map(limit_angle, [ang[2], ang[0] - 180, ang[1]]))
                    heads.append({'cx': (rb[0] + rb[2]) / 2.0, 'cy': (rb[1] + rb[3]) / 2.0,
                                  'bbox': rb, 'yaw': yaw, 'pitch': pitch, 'roll': roll})

            # ===== 4. 判定: 座位约束 + 视线夹角钝角 → 一对一匹配 =====
            pairs = one_to_one(judge_pairs(heads, img_w))
            total_chat += len(pairs)

            # ===== 5. 画结果 =====
            for (i, j, gaze) in pairs:
                a, b = heads[i], heads[j]
                cv2.line(im0, (int(a['cx']), int(a['cy'])), (int(b['cx']), int(b['cy'])),
                         (0, 0, 255), 3)
                cv2.putText(im0, f"{min(1.0, gaze / 150.0):.0%}",
                            (int((a['cx'] + b['cx']) / 2) - 20, int((a['cy'] + b['cy']) / 2) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                for p in (a, b):
                    x1, y1, x2, y2 = p['bbox']
                    cv2.rectangle(im0, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 0, 255), 3)
                    cv2.putText(im0, "CHAT!", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(im0, f"{len(heads)} Heads | {len(pairs)} Chat", (5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out = cv2.resize(im0, (int(SAVE_HEIGHT * img_w / img_h), SAVE_HEIGHT))
            for _ in range(FRAME_SKIP):     # 跳帧处理, 重复写入保持原时长
                writer.write(out)

        writer.release()
        print(f"已保存: {out_path}  (共 {total_chat} 对次)")

    print("\n全部完成!")
