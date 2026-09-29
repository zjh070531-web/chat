# -*- coding: utf-8 -*-
r"""
交头接耳检测 v1 (精简版, 从零手写)

判定逻辑五步:
  ⓪ 桌子坐标系 - 从所有人的"脚点"自动拟合桌线/消失点, 以桌子(而非摄像头)为坐标系
  ① 座位约束  - 只判同一张桌子的左右邻座 (前后排/隔桌的人直接排除)
  ② 视线夹角  - 两人头部朝向夹角达到钝角 = 相互对看 (= 交头接耳)
  ③ 一对一匹配 - 按夹角排序贪心, 一个人只能和一个人配对
  ④ 时序平滑  - 同一配对连续命中若干处理帧才确认, 确认后允许短暂漏检

已去掉的东西: yaw时序信任 / 8道关卡 / 加权评分 / 追踪补位 / 3D头模 / 多进程 / GIF
设计取舍: 宁可漏检, 不要误判; 所有可调参数都在下面【参数】区

位置: D:\chatting\SemiUHPE-main\video.py  (与 src/ utils/ models/ weights/ 同级)
运行: python D:\chatting\SemiUHPE-main\video.py
"""

import os
import sys
import glob
import math
import random
from collections import deque

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
from src.renderer import Renderer
from src.vis_plot import convert_rotmat_bbox_to_6dof
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
VIDEO_DIR = r"D:\deeplearning\ultralytics-8.3.163\pose_behavior_package_imitate\交头接耳\cutting_video\cutting_video2\segment_005.mp4" # 输入视频目录
OUTPUT_DIR = r"D:\chatting\outputs(3)"                # 输出目录
FRAME_SKIP = 3                                      # 每3帧处理1帧 (提速; 输出视频时长不变)

# ---- 检测 (防椅子靠背被当成人头) ----
HEAD_CONF_THRESH = 0.65      # ★人头置信度: 调高更严, 更少误检
BODY_CONF_THRESH = 0.62      # 身体置信度
MIN_BODY_HEIGHT = 50         # 最小身体高度(像素)
MIN_BODY_WIDTH = 30          # 最小身体宽度(像素)
BH_RATIO_MIN, BH_RATIO_MAX = 2.5, 8.0   # 身体高/头高 的合理范围
HEAD_MIN_RATIO, HEAD_MAX_RATIO = 0.015, 0.35   # 头宽占图宽的比例范围

# ---- ① 座位约束 (三种判据, SEAT_MODE 二选一) ----
# 旧判据("image")的问题: 摄像头是斜俯视角, 图像里的"上下"≠教室里的"前后",
# 同一张桌子的相邻两人在画面里是斜着排列的, 按图像 dx/dy 判同排会把他们误判成"前后排"。
SEAT_MODE = "depth"          # ★"depth"=同深度判同桌(推荐, 与摄像头视角无关)
                             #      "desk" =桌线法(规则教室, 跨帧累积脚点拟合桌线)
                             #      "image"=旧版图像坐标判定 (仅作对照)
SEAT_DEPTH_RATIO_MAX = 1.20  # ★[depth] 同深度=同桌: 两人头高比 ≤ 该值
                             #   同桌相邻的人离镜头距离一样 → 头一样大(比值≈1.0~1.1);
                             #   前后排的人深度差明显 → 头大小差得更多(实测 ≥1.2)。
                             #   这是"同排"最稳的判据: 与摄像头斜不斜、桌子怎么摆都无关。
SEAT_DIST_RATIO = 3.0        # ★邻座: 两人距离 ≤ 该倍数×平均头高 (尺度无关!)
                             #   肩并肩坐着的人头距 ≈ 肩宽 ≈ 2 倍头高, 故 3.0 留了一点余量。
HEAD_SIZE_RATIO_MAX = 1.5    # 头高比上限 (desk/image 模式的深度兜底; depth 模式用上面的
                             #   SEAT_DEPTH_RATIO_MAX, 更严)
ROW_Y_RATIO = 0.6            # [image 模式] 同排: 两头中心图像高度差 ≤ 该倍数×平均头高
NEAR_DX_RATIO = 3.0          # [image 模式] 邻座: 两头水平距离 ≤ 该倍数×平均头高
NEIGHBOR_DX_RATIO = 0.5      # [image 模式] 左右相邻: 水平距离 ≥ 垂直距离×该比例
DESK_FRAME_ENABLE = SEAT_MODE == "desk"   # 向下兼容: desk 模式才启用桌线法

# ---- ① 座位约束: 桌子坐标系 (2026-09-29 新增, 推荐) ----
# 原理: 人坐着不动, 把整段视频里所有人的"脚点"(身体框底部中心 ≈ 椅子位置, 接近地面)
#       累积起来 RANSAC 拟合桌线 —— 真实桌线会有大量支持点, 巧合共线是零星的。
#       判定两人: 是否存在一条桌线同时经过两人的脚点(垂直偏移 ≤ 阈值),
#       且两人沿这条桌线的距离 ≤ 邻座上限 —— 即"同一张桌子的左右邻座"。
#       前后排/隔桌/过道对面的人不在同一条桌线上, 直接排除。
#       这样判的是"桌子坐标系", 与摄像头是斜俯视还是正俯视无关。
DESK_FRAME_ENABLE = True     # ★True=用桌子坐标系; False=退回旧版图像坐标判定
DESK_ANGLE_MANUAL = None     # 手动指定桌子方向角(度, 图像x轴起算); None=自动拟合桌线
                             #   自动拟合失败的场景(人太少/桌线不规则)可用它手动兜底
DESK_ROW_RATIO = 0.9         # ★同排: 脚点到桌线的垂直偏移 ≤ 该倍数×平均头高
DESK_NEAR_RATIO = 3.0        # ★邻座: 同一桌线上两人沿桌距离 ≤ 该倍数×平均头高
DESK_LINE_INLIER = 0.8       # 拟合桌线时的内点距离 (×中位头高)
DESK_MIN_LINE_PTS = 40       # 一条桌线至少要积累的脚点数 (过滤巧合共线)
DESK_MIN_FEET = 60           # 脚点总数少于该值时退回旧版图像坐标判定 (视频开头)
DESK_POINTS_MAX = 4000       # 脚点缓冲上限 (滚动覆盖)
DESK_REEST_INTERVAL = 30     # 每隔多少个处理帧重新拟合一次桌线
DESK_MAX_LINES = 4           # 最多拟合几条桌线 (房间里有几排桌子)

# ---- 两人区域裁剪规格 (★训练与推理必须完全一致, 改这里两边一起变) ----
# 以两头中心连线的中点为心作正方形, 边长只由"平均头高"决定 → 取景尺度恒定,
# 使模型无法再靠"画面大小/清晰度"这类捷径偷懒, 只能学两人的交互。
PAIR_CROP_K = 5.0            # 边长 = K × 平均头高
PAIR_CROP_DY = 0.18          # 中心下移 0.18×边长, 多纳入身体
PAIR_PREP_SIZE = 256         # 裁剪区统一先缩放到该边长
PAIR_INPUT_SIZE = 224        # JEPA 实际输入边长 (=256 中心裁 224)

# ---- ② 视线夹角 (钝角 = 相互对看) ★主要调节点 ----
GAZE_OBTUSE_THRESH = 100.0   # 夹角 ≥ 该值判为交头接耳 (慢慢试: 90~130)
MIN_TURN_YAW = 10.0          # 每人至少转头该角度 (没转头不可能在说话)

# ---- ④ 时序平滑 (跨帧确认, 压掉单帧偶发误判) ★第二调节点 ----
# 语义: 同一个配对要连续命中 SMOOTH_CONFIRM_FRAMES 个"处理帧"才输出;
#       确认后即使漏检, 也能靠 SMOOTH_HOLD_FRAMES 撑住几帧不闪断。
SMOOTH_ENABLE = True          # 关掉 = 退回"单帧命中即输出"
SMOOTH_CONFIRM_FRAMES = 2     # 连续命中几帧才确认 (FRAME_SKIP=3 + 25fps 时 1帧≈0.12秒)
SMOOTH_HOLD_FRAMES = 2        # 确认后容忍连续漏检几帧 (防闪烁/缓解漏检)
SMOOTH_MATCH_DIST = 0.8       # 跨帧"同一个配对"的匹配上限 (×平均头高); 调太大易串配对

# ---- ②第二阶段: JEPA 二次确认 (视线夹角抓可疑 → JEPA 判"是否真在说话") ----
# 思路: 第一段(视线夹角)只负责"抓可疑配对", 第二段用你训练的 V-JEPA 二分类模型
#       (chatting/not_chatting) 裁剪"两人区域"的 16 帧 clip 再判一次, 过阈值才算确认。
JEPA_ENABLE = False           # 关掉 = 退回只靠视线夹角的旧版 (2026-09-29 暂时关闭, 只留第一阶段视线夹角)
JEPA_ROOT = r"D:\jepa-main"   # V-JEPA 代码根目录 (src/ 在这里)
JEPA_BACKBONE = r"D:\jepa-main\pretrained_weights\vitl16.pth.tar"          # 冻结主干
JEPA_HEAD = r"D:\jepa-main\pretrained_weights\video_classification_frozen\ssv2-16x2x3\jepa-best.pth.tar"  # 分类头(best)
JEPA_CONFIRM_THRESH = 0.5     # P(交头接耳) ≥ 该值才最终确认 (0.5=取argmax; 调高更严)
JEPA_FRAMES = 16              # 采 16 帧 (与训练一致)
JEPA_FRAME_STEP = 4           # 帧步长 4 → 16×4=64 帧 ≈ 3.2s 的时间窗
JEPA_BUF_MAX = 64             # 帧缓冲上限(原始帧数), 存 720p 降采样
JEPA_RERUN_INTERVAL = 60      # 同一配对每隔多少原始帧才重跑一次 JEPA (节流, 60≈3秒)
                              # 裁剪规格见上面 PAIR_CROP_K / PAIR_CROP_DY / PAIR_PREP_SIZE

# ---- 3D 人脸可视化 (Sim3DR: 在每个头上叠加一个 3D 头模, 朝向=估出的头姿) ----
FACE3D_ENABLE = True          # 关掉 = 不渲染 3D 人脸
FACE3D_ALPHA = 0.9            # 3D 人脸叠加透明度 (0=原图, 1=全脸)

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
# ① 座位约束: 桌子坐标系 (累积脚点 → RANSAC 拟合桌线 → 同线判同桌)
# ==============================================================================
def feet_point(h):
    """人的"脚点" = 身体框底部中心 (坐着的人≈椅子位置, 接近地面, 是最好的地面投影点)"""
    body = h.get('body')
    if not body or body[2] <= body[0] or body[3] <= body[1]:
        return None
    return ((body[0] + body[2]) / 2.0, float(body[3]))


def fit_desk_lines(pts, ref_h):
    """RANSAC 迭代提取桌线。pts: [(x,y) 脚点]; ref_h: 中位头高(尺度基准)。
    返回 [(mx,my,dx,dy,n_inliers,angle_deg), ...] (dx,dy 为单位方向)。"""
    if ref_h <= 0 or len(pts) < DESK_MIN_LINE_PTS:
        return []
    P = np.array(pts, dtype=np.float64)
    if len(P) > 1500:                        # 采样上限, 保证拟合速度
        rng0 = np.random.RandomState(0)
        P = P[rng0.choice(len(P), 1500, replace=False)]
    rng = random.Random(0)
    tol = DESK_LINE_INLIER * ref_h
    min_sep = 6.0 * ref_h                    # 采样的两点要有一定距离
    lines = []
    for _ in range(DESK_MAX_LINES):
        best, best_n = None, 0
        for _ in range(600):
            i, j = rng.randrange(len(P)), rng.randrange(len(P))
            if i == j:
                continue
            p, q = P[i], P[j]
            L = math.hypot(q[0] - p[0], q[1] - p[1])
            if L < min_sep:
                continue
            dx, dy = (q[0] - p[0]) / L, (q[1] - p[1]) / L
            d = np.abs((P[:, 0] - p[0]) * dy - (P[:, 1] - p[1]) * dx)
            n = int((d <= tol).sum())
            if n > best_n:
                best_n, best = n, (p[0], p[1], dx, dy)
        if best is None or best_n < DESK_MIN_LINE_PTS:
            break
        mx, my, dx, dy = best
        d = np.abs((P[:, 0] - mx) * dy - (P[:, 1] - my) * dx)
        inl = P[d <= tol]
        # PCA 精修方向
        c = inl.mean(axis=0)
        x = inl[:, 0] - c[0]
        y = inl[:, 1] - c[1]
        cov = np.array([[np.dot(x, x), np.dot(x, y)], [np.dot(x, y), np.dot(y, y)]])
        _, vecs = np.linalg.eigh(cov)
        dd = vecs[:, -1]
        dx, dy = float(dd[0]), float(dd[1])
        ang = ((math.degrees(math.atan2(dy, dx)) + 90.0) % 180.0) - 90.0
        lines.append((float(c[0]), float(c[1]), dx, dy, len(inl), ang))
        # 去掉这条线的内点, 继续找下一条桌线
        keep = np.abs((P[:, 0] - c[0]) * dy - (P[:, 1] - c[1]) * dx) > tol
        P = P[keep]
        if len(P) < DESK_MIN_LINE_PTS:
            break
    return lines


class DeskFrame:
    """桌子坐标系状态: 跨帧累积脚点, 周期性重拟合桌线。"""

    def __init__(self):
        self.pts = deque(maxlen=DESK_POINTS_MAX)
        self.head_hs = deque(maxlen=DESK_POINTS_MAX)
        self.lines = []                # fit_desk_lines 的结果
        self.since_est = DESK_REEST_INTERVAL   # 让第一次 update 就尝试估计

    def update(self, heads):
        """每个处理帧调用: 收集脚点/头高, 到期重拟合桌线"""
        for h in heads:
            f = feet_point(h)
            if f is not None:
                self.pts.append(f)
            bb = h['bbox']
            self.head_hs.append(bb[3] - bb[1])
        self.since_est += 1
        if self.since_est >= DESK_REEST_INTERVAL:
            self.since_est = 0
            hs = sorted(self.head_hs)
            ref_h = hs[len(hs) // 2] if hs else 0.0
            self.lines = fit_desk_lines(list(self.pts), ref_h)

    def has_lines(self):
        return len(self.lines) > 0

    def same_desk(self, ga, gb, ref_head):
        """两脚点是否"同一张桌子的邻座": 同在一条桌线附近 + 沿桌距离近"""
        for (mx, my, dx, dy, _, _) in self.lines:
            da = abs((ga[0] - mx) * dy - (ga[1] - my) * dx)
            db = abs((gb[0] - mx) * dy - (gb[1] - my) * dx)
            if da <= DESK_ROW_RATIO * ref_head and db <= DESK_ROW_RATIO * ref_head:
                if abs((gb[0] - ga[0]) * dx + (gb[1] - ga[1]) * dy) <= DESK_NEAR_RATIO * ref_head:
                    return True
        return False


# ==============================================================================
# 配对判定: 座位约束(桌子坐标系) + 视线夹角钝角
# ==============================================================================
def judge_pairs(heads, desk):
    """heads: [{cx,cy,bbox,body,yaw,...}]; desk: DeskFrame
    返回通过判定的配对列表 [(i, j, 夹角)]"""
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

            if SEAT_MODE == "depth":
                # ---- 同深度 = 同排/同桌: 头的大小就是深度, 与摄像头视角无关 ----
                if max(h_a, h_b) / min(h_a, h_b) > SEAT_DEPTH_RATIO_MAX:
                    continue                             # 深度不同 = 前后排/隔桌
                if math.hypot(a['cx'] - b['cx'], a['cy'] - b['cy']) > SEAT_DIST_RATIO * ref_head:
                    continue                             # 隔太远, 不是邻座

            elif SEAT_MODE == "desk":
                # ---- 桌线法: 两人脚点必须同在一条拟合出的桌线上 ----
                if max(h_a, h_b) / min(h_a, h_b) > HEAD_SIZE_RATIO_MAX:   # 深度兜底
                    continue
                ga, gb = feet_point(a), feet_point(b)
                if ga is None or gb is None:
                    continue
                if DESK_ANGLE_MANUAL is not None:
                    t = math.radians(DESK_ANGLE_MANUAL)
                    ux, uy = math.cos(t), math.sin(t)
                    vx, vy = gb[0] - ga[0], gb[1] - ga[1]
                    if abs(vx * uy - vy * ux) > DESK_ROW_RATIO * ref_head:
                        continue                         # 垂直桌方向偏太多 = 不同一排
                    if abs(vx * ux + vy * uy) > DESK_NEAR_RATIO * ref_head:
                        continue                         # 沿桌方向隔太远 = 不是邻座
                elif desk.has_lines():
                    if not desk.same_desk(ga, gb, ref_head):
                        continue
                else:
                    # 桌线还没拟合出来: 退回旧版图像坐标判据
                    dx = abs(a['cx'] - b['cx'])
                    dy = abs(a['cy'] - b['cy'])
                    if dy > ROW_Y_RATIO * ref_head or dx > NEAR_DX_RATIO * ref_head \
                            or dx < dy * NEIGHBOR_DX_RATIO:
                        continue

            else:
                # ---- 旧版: 图像坐标判定 (摄像头视角) ----
                if max(h_a, h_b) / min(h_a, h_b) > HEAD_SIZE_RATIO_MAX:
                    continue
                dx = abs(a['cx'] - b['cx'])
                dy = abs(a['cy'] - b['cy'])
                if dy > ROW_Y_RATIO * ref_head:          # 不在同一排 (前后排)
                    continue
                if dx > NEAR_DX_RATIO * ref_head:        # 两人隔太远, 不是邻座
                    continue
                if dx < dy * NEIGHBOR_DX_RATIO:          # 偏前后而不是左右
                    continue

            # ---------- ② 视线夹角 (钝角 = 相互对看) ----------
            ya, yb = YAW_SIGN * a['yaw'], YAW_SIGN * b['yaw']
            if abs(ya) < MIN_TURN_YAW or abs(yb) < MIN_TURN_YAW:
                continue                              # 有人没转头
            # 方向检查: 两人都得朝向对方 (否则背对背也是钝角)
            # ★符号约定(2026-09-29 实测核实, 别再改反):
            #   ya = YAW_SIGN(-1) × yaw;  用旋转矩阵的"面部朝前向量"投影到图像验证过
            #   (52 个样本无一例外): yaw>0 ⇔ fwd_x<0 ⇔ 朝图像【左】。
            #   所以 ya = -yaw ⇒ ya>0 = 朝图像【右】, ya<0 = 朝图像【左】。
            #   左边的人要朝右看(ya>0), 右边的人要朝左看(ya<0), 才是相互对看。
            a_on_left = a['cx'] < b['cx']
            a_toward_b = (ya > 0) if a_on_left else (ya < 0)
            b_toward_a = (yb < 0) if a_on_left else (yb > 0)
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
# ④ 时序平滑: 跨帧确认 (不使用追踪器, 直接按头中心就近认领"同一个配对")
# ==============================================================================
class PairSmoother:
    """同一个配对连续命中 N 帧才确认; 确认后允许短暂漏检。

    tracks 每项: {'a':(x,y), 'b':(x,y), 'consec': 连续命中帧数, 'hold': 还能容忍几次漏检}
    """

    def __init__(self, confirm_frames, hold_frames, match_dist_ratio):
        self.confirm = max(1, int(confirm_frames))
        self.hold_max = max(0, int(hold_frames))
        self.ratio = float(match_dist_ratio)
        self.tracks = []
        self._next_id = 0

    def _dist(self, t, ha, hb):
        """配对距离 = 两头中心距离的平均; 允许 a/b 顺序互换"""
        d1 = math.hypot(t['a'][0] - ha['cx'], t['a'][1] - ha['cy']) + \
             math.hypot(t['b'][0] - hb['cx'], t['b'][1] - hb['cy'])
        d2 = math.hypot(t['a'][0] - hb['cx'], t['a'][1] - hb['cy']) + \
             math.hypot(t['b'][0] - ha['cx'], t['b'][1] - ha['cy'])
        return min(d1, d2) * 0.5

    def update(self, pairs, heads):
        """pairs: [(i, j, gaze)]; heads: [{cx,cy,bbox,...}]; 返回确认后要画的 pairs"""
        avg_head = 0.0
        if heads:
            avg_head = sum(h['bbox'][3] - h['bbox'][1] for h in heads) / float(len(heads))
        limit = self.ratio * avg_head if avg_head > 0 else 0.0

        claimed = set()      # 本轮已被认领的 tracks 下标 (含新建的)
        confirmed = []

        for (i, j, gaze) in pairs:
            ha, hb = heads[i], heads[j]
            # 就近认领历史配对: 距离最小且在 limit 内
            best, best_d = -1, None
            for ti, t in enumerate(self.tracks):
                if ti in claimed:
                    continue
                d = self._dist(t, ha, hb)
                if d <= limit and (best_d is None or d < best_d):
                    best, best_d = ti, d

            if best < 0:                                  # 没见过 → 新建
                tid = self._next_id
                self._next_id += 1
                self.tracks.append({'a': (ha['cx'], ha['cy']), 'b': (hb['cx'], hb['cy']),
                                    'consec': 1, 'hold': 0, 'id': tid})
                claimed.add(len(self.tracks) - 1)
                if self.confirm <= 1:                     # 阈值=1 时首帧即确认
                    confirmed.append((i, j, gaze, tid))
            else:                                         # 命中 → 累加
                claimed.add(best)
                t = self.tracks[best]
                t['a'] = (ha['cx'], ha['cy'])
                t['b'] = (hb['cx'], hb['cy'])
                t['consec'] += 1
                t['hold'] = self.hold_max
                if t['consec'] >= self.confirm:
                    confirmed.append((i, j, gaze, t['id']))

        # 本轮没出现的配对: 消耗 hold, 同时 consec 衰减; hold 用完则删除
        alive = []
        for ti, t in enumerate(self.tracks):
            if ti in claimed:
                alive.append(t)
                continue
            t['hold'] -= 1
            if t['hold'] > 0:
                t['consec'] = max(0, t['consec'] - 1)
                alive.append(t)
        self.tracks = alive
        return confirmed


# ==============================================================================
# ②第二阶段 JEPA: V-JEPA 冻结主干 + AttentiveClassifier 二分类 (chatting/not_chatting)
#    输入 16 帧"两人区域"裁剪 → 输出 P(交头接耳)
# ==============================================================================
def _load_jepa_classes():
    """在 yolo 环境导入 JEPA 的 vit_large / AttentiveClassifier。

    坑: SemiUHPE 与 JEPA 都把顶层包叫 `src`, 且两边 `src` 都是无 __init__.py 的
    namespace 包 —— 只要两个根目录同时在 sys.path, Python 就会把两者合并成一个
    namespace 包, 导致 JEPA 的 `src.utils/`(目录) 被 SemiUHPE 的 `src/utils.py`(文件)
    顶掉, 报 "src.utils is not a package"。所以这里必须把 SemiUHPE 根目录临时移出
    sys.path, 再导入 JEPA, 导入完恢复。
    """
    saved_modules = {k: sys.modules.pop(k) for k in list(sys.modules)
                     if k == 'src' or k.startswith('src.')}
    saved_path = sys.path[:]
    py_root_abs = os.path.abspath(PY_ROOT)
    sys.path = [JEPA_ROOT] + [p for p in saved_path
                              if p and os.path.abspath(p) != py_root_abs]
    try:
        import src.models.vision_transformer as _vit
        import src.models.attentive_pooler as _ap
        return _vit.vit_large, _ap.AttentiveClassifier
    finally:
        sys.path[:] = saved_path
        for k in list(sys.modules):
            if k == 'src' or k.startswith('src.'):
                del sys.modules[k]
        sys.modules.update(saved_modules)


def load_jepa():
    """构建并加载 JEPA 模型 (fp16), 返回 (encoder, classifier); 失败返回 (None, None)。"""
    try:
        vit_large, AttentiveClassifier = _load_jepa_classes()

        encoder = vit_large(patch_size=16, tubelet_size=2, img_size=224,
                            num_frames=JEPA_FRAMES, uniform_power=True)
        main_ckpt = torch.load(JEPA_BACKBONE, map_location='cpu', weights_only=False)
        sd = main_ckpt['target_encoder']
        new_sd = {k[len('module.backbone.'):] if k.startswith('module.backbone.') else k: v
                  for k, v in sd.items()}
        encoder.load_state_dict(new_sd, strict=False)
        del main_ckpt, sd

        classifier = AttentiveClassifier(embed_dim=1024, num_heads=16, depth=1,
                                         num_classes=2, dropout=0.1)
        head_ckpt = torch.load(JEPA_HEAD, map_location='cpu', weights_only=False)
        csd = head_ckpt['classifier']
        csd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in csd.items()}
        classifier.load_state_dict(csd)
        del head_ckpt, csd

        encoder.eval().half().to(DEVICE)
        classifier.eval().half().to(DEVICE)
        print("[JEPA] 二分类模型已加载 (vit_large 主干 + AttentiveClassifier 分类头)")
        return encoder, classifier
    except Exception as e:
        print(f"[JEPA] 模型加载失败, 自动退回只靠视线夹角的旧版: {e}")
        return None, None


_JEPA_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_JEPA_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pair_region(head_a, head_b, img_w, img_h):
    """两人区域的固定规格正方形 —— 与采样脚本 sample_pairs.py 复用同一个函数。

    边长 = PAIR_CROP_K × 平均头高, 以两头中心连线的中点为心, 再整体下移
    PAIR_CROP_DY×边长(多纳入身体)。因为边长只取决于头高, 不同远近/不同机位
    取出来的样本尺度完全一致, 模型没法再靠"画面大小/清晰度"走捷径。
    返回 (x1,y1,x2,y2) 全分辨率坐标。
    """
    ax1, ay1, ax2, ay2 = head_a['bbox']
    bx1, by1, bx2, by2 = head_b['bbox']
    hh = ((ay2 - ay1) + (by2 - by1)) / 2.0
    cx = (head_a['cx'] + head_b['cx']) / 2.0
    cy = (head_a['cy'] + head_b['cy']) / 2.0
    side = max(PAIR_CROP_K * hh, 40.0)
    side = min(side, float(min(img_w, img_h)))          # 边长不超过画面短边
    cy = cy + PAIR_CROP_DY * side
    x1, y1 = cx - side / 2.0, cy - side / 2.0
    x2, y2 = x1 + side, y1 + side
    if x1 < 0:                                          # 平移到画面内(保持边长)
        x2 -= x1; x1 = 0.0
    if y1 < 0:
        y2 -= y1; y1 = 0.0
    if x2 > img_w:
        x1 -= (x2 - img_w); x2 = float(img_w)
    if y2 > img_h:
        y1 -= (y2 - img_h); y2 = float(img_h)
    x1 = max(0, x1); y1 = max(0, y1)
    return int(x1), int(y1), int(x2), int(y2)


def pair_prep(crop_bgr):
    """裁剪区 → JEPA 输入 (float32, 已归一化)。与训练端预处理逐字对应:

      短边缩放到 PAIR_PREP_SIZE(256) → 中心裁 PAIR_INPUT_SIZE(224)
      → BGR→RGB → /255 → ImageNet mean/std
    """
    h, w = crop_bgr.shape[:2]
    s = PAIR_PREP_SIZE / float(min(h, w))
    if abs(s - 1.0) > 1e-6:
        crop_bgr = cv2.resize(crop_bgr,
                              (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                              interpolation=cv2.INTER_LINEAR)
    h, w = crop_bgr.shape[:2]
    y0 = max(0, (h - PAIR_INPUT_SIZE) // 2)
    x0 = max(0, (w - PAIR_INPUT_SIZE) // 2)
    c = crop_bgr[y0:y0 + PAIR_INPUT_SIZE, x0:x0 + PAIR_INPUT_SIZE]
    x = c[..., ::-1].astype(np.float32) / 255.0          # BGR -> RGB
    return (x - _JEPA_MEAN) / _JEPA_STD


def jepa_predict_pair(encoder, classifier, buf, region, device):
    """对"两人区域"跑 JEPA, 返回 (P交头接耳, 预测类别)。

    buf: 720p BGR 帧缓冲 (list/deque); region: (x1,y1,x2,y2) 720p 坐标。
    """
    idx = np.linspace(0, len(buf) - 1, JEPA_FRAMES).astype(int)
    x1, y1, x2, y2 = region
    if x2 - x1 < 8 or y2 - y1 < 8:
        return 0.0, 0
    crops = []
    for i in idx:
        c = buf[i][y1:y2, x1:x2]
        if c.size == 0 or min(c.shape[:2]) < 8:
            return 0.0, 0
        crops.append(pair_prep(c))
    x = np.stack(crops)
    t = torch.from_numpy(x).permute(3, 0, 1, 2).unsqueeze(0).half().to(device)
    with torch.no_grad():
        feat = encoder(t)
        logits = classifier(feat)
        probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
    return float(probs[1]), int(probs[1] >= probs[0])


# ==============================================================================
# 视频写出 (H.264 via ffmpeg libx264; 无 ffmpeg 时回落 cv2 mp4v)
# ==============================================================================
def _find_ffmpeg():
    """找到一个带 libx264 的 ffmpeg; 找不到返回 None"""
    import shutil
    import subprocess
    cands = [
        r"D:\ffmpeg\ffmpeg-2026-09-07-git-ecc7eb519e-essentials_build\bin\ffmpeg.EXE",
        r"D:\ffmpeg\bin\ffmpeg.exe",
        "ffmpeg",
    ]
    try:
        import imageio_ffmpeg
        cands.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    seen = set()
    for c in cands:
        if not c or c in seen:
            continue
        seen.add(c)
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


class H264Writer:
    """把 BGR 帧流式写入 ffmpeg(libx264) → H.264 mp4; 接口对齐 cv2.VideoWriter。"""
    def __init__(self, out_path, size, fps, crf=20):
        import subprocess
        self.out_path = out_path
        self.fps = fps
        self.proc = None
        self.cv = None
        ffmpeg = _find_ffmpeg()
        if ffmpeg:
            cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                   "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", "%dx%d" % (size[0], size[1]), "-r", "%.6f" % float(fps), "-i", "-",
                   "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            print("[写出] ffmpeg(libx264) → H.264")
        else:
            self.cv = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            print("[写出] 无 ffmpeg, 回落 cv2 mp4v")

    def isOpened(self):
        return self.proc is not None or (self.cv is not None and self.cv.isOpened())

    def write(self, frame):
        if self.proc is not None:
            try:
                self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            except Exception:
                pass
        elif self.cv is not None:
            self.cv.write(frame)

    def release(self):
        ok = True
        if self.proc is not None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=300)
                if self.proc.returncode != 0:
                    ok = False
            except Exception:
                ok = False
        if self.cv is not None:
            self.cv.release()
        if not ok or not os.path.exists(self.out_path) or os.path.getsize(self.out_path) <= 0:
            return False
        return True


# ==============================================================================
# 主流程
# ==============================================================================
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.isfile(VIDEO_DIR):                       # 指向单个视频文件
        video_files = [VIDEO_DIR]
    else:                                               # 指向目录: 处理里面所有 mp4
        video_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    print(f"[输入] {VIDEO_DIR}\n[输出] {OUTPUT_DIR}\n[视频] {len(video_files)} 个")
    if SMOOTH_ENABLE:
        print(f"[时序] 平滑开: 连续 {SMOOTH_CONFIRM_FRAMES} 个处理帧确认, "
              f"确认后容忍漏检 {SMOOTH_HOLD_FRAMES} 帧 "
              f"(1 个处理帧 = {FRAME_SKIP} 帧原始画面)")
    else:
        print("[时序] 平滑关: 单帧命中即输出")

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

    # ---------- 加载 3D 人脸渲染 (Sim3DR) ----------
    renderer = None
    if FACE3D_ENABLE:
        try:
            renderer = Renderer(
                vertices_path=os.path.join(PY_ROOT, "pose_references", "vertices_trans.npy"),
                triangles_path=os.path.join(PY_ROOT, "pose_references", "triangles.npy"),
            )
            print("[模型] Sim3DR 3D 人脸渲染已加载")
        except Exception as e:
            print(f"[3D人脸] 加载失败, 本次不渲染 3D 人脸: {e}")

    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

    # ---------- 加载 JEPA (第二阶段二次确认) ----------
    jepa_enc, jepa_clf = None, None
    if JEPA_ENABLE:
        jepa_enc, jepa_clf = load_jepa()
        if jepa_enc is None:
            print("[JEPA] 第二阶段不可用, 本次运行退回旧版 (只靠视线夹角判定)")

    # ---------- 逐个视频处理 ----------
    for vi, video_path in enumerate(video_files):
        name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n===== [{vi + 1}/{len(video_files)}] {name} =====")
        dataset = LoadImages(video_path, img_size=imgsz, stride=stride, auto=True)
        fps = dataset.cap.get(cv2.CAP_PROP_FPS) or 25.0

        w0 = int(dataset.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h0 = int(dataset.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_w = int(SAVE_HEIGHT * w0 / h0)
        out_h = SAVE_HEIGHT
        out_path = os.path.join(OUTPUT_DIR, name + "_chat.mp4")
        writer = H264Writer(out_path, (out_w, out_h), fps)
        if not writer.isOpened():
            print("!! 无法创建输出视频, 跳过"); continue

        # 时序平滑状态 + JEPA 缓冲 (每个视频重置)
        smoother = PairSmoother(SMOOTH_CONFIRM_FRAMES if SMOOTH_ENABLE else 1,
                                SMOOTH_HOLD_FRAMES if SMOOTH_ENABLE else 0,
                                SMOOTH_MATCH_DIST)
        desk_frame = DeskFrame()                   # 桌子坐标系状态 (跨帧累积脚点)
        jepa_buf = deque(maxlen=JEPA_BUF_MAX)      # 720p 原始帧(无标注)缓冲
        jepa_cache = {}                            # tid -> {'fi': 上次JEPA帧号, 'p_chat': P交头接耳}
        total_raw = 0
        total_susp = 0
        total_conf = 0
        annotated = None                           # 上一帧标注画面(720p), 跳帧时复用
        for fi, (path, img, im0, _) in enumerate(tqdm(dataset, desc=name)):
            img_h, img_w = im0.shape[:2]
            if fi % FRAME_SKIP != 0:
                jepa_buf.append(cv2.resize(im0, (out_w, out_h)))   # 跳帧: 原始帧也进缓冲
                if annotated is not None:
                    writer.write(annotated)                        # 复用上一帧标注
                continue
            jepa_buf.append(cv2.resize(im0, (out_w, out_h)))       # 当前原始帧进缓冲(供JEPA)

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
                head_boxes.append({'head': [int(hx1), int(hy1), int(hx2), int(hy2)],
                                   'body': [int(bx1), int(by1), int(bx2), int(by2)]})

            # ===== 3. SemiUHPE 头姿 (yaw/pitch/roll) =====
            heads = []
            face_poses = []                    # (rot_mat, head_bbox) 供 3D 人脸渲染
            if head_boxes:
                crops, tensors = [], []
                for hb in head_boxes:
                    hx1, hy1, hx2, hy2 = hb['head']
                    pcx, pcy = (hx1 + hx2) / 2.0, (hy1 + hy2) / 2.0
                    hs = max(hx2 - hx1, hy2 - hy1)
                    ex, ey = (0.5 - EDGES_SCALE) * hs, (0.5 - EDGES_SCALE) * hs
                    nx1 = max(0, int(pcx - ex)); nx2 = min(img_w - 1, int(pcx + ex))
                    ny1 = max(0, int(pcy - ey)); ny2 = min(img_h - 1, int(pcy + ey))
                    crop = im0[ny1:ny2, nx1:nx2]
                    rgb = cv2.cvtColor(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR),
                                       cv2.COLOR_BGR2RGB)
                    tensors.append(torch.from_numpy(rgb.transpose(2, 0, 1)).float().div_(255.0))
                    crops.append([nx1, ny1, nx2, ny2, hb['body']])
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
                                  'bbox': rb[:4], 'body': rb[4],
                                  'yaw': yaw, 'pitch': pitch, 'roll': roll})
                    face_poses.append((rot_mats[k], head_boxes[k]['head']))

            # ===== 3.5 3D 人脸可视化 (Sim3DR) =====
            if FACE3D_ENABLE and renderer is not None and face_poses:
                try:
                    global_intrinsics = np.array(
                        [[img_w + img_h, 0, img_w // 2], [0, img_w + img_h, img_h // 2], [0, 0, 1]])
                    poses = [convert_rotmat_bbox_to_6dof(rm, bbox, global_intrinsics)
                             for (rm, bbox) in face_poses]
                    trans_verts = renderer.transform_vertices(im0, poses, global_intrinsics)
                    im0 = renderer.render(im0, trans_verts, alpha=FACE3D_ALPHA)
                except Exception as e:
                    print(f"[3D人脸] 渲染失败: {e}")

            # ===== 4. 桌子坐标系: 累积脚点 → 周期性拟合桌线 =====
            desk.update(heads)

            # ===== 5. 判定: 座位约束(桌子坐标) + 视线夹角钝角 → 一对一匹配 =====
            raw_pairs = one_to_one(judge_pairs(heads, desk_frame))
            total_raw += len(raw_pairs)

            # ===== 6. 时序平滑 → 可疑配对 (带 track id) =====
            suspicious = smoother.update(raw_pairs, heads)
            total_susp += len(suspicious)

            # ===== 7. JEPA 二次确认 =====
            sx, sy = out_w / img_w, out_h / img_h   # 全分辨率 → 720p 缩放比
            confirmed = []      # JEPA 确认: (i, j, p_chat)
            yellow = []         # 可疑但 JEPA 判否: (i, j)
            for (i, j, gaze, tid) in suspicious:
                if jepa_enc is not None and len(jepa_buf) >= JEPA_FRAMES:
                    c = jepa_cache.get(tid)
                    if c is None or (fi - c['fi']) >= JEPA_RERUN_INTERVAL:
                        a, b = heads[i], heads[j]
                        r = pair_region(a, b, img_w, img_h)
                        r720 = (int(r[0] * sx), int(r[1] * sy), int(r[2] * sx), int(r[3] * sy))
                        p_chat, _ = jepa_predict_pair(jepa_enc, jepa_clf, jepa_buf, r720, DEVICE)
                        jepa_cache[tid] = {'fi': fi, 'p_chat': p_chat}
                    else:
                        p_chat = c['p_chat']
                    if p_chat >= JEPA_CONFIRM_THRESH:
                        confirmed.append((i, j, p_chat))
                    else:
                        yellow.append((i, j))
                else:
                    confirmed.append((i, j, gaze))   # 无 JEPA: 直接确认, 附上视线夹角(度)
            total_conf += len(confirmed)

            # ===== 8. 画结果: 红=确认交头接耳, 黄=可疑但被JEPA否决 =====
            for (i, j, p_chat) in confirmed:
                a, b = heads[i], heads[j]
                cv2.line(im0, (int(a['cx']), int(a['cy'])), (int(b['cx']), int(b['cy'])),
                         (0, 0, 255), 3)
                lbl = f"CHAT {p_chat:.0%}" if jepa_enc is not None else f"CHAT {p_chat:.0f}°"
                cv2.putText(im0, lbl,
                            (int((a['cx'] + b['cx']) / 2) - 20, int((a['cy'] + b['cy']) / 2) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                for p in (a, b):
                    x1, y1, x2, y2 = p['bbox']
                    cv2.rectangle(im0, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 0, 255), 3)
            for (i, j) in yellow:
                a, b = heads[i], heads[j]
                for p in (a, b):
                    x1, y1, x2, y2 = p['bbox']
                    cv2.rectangle(im0, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 255, 255), 2)
                cv2.putText(im0, "?", (int((a['cx'] + b['cx']) / 2) - 10, int((a['cy'] + b['cy']) / 2) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(im0, f"{len(heads)} Heads | {len(confirmed)} CHAT | {len(yellow)} ?", (5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 桌线可视化 (仅 desk 模式): 绿色直线 = 拟合出的桌线, 用于人工核对
            if SEAT_MODE == "desk":
                L0 = max(img_w, img_h) * 0.8
                if desk_frame.has_lines():
                    for k, (mx, my, dx, dy, npts, ang) in enumerate(desk_frame.lines):
                        cv2.line(im0, (int(mx - dx * L0), int(my - dy * L0)),
                                 (int(mx + dx * L0), int(my + dy * L0)), (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(im0, f"desk lines: {len(desk_frame.lines)}", (5, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif DESK_ANGLE_MANUAL is not None:
                    t = math.radians(DESK_ANGLE_MANUAL)
                    ux, uy = math.cos(t), math.sin(t)
                    cx0, cy0 = img_w / 2.0, img_h / 2.0
                    cv2.line(im0, (int(cx0 - ux * L0), int(cy0 - uy * L0)),
                             (int(cx0 + ux * L0), int(cy0 + uy * L0)), (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(im0, f"desk {DESK_ANGLE_MANUAL:.0f}deg", (5, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(im0, "desk: waiting (feet pts...)", (5, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            elif SEAT_MODE == "depth":
                cv2.putText(im0, f"seat: depth<={SEAT_DEPTH_RATIO_MAX}", (5, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            annotated = cv2.resize(im0, (out_w, out_h))
            writer.write(annotated)

        writer.release()
        stage3 = "JEPA确认" if jepa_enc is not None else "视线夹角确认"
        print(f"已保存: {out_path}  (原始 {total_raw} 对次 → 可疑 {total_susp} 对次 → {stage3} {total_conf} 对次)")

    print("\n全部完成!")
