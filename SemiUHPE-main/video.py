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

位置: SemiUHPE-main/video.py  (与 src/ utils/ models/ weights/ 同级)
运行: python SemiUHPE-main/video.py --video-dir <视频目录> --output-dir <输出目录>
"""

import os
import sys
import glob
import json
import math
import time
import random
from collections import deque

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from pathlib import Path
from tqdm import tqdm

# ============ 路径: 代码/权重在 SemiUHPE-main 子目录 ============
PY_ROOT = os.path.dirname(os.path.abspath(__file__))  # 本文件所在目录; 随仓库移动, 不写死绝对路径
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
# 输入/输出: 优先级 = 命令行 --video-dir/--output-dir > 环境变量 > 下面这两个默认值
# 不再写死本机绝对路径, 换机器 / clone 仓库后可直接用
VIDEO_DIR = os.environ.get(
    "CHATTING_VIDEO_DIR",
    r"D:\video\2\教室中后-右\cutting"
)  # 输入视频目录(或单个视频文件)
OUTPUT_DIR = os.environ.get("CHATTING_OUTPUT_DIR", r"D:\chatting\outputs(15)")  # 输出目录
FRAME_SKIP = 3                                      # 每3帧处理1帧 (提速; 输出视频时长不变)

# ---- 检测 (防椅子靠背被当成人头) ----
HEAD_CONF_THRESH = 0.65      # ★人头置信度: 调高更严, 更少误检
BODY_CONF_THRESH = 0.62      # 身体置信度
MIN_BODY_HEIGHT = 50         # 最小身体高度(像素)
MIN_BODY_WIDTH = 30          # 最小身体宽度(像素)
BH_RATIO_MIN, BH_RATIO_MAX = 2.5, 8.0   # 身体高/头高 的合理范围
HEAD_MIN_RATIO, HEAD_MAX_RATIO = 0.015, 0.35   # 头宽占图宽的比例范围

# ---- ① 座位约束: 以"桌子"为坐标系判相邻 (2026-09-29) ----
# 已废弃旧的图像坐标判据 (ROW_Y_RATIO/NEAR_DX_RATIO/NEIGHBOR_DX_RATIO: 按画面里的上下左右
# 判"同排/邻座"): 摄像头是斜俯视角, 画面里的"上下"≠教室里的"前后", 同一张桌子的相邻两人
# 在画面里是斜着排列的, 会被误判成"前后排"而漏检。现在只按桌子(深度)来判。
SEAT_MODE = "depth"          # ★"depth"=同深度即同桌(推荐, 与摄像头视角无关)
                             #   "desk" =桌线法(规则教室: 跨帧累积脚点拟合桌线, 两人同线才算同桌)
SEAT_DEPTH_RATIO_MAX = 1.20  # ★[depth] 同深度=同桌: 两人头高比 ≤ 该值
                             #   同桌相邻的人离镜头一样远 → 头一样大(比值≈1.0~1.1);
                             #   前后排的人深度差明显 → 头大小差得更多(实测 ≥1.2)。
                             #   头的大小就是深度, 与摄像头斜不斜、桌子怎么摆都无关。
SEAT_DIST_RATIO = 3.5        # ★邻座: 两人距离 ≤ 该倍数×平均头高 (尺度无关!)
                             #   2026-09-29 3.0→3.5: 缩放截图/近镜头下邻座距离偏大, 3.0 会漏。
                             #   肩并肩头距≈2×头高, 3.5 对真邻座仍安全, 隔一桌(≈4×头高)仍挡掉。
HEAD_SIZE_RATIO_MAX = 1.5    # [desk 模式] 头高比上限(深度兜底); depth 模式用上面更严的
                             #   SEAT_DEPTH_RATIO_MAX。desk 模式在桌线还没拟合出来时也用它。

# ---- ① 座位约束: 桌线法参数 (仅 SEAT_MODE="desk" 时生效) ----
# 人坐着不动 → 把整段视频所有人的"脚点"(身体框底部中心 ≈ 椅子位置, 接近地面)累积起来
# RANSAC 拟合桌线: 真实桌线支持点多, 巧合共线是零星的。判定两人是否同桌: 是否存在一条
# 桌线同时经过两人脚点(垂直偏移 ≤ 阈值), 且沿桌距离 ≤ 邻座上限。
DESK_ANGLE_MANUAL = None     # 手动指定桌向角(度, 图像x轴起算); None=自动拟合桌线
DESK_ROW_RATIO = 0.9         # 脚点到桌线的垂直偏移 ≤ 该倍数×平均头高
DESK_NEAR_RATIO = 3.0        # 同一桌线上两人沿桌距离 ≤ 该倍数×平均头高
DESK_LINE_INLIER = 0.8       # 拟合桌线时的内点距离 (×中位头高)
DESK_MIN_LINE_PTS = 40       # 一条桌线至少要积累的脚点数 (过滤巧合共线)
DESK_MIN_FEET = 60           # 脚点总数少于该值时暂用 depth 判据兜底 (视频开头)
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
GAZE_OBTUSE_THRESH = 70.0    # 夹角 ≥ 该值判为交头接耳 (2026-09-29 100→80→70:
                             #   4 真实交谈样本 92~141°; "快要对上"的临界样本 76.4° 也要能过)
MIN_TURN_YAW = 7.0           # 每人至少转头该角度 (没转头不可能在说话)
                             #   2026-09-29 10→7: 教室监控/俯身辅导样本里有人只转头 7.7~8.2°,

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
JEPA_ENABLE = True            # 默认开启; 命令行 --no-jepa 退回只靠视线夹角的旧版
# JEPA 代码与权重不随本仓库分发, 默认按下面的路径找; 可用环境变量或命令行覆盖
JEPA_ROOT = os.environ.get("JEPA_ROOT", r"D:\jepa-main")  # V-JEPA 代码根目录 (src/ 在这里)
JEPA_BACKBONE = os.environ.get("JEPA_BACKBONE", os.path.join(JEPA_ROOT, "pretrained_weights", "vitl16.pth.tar"))  # 冻结主干 ViT-L/16
# 分类头(best/latest): 默认用 2026-11-04 训练的 chatting-ssv2-v2b
#   SSv2 探针热启动 + 16×2×1 + 按源簇划分的 my_dataset_v2 (无同源泄漏)
#   val_acc 100% / val_loss 0.2073 —— 用 latest(ep6), 因为 best 停在 val_acc 首次达 100% 的 ep1
JEPA_HEAD = os.environ.get("JEPA_HEAD", os.path.join(
    JEPA_ROOT, "pretrained_weights", "video_classification_frozen", "chatting-ssv2-v2b", "jepa-latest.pth.tar"))
JEPA_CONFIRM_THRESH = 0.5     # P(交头接耳) ≥ 该值才最终确认 (0.5=取argmax; 命令行 --jepa-thresh 调高更严)
JEPA_FRAMES = 16              # 采 16 帧 (与训练 frames_per_clip 一致)
JEPA_FRAME_STEP = 3           # 训练端 frame_step (ssv2-v2b 用 3); 缓冲里相邻两个"处理帧"实际间隔就是 FRAME_SKIP 个原始帧
JEPA_BUF_MAX = 64             # 帧缓冲上限(处理帧数), 存 720p 降采样
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

# ---- 进度条 / 超时 (防止单个大视频把整晚都占住) ----
TIMEOUT_MIN = 30.0           # ★单个视频处理超时(分钟); 0 = 不限。超时后取消该视频
STOP_ON_TIMEOUT = False      # True = 任一视频超时后整个脚本退出; False = 只放弃该视频, 继续下一个
DELETE_PARTIAL = False       # True = 超时取消时把半成品视频一起删掉
DEBUG_PAIRS = False          # True = 打印每一对成交接耳的判据数值 (也可用 --debug-pairs)

# ---- 左上角结果面板 (HUD) ----
HUD_ENABLE = True            # 关掉 = 回到原来那行小字
HUD_SCALE = 1.25             # ★整体大小 (以 720p 为基准); 1.2 = 更大更醒目
HUD_OFFSET_X = 0.0125        # 左边距 (占画面宽度比例)
HUD_OFFSET_Y = 0.150         # 上边距 (占画面高度比例); 用来避开摄像头自带的时间水印, 想贴顶就改 0.02
HUD_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyhbd.ttc",   # 微软雅黑 Bold
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]
# 面板位置可以拖动: python video.py --pick-hud 拖好后回车, 位置存到这个文件, 之后每次渲染都按它画
HUD_POS_FILE = os.path.join(PY_ROOT, "hud_position.json")


def load_hud_pos(path=HUD_POS_FILE):
    """读取拖动保存的面板位置 (画面宽/高比例); 没有或读不了就返回 None"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        x, y = float(d["x"]), float(d["y"])
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            return x, y
    except Exception:
        pass
    return None


def save_hud_pos(x, y, path=HUD_POS_FILE):
    """保存面板位置 (比例坐标)"""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"x": round(float(x), 4), "y": round(float(y), 4),
                       "note": "画面宽高比例; 改这里或重新 --pick-hud 拖动即可"}, f,
                      ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[HUD] 保存位置失败: {e}")
        return False


_saved_hud_pos = load_hud_pos()          # 有保存过就优先用保存的位置
if _saved_hud_pos is not None:
    HUD_OFFSET_X, HUD_OFFSET_Y = _saved_hud_pos


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
# 配对判定: 座位约束(桌子坐标系: 同深度即同桌 / 桌线法) + 视线夹角钝角
# ==============================================================================
def _too_far_or_diff_depth(a, b, ref_head):
    """同桌相邻判据: 同深度(头一样大) + 距离近。两者都是尺度无关的。"""
    h_a = a['bbox'][3] - a['bbox'][1]
    h_b = b['bbox'][3] - b['bbox'][1]
    if h_a <= 0 or h_b <= 0:
        return True
    if max(h_a, h_b) / min(h_a, h_b) > SEAT_DEPTH_RATIO_MAX:      # 深度不同 = 前后排/隔桌
        return True
    if math.hypot(a['cx'] - b['cx'], a['cy'] - b['cy']) > SEAT_DIST_RATIO * ref_head:
        return True                                               # 隔太远, 不是邻座
    return False


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

            if SEAT_MODE == "desk":
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
                    # 桌线还没拟合出来(视频开头/人太少): 用同深度判据兜底
                    if _too_far_or_diff_depth(a, b, ref_head):
                        continue
            else:
                # ---- 同深度 = 同排/同桌: 头的大小就是深度, 与摄像头视角无关 ----
                if _too_far_or_diff_depth(a, b, ref_head):
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

    取帧方式: 用缓冲里**最近**的 JEPA_FRAMES 个处理帧。
    缓冲里存的是"处理帧"(主循环每 FRAME_SKIP 个原始帧才 append 一个), 所以
    16 个处理帧 ≈ 16×FRAME_SKIP = 48 个原始帧, 与训练端(16 帧 × frame_step 3)
    的时间跨度一致。早先用 linspace(0, len(buf)-1) 跨整个 64 帧缓冲, 窗口会拉到
    ~9.6 秒, 是训练窗口的 4 倍, 模型看到的运动幅度完全失真。
    """
    if len(buf) < JEPA_FRAMES:
        return 0.0, 0
    x1, y1, x2, y2 = region
    if x2 - x1 < 8 or y2 - y1 < 8:
        return 0.0, 0
    frames = list(buf)[-JEPA_FRAMES:]
    crops = []
    for fr in frames:
        c = fr[y1:y2, x1:x2]
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
# 左上角结果面板 (HUD): 半透明圆角底板 + 大号数字, 按画面高度自动缩放
# ==============================================================================
_hud_fonts = {}
_hud_cache = {}


def _hud_font(px):
    """按像素大小取字体 (带缓存); 找不到中文字体时退回 PIL 默认字体 (中文会显示成方块)"""
    from PIL import ImageFont
    px = max(8, int(px))
    if px in _hud_fonts:
        return _hud_fonts[px]
    font = None
    for path in HUD_FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, px)
                break
            except Exception:
                font = None
    if font is None:
        font = ImageFont.load_default()
    _hud_fonts[px] = font
    return font


def _make_hud(n_heads, n_chat, n_susp, sub_text, frame_h):
    """渲染一块带 alpha 的 HUD 小图, 返回 (bgr, alpha[h,w,1] in 0..1)。尺寸随 frame_h 缩放。"""
    from PIL import Image, ImageDraw

    s = frame_h / 720.0 * HUD_SCALE            # 以 720p 为基准的整体缩放系数
    f_num = _hud_font(round(44 * s))           # 大号数字
    f_lab = _hud_font(round(17 * s))           # 数字下方标签
    f_sub = _hud_font(round(15 * s))           # 底部说明
    pad = round(16 * s)
    gap = round(34 * s)
    metric_gap = round(16 * s)
    blk_pad = round(6 * s)

    metrics = [(f"{n_heads}", "人头", (255, 255, 255)),
               (f"{n_chat}", "交头接耳", (255, 92, 96)),       # RGB 红
               (f"{n_susp}", "可疑", (255, 191, 74))]         # RGB 琥珀

    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))

    def text_wh(txt, font):
        l, t, r, b = probe.textbbox((0, 0), txt, font=font)
        return r - l, b - t

    num_wh = [text_wh(m[0], f_num) for m in metrics]
    lab_wh = [text_wh(m[1], f_lab) for m in metrics]
    blk_w = [max(n[0], l[0]) for n, l in zip(num_wh, lab_wh)]
    num_h = max(n[1] for n in num_wh)
    lab_h = max(l[1] for l in lab_wh)
    sub_h = text_wh(sub_text, f_sub)[1] if sub_text else 0

    w = pad * 2 + sum(blk_w) + blk_pad * len(blk_w) + gap * (len(metrics) - 1)
    h = pad * 2 + num_h + metric_gap + lab_h + ((round(14 * s) + sub_h) if sub_text else 0)
    w, h = max(w, round(150 * s)), max(h, round(56 * s))

    img = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=round(12 * s),
                        fill=(10, 12, 16, 190), outline=(255, 255, 255, 26), width=max(1, round(s)))

    x = pad
    for i, (val, lab, rgb) in enumerate(metrics):
        bw = blk_w[i] + blk_pad
        tx_num = x + (bw - num_wh[i][0]) / 2.0
        tx_lab = x + (bw - lab_wh[i][0]) / 2.0
        ty_num, ty_lab = pad, pad + num_h + metric_gap
        # 数字带一点投影, 压住亮背景
        d.text((tx_num + 1, ty_num + 2), val, font=f_num, fill=(0, 0, 0, 120))
        d.text((tx_num, ty_num), val, font=f_num, fill=rgb + (255,))
        d.text((tx_lab, ty_lab), lab, font=f_lab, fill=(184, 192, 204, 240))
        x += bw
        if i < len(metrics) - 1:                      # 竖分隔线
            d.line([(x + gap / 2 - blk_pad / 2, pad), (x + gap / 2 - blk_pad / 2, pad + num_h + metric_gap + lab_h)],
                   fill=(255, 255, 255, 34), width=max(1, round(1.2 * s)))
            x += gap

    if sub_text:
        d.text((pad, h - pad - sub_h - round(2 * s)), sub_text, font=f_sub,
               fill=(150, 158, 170, 240))

    arr = np.asarray(img)
    bgr = np.ascontiguousarray(arr[:, :, 2::-1])       # RGB -> BGR
    alpha = (arr[:, :, 3:4].astype(np.float32) / 255.0)
    return bgr, alpha


def _get_hud(n_heads, n_chat, n_susp, sub_text, frame_h):
    """取 (带 alpha 的) 面板小图; 内容不变时复用缓存"""
    key = (n_heads, n_chat, n_susp, sub_text, frame_h)
    item = _hud_cache.get(key)
    if item is None:
        item = _make_hud(n_heads, n_chat, n_susp, sub_text, frame_h)
        if len(_hud_cache) > 400:
            _hud_cache.clear()
        _hud_cache[key] = item
    return item


def _blend_hud(img, bgr, alpha, x0, y0):
    """把面板叠加到 img 的 (x0,y0) 位置 (就地修改); 超出边界自动裁剪"""
    h, w = img.shape[:2]
    ph, pw = bgr.shape[:2]
    x0, y0 = int(round(x0)), int(round(y0))
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    pw, ph = min(pw, w - x0), min(ph, h - y0)
    if pw <= 0 or ph <= 0:
        return img
    roi = img[y0:y0 + ph, x0:x0 + pw]
    a = alpha[:ph, :pw]
    img[y0:y0 + ph, x0:x0 + pw] = (roi.astype(np.float32) * (1.0 - a)
                                   + bgr[:ph, :pw].astype(np.float32) * a).astype(np.uint8)
    return img


def draw_hud(img, n_heads, n_chat, n_susp, sub_text=""):
    """把结果面板贴到 HUD_OFFSET_X/Y 指定的位置 (就地修改 img); 数值没变时直接复用缓存, 几乎不耗时"""
    if not HUD_ENABLE:
        return img
    h, w = img.shape[:2]
    try:
        bgr, alpha = _get_hud(n_heads, n_chat, n_susp, sub_text, h)
    except Exception:                                   # 没装 Pillow 等情况: 退回单行文字
        cv2.putText(img, f"{n_heads} Heads | {n_chat} CHAT | {n_susp} ?",
                    (int(8 * h / 720), int(30 * h / 720)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1 * h / 720, (255, 255, 255), 2)
        return img
    return _blend_hud(img, bgr, alpha, HUD_OFFSET_X * w, HUD_OFFSET_Y * h)


def _hud_drag_handler(state, pw, ph, W, H):
    """鼠标拖动回处理: 按在面板上才拖, 拖出画面会被夹住。state 里存 x/y/drag/ox/oy"""
    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if state['x'] <= mx <= state['x'] + pw and state['y'] <= my <= state['y'] + ph:
                state['drag'] = True
                state['ox'], state['oy'] = mx - state['x'], my - state['y']
        elif event == cv2.EVENT_MOUSEMOVE and state['drag']:
            state['x'] = max(0, min(W - pw, mx - state['ox']))
            state['y'] = max(0, min(H - ph, my - state['oy']))
        elif event == cv2.EVENT_LBUTTONUP:
            state['drag'] = False
    return on_mouse


def place_hud_interactive(video_path, pos_file=HUD_POS_FILE, counts=(9, 1, 0), sub_text="", win="HUD drag"):
    """打开窗口, 用鼠标把结果面板拖到想要的位置, 回车保存 (存成画面比例), ESC 取消。
    返回保存的 (x, y) 比例坐标; 取消返回 None。"""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print(f"[定位] 读不到视频画面: {video_path}")
        return None

    H0, W0 = frame.shape[:2]
    k = min(1.0, 1280.0 / W0, 760.0 / H0)           # 大画面缩小到能放进窗口
    if k < 1.0:
        frame = cv2.resize(frame, (int(W0 * k), int(H0 * k)), interpolation=cv2.INTER_AREA)
    H, W = frame.shape[:2]
    bgr, alpha = _get_hud(counts[0], counts[1], counts[2], sub_text, H)
    ph, pw = bgr.shape[:2]

    px, py = load_hud_pos(pos_file) or (HUD_OFFSET_X, HUD_OFFSET_Y)
    state = {'x': int(round(px * W)), 'y': int(round(py * H)), 'drag': False, 'ox': 0, 'oy': 0}
    state['x'] = max(0, min(W - pw, state['x']))
    state['y'] = max(0, min(H - ph, state['y']))

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, _hud_drag_handler(state, pw, ph, W, H))
    print("[定位] 用鼠标拖动面板 → 回车保存 / ESC 取消 / 方向键微调 / r 复位")

    while True:
        shown = _blend_hud(frame.copy(), bgr, alpha, state['x'], state['y'])
        cv2.putText(shown, "drag: mouse   save: Enter   cancel: ESC   nudge: arrows",
                    (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(shown, "drag: mouse   save: Enter   cancel: ESC   nudge: arrows",
                    (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, shown)
        key = cv2.waitKey(30) & 0xFFFF
        if key in (13, 10, ord('s'), ord('S')):      # 回车 / s 保存
            x, y = state['x'] / float(W), state['y'] / float(H)
            cv2.destroyWindow(win)
            if save_hud_pos(x, y, pos_file):
                print(f"[定位] 已保存: {pos_file}  (x={x:.3f}, y={y:.3f})")
                return x, y
            return None
        if key in (27, ord('q'), ord('Q')):          # ESC / q 取消
            cv2.destroyWindow(win)
            print("[定位] 已取消, 位置未改动")
            return None
        if key == ord('r'):                          # 复位到默认
            state['x'], state['y'] = 0, 0
        step = 5
        if key == 2424832:  state['x'] = max(0, state['x'] - step)          # ←
        elif key == 2555904: state['x'] = min(W - pw, state['x'] + step)    # →
        elif key == 2490368: state['y'] = max(0, state['y'] - step)         # ↑
        elif key == 2621440: state['y'] = min(H - ph, state['y'] + step)    # ↓


# ==============================================================================
# 主流程
# ==============================================================================
if __name__ == '__main__':
    import argparse
    _ap = argparse.ArgumentParser(description="交头接耳检测 v1 (精简版)")
    _ap.add_argument("--video-dir", default=VIDEO_DIR,
                     help="输入视频目录或单个视频文件 (默认: 环境变量 CHATTING_VIDEO_DIR / ./videos)")
    _ap.add_argument("--output-dir", default=OUTPUT_DIR,
                     help="输出目录 (默认: 环境变量 CHATTING_OUTPUT_DIR / ./outputs)")
    _ap.add_argument("--timeout-min", type=float, default=TIMEOUT_MIN,
                     help="单个视频处理超时分钟数, 超过则取消该视频 (默认: 30; 0=不限)")
    _ap.add_argument("--stop-on-timeout", action="store_true", default=STOP_ON_TIMEOUT,
                     help="任一视频超时后直接退出整个脚本 (默认: 只放弃该视频, 继续下一个)")
    _ap.add_argument("--delete-partial", action="store_true", default=DELETE_PARTIAL,
                     help="超时取消时删除半成品视频文件 (默认: 保留可播放的前半段)")
    _ap.add_argument("--pick-hud", action="store_true",
                     help="打开窗口用鼠标拖动结果面板, 回车保存位置后退出 (下次渲染生效)")
    _ap.add_argument("--debug-pairs", action="store_true",
                     help="把每一对判定成交头接耳的判据数值打印出来 (排查误报用)")
    _ap.add_argument("--hud-pos", type=str, default=None,
                     help="直接指定面板位置, 格式 'x,y' (0~1 的画面比例), 例: --hud-pos 0.02,0.02")
    _ap.add_argument("--no-jepa", action="store_true",
                     help="关闭 JEPA 二次确认, 只用视线夹角判据 (旧行为)")
    _ap.add_argument("--jepa-thresh", type=float, default=JEPA_CONFIRM_THRESH,
                     help=f"P(交头接耳) 确认阈值, 越高越严 (默认 {JEPA_CONFIRM_THRESH})")
    _ap.add_argument("--jepa-head", type=str, default=None,
                     help="分类头权重 .pth.tar 路径 (默认: chatting-ssv2-v2b/jepa-latest.pth.tar)")
    _ap.add_argument("--jepa-root", type=str, default=None,
                     help=r"JEPA 仓库根目录, 里面有 src/ 和 pretrained_weights/ (默认 D:\jepa-main)")
    _args = _ap.parse_args()
    VIDEO_DIR, OUTPUT_DIR = _args.video_dir, _args.output_dir
    TIMEOUT_MIN, STOP_ON_TIMEOUT, DELETE_PARTIAL = _args.timeout_min, _args.stop_on_timeout, _args.delete_partial
    DEBUG_PAIRS = _args.debug_pairs
    # ---- JEPA 二次确认的开关与路径 (命令行优先于环境变量/默认值) ----
    if _args.no_jepa:
        JEPA_ENABLE = False
    if _args.jepa_root:
        JEPA_ROOT = _args.jepa_root
        if not _args.jepa_head:
            JEPA_BACKBONE = os.path.join(JEPA_ROOT, "pretrained_weights", "vitl16.pth.tar")
            JEPA_HEAD = os.path.join(JEPA_ROOT, "pretrained_weights", "video_classification_frozen",
                                     "chatting-ssv2-v2b", "jepa-latest.pth.tar")
    if _args.jepa_head:
        JEPA_HEAD = _args.jepa_head
    JEPA_CONFIRM_THRESH = _args.jepa_thresh
    if _args.hud_pos:
        try:
            _hx, _hy = [float(v) for v in _args.hud_pos.replace("，", ",").split(",")[:2]]
            HUD_OFFSET_X, HUD_OFFSET_Y = _hx, _hy
            save_hud_pos(_hx, _hy)
            print(f"[HUD] 面板位置已设为 x={_hx:.3f}, y={_hy:.3f} (已写入 {HUD_POS_FILE})")
        except Exception:
            print(f"[HUD] --hud-pos 格式不对: {_args.hud_pos!r}, 应形如 0.02,0.02")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.isfile(VIDEO_DIR):                       # 指向单个视频文件
        video_files = [VIDEO_DIR]
    else:                                               # 指向目录: 处理里面所有 mp4
        video_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    print(f"[输入] {VIDEO_DIR}\n[输出] {OUTPUT_DIR}\n[视频] {len(video_files)} 个")
    if JEPA_ENABLE:
        print(f"[JEPA] 二次确认开启  阈值 P≥{JEPA_CONFIRM_THRESH:.2f}  窗口 {JEPA_FRAMES} 帧")
        print(f"[JEPA]   主干 {JEPA_BACKBONE}")
        print(f"[JEPA]   分类头 {JEPA_HEAD}")
        for _tag, _p in (("主干", JEPA_BACKBONE), ("分类头", JEPA_HEAD)):
            if not os.path.isfile(_p):
                print(f"[JEPA]   !! {_tag} 不存在: {_p}  (可用 --jepa-root / --jepa-head 指定)")
        if not os.path.isdir(os.path.join(JEPA_ROOT, "src")):
            print(f"[JEPA]   !! JEPA_ROOT 下没有 src/: {JEPA_ROOT}")
    else:
        print("[JEPA] 已关闭 (--no-jepa), 只用视线夹角判据")
    if HUD_ENABLE:
        print(f"[HUD] 面板位置 x={HUD_OFFSET_X:.3f}, y={HUD_OFFSET_Y:.3f}"
              + (" (来自 hud_position.json)" if load_hud_pos() is not None else " (默认)")
              + "；想拖动定位就运行: python video.py --pick-hud")

    # ---------- 拖动定位面板: 拖好回车保存, 然后退出 ----------
    if _args.pick_hud:
        if not video_files:
            print("[定位] 没有可用的视频文件, 无法取画面")
            sys.exit(1)
        _sub = (f"座位: 深度比 ≤ {SEAT_DEPTH_RATIO_MAX:g}" if SEAT_MODE == "depth" else "座位: 桌线")
        place_hud_interactive(video_files[0], HUD_POS_FILE, (9, 1, 0), _sub)
        print("位置已定, 直接重新运行本脚本(不加 --pick-hud)即可按新位置渲染")
        sys.exit(0)
    if TIMEOUT_MIN > 0:
        print(f"[超时] 单个视频上限 {TIMEOUT_MIN:g} 分钟, 超过自动取消"
              + ("并退出脚本" if STOP_ON_TIMEOUT else ", 继续下一个视频"))
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
    stopped_early = False
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

        # ---------- 进度条: 用真实总帧数, 显示百分比 / 速度 / ETA / 超时倒计时 ----------
        n_frames = 0
        try:
            n_frames = int(getattr(dataset, "frames", 0) or 0)
        except Exception:
            n_frames = 0
        if n_frames <= 0:
            try:
                cap_n = int(dataset.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                n_frames = cap_n if cap_n > 0 else 0
            except Exception:
                n_frames = 0
        bar = tqdm(total=n_frames if n_frames > 0 else None,
                   desc=f"[{vi + 1}/{len(video_files)}] {name}",
                   unit="帧", dynamic_ncols=True, mininterval=0.5, leave=True)
        t_video = time.time()                      # 本视频计时起点 (只算处理时间, 不含加载模型)
        timed_out = False
        for fi, (path, img, im0, _) in enumerate(dataset):
            bar.update(1)

            # ---- 超时保护: 单个视频处理超过 TIMEOUT_MIN 分钟就取消 ----
            if TIMEOUT_MIN > 0 and (time.time() - t_video) > TIMEOUT_MIN * 60.0:
                timed_out = True
                break

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
            desk_frame.update(heads)

            # ===== 5. 判定: 座位约束(桌子坐标) + 视线夹角钝角 → 一对一匹配 =====
            raw_pairs = one_to_one(judge_pairs(heads, desk_frame))
            total_raw += len(raw_pairs)
            if DEBUG_PAIRS:
                for (pi, pj, pg) in raw_pairs:
                    pa, pb = heads[pi], heads[pj]
                    ph_a = pa['bbox'][3] - pa['bbox'][1]
                    ph_b = pb['bbox'][3] - pb['bbox'][1]
                    ph_ref = (ph_a + ph_b) / 2.0
                    pdist = math.hypot(pa['cx'] - pb['cx'], pa['cy'] - pb['cy'])
                    print(f"[配对] f{fi} 头{pi}-头{pj}: 外角 {pg:.1f}° (≥{GAZE_OBTUSE_THRESH:g}) | "
                          f"yaw {pa['yaw']:+.1f}°/{pb['yaw']:+.1f}° (各需≥{MIN_TURN_YAW:g}) | "
                          f"pitch {pa['pitch']:+.1f}°/{pb['pitch']:+.1f}° | "
                          f"头高 {ph_a:.0f}/{ph_b:.0f}px 比 {max(ph_a, ph_b) / min(ph_a, ph_b):.2f} "
                          f"(≤{SEAT_DEPTH_RATIO_MAX:g}) | 中心距 {pdist:.0f}px = "
                          f"{pdist / ph_ref:.2f}×头高 (≤{SEAT_DIST_RATIO:g})")

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
                        if DEBUG_PAIRS:
                            print(f"[JEPA] f{fi} 头{i}-头{j}: P(交头接耳)={p_chat:.3f} "
                                  f"→ {'确认' if p_chat >= JEPA_CONFIRM_THRESH else '否决'}"
                                  f"  (阈值 {JEPA_CONFIRM_THRESH:.2f}, 区域 {r720})")
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
            k = max(1.0, img_h / 720.0)        # 线宽/字号随分辨率缩放 (4K 输入不会被缩小到看不见)
            for (i, j, p_chat) in confirmed:
                a, b = heads[i], heads[j]
                cv2.line(im0, (int(a['cx']), int(a['cy'])), (int(b['cx']), int(b['cy'])),
                         (0, 0, 255), max(2, int(3 * k)))
                lbl = f"CHAT {p_chat:.0%}" if jepa_enc is not None else f"CHAT {p_chat:.0f}°"
                cv2.putText(im0, lbl,
                            (int((a['cx'] + b['cx']) / 2) - int(20 * k), int((a['cy'] + b['cy']) / 2) - int(10 * k)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6 * k, (0, 0, 255), max(2, int(2 * k)), cv2.LINE_AA)
                for p in (a, b):
                    x1, y1, x2, y2 = p['bbox']
                    cv2.rectangle(im0, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 0, 255), max(2, int(3 * k)))
            for (i, j) in yellow:
                a, b = heads[i], heads[j]
                for p in (a, b):
                    x1, y1, x2, y2 = p['bbox']
                    cv2.rectangle(im0, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 255, 255), max(2, int(2 * k)))
                cv2.putText(im0, "?", (int((a['cx'] + b['cx']) / 2) - int(10 * k), int((a['cy'] + b['cy']) / 2) - int(10 * k)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6 * k, (0, 255, 255), max(2, int(2 * k)), cv2.LINE_AA)

            # ===== 9. 左上角结果面板 (半透明圆角底板 + 大号数字, 自动缩放) =====
            if SEAT_MODE == "desk":
                L0 = max(img_w, img_h) * 0.8
                if desk_frame.has_lines():
                    for kk, (mx, my, dx, dy, npts, ang) in enumerate(desk_frame.lines):
                        cv2.line(im0, (int(mx - dx * L0), int(my - dy * L0)),
                                 (int(mx + dx * L0), int(my + dy * L0)), (0, 255, 0), 1, cv2.LINE_AA)
                    hud_sub = f"座位: 桌线 {len(desk_frame.lines)} 条"
                elif DESK_ANGLE_MANUAL is not None:
                    t = math.radians(DESK_ANGLE_MANUAL)
                    ux, uy = math.cos(t), math.sin(t)
                    cx0, cy0 = img_w / 2.0, img_h / 2.0
                    cv2.line(im0, (int(cx0 - ux * L0), int(cy0 - uy * L0)),
                             (int(cx0 + ux * L0), int(cy0 + uy * L0)), (0, 255, 0), 1, cv2.LINE_AA)
                    hud_sub = f"座位: 桌向 {DESK_ANGLE_MANUAL:.0f}°"
                else:
                    hud_sub = "座位: 桌线拟合中"
            elif SEAT_MODE == "depth":
                hud_sub = f"座位: 深度比 ≤ {SEAT_DEPTH_RATIO_MAX:g}"
            else:
                hud_sub = ""

            annotated = cv2.resize(im0, (out_w, out_h))
            draw_hud(annotated, len(heads), len(confirmed), len(yellow), hud_sub)   # 在输出分辨率上画, 又快又清晰
            writer.write(annotated)

            # 进度条后缀: 当前帧结果 + 距离自动取消还剩多久
            try:
                post = f"人头 {len(heads)} | 确认 {len(confirmed)}"
                if TIMEOUT_MIN > 0:
                    left = TIMEOUT_MIN * 60.0 - (time.time() - t_video)
                    post += f" | 剩余 {max(0.0, left) / 60.0:.1f}分自动取消"
                bar.set_postfix_str(post)
            except Exception:
                pass

        bar.close()
        writer.release()
        elapsed = time.time() - t_video
        stage3 = "JEPA确认" if jepa_enc is not None else "视线夹角确认"
        if timed_out:
            info = (f"[超时] {name}: 已用 {elapsed / 60.0:.1f} 分钟 > 上限 {TIMEOUT_MIN:g} 分钟, "
                    f"取消该视频; 半成品: {out_path}")
            if DELETE_PARTIAL:
                try:
                    if os.path.exists(out_path):
                        os.remove(out_path)
                    info = (f"[超时] {name}: 已用 {elapsed / 60.0:.1f} 分钟 > 上限 {TIMEOUT_MIN:g} 分钟, "
                            f"已取消并删除半成品")
                except Exception as e:
                    info += f" (删除半成品失败: {e})"
            print("\n" + info)
            if STOP_ON_TIMEOUT:
                print("[超时] 已按 --stop-on-timeout 退出整个脚本")
                stopped_early = True
                break
        else:
            print(f"已保存: {out_path}  (原始 {total_raw} 对次 → 可疑 {total_susp} 对次 → "
                  f"{stage3} {total_conf} 对次, 用时 {elapsed / 60.0:.1f} 分钟)")

    print("\n已结束 (因超时提前退出)!" if stopped_early else "\n全部完成!")
