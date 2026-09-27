# -*- coding: utf-8 -*-
r"""
交头接耳检测 v1 (精简版, 从零手写)

判定逻辑四步:
  ① 座位约束  - 只判同一排的左右邻座 (前后排的人直接排除)
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
OUTPUT_DIR = r"D:\chatting\outputs(2)"                # 输出目录
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
JEPA_ENABLE = True            # 关掉 = 退回只靠视线夹角的旧版
JEPA_ROOT = r"D:\jepa-main"   # V-JEPA 代码根目录 (src/ 在这里)
JEPA_BACKBONE = r"D:\jepa-main\pretrained_weights\vitl16.pth.tar"          # 冻结主干
JEPA_HEAD = r"D:\jepa-main\pretrained_weights\video_classification_frozen\ssv2-16x2x3\jepa-best.pth.tar"  # 分类头(best)
JEPA_CONFIRM_THRESH = 0.5     # P(交头接耳) ≥ 该值才最终确认 (0.5=取argmax; 调高更严)
JEPA_FRAMES = 16              # 采 16 帧 (与训练一致)
JEPA_FRAME_STEP = 4           # 帧步长 4 → 16×4=64 帧 ≈ 3.2s 的时间窗
JEPA_BUF_MAX = 64             # 帧缓冲上限(原始帧数), 存 720p 降采样
JEPA_RERUN_INTERVAL = 60      # 同一配对每隔多少原始帧才重跑一次 JEPA (节流, 60≈3秒)
JEPA_CROP_MARGIN = 0.25       # "两人区域"向外扩边比例 (给一点上下文/头动余量)

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


def pair_region(body_a, body_b, img_w, img_h, margin=JEPA_CROP_MARGIN):
    """两人区域的联合框(身体框取并集) + 向外扩边, 返回 (x1,y1,x2,y2) 全分辨率坐标。"""
    x1 = min(body_a[0], body_b[0]); y1 = min(body_a[1], body_b[1])
    x2 = max(body_a[2], body_b[2]); y2 = max(body_a[3], body_b[3])
    w = x2 - x1; h = y2 - y1
    mx = w * margin; my = h * margin
    x1 = max(0, int(x1 - mx)); y1 = max(0, int(y1 - my))
    x2 = min(img_w, int(x2 + mx)); y2 = min(img_h, int(y2 + my))
    return x1, y1, x2, y2


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
        crops.append(cv2.resize(c, (224, 224), interpolation=cv2.INTER_LINEAR))
    x = np.stack(crops).astype(np.float32) / 255.0
    x = x[..., ::-1]                                # BGR -> RGB
    x = (x - _JEPA_MEAN) / _JEPA_STD
    t = torch.from_numpy(x).permute(3, 0, 1, 2).unsqueeze(0).half().to(device)
    with torch.no_grad():
        feat = encoder(t)
        logits = classifier(feat)
        probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
    return float(probs[1]), int(probs[1] >= probs[0])


# ==============================================================================
# 主流程
# ==============================================================================
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
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
        writer, out_path = None, os.path.join(OUTPUT_DIR, name + "_chat.mp4")
        for codec in ['avc1', 'mp4v', 'XVID']:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*codec), fps, (out_w, out_h))
            if writer.isOpened():
                break
        if writer is None or not writer.isOpened():
            print("!! 无法创建输出视频, 跳过"); continue

        # 时序平滑状态 + JEPA 缓冲 (每个视频重置)
        smoother = PairSmoother(SMOOTH_CONFIRM_FRAMES if SMOOTH_ENABLE else 1,
                                SMOOTH_HOLD_FRAMES if SMOOTH_ENABLE else 0,
                                SMOOTH_MATCH_DIST)
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

            # ===== 4. 判定: 座位约束 + 视线夹角钝角 → 一对一匹配 =====
            raw_pairs = one_to_one(judge_pairs(heads, img_w))
            total_raw += len(raw_pairs)

            # ===== 5. 时序平滑 → 可疑配对 (带 track id) =====
            suspicious = smoother.update(raw_pairs, heads)
            total_susp += len(suspicious)

            # ===== 6. JEPA 二次确认 =====
            sx, sy = out_w / img_w, out_h / img_h   # 全分辨率 → 720p 缩放比
            confirmed = []      # JEPA 确认: (i, j, p_chat)
            yellow = []         # 可疑但 JEPA 判否: (i, j)
            for (i, j, gaze, tid) in suspicious:
                if jepa_enc is not None and len(jepa_buf) >= JEPA_FRAMES:
                    c = jepa_cache.get(tid)
                    if c is None or (fi - c['fi']) >= JEPA_RERUN_INTERVAL:
                        a, b = heads[i], heads[j]
                        r = pair_region(a['body'], b['body'], img_w, img_h)
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
                    confirmed.append((i, j, 1.0))   # 无 JEPA 或历史不足 → 直接确认
            total_conf += len(confirmed)

            # ===== 7. 画结果: 红=确认交头接耳, 黄=可疑但被JEPA否决 =====
            for (i, j, p_chat) in confirmed:
                a, b = heads[i], heads[j]
                cv2.line(im0, (int(a['cx']), int(a['cy'])), (int(b['cx']), int(b['cy'])),
                         (0, 0, 255), 3)
                cv2.putText(im0, f"CHAT {p_chat:.0%}",
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

            annotated = cv2.resize(im0, (out_w, out_h))
            writer.write(annotated)

        writer.release()
        print(f"已保存: {out_path}  (原始 {total_raw} 对次 → 可疑 {total_susp} 对次 → JEPA确认 {total_conf} 对次)")

    print("\n全部完成!")
