
import os
import sys
import cv2
import json
import math
import time
import glob
import torch
import shutil
import imageio
import subprocess
import threading
import queue
import torchvision

import numpy as np
import torch.nn as nn
import torchvision.transforms as tfs

from scipy.spatial.transform import Rotation
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ============ 路径设置 ============
# 本脚本位于 D:\chatting\SemiUHPE-main (代码/权重同目录), PY_ROOT 显式指向该目录
PY_ROOT = r"D:\chatting\SemiUHPE-main"
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

from src.config import get_config
from src.agent import get_agent
from src.fisher.fisher_utils import batch_torch_A_to_R
from src.utils import compute_euler_angles_from_rotation_matrices
from src.utils import limit_angle
from src.vis_plot import convert_euler_bbox_to_6dof
from src.vis_plot import convert_rotmat_bbox_to_6dof

from src.renderer import Renderer
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
renderer = Renderer(
    vertices_path=os.path.join(PY_ROOT, "pose_references", "vertices_trans.npy"),
    triangles_path=os.path.join(PY_ROOT, "pose_references", "triangles.npy")
)

###############################################################################
# BPJDet utils (the body-head model trained on CrowdHuman and using YOLOv5-L)
###############################################################################

from utils.torch_utils import select_device, time_sync
from utils.general import check_img_size, scale_coords, non_max_suppression
from utils.datasets import LoadImages
from models.experimental import attempt_load

colors_list = [
        # [255, 0, 0], [255, 127, 0], [255, 255, 0], [127, 255, 0], [0, 255, 0], [0, 255, 127], 
        # [0, 255, 255], [0, 127, 255], [0, 0, 255], [127, 0, 255], [255, 0, 255], [255, 0, 127],
        [255, 127, 0], [127, 255, 0], [0, 255, 127], [0, 127, 255], [127, 0, 255], [255, 0, 127],
        [255, 255, 255],
        [127, 0, 127], [0, 127, 127], [127, 127, 0], [127, 0, 0], [127, 0, 0], [0, 127, 0],
        [127, 127, 127],
        [255, 0, 255], [0, 255, 255], [255, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 0],
        [0, 0, 0],
        [255, 127, 255], [127, 255, 255], [255, 255, 127], [127, 127, 255], [255, 127, 127], [255, 127, 127],
    ]  # 27 colors

def cal_inside_iou(bigBox, smallBox):  # body_box, part_box
    # calculate small rectangle inside big box ratio, calSmallBoxInsideRatio
    [Ax0, Ay0, Ax1, Ay1] = bigBox[0:4]
    [Bx0, By0, Bx1, By1] = smallBox[0:4]
    W = min(Ax1, Bx1) - max(Ax0, Bx0)
    H = min(Ay1, By1) - max(Ay0, By0)
    if W <= 0 or H <= 0:
        return 0
    else:
        areaA = (Ax1 - Ax0)*(Ay1 - Ay0)
        areaB = (Bx1 - Bx0)*(By1 - By0)
        crossArea = W * H
        # return crossArea/(areaA + areaB - crossArea)
        return crossArea/areaB  # range [0, 1]

def post_process_batch(imgs, paths, shapes, body_dets, part_dets, num_offsets, match_iou_thres):

    batch_bboxes, batch_points, batch_scores = [], [], []

    # process each image in batch
    for si, (bdet, pdet) in enumerate(zip(body_dets, part_dets)):
        nbody, npart = bdet.shape[0], pdet.shape[0]
        
        if nbody:  # one batch
            path, shape = Path(paths[si]) if len(paths) else '', shapes[si][0]

            scores = bdet[:, 4].cpu().numpy()  # body detection score
            bboxes = scale_coords(imgs[si].shape[1:], bdet[:, :4], shape).round().cpu().numpy()

            points = scale_coords(imgs[si].shape[1:], bdet[:, -num_offsets:], shape).cpu().numpy()
            points = points.reshape((nbody, -1, 2))
            # points = np.concatenate((points, np.zeros((nbody, points.shape[1], 1))), axis=-1)  # n*c*2 --> n*c*3
            points = np.concatenate((points, np.zeros((nbody, points.shape[1], 5))), axis=-1)  # n*c*2 --> n*c*7
                
            # batch_parts_dict[str(img_id)] = []
            if npart:
                pdet[:, :4] = scale_coords(imgs[si].shape[1:], pdet[:, :4].clone(), shape)
                pdet_slim = pdet[:, :6].cpu().numpy()

                left_pdet = []
                matched_part_ids = [-1 for i in range(points.shape[0])]  # points shape is n*c*7, add in 2022-12-09
                for id, (x1, y1, x2, y2, conf, cls) in enumerate(pdet_slim):
                    p_xc, p_yc = np.mean((x1, x2)), np.mean((y1, y2))  # the body-part's part bbox center point
                    part_pts = points[:, int(cls - 1)]
                    dist = np.linalg.norm(part_pts[:, :2] - np.array([[p_xc, p_yc]]), axis=-1)
                    pt_match = np.argmin(dist)

                    '''association alg version 4.0'''   
                    tmp_iou = cal_inside_iou(bboxes[pt_match], [x1, y1, x2, y2])  # add in 2022-12-11, body-part must inside the body
                    if conf > part_pts[pt_match][2] and tmp_iou > match_iou_thres:  # add in 2022-12-09, we fetch the part bbox with highest conf
                        part_pts[pt_match] = [p_xc, p_yc, conf, x1, y1, x2, y2]  # update points[:, int(cls - 1), 7]
                        matched_part_ids[pt_match] = id  # only for dataset BodyHands without left/right label of hands

            batch_bboxes.extend(bboxes)
            batch_points.extend(points)
            batch_scores.extend(scores)

        # else:
            # print("This image has no object detected!")
        
    return batch_bboxes, batch_points, batch_scores

###############################################################################

if __name__ == '__main__':

    alpha_value = 0.3
    edges_scale = -0.05
    DRAW_HEAD_MESH = False       # 关闭3D头部渲染 (只保留交头接耳标注)

    # ============ 交头接耳检测参数 v1 (精简版: 限制少, 宁可漏检不误判) ============
    # 只用两个最可靠的信号: ① 座位约束(同排左右邻座) ② 视线夹角(钝角=相互对看)
    # ---- 座位约束 ----
    ROW_Y_RATIO = 1.2            # 同排判定: 两头中心高度差 ≤ 该倍数×平均头高 (前后排排除)
    SEAT_DIST_RATIO = 0.35       # 邻座判定: 两头水平距离 ≤ 该比例×图宽
    NEIGHBOR_DX_RATIO = 0.5      # 左右相邻: 水平距离 ≥ 垂直距离×该比例
    # ---- 视线夹角 (钝角) ★ 主要调节点 ----
    GAZE_OBTUSE_THRESH = 100.0   # 视线夹角 ≥ 该值(钝角) → 判为交头接耳 (慢慢调: 90~130)
    MIN_TURN_YAW = 10.0          # 每人至少转头该角度 (没转头不可能在说话)
    # ---- 时序平滑 ----
    CHAT_MIN_DURATION_SEC = 0.1  # 持续 ≥0.1 秒才标注 (配合 FRAME_SKIP=3 → 1个处理帧)
    CHAT_HOLD_FRAMES = 2         # 确认后保持N帧 (防闪烁)
    CHAT_CONFIRM_THRESH = 0.55   # 分数高于此值显示红色 CHAT!, 否则黄色 SUSPECT
    CHAT_ONE_TO_ONE = True       # 一对一匹配 (防止一人配多人)
    # ---- 显示 ----
    DRAW_GAZE_ARROWS = False
    YAW_SIGN = -1                # yaw 符号约定 (与 SemiUHPE 输出对接)

    # ============ 跨帧人脸追踪补漏参数 (解决漏帧问题) ============
    TRACK_ENABLE = True             # 是否启用追踪补漏
    TRACK_N_PRE = 3                 # 历史参考帧数 (可测试2/3/5, 越大越稳但延迟越大)
    TRACK_MAX_HOLD = 2              # 单个目标最多连续补位帧数 (降低解决拖尾)
    TRACK_IOU_THRESH = 0.20         # 检测-轨迹匹配的IoU阈值
    TRACK_PREDICT_MODE = 'linear'   # 'last'用最后一帧位置, 'linear'用线性预测

    os.chdir(_BASE_DIR)  # 切换到脚本所在目录, 确保相对路径(weights/config等)生效

    # ============ Step 2 (提前): 视频列表 + 多进程分发 ============
    # 【关键修复】多进程分发必须放在模型加载之前, 否则主进程占着显存, 子进程再加载会OOM挂死
    _VIDEO_DIR = r"D:\video\1\1\right\cutting_video2"
    video_files = sorted(glob.glob(os.path.join(_VIDEO_DIR, "*.mp4")))
    print(f"[输入] {_VIDEO_DIR}  共 {len(video_files)} 个视频")

    # 输出目录
    output_folder = r"D:\chatting\outputs"
    os.makedirs(output_folder, exist_ok=True)

    # ============ 多进程并行处理 ============
    NUM_WORKERS = 2  # 并行进程数 (设为1则顺序处理; 单GPU建议先试2, 显存不足降到1)

    # 如果是 worker 进程, 只处理分配的视频
    _worker_indices = os.environ.get('VIDEO_WORKER_INDICES')
    if _worker_indices:
        _indices = [int(x) for x in _worker_indices.split(',')]
        video_files = [video_files[i] for i in _indices]
        print(f"[Worker] 处理 {len(video_files)} 个视频: {[os.path.basename(v) for v in video_files]}")
    elif NUM_WORKERS > 1 and len(video_files) > 1:
        # 主进程: 启动多个 worker 子进程后立即退出 (不加载模型, 不占显存)
        print(f"启动 {NUM_WORKERS} 个并行进程处理 {len(video_files)} 个视频...")
        processes = []
        chunk_size = (len(video_files) + NUM_WORKERS - 1) // NUM_WORKERS
        for w in range(NUM_WORKERS):
            start = w * chunk_size
            end = min(start + chunk_size, len(video_files))
            if start >= end:
                break
            indices = ','.join(str(i) for i in range(start, end))
            env = {**os.environ, 'VIDEO_WORKER_INDICES': indices}
            p = subprocess.Popen([sys.executable, os.path.abspath(__file__)], env=env)
            processes.append(p)
            print(f"  进程 {w+1}: 视频 {start+1}-{end} (共{end-start}个)")
        for p in processes:
            p.wait()
        print(f"\n全部完成! {len(video_files)} 个视频已处理, 输出目录: {output_folder}")
        sys.exit(0)

    # 自动设置命令行参数 (无需手动传参)
    sys.argv = [
        'video.py',
        'DAD-COCOHead-ResNet50-best',           # test_ckpt (released checkpoint in weights/)
        '--is_full_range',
        '--config', 'settings/DAD3DHeads_COCOHead.yml',
        '--network', 'resnet50',
        '--gpu_ids', '0',
    ]

    config = get_config('test')
    config.device = 'cuda:0'
    agent = get_agent(config)
    agent.load_ckpt(config.ckpt)
    
    
    """Step 1: load model weights"""
    weights_path = os.path.join(PY_ROOT, "weights", "ch_head_l_1536_e150_best_mMR.pt")
    conf_thres = 0.50
    iou_thres = 0.75
    match_iou_thres = 0.6
    HEAD_CONF_THRESH = 0.65  # 【v1调高】人头置信度, 防止椅子靠背被误认为人头
    # 尺寸过滤
    HEAD_MIN_RATIO = 0.015
    HEAD_MAX_RATIO = 0.35
    # 必须同时检测到身体+头部
    REQUIRE_BODY_HEAD_PAIR = True
    # 头部必须在身体的上方/上部区域
    HEAD_IN_BODY_UPPER = True
    # 身体-头部尺寸约束 (防止椅子靠背)
    ENFORCE_BODY_HEAD_RATIO = True
    # 坐着的情况: 身体高/头高 ∈ [2.5, 8.0] (提高下限过滤椅子)
    BH_RATIO_MIN = 2.5  # 提高身体/头部高度比下限 (原2.0, 椅子身头比通常偏小)
    BH_RATIO_MAX = 8.0
    # 身体高/头高 的最优值 (用于打分, 偏离越远惩罚越大)
    BH_RATIO_IDEAL = 4.0
    # 头部必须贴近身体顶部 (head_bottom 和 body_top 的相对差 < body_h * 0.40)
    HEAD_NEAR_BODY_TOP = True
    # 最小身体高度 (像素)
    MIN_BODY_HEIGHT = 50  # 提高最小身体高度，过滤椅子靠背
    BODY_CONF_THRESH = 0.62  # 【v1调高】身体置信度
    MIN_BODY_WIDTH = 30      # 最小身体宽度 (过滤窄椅子靠背)
    scales = [1]
    imgsz = 960
    num_offsets = 2
    FRAME_SKIP = 3  # 每3帧处理1帧, 提速1.5x (零精度损失)

    device = 'cuda:0'
    print('Using device: {}'.format(device))

    # Windows 中文环境: 强制 UTF-8 编码, 避免 torch.compile 的 GBK 错误
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except Exception:
        pass
    import os
    os.environ.setdefault('PYTHONUTF8', '1')
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'backend:cudaMallocAsync')

    # GPU 优化
    torch.backends.cudnn.benchmark = True  # 自动选最快卷积算法
    torch.backends.cudnn.deterministic = False

    BPJDet_model = attempt_load(weights_path, map_location=device)
    stride = int(BPJDet_model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    BPJDet_model.eval()

    # FP16 半精度推理
    USE_FP16 = True
    if USE_FP16:
        BPJDet_model = BPJDet_model.half()
        print("FP16 半精度推理已开启")

    # torch.compile 在 Windows 上需要 triton, 但 triton 不支持 Windows, 直接跳过
    # 保留 cudnn.benchmark + FP16 + inference_mode 等其他加速手段
    print("torch.compile 跳过 (Windows 无 triton 支持)")

    # 预初始化 head pose 模型
    agent.net.eval()
    agent.net.to(device)
    if USE_FP16:
        agent.net = agent.net.half()

    print(f"BPJDet imgsz={imgsz}, FRAME_SKIP={FRAME_SKIP}")

    # CUDA 预热: 一次性分配显存 + 走一遍完整推理, 消除视频首帧JIT/编译开销
    if device.startswith('cuda'):
        print("CUDA 预热中...")
        if USE_FP16:
            amp_ctx_warm = torch.autocast(device_type='cuda', dtype=torch.float16)
        else:
            amp_ctx_warm = torch.autocast(device_type='cuda', dtype=torch.float32)
        with torch.inference_mode():
            with amp_ctx_warm:
                # BPJDet 预热
                dummy = torch.zeros((1, 3, imgsz, imgsz), dtype=torch.float16 if USE_FP16 else torch.float32, device=device)
                try:
                    _ = BPJDet_model(dummy, augment=False)
                except Exception:
                    pass
                # HeadPose 预热
                dummy_head = torch.zeros((2, 3, 224, 224), dtype=torch.float16 if USE_FP16 else torch.float32, device=device)
                try:
                    _ = agent.net(dummy_head)
                except Exception:
                    pass
            torch.cuda.synchronize()
        print("CUDA 预热完成")

    # 全局 amp_ctx (预热+主循环共用)
    if USE_FP16:
        amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
    else:
        amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float32)

    gif = False
    video_save_size = 720
    video_start, video_end = 0, -1
    body_bbox, head_bbox = False, True
    fps_fontsize = 0  # 0 or 1
    display = False
    is_save_gif = gif  # save gif (.gif) or video (.mp4)
    write_video = not display and not is_save_gif

    print(f"共 {len(video_files)} 个视频待处理, 输出目录: {output_folder}")

    for vid_idx, video_path in enumerate(video_files):
        print(f"\n{'='*60}")
        print(f"[{vid_idx+1}/{len(video_files)}] {os.path.basename(video_path)}")
        print(f"{'='*60}")

        dataset = LoadImages(video_path, img_size=imgsz, stride=stride, auto=True)
        cap = dataset.cap
        cap.set(cv2.CAP_PROP_POS_MSEC, video_start * 1000)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if video_end == -1:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - fps * video_start)
        else:
            n = int(fps * (video_end - video_start))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        gif_frames = []
        out_path = os.path.join(output_folder, os.path.splitext(os.path.basename(video_path))[0] + "_chat")
        print("fps:", fps, "\t total frames:", n, "\n", "out_path:", out_path)

        # 根据FPS和FRAME_SKIP动态计算确认帧数 (保证持续0.4秒才标注)
        # 有效帧率 = fps / FRAME_SKIP, 确认帧数 = ceil(0.4秒 * 有效帧率)
        _effective_fps = max(1.0, fps / max(1, FRAME_SKIP))
        CHAT_CONFIRM_FRAMES = max(1, int(-(-CHAT_MIN_DURATION_SEC * _effective_fps // 1)))
        print(f"[时序] 交头接耳确认: {CHAT_MIN_DURATION_SEC}秒 → 需要连续{CHAT_CONFIRM_FRAMES}帧 (有效FPS={_effective_fps:.1f})")

        writer_mp4 = None
        if write_video:
            out_size = (int(video_save_size*w/h), video_save_size)
            # 尝试多个编码器, 确保兼容性
            codecs = ['avc1', 'XVID', 'mp4v', 'H264']
            for codec in codecs:
                writer_mp4 = cv2.VideoWriter(out_path + '.mp4',
                    cv2.VideoWriter_fourcc(*codec), fps, out_size)
                if writer_mp4.isOpened():
                    print(f"使用编码器: {codec}")
                    break
            if writer_mp4 is None or not writer_mp4.isOpened():
                print("警告: 视频写入器未打开, 跳过保存")
                write_video = False


        """Step 3: processing frames"""
        dataset = tqdm(dataset, desc='Running inference', total=n)
        t0 = time_sync()

        # 用生产者-消费者模式: 解码线程预读帧, GPU线程做推理, 实现CPU/GPU并行
        frame_queue = queue.Queue(maxsize=8)
        STOP_TOKEN = None

        def decoder_thread():
            """后台线程: 提前解码帧放入队列"""
            for item in dataset:
                frame_queue.put(item)
            frame_queue.put(STOP_TOKEN)

        dec_thread = threading.Thread(target=decoder_thread, daemon=True)
        dec_thread.start()

        # amp_ctx 已在全局定义, 这里不再重复定义

        # 时序平滑状态 (跨帧持久化)
        video_processor = type('VideoProcessor', (), {'chat_history': {}})()

        # ========== 跨帧追踪器 (解决漏帧问题) ==========
        # 轨迹结构: {track_id: {'history': deque[(head_bbox, body_bbox, frame_i)], 'hold': 连续补位次数, 'last_seen': 最后匹配帧idx, 'prev_yaw': 上一帧yaw}}
        tracker = {
            'tracks': {},
            'next_id': 0,
        }
        # 帧间 track_id → head_bbox_list 索引的映射 (用于 yaw 时序检查)
        _track_id_map = {}  # {head_bbox_list_index: track_id}
        from collections import deque

        def _bbox_iou(a, b):
            """两个bbox的IoU, a=[x1,y1,x2,y2], b相同"""
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
            iw = max(0, inter_x2 - inter_x1); ih = max(0, inter_y2 - inter_y1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
            return inter / (union + 1e-6)

        def _to_int_bbox(box):
            """统一转 Python int, 避免 cv2.rectangle 在 4.11+ 拒绝 numpy 类型"""
            if box is None:
                return None
            return [int(round(float(v))) for v in box]

        def _predict_bbox(track, cur_frame_i, img_w, img_h, mode='linear'):
            """用轨迹历史预测当前帧bbox (head_bbox, body_bbox)
            随hold增加降低预测速度 (防拖尾)"""
            hist = list(track['history'])
            if len(hist) == 0:
                return None, None
            hold = track.get('hold', 0)
            # 衰减因子: hold越大, 速度外推越保守
            # hold=0: 完整速度; hold=1: 80%速度; hold=2: 50%速度; hold>=3: 停止外推
            if hold >= TRACK_MAX_HOLD:
                decay = 0.0
            else:
                decay = max(0.0, 1.0 - hold * 0.5)
            # 最近的参考点 (不超过 TRACK_N_PRE 个)
            ref = hist[-TRACK_N_PRE:]
            last_head, last_body, last_idx = ref[-1]
            if mode == 'last' or len(ref) < 2 or decay <= 0:
                return _to_int_bbox(last_head), _to_int_bbox(last_body)
            # 线性预测: 用近2帧的速度外推, 并应用衰减
            prev_head, prev_body, prev_idx = ref[-2]
            dt = last_idx - prev_idx
            if dt <= 0:
                return _to_int_bbox(last_head), _to_int_bbox(last_body)
            dt_pred = (cur_frame_i - last_idx) * decay
            # head/body 预测: 用近2帧中心点的速度外推
            def pred_box(p_box, c_box, d_t):
                pcx = (p_box[0]+p_box[2])/2.0; pcy = (p_box[1]+p_box[3])/2.0
                ccx = (c_box[0]+c_box[2])/2.0; ccy = (c_box[1]+c_box[3])/2.0
                vx = (ccx - pcx)/dt; vy = (ccy - pcy)/dt
                nc_x = ccx + vx*d_t; nc_y = ccy + vy*d_t
                w = c_box[2]-c_box[0]; h = c_box[3]-c_box[1]
                return [max(0, int(round(nc_x-w/2.0))), max(0, int(round(nc_y-h/2.0))),
                        min(img_w-1, int(round(nc_x+w/2.0))), min(img_h-1, int(round(nc_y+h/2.0)))]
            pred_h = pred_box(prev_head, last_head, dt_pred)
            if prev_body is not None and last_body is not None:
                pred_b = pred_box(prev_body, last_body, dt_pred)
            else:
                pred_b = _to_int_bbox(last_body)
            return pred_h, pred_b

        # 预计算常量
        _h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        _line_thick = max(_h, _w) // 1000 + 3
        # 全局内参矩阵 (原每帧重建 → 复用)
        _global_intrinsics = np.array([
            [_h * 1.2, 0,      _w / 2.0],
            [0,      _h * 1.2, _h / 2.0],
            [0,      0,        1.0],
        ], dtype=np.float32)


        for i in range(n):
            item = frame_queue.get()
            if item is STOP_TOKEN:
                break
            path, img, im0, _ = item

            # 跳帧加速
            if i % FRAME_SKIP != 0:
                continue

            img = torch.from_numpy(img).to(device, non_blocking=True)
            img = img / 255.0
            if len(img.shape) == 3:
                img = img[None]

            # FP16 autocast + inference_mode (比no_grad快~10%)
            with torch.inference_mode():
                with amp_ctx:
                    out_ori = BPJDet_model(img, augment=False, scales=scales)[0]
                    body_dets = non_max_suppression(out_ori, conf_thres, iou_thres,
                        classes=[0], num_offsets=num_offsets)
                    part_dets = non_max_suppression(out_ori, conf_thres, iou_thres,
                        classes=list(range(1, 1 + num_offsets//2)), num_offsets=num_offsets)
                # Post-processing 必须在 inference_mode 内, 否则 inplace 修改会报错
                bboxes, points, scores = post_process_batch(
                    img, [], [[(_h, _w)]], body_dets, part_dets, num_offsets, match_iou_thres)

            line_thick = _line_thick
            im0_ori = im0.copy()  # 需要原图做 head pose 裁剪
            img_h, img_w = _h, _w

            head_bbox_list = []   # [(head_bbox, body_bbox), ...]  同时存身体框, 用于后续邻座判断
            for ind, (bbox, point, score) in enumerate(zip(bboxes, points, scores)):
                f_score, f_bbox = point[0][2], point[0][3:]
                if f_score < HEAD_CONF_THRESH:
                    continue
                # 身体检测置信度过滤 (椅子靠背通常身体分数较低)
                if score < BODY_CONF_THRESH:
                    continue

                # 必须同时有身体检测框
                if REQUIRE_BODY_HEAD_PAIR:
                    body_detected = bbox is not None and len(bbox) > 0
                    if not body_detected:
                        continue

                # 头部必须在身体上方
                if HEAD_IN_BODY_UPPER and bbox is not None and len(bbox) > 0:
                    bx1, by1, bx2, by2 = bbox
                    body_h = by2 - by1
                    body_w = bx2 - bx1
                    # 身体必须足够高
                    if body_h < MIN_BODY_HEIGHT:
                        continue
                    if body_w < MIN_BODY_WIDTH:
                        continue
                    hx1, hy1, hx2, hy2 = f_bbox
                    head_cy = (hy1 + hy2) / 2
                    head_h = hy2 - hy1
                    head_w = hx2 - hx1

                    # 头部中心必须在身体上50% (放宽, 坐着姿势可能头部位置偏下)
                    if head_cy > by1 + body_h * 0.50:
                        continue
                    # 水平范围 (放宽±25%, 允许坐姿头部偏移)
                    head_cx = (hx1 + hx2) / 2
                    if head_cx < bx1 - body_w * 0.25 or head_cx > bx2 + body_w * 0.25:
                        continue

                    # 身体-头部高度比例
                    if ENFORCE_BODY_HEAD_RATIO and head_h > 0:
                        bh_ratio = body_h / head_h
                        if bh_ratio < BH_RATIO_MIN or bh_ratio > BH_RATIO_MAX:
                            continue

                    # 头-身宽度比: 人头宽度通常是身体的 15-60% (放宽)
                    if head_w > 0 and body_w > 0:
                        hw_ratio = head_w / body_w
                        if hw_ratio < 0.15 or hw_ratio > 0.65:
                            continue

                    # 身体形状: 人体宽高比 0.08~0.95 (坐姿身体框可能很扁)
                    if body_w > 0:
                        body_aspect = body_w / body_h
                        if body_aspect < 0.15 or body_aspect > 0.90:  # 提高下限过滤椅子靠背
                            continue

                    # 头部底部贴近身体顶部 (放宽到0.55, 允许坐姿头部稍低)
                    if HEAD_NEAR_BODY_TOP:
                        head_bottom = hy2
                        dist_rel = (head_bottom - by1) / body_h
                        if dist_rel > 0.40:  # 更严格要求头部贴近身体顶部
                            continue
                        if (by1 - hy1) < -head_h * 0.6:
                            continue

                # 尺寸过滤
                hx1, hy1, hx2, hy2 = f_bbox
                head_w = hx2 - hx1
                head_ratio = head_w / img_w
                if head_ratio < HEAD_MIN_RATIO or head_ratio > HEAD_MAX_RATIO:
                    continue
                # 长宽比过滤
                head_h = hy2 - hy1
                if head_h > 0:
                    aspect = head_w / head_h
                    if aspect < 0.5 or aspect > 1.8:
                        continue
                color = colors_list[ind%len(colors_list)]
                # 同时保存 head_bbox 和通过所有过滤的 body_bbox
                body_box_saved = list(bbox) if (bbox is not None and len(bbox) > 0) else None
                head_bbox_list.append((list(f_bbox), body_box_saved))
                # 可视化精简: 不画普通检测框, 只在交头接耳判定后画

            # ========== 跨帧追踪补漏 (解决漏帧) ==========
            if TRACK_ENABLE:
                # --- Step 1: 当前检测和历史轨迹做IoU匹配 ---
                cur_head_bboxes = [h for (h, _) in head_bbox_list]
                matched_cur = set()   # 已被轨迹匹配的当前检测序号
                matched_trk = set()   # 已匹配到检测的轨迹ID
                tracks = tracker['tracks']
                for tid, trk in list(tracks.items()):
                    if trk['hold'] >= TRACK_MAX_HOLD:
                        del tracks[tid]; continue
                    best_iou = TRACK_IOU_THRESH
                    best_ci = -1
                    pred_h, _ = _predict_bbox(trk, i, _w, _h, TRACK_PREDICT_MODE)
                    if pred_h is None:
                        del tracks[tid]; continue
                    for ci, ch in enumerate(cur_head_bboxes):
                        if ci in matched_cur: continue
                        iou = _bbox_iou(pred_h, ch)
                        if iou > best_iou:
                            best_iou = iou; best_ci = ci
                    if best_ci >= 0:
                        matched_cur.add(best_ci); matched_trk.add(tid)
                        _track_id_map[best_ci] = tid
                        trk['history'].append((head_bbox_list[best_ci][0], head_bbox_list[best_ci][1], i))
                        if len(trk['history']) > TRACK_N_PRE + 5:
                            trk['history'].popleft()
                        trk['hold'] = 0
                        trk['last_seen'] = i
                # --- Step 2: 未匹配轨迹 → 预测补位 ---
                _tracked_indices = set()
                unmatched_track_ids = set()
                for tid, trk in list(tracks.items()):
                    if tid in matched_trk: continue
                    if trk['hold'] >= TRACK_MAX_HOLD:
                        del tracks[tid]; continue
                    pred_h, pred_b = _predict_bbox(trk, i, _w, _h, TRACK_PREDICT_MODE)
                    if pred_h is None:
                        del tracks[tid]; continue
                    # 检测是否有 "真正的新检测" 已经在附近
                    has_nearby_real = False
                    for ci, ch in enumerate(cur_head_bboxes):
                        if ci in matched_cur: continue
                        if _bbox_iou(pred_h, ch) > 0.15:
                            has_nearby_real = True; break
                    if has_nearby_real:
                        # 有真实检测在附近 → 不补位 (避免双轨拖尾), 只清空hold
                        trk['hold'] = 0
                        continue
                    # 补位: 加入 head_bbox_list
                    trk['hold'] += 1
                    trk['history'].append((pred_h, pred_b, i))
                    if len(trk['history']) > TRACK_N_PRE + 5:
                        trk['history'].popleft()
                    idx = len(head_bbox_list)
                    _tracked_indices.add(idx)
                    unmatched_track_ids.add(tid)
                    _track_id_map[idx] = tid
                    head_bbox_list.append((pred_h, pred_b))
                    # 补位框: 仅加入检测列表, 不在此绘制 (正常人不标注)
                    # 补位框若属于交头接耳, 会在后续可视化阶段绘制黄/红框
                # --- Step 3: 未匹配的当前检测 → 新建轨迹 ---
                for ci, (hh, bb) in enumerate(head_bbox_list):
                    if ci in matched_cur or ci in _tracked_indices: continue
                    nid = tracker['next_id']; tracker['next_id'] += 1
                    q = deque()
                    q.append((hh, bb, i))
                    tracks[nid] = {'history': q, 'hold': 0, 'last_seen': i, 'prev_yaw': None}
                    _track_id_map[ci] = nid
                # --- Step 4: 反向吸收 - 把新检测和残留幽灵轨迹合并 ---
                new_track_ids = set()
                for nid in list(tracks.keys()):
                    if nid not in new_track_ids:
                        new_track_ids.add(nid)
                # 如果某新检测与残留幽灵轨迹IoU高 → 合并(删除幽灵轨迹, 保留新检测所在轨迹)
                for tid in list(tracks.keys()):
                    trk = tracks.get(tid)
                    if trk is None or trk['hold'] == 0:
                        continue  # 真实检测轨迹, 跳过
                    # 检查是否有hold=0的真实检测轨迹和它IoU高
                    for real_id, real_trk in tracks.items():
                        if real_id == tid or real_trk['hold'] != 0: continue
                        rh = list(real_trk['history'])[-1][0]
                        gh = list(trk['history'])[-1][0]
                        if _bbox_iou(rh, gh) > 0.25:
                            # 幽灵轨迹已被真实检测吸收, 删除
                            if tid in tracks:
                                del tracks[tid]
                            break

            t = time_sync() - t0
            t0 = time_sync()

            # 收集每人的头部信息
            head_infos = []
            if head_bbox_list:
                head_tensors = []
                head_crops = []
                body_bboxes_for_infos = []
                for (f_bbox, body_box) in head_bbox_list:
                    [px1, py1, px2, py2] = f_bbox
                    pcx, pcy = (px1 + px2)/2.0, (py1 + py2)/2.0
                    head_size = max(px2 - px1, py2 - py1)
                    new_px1 = max(0, int(pcx - (0.5 - edges_scale) * head_size))
                    new_px2 = min(img_w-1, int(pcx + (0.5 - edges_scale) * head_size))
                    new_py1 = max(0, int(pcy - (0.5 - edges_scale) * head_size))
                    new_py2 = min(img_h-1, int(pcy + (0.5 - edges_scale) * head_size))

                    img_head_cv2 = im0_ori[new_py1:new_py2, new_px1:new_px2]
                    # cv2.resize (双线性) 替代 PIL.resize, 速度快 3-5x
                    img_resized = cv2.resize(img_head_cv2, (224, 224), interpolation=cv2.INTER_LINEAR)
                    # BGR→RGB + HWC→CHW + uint8→float32/255
                    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                    img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().div_(255.0)
                    # 用 GPU 批量 normalize (原每帧单独 PIL normalize → 移到 batch_tensor 后 GPU 做)
                    head_tensors.append(img_tensor)
                    head_crops.append([new_px1, new_py1, new_px2, new_py2])
                    body_bboxes_for_infos.append(body_box)

                # 批量推理 (FP16 + non_blocking)
                batch_tensor = torch.stack(head_tensors).to(device, non_blocking=True)
                # GPU 批量 normalize (原 PIL 每帧单独做, 速度提升 ~ 1.1x)
                mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                batch_tensor = (batch_tensor - mean) / std
                with torch.inference_mode():
                    with amp_ctx:
                        fisher_out = agent.net(batch_tensor)
                        pd_m = batch_torch_A_to_R(fisher_out.float())

                rot_mats = pd_m.detach().cpu().numpy()
                all_poses = []
                global_intrinsics = _global_intrinsics
                for idx in range(len(head_crops)):
                    rot_mat = rot_mats[idx]
                    rot_mat_2 = np.transpose(rot_mat)
                    angle = Rotation.from_matrix(rot_mat_2).as_euler("xyz", degrees=True)
                    roll, pitch, yaw = list(map(limit_angle, [angle[2], angle[0] - 180, angle[1]]))

                    hb = head_crops[idx]
                    global_pose = convert_rotmat_bbox_to_6dof(rot_mat, hb, global_intrinsics)
                    all_poses.append(global_pose)

                    cx = (hb[0] + hb[2]) / 2.0
                    cy = (hb[1] + hb[3]) / 2.0
                    body_bx = body_bboxes_for_infos[idx] if idx < len(body_bboxes_for_infos) else None
                    head_infos.append({
                        'cx': cx, 'cy': cy, 'yaw': yaw, 'roll': roll, 'pitch': pitch,
                        'bbox': [int(hb[0]), int(hb[1]), int(hb[2]), int(hb[3])],
                        'body_bbox': body_bx,
                        # 【bug修复】补位框(预测框)标记, 之前 is_tracked 从未被赋值导致补位框
                        # 按全置信度参与配对; 后脑勺最容易丢检触发补位, 必须按严格门槛处理
                        'is_tracked': (idx in _tracked_indices) if TRACK_ENABLE else False,
                    })

                # 批量渲染 3D 头部 (缩放到 head bbox 大小, 避免 mesh 过大)
                if DRAW_HEAD_MESH and all_poses:
                    trans_vertices = renderer.transform_vertices(im0, all_poses)
                    for idx, verts in enumerate(trans_vertices):
                        if idx < len(head_crops):
                            hb = head_crops[idx]
                            hx1, hy1, hx2, hy2 = hb
                            target_w = hx2 - hx1
                            target_h = hy2 - hy1
                            vert_x_min, vert_x_max = verts[:, 0].min(), verts[:, 0].max()
                            vert_y_min, vert_y_max = verts[:, 1].min(), verts[:, 1].max()
                            mesh_w = vert_x_max - vert_x_min
                            mesh_h = vert_y_max - vert_y_min
                            if mesh_w > 1 and mesh_h > 1:
                                scale_x = target_w / mesh_w * 0.95
                                scale_y = target_h / mesh_h * 0.95
                                cx_head = (hx1 + hx2) / 2.0
                                cy_head = (hy1 + hy2) / 2.0
                                verts[:, 0] = cx_head + (verts[:, 0] - (vert_x_min + vert_x_max) / 2) * scale_x
                                verts[:, 1] = cy_head + (verts[:, 1] - (vert_y_min + vert_y_max) / 2) * scale_y
                    im0 = renderer.render(im0, trans_vertices, alpha=alpha_value)

            # ============ 画视线箭头 ============
            for hi in head_infos:
                ya = YAW_SIGN * hi['yaw']
                yaw_rad = math.radians(ya)
                gx = math.sin(yaw_rad)
                gy = -math.cos(yaw_rad)
                arrow_len = max(30, (hi['bbox'][2] - hi['bbox'][0]) * 2.5)
                ex = int(hi['cx'] + gx * arrow_len)
                ey = int(hi['cy'] + gy * arrow_len)
                if DRAW_GAZE_ARROWS:
                    cv2.arrowedLine(im0, (int(hi['cx']), int(hi['cy'])), (ex, ey),
                                    (0, 255, 255), 2, tipLength=0.3)

            # ============ 交头接耳检测 v1 (精简版) ============
            # 设计原则: 限制尽量少, 宁可漏检不误判。只用两个可靠信号:
            #   ① 座位约束 - 只有同一排的左右邻座才算 (前后排的人排除)
            #   ② 视线夹角 - 两人头部朝向夹角达到钝角(≥GAZE_OBTUSE_THRESH) = 相互对看
            raw_chat_pairs = []
            n_heads = len(head_infos)
            if n_heads >= 2:
                for ai in range(n_heads):
                    for bi in range(ai + 1, n_heads):
                        ha, hb = head_infos[ai], head_infos[bi]

                        # ---- 门0: 双方都是补位框(预测框) → 不参与判定 ----
                        if ha.get('is_tracked', False) and hb.get('is_tracked', False):
                            continue

                        # ---- 门1: 座位约束 (同一排的左右邻座) ----
                        _hha = ha['bbox'][3] - ha['bbox'][1]
                        _hhb = hb['bbox'][3] - hb['bbox'][1]
                        if _hha <= 0 or _hhb <= 0:
                            continue
                        ref_head = (_hha + _hhb) * 0.5
                        dx_head = abs(ha['cx'] - hb['cx'])
                        dy_head = abs(ha['cy'] - hb['cy'])
                        # ① 同排: 头中心高度差 ≤ ROW_Y_RATIO 倍头高 (前后排的人高度差明显更大)
                        if dy_head > ROW_Y_RATIO * ref_head:
                            continue
                        # ② 邻座: 水平距离 ≤ SEAT_DIST_RATIO 倍图宽
                        if dx_head > SEAT_DIST_RATIO * img_w:
                            continue
                        # ③ 左右相邻: 水平距离 ≥ 垂直距离 × NEIGHBOR_DX_RATIO
                        if dx_head < dy_head * NEIGHBOR_DX_RATIO:
                            continue

                        # ---- 门2: 视线夹角 (钝角 = 相互对看) ★主要调节点 ----
                        ya_raw = YAW_SIGN * ha['yaw']
                        yb_raw = YAW_SIGN * hb['yaw']
                        # 每人必须真的转头了
                        if abs(ya_raw) < MIN_TURN_YAW or abs(yb_raw) < MIN_TURN_YAW:
                            continue
                        # 方向检查: 两人都得朝向对方 (否则背对背也会形成钝角)
                        a_on_left = ha['cx'] < hb['cx']
                        a_toward_b = (ya_raw < 0) if a_on_left else (ya_raw > 0)
                        b_toward_a = (yb_raw > 0) if a_on_left else (yb_raw < 0)
                        if not (a_toward_b and b_toward_a):
                            continue
                        # 视线夹角 = 两人头部转向的夹角, 相互对看时是钝角
                        gaze_angle = abs(ya_raw - yb_raw)
                        if gaze_angle > 180.0:
                            gaze_angle = 360.0 - gaze_angle
                        if gaze_angle < GAZE_OBTUSE_THRESH:
                            continue

                        # ---- 通过全部门: 记录 (分数仅用于显示和一对一排序) ----
                        score = min(1.0, gaze_angle / 150.0)
                        tid_a, tid_b = _track_id_map.get(ai), _track_id_map.get(bi)
                        if tid_a is not None and tid_b is not None:
                            pair_key = ('t', min(tid_a, tid_b), max(tid_a, tid_b))
                        else:
                            pair_key = ('i', min(ai, bi), max(ai, bi))
                        raw_chat_pairs.append({
                            'key': pair_key, 'a': ai, 'b': bi,
                            'ha': ha, 'hb': hb, 'score': score,
                            'gaze_angle': gaze_angle,
                        })

            # --- 调试: 打印前5个多人的帧的配对详情 (调参用) ---
            if not hasattr(video_processor, '_pair_dbg'):
                video_processor._pair_dbg = 0
            if video_processor._pair_dbg < 5 and n_heads >= 2:
                video_processor._pair_dbg += 1
                print(f"  [调参] {n_heads}人, 判定为交头接耳 {len(raw_chat_pairs)} 对")
                for ai in range(n_heads):
                    for bi in range(ai + 1, n_heads):
                        _a, _b = head_infos[ai], head_infos[bi]
                        _ya, _yb = YAW_SIGN * _a['yaw'], YAW_SIGN * _b['yaw']
                        _dy = abs(_a['cy'] - _b['cy'])
                        _dx = abs(_a['cx'] - _b['cx'])
                        _hh = ((_a['bbox'][3] - _a['bbox'][1]) + (_b['bbox'][3] - _b['bbox'][1])) * 0.5
                        _ga = abs(_ya - _yb); _ga = min(_ga, 360.0 - _ga)
                        print(f"     a{ai}-b{bi}: yaw={_ya:6.1f}/{_yb:6.1f} 夹角={_ga:5.1f} "
                              f"dy/头高={_dy / max(_hh, 1):4.2f} dx/图宽={_dx / max(img_w, 1):4.2f}")

            # --- 一对一贪心匹配 (避免一人配多人) ---
            if CHAT_ONE_TO_ONE and len(raw_chat_pairs) > 1:
                raw_chat_pairs.sort(key=lambda x: -x['score'])  # 按分数降序
                matched_ids = set()
                unique_pairs = []
                for cp in raw_chat_pairs:
                    if cp['a'] in matched_ids or cp['b'] in matched_ids:
                        continue
                    matched_ids.add(cp['a'])
                    matched_ids.add(cp['b'])
                    unique_pairs.append(cp)
                raw_chat_pairs = unique_pairs



            # --- 时序平滑 ---
            if not hasattr(video_processor, 'chat_history'):
                video_processor.chat_history = {}

            confirmed_pairs = []
            raw_keys = set(cp['key'] for cp in raw_chat_pairs)

            # 调试: 打印前3帧的检测详情
            if not hasattr(video_processor, '_debug_count'):
                video_processor._debug_count = 0
            if video_processor._debug_count < 3 and raw_chat_pairs:
                video_processor._debug_count += 1
                for cp in raw_chat_pairs[:3]:
                    ha, hb = cp['ha'], cp['hb']
                    ya, yb = YAW_SIGN * ha['yaw'], YAW_SIGN * hb['yaw']
                    dx = ha['cx'] - hb['cx']
                    dy = ha['cy'] - hb['cy']
                    dist_val = math.hypot(dx, dy)
                    la = math.degrees(math.acos(max(-1, min(1, 
                        math.sin(math.radians(ya)) * (dx/dist_val) + 
                        (-math.cos(math.radians(ya))) * (dy/dist_val)))))
                    lb = math.degrees(math.acos(max(-1, min(1, 
                        math.sin(math.radians(yb)) * (-dx/dist_val) + 
                        (-math.cos(math.radians(yb))) * (-dy/dist_val)))))
                    print(f"  [DEBUG] pair score={cp['score']:.2f}  look_a={la:.1f}° look_b={lb:.1f}° "
                          f"yaw_a={ya:.1f}° yaw_b={yb:.1f}° dist={dist_val:.0f}px")

            for key, (consec, hold) in list(video_processor.chat_history.items()):
                if key in raw_keys:
                    video_processor.chat_history[key] = (consec + 1, CHAT_HOLD_FRAMES)
                else:
                    new_hold = hold - 1
                    if new_hold > 0:
                        video_processor.chat_history[key] = (max(0, consec - 1), new_hold)
                    else:
                        del video_processor.chat_history[key]

            # 新出现的配对: 连续计数+1
            for cp in raw_chat_pairs:
                key = cp['key']
                if key not in video_processor.chat_history:
                    video_processor.chat_history[key] = (1, 0)

            # 筛选: 达到确认帧数 或 在保持帧内
            for cp in raw_chat_pairs:
                key = cp['key']
                consec, hold = video_processor.chat_history.get(key, (0, 0))
                if consec >= CHAT_CONFIRM_FRAMES or (hold > 0 and consec >= 1):
                    confirmed_pairs.append(cp)

            chat_pairs = confirmed_pairs

            # ============ 交头接耳可视化 (根据分数区分 SUSPECT 和 CHAT) ============
            chat_head_ids = set()      # 认定交头接耳的人 (score > CHAT_CONFIRM_THRESH)
            suspect_head_ids = set()   # 疑似交头接耳的人 (score <= CHAT_CONFIRM_THRESH)
            for cp in chat_pairs:
                ha, hb = cp['ha'], cp['hb']
                score = cp['score']
                
                # 根据分数区分: >CHAT_CONFIRM_THRESH 为CHAT! (红), <= 为SUSPECT (黄)
                is_confirmed = score > CHAT_CONFIRM_THRESH
                
                if is_confirmed:
                    # 认定: 红色连线 + CHAT!
                    chat_head_ids.add(cp['a'])
                    chat_head_ids.add(cp['b'])
                    line_color = (0, 0, 255)      # 红色
                    label = "CHAT!"
                else:
                    # 疑似: 黄色连线 + SUSPECT
                    suspect_head_ids.add(cp['a'])
                    suspect_head_ids.add(cp['b'])
                    line_color = (0, 200, 255)   # 黄色
                    label = "SUSPECT"
                
                cv2.line(im0, (int(ha['cx']), int(ha['cy'])),
                         (int(hb['cx']), int(hb['cy'])), line_color, 3)
                mid_x = int((ha['cx'] + hb['cx']) / 2)
                mid_y = int((ha['cy'] + hb['cy']) / 2)
                cv2.putText(im0, f"{score:.0%}", (mid_x - 20, mid_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)

            # 认定交头接耳: 红框 + CHAT!
            for idx, hi in enumerate(head_infos):
                if idx in chat_head_ids:
                    x1, y1, x2, y2 = hi['bbox']
                    cv2.rectangle(im0, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 0, 255), 3)
                    cv2.putText(im0, "CHAT!", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                elif idx in suspect_head_ids:
                    # 疑似: 黄色框 + SUSPECT标签
                    x1, y1, x2, y2 = hi['bbox']
                    cv2.rectangle(im0, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 200, 255), 2)
                    cv2.putText(im0, "SUSPECT", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

            info_text = '{:2d} Heads | {:2d} Chat'.format(len(head_bbox_list), len(chat_pairs))
            cv2.putText(im0, info_text, (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)

            if not is_save_gif and fps_fontsize:
                cv2.putText(im0, '{:.1f} FPS'.format(1 / t), (5 * fps_fontsize, 25 * fps_fontsize),
                    cv2.FONT_HERSHEY_SIMPLEX, fps_fontsize, (255, 255, 255), thickness=2 * fps_fontsize)

            if is_save_gif:
                gif_img = cv2.cvtColor(cv2.resize(im0, dsize=tuple(gif_size)), cv2.COLOR_RGB2BGR)
                if fps_fontsize:
                    cv2.putText(gif_img, '{:.1f} FPS'.format(1 / t), (5 * fps_fontsize, 25 * fps_fontsize),
                        cv2.FONT_HERSHEY_SIMPLEX, fps_fontsize, (255, 255, 255), thickness=2 * fps_fontsize)
                gif_frames.append(gif_img)
            elif write_video:
                im0 = cv2.resize(im0, dsize=(int(video_save_size*w/h), video_save_size))
                # 重复写入 FRAME_SKIP 次, 保持原视频时长不变
                for _ in range(FRAME_SKIP):
                    writer_mp4.write(im0)
            else:
                cv2.imshow('', im0)
                cv2.waitKey(1)

        dec_thread.join(timeout=30)
        cap.release()
        if write_video and writer_mp4 is not None:
            writer_mp4.release()
            print(f"已保存: {out_path}.mp4")
        if is_save_gif:
            print('Saving GIF...')
            with imageio.get_writer(out_path + '.gif', mode="I", fps=fps) as writer:
                for idx, frame in tqdm(enumerate(gif_frames)):
                    writer.append_data(frame)

    print(f"\n全部完成! {len(video_files)} 个视频已处理, 输出目录: {output_folder}")


'''
python video.py SSL1.0_r0.05_ce_tDyna0.75_b16_ema_RO_CO_CM_full/Sep20_195132/best \
    --is_full_range --config settings/DAD3DHeads_COCOHead.yml --network resnet50 --gpu_ids 1

python video.py SSL1.0_r0.05_ce_tDyna0.75_b32_ema_RO_CO_CM_full/Sep30_130637/best \
    --is_full_range --config settings/DAD3DHeads_COCOHead.yml --network repvgg --gpu_ids 1
'''
