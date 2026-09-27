
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
    vertices_path=os.path.join(_BASE_DIR, "pose_references", "vertices_trans.npy"),
    triangles_path=os.path.join(_BASE_DIR, "pose_references", "triangles.npy")
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
    DRAW_HEAD_MESH = True

    # ============ 交头接耳检测参数 ============
    CHAT_MAX_DIST_RATIO = 0.30      # 两人头部最大距离 (相对图像宽度, 邻座级别)
    CHAT_MAX_VERT_RATIO = 0.20      # 最大垂直距离 (相对图像高度, 同排约束)
    CHAT_GAZE_ANGLE_THRESH = 30.0   # 视线夹角下限
    CHAT_LOOK_ANGLE_THRESH = 85.0   # 看向对方角度上限
    CHAT_YAW_DIFF_THRESH = 8.0      # yaw差下限
    DRAW_GAZE_ARROWS = False
    YAW_SIGN = -1
    CHAT_SCORE_THRESH = 0.58        # 综合评分门槛
    CHAT_SCORE_STRICT = 0.65       # 单人朝向时的严格门槛

    # 时序平滑
    CHAT_CONFIRM_FRAMES = 1         # 连续帧数确认 (配合SKIP=3, 相当于原3帧)
    CHAT_HOLD_FRAMES = 2            # 确认后保持帧数

    # ============ 跨帧人脸追踪补漏参数 (解决漏帧问题) ============
    TRACK_ENABLE = True             # 是否启用追踪补漏
    TRACK_N_PRE = 3                 # 历史参考帧数 (可测试2/3/5, 越大越稳但延迟越大)
    TRACK_MAX_HOLD = 2              # 单个目标最多连续补位帧数 (降低解决拖尾)
    TRACK_IOU_THRESH = 0.20         # 检测-轨迹匹配的IoU阈值
    TRACK_PREDICT_MODE = 'linear'   # 'last'用最后一帧位置, 'linear'用线性预测

    os.chdir(_BASE_DIR)  # 切换到脚本所在目录, 确保相对路径(weights/config等)生效

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
    weights_path = "weights/ch_head_l_1536_e150_best_mMR.pt"
    conf_thres = 0.50
    iou_thres = 0.75
    match_iou_thres = 0.6
    HEAD_CONF_THRESH = 0.45
    # 尺寸过滤
    HEAD_MIN_RATIO = 0.015
    HEAD_MAX_RATIO = 0.35
    # 必须同时检测到身体+头部
    REQUIRE_BODY_HEAD_PAIR = True
    # 头部必须在身体的上方/上部区域
    HEAD_IN_BODY_UPPER = True
    # 身体-头部尺寸约束 (防止椅子靠背)
    ENFORCE_BODY_HEAD_RATIO = True
    # 坐着的情况: 身体高/头高 ∈ [1.5, 8.0] (放宽, 坐着身体框可能很扁)
    BH_RATIO_MIN = 1.5
    BH_RATIO_MAX = 8.0
    # 身体高/头高 的最优值 (用于打分, 偏离越远惩罚越大)
    BH_RATIO_IDEAL = 4.0
    # 头部必须贴近身体顶部 (head_bottom 和 body_top 的相对差 < body_h * 0.40)
    HEAD_NEAR_BODY_TOP = True
    # 最小身体高度 (像素)
    MIN_BODY_HEIGHT = 30
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

    """Step 2: 指定要检测的视频文件 (直接列出路径即可)"""
    video_files = [
        r"D:\deeplearning\ultralytics-8.3.163\pose_behavior_package_imitate\交头接耳\cutting_video2\segment_004.mp4",
        r"D:\deeplearning\ultralytics-8.3.163\pose_behavior_package_imitate\交头接耳\cutting_video2\segment_005.mp4",
        r"D:\deeplearning\ultralytics-8.3.163\pose_behavior_package_imitate\交头接耳\cutting_video2\segment_006.mp4",
        r"D:\deeplearning\ultralytics-8.3.163\pose_behavior_package_imitate\交头接耳\cutting_video2\segment_007.mp4",
    ]

    # 输出目录
    output_folder = r"D:\deeplearning\ultralytics-8.3.163\pose_behavior_package_imitate\交头接耳\outputs2(7)"
    os.makedirs(output_folder, exist_ok=True)

    # ============ 多进程并行处理 ============
    NUM_WORKERS = 2  # 并行进程数 (设为1则顺序处理)

    # 如果是 worker 进程, 只处理分配的视频
    _worker_indices = os.environ.get('VIDEO_WORKER_INDICES')
    if _worker_indices:
        _indices = [int(x) for x in _worker_indices.split(',')]
        video_files = [video_files[i] for i in _indices]
        print(f"[Worker] 处理 {len(video_files)} 个视频: {[os.path.basename(v) for v in video_files]}")
    elif NUM_WORKERS > 1 and len(video_files) > 1:
        # 主进程: 启动多个 worker 子进程
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
        # 轨迹结构: {track_id: {'history': deque[(head_bbox, body_bbox, frame_i)], 'hold': 连续补位次数, 'last_seen': 最后匹配帧idx}}
        tracker = {
            'tracks': {},
            'next_id': 0,
        }
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
                    hx1, hy1, hx2, hy2 = f_bbox
                    head_cy = (hy1 + hy2) / 2
                    head_h = hy2 - hy1
                    head_w = hx2 - hx1

                    # 头部中心必须在身体上50% (放宽, 坐着姿势可能头部位置偏下)
                    if head_cy > by1 + body_h * 0.50:
                        continue
                    # 水平范围
                    head_cx = (hx1 + hx2) / 2
                    if head_cx < bx1 - body_w * 0.15 or head_cx > bx2 + body_w * 0.15:
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

                    # 身体形状: 人体宽高比 0.10~0.75 (坐着身体框可能很扁)
                    if body_w > 0:
                        body_aspect = body_w / body_h
                        if body_aspect < 0.10 or body_aspect > 0.75:
                            continue

                    # 头部底部贴近身体顶部 (放宽到0.45, 允许坐姿头部稍低)
                    if HEAD_NEAR_BODY_TOP:
                        head_bottom = hy2
                        dist_rel = (head_bottom - by1) / body_h
                        if dist_rel > 0.45:
                            continue
                        if (by1 - hy1) < -head_h * 0.5:
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
                if body_bbox:
                    [x1, y1, x2, y2] = bbox
                    cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=line_thick)
                if head_bbox:
                    [px1, py1, px2, py2] = f_bbox
                    cv2.rectangle(im0, (int(px1), int(py1)), (int(px2), int(py2)), color, thickness=line_thick)

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
                    head_bbox_list.append((pred_h, pred_b))
                    # 可视化: 补位框用不同样式 (细线+不同颜色区分)
                    th = max(1, line_thick - 1)
                    cv2.rectangle(im0, (pred_h[0], pred_h[1]), (pred_h[2], pred_h[3]),
                                  (0, 210, 100), thickness=th, lineType=cv2.LINE_AA)
                # --- Step 3: 未匹配的当前检测 → 新建轨迹 ---
                for ci, (hh, bb) in enumerate(head_bbox_list):
                    if ci in matched_cur or ci in _tracked_indices: continue
                    nid = tracker['next_id']; tracker['next_id'] += 1
                    q = deque()
                    q.append((hh, bb, i))
                    tracks[nid] = {'history': q, 'hold': 0, 'last_seen': i}
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

            # ============ 交头接耳检测 (加权评分 + 时序平滑) ============
            raw_chat_pairs = []
            max_dist = CHAT_MAX_DIST_RATIO * img_w
            n_heads = len(head_infos)
            if n_heads >= 2:
                for ai in range(n_heads):
                    for bi in range(ai + 1, n_heads):
                        ha, hb = head_infos[ai], head_infos[bi]
                        bba = ha.get('body_bbox')
                        bbb = hb.get('body_bbox')

                        # ========= 补位框过滤 =========
                        a_tracked = ha.get('is_tracked', False)
                        b_tracked = hb.get('is_tracked', False)
                        if a_tracked and b_tracked:
                            continue  # 双方都是补位框, 跳过
                        one_tracked = a_tracked or b_tracked  # 一方为补位框

                        # ========= 身体尺度约束 (邻座级别, 最可靠) =========
                        if bba is None or bbb is None:
                            continue
                        ax1, ay1, ax2, ay2 = bba
                        bx1, by1, bx2, by2 = bbb
                        body_w_a, body_h_a = ax2 - ax1, ay2 - ay1
                        body_w_b, body_h_b = bx2 - bx1, by2 - by1
                        if body_w_a <= 0 or body_h_a <= 0 or body_w_b <= 0 or body_h_b <= 0:
                            continue
                        avg_body_w = (body_w_a + body_w_b) * 0.5
                        avg_body_h = (body_h_a + body_h_b) * 0.5
                        acx, acy = (ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5
                        bcx, bcy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5

                        # 约束1a: 身体水平距离 ≤ 3.5 个身宽 (覆盖坐姿身体框偏小)
                        if abs(acx - bcx) > avg_body_w * 3.5:
                            continue
                        # 约束1b: 身体垂直距离 ≤ 1.5 个身高 (放宽, 允许前后明显错开)
                        if abs(acy - bcy) > avg_body_h * 1.5:
                            continue
                        # 约束1c: 不能在x方向重叠过多 (排除同一人误检)
                        if max(ax1, bx1) < min(ax2, bx2) * 0.7:
                            continue

                        # 约束2: 头中心距离 (辅助约束, 用像素比例作为上限)
                        dx = ha['cx'] - hb['cx']
                        dy = ha['cy'] - hb['cy']
                        dist = math.hypot(dx, dy)
                        if dist > max_dist:
                            continue

                        # 约束3: 左右相邻 (水平距离 ≥ 垂直距离*0.7, 允许前后错开)
                        if abs(acx - bcx) < abs(acy - bcy) * 0.7:
                            continue

                        # 约束4: 朝向匹配 - 必须双方都朝向对方
                        # 交头接耳 = 双方都转头互看, 不是单方面看
                        ya_raw, yb_raw = YAW_SIGN * ha['yaw'], YAW_SIGN * hb['yaw']
                        dx_ab_b = bcx - acx
                        # 双方必须有明显转头 (≥5°), 防止面向屏幕被误判
                        FACING_YAW_THRESH = 5.0
                        facing_a = (dx_ab_b > 0 and ya_raw > FACING_YAW_THRESH) or (dx_ab_b < 0 and ya_raw < -FACING_YAW_THRESH)
                        facing_b = (dx_ab_b < 0 and yb_raw > FACING_YAW_THRESH) or (dx_ab_b > 0 and yb_raw < -FACING_YAW_THRESH)
                        facing_count = int(facing_a) + int(facing_b)

                        # 条件2: 计算各维度得分
                        ya, yb = ya_raw, yb_raw
                        yaw_diff = abs(ya - yb)

                        yaw_a_rad = math.radians(ya)
                        yaw_b_rad = math.radians(yb)
                        gaze_ax, gaze_ay = math.sin(yaw_a_rad), -math.cos(yaw_a_rad)
                        gaze_bx, gaze_by = math.sin(yaw_b_rad), -math.cos(yaw_b_rad)

                        # 用身体中心的方向计算 look 角度 (更符合座位位置)
                        dir_ab_x = bcx - acx
                        dir_ab_y = bcy - acy
                        dir_ab_norm = math.hypot(dir_ab_x, dir_ab_y) or 1
                        dir_ab_x /= dir_ab_norm
                        dir_ab_y /= dir_ab_norm
                        look_a = math.degrees(math.acos(max(-1, min(1, gaze_ax * dir_ab_x + gaze_ay * dir_ab_y))))
                        look_b = math.degrees(math.acos(max(-1, min(1, gaze_bx * (-dir_ab_x) + gaze_by * (-dir_ab_y)))))
                        gaze_angle = math.degrees(math.acos(max(-1, min(1, gaze_ax * gaze_bx + gaze_ay * gaze_by))))

                        # --- 加权评分 ---
                        look_score = max(0.0, 1.0 - max(look_a, look_b) / CHAT_LOOK_ANGLE_THRESH)
                        gaze_score = max(0.0, min(1.0, (gaze_angle - CHAT_GAZE_ANGLE_THRESH) / 80.0))
                        yaw_score = max(0.0, min(1.0, (yaw_diff - CHAT_YAW_DIFF_THRESH) / 60.0))
                        dist_score = max(0.0, 1.0 - dist / max_dist)

                        # --- 基础门限: 双方必须都朝向 ---
                        if facing_count < 2:
                            continue  # 必须双方都朝向对方, 单人朝向不算交头接耳
                        # 两人都朝向: 检查look角
                        if look_a > 85.0 and look_b > 85.0:
                            continue  # 两人视线都没对准对方

                        # --- 侧脸模式: 两人都是侧脸 + yaw方向相反 + 近距离 → 强信号 ---
                        PROFILE_YAW_THRESH = 20.0  # 原28→降低, 捕捉不那么极端的侧脸
                        both_profile = abs(ya) > PROFILE_YAW_THRESH and abs(yb) > PROFILE_YAW_THRESH
                        yaw_opposite = (ya * yb < 0)
                        near_distance = dist < max_dist * 0.75

                        # 评分: 看向对方(50%) + 视线夹角(20%) + yaw差(15%) + 距离(15%)
                        # 看向对方权重最高, 确保互看是必要条件
                        score = (0.50 * look_score + 0.20 * gaze_score +
                                 0.15 * yaw_score  + 0.15 * dist_score)

                        # 距离-视线联动: 距离越远, 要求夹角越大 (系数从35降到30, 更宽容)
                        dist_ratio = dist / max_dist
                        required_gaze = CHAT_GAZE_ANGLE_THRESH + dist_ratio * 30.0
                        if gaze_angle < required_gaze:
                            gaze_score *= 0.5  # 远距离惩罚减轻 (0.4→0.5)
                            score = (0.50 * look_score + 0.20 * gaze_score +
                                     0.15 * yaw_score  + 0.15 * dist_score)

                        if both_profile and yaw_opposite and near_distance:
                            profile_yaw_score = min(1.0, yaw_diff / 80.0)
                            profile_dist_score = max(0.0, 1.0 - dist / (max_dist * 0.75))
                            profile_score = 0.6 * profile_yaw_score + 0.4 * profile_dist_score
                            score = max(score, profile_score)

                        # 硬门槛: 双方都朝向, 有实质的看向
                        # 双方朝向是必须条件，移除单人朝向分支
                        if facing_count < 2:
                            continue
                        if look_score < 0.05:
                            continue
                        # 一方为补位框: 提高门槛
                        effective_thresh = CHAT_SCORE_THRESH  # 0.55
                        if one_tracked:
                            effective_thresh = max(effective_thresh, 0.65)

                        if score >= effective_thresh:
                            pair_key = (min(ai, bi), max(ai, bi))
                            raw_chat_pairs.append({
                                'key': pair_key,
                                'a': ai, 'b': bi,
                                'ha': ha, 'hb': hb,
                                'score': score,
                            })

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

            # ============ 交头接耳可视化 ============
            chat_head_ids = set()
            for cp in chat_pairs:
                ha, hb = cp['ha'], cp['hb']
                chat_head_ids.add(cp['a'])
                chat_head_ids.add(cp['b'])
                cv2.line(im0, (int(ha['cx']), int(ha['cy'])),
                         (int(hb['cx']), int(hb['cy'])), (0, 0, 255), 3)
                mid_x = int((ha['cx'] + hb['cx']) / 2)
                mid_y = int((ha['cy'] + hb['cy']) / 2)
                cv2.putText(im0, f"{cp['score']:.0%}", (mid_x - 20, mid_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            for idx, hi in enumerate(head_infos):
                x1, y1, x2, y2 = hi['bbox']
                if idx in chat_head_ids:
                    cv2.rectangle(im0, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 0, 255), 3)
                    cv2.putText(im0, "CHAT!", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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