# -*- coding: utf-8 -*-
"""批量: 对多张"交头接耳"示例图用 video.py 的算法算两人视线夹角,
最后汇总统计并给出阈值建议。
用法: python _diag_crop_batch.py 图1 [图2 ...]
"""
import sys, os, math
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\chatting\_dataset")
import cv2
import numpy as np
import torch
import sample_pairs as SP

V = SP.V
IMGS = sys.argv[1:]                # ★必须先存: SP.load_models 内部会重写 sys.argv(video.py:706)
CONF, MIN_FS = 0.15, 0.30          # 小截图要放宽
agent, det_model = SP.load_models(1280)

def analyze(img_path):
    im0 = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if im0 is None:
        print(f"!!! 读不到 {img_path}")
        return None
    img_h, img_w = im0.shape[:2]
    print(f"\n{'='*66}\n{os.path.basename(img_path)}  {img_w}x{img_h}")
    img = SP.letterbox(im0, 1280, stride=SP.STRIDE, auto=True)[0]
    img = np.ascontiguousarray(img.transpose((2, 0, 1))[::-1])
    x = torch.from_numpy(img).to(SP.DEVICE) / 255.0
    x = x[None]
    with torch.inference_mode():
        with SP.AMP:
            out = det_model(x, augment=False, scales=V.SCALES)[0]
            body_dets = V.non_max_suppression(out, CONF, V.IOU_THRES,
                                              classes=[0], num_offsets=V.NUM_OFFSETS)
            part_dets = V.non_max_suppression(out, CONF, V.IOU_THRES,
                                              classes=list(range(1, 1 + V.NUM_OFFSETS // 2)),
                                              num_offsets=V.NUM_OFFSETS)
        bboxes, points, scores = V.post_process_batch(
            x, [], [[(img_h, img_w)]], body_dets, part_dets, V.NUM_OFFSETS, V.MATCH_IOU_THRES)

    head_boxes = []
    for bbox, point, score in zip(bboxes, points, scores):
        fs, fb = point[0][2], point[0][3:]
        hx1, hy1, hx2, hy2 = fb
        if fs < MIN_FS or hx2 - hx1 <= 0 or hy2 - hy1 <= 0:
            continue
        head_boxes.append([int(hx1), int(hy1), int(hx2), int(hy2)])
    head_boxes.sort(key=lambda b: -(b[3] - b[1]))
    print(f"  检出头数: {len(head_boxes)} → 取头最大的两人")
    if len(head_boxes) < 2:
        print("  !! 不足两人")
        return None
    head_boxes = head_boxes[:2]

    crops, tensors, heads = [], [], []
    for hx1, hy1, hx2, hy2 in head_boxes:
        pcx, pcy = (hx1 + hx2) / 2.0, (hy1 + hy2) / 2.0
        hs = max(hx2 - hx1, hy2 - hy1)
        ex = ey = (0.5 - V.EDGES_SCALE) * hs
        nx1 = max(0, int(pcx - ex)); nx2 = min(img_w - 1, int(pcx + ex))
        ny1 = max(0, int(pcy - ey)); ny2 = min(img_h - 1, int(pcy + ey))
        rgb = cv2.cvtColor(cv2.resize(im0[ny1:ny2, nx1:nx2], (224, 224),
                                      interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        tensors.append(torch.from_numpy(rgb.transpose(2, 0, 1)).float().div_(255.0))
        crops.append([nx1, ny1, nx2, ny2])
    batch = ((torch.stack(tensors).to(SP.DEVICE) - SP.MEAN) / SP.STD)
    with torch.inference_mode():
        with SP.AMP:
            pd_m = V.batch_torch_A_to_R(agent.net(batch).float())
    rot_mats = pd_m.detach().cpu().numpy()
    for k, rb in enumerate(crops):
        rm = np.transpose(rot_mats[k])
        ang = V.Rotation.from_matrix(rm).as_euler("xyz", degrees=True)
        roll, pitch, yaw = list(map(V.limit_angle, [ang[2], ang[0] - 180, ang[1]]))
        heads.append({"cx": (rb[0] + rb[2]) / 2.0, "cy": (rb[1] + rb[3]) / 2.0,
                      "bbox": rb, "ya": V.YAW_SIGN * yaw, "pitch": pitch})
        print(f"  P{k}: 中心=({heads[-1]['cx']:.0f},{heads[-1]['cy']:.0f}) 头高={rb[3]-rb[1]} "
              f"ya={heads[-1]['ya']:.1f} ({'朝右' if heads[-1]['ya'] > 0 else '朝左'}) pitch={pitch:.1f}")

    a, b = heads
    h_a, h_b = a["bbox"][3] - a["bbox"][1], b["bbox"][3] - b["bbox"][1]
    ref = (h_a + h_b) * 0.5
    szr = max(h_a, h_b) / min(h_a, h_b)
    dist = math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])
    ya, yb = a["ya"], b["ya"]
    a_on_left = a["cx"] < b["cx"]
    a_toward = (ya > 0) if a_on_left else (ya < 0)
    b_toward = (yb < 0) if a_on_left else (yb > 0)
    gaze = abs(ya - yb)
    if gaze > 180.0:
        gaze = 360.0 - gaze
    print(f"  头高比={szr:.2f}(≤{V.SEAT_DEPTH_RATIO_MAX}) 距离={dist:.0f}px "
          f"(门槛 {V.SEAT_DIST_RATIO * ref:.0f}) 转头={abs(ya):.0f}/{abs(yb):.0f}° "
          f"互朝={a_toward and b_toward}")
    print(f"  >>> 视线夹角 = {gaze:.1f}°")
    return {"file": os.path.basename(img_path), "gaze": gaze, "ya": ya, "yb": yb,
            "toward": a_toward and b_toward, "szr": szr, "dist_ratio": dist / ref,
            "min_turn": min(abs(ya), abs(yb))}

results = []
for p in IMGS:
    try:
        r = analyze(p)
        if r:
            results.append(r)
    except Exception as e:
        print(f"!!! {p} 出错: {e}")

if results:
    print(f"\n{'='*66}\n===== 汇总 ({len(results)} 对) =====")
    gs = sorted(r["gaze"] for r in results)
    for r in results:
        print(f"  {r['file'][:44]:46s} 夹角={r['gaze']:6.1f}°  互朝={r['toward']}  "
              f"转头min={r['min_turn']:.0f}°  距离/头高={r['dist_ratio']:.2f}")
    n = len(gs)
    mean = sum(gs) / n
    med = gs[n // 2] if n % 2 else (gs[n // 2 - 1] + gs[n // 2]) / 2
    print(f"  min={gs[0]:.1f}  中位={med:.1f}  mean={mean:.1f}  max={gs[-1]:.1f}")
    print(f"  → 若阈值取 min 向下留余量: GAZE_OBTUSE_THRESH ≈ {max(45.0, gs[0] - 8):.0f}°")
