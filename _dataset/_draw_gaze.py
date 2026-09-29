# -*- coding: utf-8 -*-
"""在截图上画出每人的视线方向箭头 + 两人视线夹角标注"""
import sys, os, math
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\chatting\_dataset")
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import sample_pairs as SP

img_path = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) > 2 else r"D:\chatting\outputs\gaze_debug.png"
im0 = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
img_h, img_w = im0.shape[:2]
print("image", im0.shape)

agent, det_model = SP.load_models(1280)
img = SP.letterbox(im0, 1280, stride=SP.STRIDE, auto=True)[0]
img = np.ascontiguousarray(img.transpose((2, 0, 1))[::-1])
x = torch.from_numpy(img).to(SP.DEVICE) / 255.0
x = x[None]
CONF = 0.15   # 降低以检出暗处/被3D网格覆盖的人
MIN_FS = 0.30 # 只按头框置信度筛选(身体框弱也能用)
with torch.inference_mode():
    with SP.AMP:
        out = det_model(x, augment=False, scales=SP.V.SCALES)[0]
        body_dets = SP.V.non_max_suppression(out, CONF, SP.V.IOU_THRES,
                                             classes=[0], num_offsets=SP.V.NUM_OFFSETS)
        part_dets = SP.V.non_max_suppression(out, CONF, SP.V.IOU_THRES,
                                             classes=list(range(1, 1 + SP.V.NUM_OFFSETS // 2)),
                                             num_offsets=SP.V.NUM_OFFSETS)
    bboxes, points, scores = SP.V.post_process_batch(
        x, [], [[(img_h, img_w)]], body_dets, part_dets, SP.V.NUM_OFFSETS, SP.V.MATCH_IOU_THRES)

head_boxes = []
for bbox, point, score in zip(bboxes, points, scores):
    f_score, f_bbox = point[0][2], point[0][3:]
    hx1, hy1, hx2, hy2 = f_bbox
    if f_score < MIN_FS or hx2 - hx1 <= 0 or hy2 - hy1 <= 0:
        continue
    head_boxes.append([int(hx1), int(hy1), int(hx2), int(hy2)])
print(f"送头姿的头数: {len(head_boxes)}")

heads = []
crops, tensors = [], []
for hb in head_boxes:
    hx1, hy1, hx2, hy2 = hb
    pcx, pcy = (hx1 + hx2) / 2.0, (hy1 + hy2) / 2.0
    hs = max(hx2 - hx1, hy2 - hy1)
    ex = ey = (0.5 - SP.V.EDGES_SCALE) * hs
    nx1 = max(0, int(pcx - ex)); nx2 = min(img_w - 1, int(pcx + ex))
    ny1 = max(0, int(pcy - ey)); ny2 = min(img_h - 1, int(pcy + ey))
    crop = im0[ny1:ny2, nx1:nx2]
    rgb = cv2.cvtColor(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR),
                       cv2.COLOR_BGR2RGB)
    tensors.append(torch.from_numpy(rgb.transpose(2, 0, 1)).float().div_(255.0))
    crops.append([nx1, ny1, nx2, ny2])
batch = torch.stack(tensors).to(SP.DEVICE)
batch = (batch - SP.MEAN) / SP.STD
with torch.inference_mode():
    with SP.AMP:
        fisher = agent.net(batch)
        pd_m = SP.V.batch_torch_A_to_R(fisher.float())
rot_mats = pd_m.detach().cpu().numpy()
for k, rb in enumerate(crops):
    rm = np.transpose(rot_mats[k])
    ang = SP.V.Rotation.from_matrix(rm).as_euler("xyz", degrees=True)
    roll, pitch, yaw = list(map(SP.V.limit_angle, [ang[2], ang[0] - 180, ang[1]]))
    ya = SP.V.YAW_SIGN * yaw
    fwd = rm @ np.array([0.0, 0.0, 1.0])   # 面部朝前向量(相机系: x右 y下 z向前)
    heads.append({"cx": (rb[0] + rb[2]) / 2.0, "cy": (rb[1] + rb[3]) / 2.0,
                  "bbox": rb, "ya": ya, "pitch": pitch, "fwd": fwd,
                  "hh": rb[3] - rb[1]})
    print(f"head{k}: cx={heads[-1]['cx']:.0f} cy={heads[-1]['cy']:.0f} "
          f"头高={heads[-1]['hh']} yaw={yaw:.1f} (ya={ya:.1f}) pitch={pitch:.1f} "
          f"fwd=({fwd[0]:.2f},{fwd[1]:.2f},{fwd[2]:.2f})")

# ---- 画图 ----
vis = im0.copy()
def draw_cn(bgr, xy, text, size=18, color=(255, 255, 255)):
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", size)
    dr.text(xy, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def arrow(bgr, p0, p1, color, thick=3):
    cv2.arrowedLine(bgr, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                    color, thick, cv2.LINE_AA, tipLength=0.15)

def gaze_dir(h, scale):
    """用旋转矩阵的面部朝前向量投影到2D(与3D网格渲染同源, 方向必然一致)"""
    f = h["fwd"]
    vx, vy = f[0], f[1]
    n = math.hypot(vx, vy)
    if n < 0.15:          # 几乎正对/背对镜头, 水平分量太小, 箭头无明显方向
        n = max(n, 0.15)
    return (h["cx"] + vx / n * scale, h["cy"] + vy / n * scale)

colors = [(0, 255, 255), (255, 160, 0), (0, 200, 0), (255, 0, 255)]
for k, h in enumerate(heads):
    c = colors[k % len(colors)]
    L = h["hh"] * 2.2
    tip = gaze_dir(h, L)
    # 画头框
    x1, y1, x2, y2 = h["bbox"]
    cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
    arrow(vis, (h["cx"], h["cy"]), tip, c, 4)
    vis = draw_cn(vis, (max(2, x1), max(2, y1 - 26)),
                  f"P{k} 朝向 {h['ya']:.0f}°", 20, c)

# 两两之间的夹角标注(取头高最大的两个作主对)
order = sorted(range(len(heads)), key=lambda k: -heads[k]["hh"])
if len(order) >= 2:
    i, j = order[0], order[1]
    a, b = heads[i], heads[j]
    gaze = abs(a["ya"] - b["ya"])
    if gaze > 180.0:
        gaze = 360.0 - gaze
    mx, my = (a["cx"] + b["cx"]) / 2, (a["cy"] + b["cy"]) / 2
    cv2.line(vis, (int(a["cx"]), int(a["cy"])), (int(b["cx"]), int(b["cy"])),
             (255, 255, 255), 1, cv2.LINE_AA)
    ok = gaze >= SP.V.GAZE_OBTUSE_THRESH
    label = f"夹角 {gaze:.1f}° ({'≥100° 达标' if ok else '<100° 不达标'})"
    vis = draw_cn(vis, (int(mx) + 8, int(my) + 8), label, 22,
                  (0, 255, 0) if ok else (0, 0, 255))
    print(f"主对 P{i}-P{j}: 夹角 {gaze:.1f}° (阈值 {SP.V.GAZE_OBTUSE_THRESH}°)")

os.makedirs(os.path.dirname(out_path), exist_ok=True)
cv2.imencode(os.path.splitext(out_path)[1], vis)[1].tofile(out_path)
print("saved:", out_path)
