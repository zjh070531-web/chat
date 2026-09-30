# -*- coding: utf-8 -*-
"""在原始截图上画出检出人头的视线箭头 + 夹角 (极小截图专用, 原尺寸检测优先)"""
import sys, os, math
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\chatting\_dataset")
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import sample_pairs as SP

IMGS = []
for a in sys.argv[1:]:
    if a.startswith("@"):
        IMGS += [ln.strip() for ln in open(a[1:], encoding="utf-8") if ln.strip()]
    else:
        IMGS.append(a)

V = SP.V
CONF, MIN_FS = 0.10, 0.20
agent, det_model = SP.load_models(1280)
OUT = r"D:\chatting\outputs\gaze10"
os.makedirs(OUT, exist_ok=True)
FONT = r"C:\Windows\Fonts\msyh.ttc"


def detect(im0, conf, min_fs):
    h, w = im0.shape[:2]
    img = SP.letterbox(im0, 1280, stride=SP.STRIDE, auto=True)[0]
    img = np.ascontiguousarray(img.transpose((2, 0, 1))[::-1])
    x = torch.from_numpy(img).to(SP.DEVICE) / 255.0
    x = x[None]
    with torch.inference_mode():
        with SP.AMP:
            out = det_model(x, augment=False, scales=V.SCALES)[0]
            bd = V.non_max_suppression(out, conf, V.IOU_THRES, classes=[0], num_offsets=V.NUM_OFFSETS)
            pd = V.non_max_suppression(out, conf, V.IOU_THRES,
                                       classes=list(range(1, 1 + V.NUM_OFFSETS // 2)),
                                       num_offsets=V.NUM_OFFSETS)
        bboxes, points, scores = V.post_process_batch(
            x, [], [[(h, w)]], bd, pd, V.NUM_OFFSETS, V.MATCH_IOU_THRES)
    res = []
    for bb, pt, sc in zip(bboxes, points, scores):
        fs, fb = pt[0][2], pt[0][3:]
        hx1, hy1, hx2, hy2 = fb
        if fs < min_fs or hx2 - hx1 <= 0 or hy2 - hy1 <= 0:
            continue
        res.append([int(hx1), int(hy1), int(hx2), int(hy2)])
    return res


def yaw_of(im0, hb):
    h, w = im0.shape[:2]
    hx1, hy1, hx2, hy2 = hb
    pcx, pcy = (hx1 + hx2) / 2.0, (hy1 + hy2) / 2.0
    hs = max(hx2 - hx1, hy2 - hy1)
    ex = ey = (0.5 - V.EDGES_SCALE) * hs
    nx1 = max(0, int(pcx - ex)); nx2 = min(w - 1, int(pcx + ex))
    ny1 = max(0, int(pcy - ey)); ny2 = min(h - 1, int(pcy + ey))
    rgb = cv2.cvtColor(cv2.resize(im0[ny1:ny2, nx1:nx2], (224, 224),
                                  interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().div_(255.0)
    t = t[None].to(SP.DEVICE)
    t = (t - SP.MEAN) / SP.STD
    with torch.inference_mode():
        with SP.AMP:
            pd_m = V.batch_torch_A_to_R(agent.net(t).float())
    rm = np.transpose(pd_m.detach().cpu().numpy()[0])
    ang = V.Rotation.from_matrix(rm).as_euler("xyz", degrees=True)
    roll, pitch, yaw = list(map(V.limit_angle, [ang[2], ang[0] - 180, ang[1]]))
    ya = V.YAW_SIGN * yaw
    # 面部朝前向量投影 (与 _draw_gaze.py 同源)
    fwd = rm @ np.array([0.0, 0.0, 1.0])
    fwd = np.array([fwd[0], -fwd[1], -fwd[2]])
    return yaw, pitch, ya, fwd


def draw_cn(bgr, xy, text, size, color):
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    font = ImageFont.truetype(FONT, size)
    dr.text(xy, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


colors = [(0, 255, 255), (255, 160, 0), (0, 200, 0), (255, 0, 255)]


def run(path):
    im0 = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if im0 is None:
        print("!!! 读不到", path); return
    h, w = im0.shape[:2]
    name = os.path.basename(path)
    print(f"\n{name}  {w}x{h}")
    # 原尺寸优先; 不够2人再依次降阈值、放大
    dets = detect(im0, CONF, MIN_FS)
    scale = 1.0
    if len(dets) < 2:
        dets = detect(im0, 0.05, 0.05)          # 极低阈值兜底(糊图/小头)
    if len(dets) < 2:
        big = cv2.resize(im0, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        d2 = detect(big, 0.05, 0.05)
        if len(d2) > len(dets):
            dets = [[int(v / 2) for v in bb] for bb in d2]
            scale = 2.0
    dets.sort(key=lambda b: -(b[3] - b[1]))
    print(f"  检出 {len(dets)} 人 (scale={scale})")
    heads = []
    for hb in dets[:4]:
        yaw, pitch, ya, fwd = yaw_of(im0, hb)
        heads.append({"bbox": hb, "ya": ya, "pitch": pitch, "fwd": fwd})
        print(f"    {hb} 头高={hb[3]-hb[1]} yaw={yaw:.1f} ya={ya:.1f} pitch={pitch:.1f}")

    vis = im0.copy()
    for k, hd in enumerate(heads):
        c = colors[k % len(colors)]
        x1, y1, x2, y2 = hd["bbox"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
        f = hd["fwd"]
        L = (y2 - y1) * 2.2
        vx, vy = f[0], f[1]
        n = math.hypot(vx, vy)
        if n < 0.15:
            n = max(n, 0.15)
        tip = (int(cx + vx / n * L), int(cy + vy / n * L))
        cv2.arrowedLine(vis, (int(cx), int(cy)), tip, c, 3, cv2.LINE_AA, tipLength=0.18)
        vis = draw_cn(vis, (max(1, x1), max(1, y1 - 16)),
                      f"P{k} {hd['ya']:+.0f}°", 13, c)

    # 夹角 (头最大的两人)
    if len(heads) >= 2:
        a, b = heads[0], heads[1]
        gaze = abs(a["ya"] - b["ya"])
        if gaze > 180.0:
            gaze = 360.0 - gaze
        (x1, y1, x2, y2) = a["bbox"]; ax, ay = (x1 + x2) / 2, (y1 + y2) / 2
        (x1, y1, x2, y2) = b["bbox"]; bx, by = (x1 + x2) / 2, (y1 + y2) / 2
        cv2.line(vis, (int(ax), int(ay)), (int(bx), int(by)), (255, 255, 255), 1, cv2.LINE_AA)
        ok = gaze >= V.GAZE_OBTUSE_THRESH
        vis = draw_cn(vis, (int((ax + bx) / 2), int((ay + by) / 2)),
                      f"夹角 {gaze:.1f}°", 16, (0, 0, 255) if not ok else (0, 255, 0))
        print(f"  夹角 {gaze:.1f}° (阈值 {V.GAZE_OBTUSE_THRESH}°)")
    else:
        vis = draw_cn(vis, (5, 5), f"只检出 {len(heads)} 人, 无法配对", 16, (0, 0, 255))

    out = os.path.join(OUT, name)
    cv2.imencode(".png", vis)[1].tofile(out)
    print("  saved", out)


for p in IMGS:
    try:
        run(p)
    except Exception as e:
        print("!!! ", p, e)
