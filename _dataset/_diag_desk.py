# -*- coding: utf-8 -*-
"""诊断: 桌子坐标系下的座位判定 (配合 video.py 新版 judge_pairs)"""
import sys, os, math
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\chatting\_dataset")
import cv2
import numpy as np
import sample_pairs as SP
import video as V

img_path = sys.argv[1]
im0 = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
img_h, img_w = im0.shape[:2]
print("image", im0.shape)

agent, det_model = SP.load_models(1280)
img = SP.letterbox(im0, 1280, stride=SP.STRIDE, auto=True)[0]
img = np.ascontiguousarray(img.transpose((2, 0, 1))[::-1])
heads = SP.detect_heads(img, im0, agent, det_model)
print(f"检出人头: {len(heads)}")
for k, h in enumerate(heads):
    x1, y1, x2, y2 = h['bbox']
    print(f"  head{k}: cx={h['cx']:.0f} cy={h['cy']:.0f} 头高={y2-y1} yaw={h['yaw']:.1f}")

# ---- 桌子坐标系 ----
avg_head = sum(h['bbox'][3] - h['bbox'][1] for h in heads) / len(heads)
desk = {'vp': None, 'angle': 0.0}
desk = V.estimate_desk_geometry([V.feet_point(h) for h in heads], avg_head, desk, img_w, img_h)
print(f"\n桌子方向角={desk['angle']:.1f}°  VP={desk['vp']}")

ux, uy = V.desk_dir(desk, img_w / 2.0, img_h / 2.0)
print(f"全局沿桌方向 u=({ux:.2f},{uy:.2f})  (画面中心处)")

# ---- 找用户关注的两人: 黑衣(约300,235) 与 白衣(约485,330) ----
def nearest(hs, px, py):
    return min(range(len(hs)), key=lambda k: (hs[k]['cx'] - px) ** 2 + (hs[k]['cy'] - py) ** 2)

focus = []
if len(heads) >= 2:
    i1 = nearest(heads, 300, 235)
    i0 = nearest(heads, 485, 330)
    if i0 != i1:
        focus = [(i0, i1, "★用户关注对")]

n = len(heads)
seat_pass = []
for i in range(n):
    for j in range(i + 1, n):
        a, b = heads[i], heads[j]
        h_a = a['bbox'][3] - a['bbox'][1]
        h_b = b['bbox'][3] - b['bbox'][1]
        ref = (h_a + h_b) * 0.5
        tag = f"({i},{j})"
        if max(h_a, h_b) / min(h_a, h_b) > V.HEAD_SIZE_RATIO_MAX:
            print(f"{tag} 头高比{max(h_a,h_b)/min(h_a,h_b):.2f} 超限 → 排除")
            continue
        ga, gb = V.feet_point(a), V.feet_point(b)
        dx, dy = gb[0] - ga[0], gb[1] - ga[1]
        mx, my = (ga[0] + gb[0]) / 2.0, (ga[1] + gb[1]) / 2.0
        ux2, uy2 = V.desk_dir(desk, mx, my)
        along = abs(dx * ux2 + dy * uy2)
        across = abs(dx * uy2 - dy * ux2)
        ok_row = across <= V.DESK_ROW_RATIO * ref
        ok_near = along <= V.DESK_NEAR_RATIO * ref
        seat = ok_row and ok_near
        if seat:
            seat_pass.append((i, j))
        if (i, j) in [(f[0], f[1]) for f in focus] or seat:
            print(f"{tag} 沿桌={along:.0f} (≤{V.DESK_NEAR_RATIO*ref:.0f}) "
                  f"垂直桌={across:.0f} (≤{V.DESK_ROW_RATIO*ref:.0f}) "
                  f"座位{'✓' if seat else '✗'}")

print(f"\n座位约束通过的对: {seat_pass}")
for (i, j) in seat_pass:
    a, b = heads[i], heads[j]
    ya, yb = V.YAW_SIGN * a['yaw'], V.YAW_SIGN * b['yaw']
    turned = abs(ya) >= V.MIN_TURN_YAW and abs(yb) >= V.MIN_TURN_YAW
    a_left = a['cx'] < b['cx']
    a_t = (ya < 0) if a_left else (ya > 0)
    b_t = (yb > 0) if a_left else (yb < 0)
    gaze = abs(ya - yb)
    if gaze > 180:
        gaze = 360 - gaze
    ok = turned and a_t and b_t and gaze >= V.GAZE_OBTUSE_THRESH
    print(f"  ({i},{j}) 转头{turned} 互朝{a_t and b_t} 夹角{gaze:.1f}° → "
          f"{'✓交头接耳' if ok else '✗视线不够'}")
