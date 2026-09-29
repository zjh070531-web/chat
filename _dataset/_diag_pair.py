# -*- coding: utf-8 -*-
"""诊断单张图为什么没被判为交头接耳: 跑完整检测+头姿, 打印每步判定"""
import sys, os, math
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\chatting\_dataset")
import cv2
import numpy as np
import sample_pairs as SP

img_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\31026\.workbuddy\clipboard-images\clipboard-2026-09-29T09-11-38-130Z-261c5d70.png"
im0 = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
img_h, img_w = im0.shape[:2]
print("image", im0.shape)

agent, det_model = SP.load_models(1280)
img = SP.letterbox(im0, 1280, stride=SP.STRIDE, auto=True)[0]
img = np.ascontiguousarray(img.transpose((2, 0, 1))[::-1])
heads = SP.detect_heads(img, im0, agent, det_model)
print("检出人头:", len(heads))
for i, h in enumerate(heads):
    x1, y1, x2, y2 = h["bbox"]
    print(f"  head{i}: cx={h['cx']:.0f} cy={h['cy']:.0f} 头高={y2-y1:.0f} "
          f"yaw={h['yaw']:.1f} pitch={h['pitch']:.1f} roll={h['roll']:.1f}")

V = SP.V
n = len(heads)
if n < 2:
    print(f"\n只检出 {n} 个头(至少 2 个才能配对) → 直接不算")
for i in range(n):
    for j in range(i + 1, n):
        a, b = heads[i], heads[j]
        h_a = a["bbox"][3] - a["bbox"][1]
        h_b = b["bbox"][3] - b["bbox"][1]
        ref = (h_a + h_b) * 0.5
        dx = abs(a["cx"] - b["cx"]); dy = abs(a["cy"] - b["cy"])
        szr = max(h_a, h_b) / min(h_a, h_b)
        ya = V.YAW_SIGN * a["yaw"]; yb = V.YAW_SIGN * b["yaw"]
        a_on_left = a["cx"] < b["cx"]
        a_toward = (ya > 0) if a_on_left else (ya < 0)   # 2026-09-29 实测符号修正
        b_toward = (yb < 0) if a_on_left else (yb > 0)
        gaze = abs(ya - yb)
        if gaze > 180.0:
            gaze = 360.0 - gaze
        turned = abs(ya) >= V.MIN_TURN_YAW and abs(yb) >= V.MIN_TURN_YAW
        ok_seat = szr <= V.SEAT_DEPTH_RATIO_MAX          # ①同深度=同桌(头一样大)
        ok_depth = True
        ok_near = math.hypot(dx, dy) <= V.SEAT_DIST_RATIO * ref   # ②邻座距离
        ok_lr = True
        ok_turn = turned
        ok_toward = a_toward and b_toward
        ok_gaze = gaze >= V.GAZE_OBTUSE_THRESH
        print(f"\n=== 配对 ({i},{j}) ===")
        print(f"  ①同深度=同桌  头高比={szr:.2f} ≤ {V.SEAT_DEPTH_RATIO_MAX}   {'✓' if ok_seat else '✗ 前后排/隔桌'}")
        print(f"  ②同深度  头高比={szr:.2f} ≤ {V.HEAD_SIZE_RATIO_MAX}   {'✓' if ok_depth else '✗ 前后排'}")
        print(f"  ③邻座    距={math.hypot(dx, dy):.0f} ≤ {V.SEAT_DIST_RATIO * ref:.0f}   {'✓' if ok_near else '✗ 隔太远'}")
        print(f"  ④(旧判据'左右相邻'已废弃: 摄像头斜俯视下画面左右≠教室左右)")
        print(f"  ⑤转头    |ya|={abs(ya):.0f}° |yb|={abs(yb):.0f}° ≥ {V.MIN_TURN_YAW}°   {'✓' if ok_turn else '✗ 有人没转头'}")
        print(f"  ⑥互朝    a→b={a_toward} b→a={b_toward}   {'✓' if ok_toward else '✗ 背对背/同向'}")
        print(f"  ⑦夹角    {gaze:.1f}° ≥ {V.GAZE_OBTUSE_THRESH}°   {'✓' if ok_gaze else '✗ 不够钝'}")
        passed = ok_seat and ok_depth and ok_near and ok_lr and ok_turn and ok_toward and ok_gaze
        print(f"  => {'✓ 会判为交头接耳' if passed else '✗ 不算(卡在上面 ✗ 的那步)'}")
