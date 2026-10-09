# -*- coding: utf-8 -*-
"""小截图(宽<300px)专用: 先放大到长边>=960 再检测, 否则 BPJDet 检不出人。
同时 OCR 读图上 video.py 打的标注(CHAT xx???), 用于交叉验证。
用法: python _diag_small.py @列表文件
"""
import sys, os, math
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\chatting\_dataset")
import cv2
import numpy as np
import torch
import sample_pairs as SP

IMGS = []
for a in sys.argv[1:]:
    if a.startswith("@"):
        IMGS += [ln.strip() for ln in open(a[1:], encoding="utf-8") if ln.strip()]
    else:
        IMGS.append(a)

V = SP.V
CONF, MIN_FS = 0.10, 0.20          # 这类小图必须放很宽, 否则检不出
agent, det_model = SP.load_models(1280)

try:
    from rapidocr_onnxruntime import RapidOCR
    OCR = RapidOCR()
except Exception:
    OCR = None


def ocr(im, tag):
    if OCR is None:
        return
    res, _ = OCR(im)
    if not res:
        print(f"  [OCR {tag}] 无文字")
        return
    txt = " | ".join(str(r[1]) for r in res)
    print(f"  [OCR {tag}] {txt}")


def detect_heads(im0):
    """返回 [(head_bbox, body_bbox, head_score)], 多尺度重试(小图放大后会失真)"""
    img_h, img_w = im0.shape[:2]
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
    res = []
    for bbox, point, score in zip(bboxes, points, scores):
        fs, fb = point[0][2], point[0][3:]
        hx1, hy1, hx2, hy2 = fb
        if fs < MIN_FS or hx2 - hx1 <= 0 or hy2 - hy1 <= 0:
            continue
        res.append(([int(hx1), int(hy1), int(hx2), int(hy2)], bbox, float(fs)))
    return res


def head_yaw(im0, hb):
    img_h, img_w = im0.shape[:2]
    hx1, hy1, hx2, hy2 = hb
    pcx, pcy = (hx1 + hx2) / 2.0, (hy1 + hy2) / 2.0
    hs = max(hx2 - hx1, hy2 - hy1)
    ex = ey = (0.5 - V.EDGES_SCALE) * hs
    nx1 = max(0, int(pcx - ex)); nx2 = min(img_w - 1, int(pcx + ex))
    ny1 = max(0, int(pcy - ey)); ny2 = min(img_h - 1, int(pcy + ey))
    rgb = cv2.cvtColor(cv2.resize(im0[ny1:ny2, nx1:nx2], (224, 224),
                                  interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().div_(255.0)
    # ★MEAN/STD 是 4D (1,3,1,1), 必须先补 batch 维再归一化, 否则会变成 5D 喂残网络
    t = t[None].to(SP.DEVICE)
    t = (t - SP.MEAN) / SP.STD
    with torch.inference_mode():
        with SP.AMP:
            pd_m = V.batch_torch_A_to_R(agent.net(t).float())
    rm = np.transpose(pd_m.detach().cpu().numpy()[0])
    ang = V.Rotation.from_matrix(rm).as_euler("xyz", degrees=True)
    roll, pitch, yaw = list(map(V.limit_angle, [ang[2], ang[0] - 180, ang[1]]))
    return yaw, pitch, V.YAW_SIGN * yaw


def analyze(path):
    im0 = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if im0 is None:
        print(f"!!! 读不到 {path}")
        return None
    h, w = im0.shape[:2]
    print(f"\n{'='*70}\n{os.path.basename(path)}  原始 {w}x{h}")
    ocr(im0, "原图")
    # ★这类 172~247px 的小图在原尺寸检出最好, 放大反而丢了特征(x3 以上常常归零)
    big = im0
    print(f"  用原始尺寸检测 (放大反而掉检出)")
    dets = detect_heads(big)
    print(f"  检出头数: {len(dets)}")
    for i, (hb, bb, fs) in enumerate(dets):
        print(f"    #{i} 头框={hb} 头高={hb[3]-hb[1]} conf={fs:.2f}")
    if len(dets) < 2:
        print("  !! 不足两人")
        return None
    dets.sort(key=lambda d: -(d[0][3] - d[0][1]))
    use = dets[:2]
    heads = []
    for hb, bb, fs in use:
        hx1, hy1, hx2, hy2 = hb
        yaw, pitch, ya = head_yaw(big, hb)
        heads.append({"cx": (hx1 + hx2) / 2.0, "cy": (hy1 + hy2) / 2.0,
                      "bbox": hb, "ya": ya, "pitch": pitch})
        print(f"  P{len(heads)-1}: 中心=({heads[-1]['cx']:.0f},{heads[-1]['cy']:.0f}) "
              f"头高={hy2-hy1} yaw={yaw:.1f} → ya={ya:.1f} "
              f"({'朝右' if ya > 0 else '朝左'}) pitch={pitch:.1f}")
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
    ok_seat = szr <= V.SEAT_DEPTH_RATIO_MAX
    ok_near = dist <= V.SEAT_DIST_RATIO * ref
    ok_turn = abs(ya) >= V.MIN_TURN_YAW and abs(yb) >= V.MIN_TURN_YAW
    ok_toward = a_toward and b_toward
    ok_gaze = gaze >= V.GAZE_OBTUSE_THRESH
    ok = ok_seat and ok_near and ok_turn and ok_toward and ok_gaze
    print(f"  ①同深度 头高比={szr:.2f}≤{V.SEAT_DEPTH_RATIO_MAX} {'✓' if ok_seat else '✗'}")
    print(f"  ②邻座 距离={dist:.0f}≤{V.SEAT_DIST_RATIO*ref:.0f} {'✓' if ok_near else '✗'}")
    print(f"  ③转头 |ya|={abs(ya):.0f}/{abs(yb):.0f} ≥{V.MIN_TURN_YAW} {'✓' if ok_turn else '✗'}")
    print(f"  ④互朝 a→b={a_toward} b→a={b_toward} {'✓' if ok_toward else '✗'}")
    print(f"  ⑤夹角 {gaze:.1f}≥{V.GAZE_OBTUSE_THRESH} {'✓' if ok_gaze else '✗'}")
    print(f"  => {'✓ 判为交头接耳(误检?)' if ok else '✗ 不算'}")
    return {"file": os.path.basename(path), "gaze": gaze, "ok": ok, "toward": ok_toward,
            "szr": szr, "dist_ratio": dist / ref, "min_turn": min(abs(ya), abs(yb)),
            "n_heads": len(dets), "ok_seat": ok_seat, "ok_near": ok_near,
            "ok_turn": ok_turn, "ok_gaze": ok_gaze}


res = []
for p in IMGS:
    try:
        r = analyze(p)
        if r:
            res.append(r)
    except Exception as e:
        print(f"!!! {p}: {e}")

print(f"\n{'='*70}\n===== 汇总 =====")
for r in res:
    fails = [k for k in ("ok_seat", "ok_near", "ok_turn", "ok_gaze") if not r[k]]
    fail_cn = {"ok_seat": "深度", "ok_near": "距离", "ok_turn": "转头", "ok_gaze": "夹角"}
    print(f"  {r['file'][-16:]:18s} 夹角={r['gaze']:6.1f}° 互朝={str(r['toward']):5s} "
          f"头数={r['n_heads']} 距离/头高={r['dist_ratio']:.2f} "
          f"{'✓报警' if r['ok'] else '✗卡在:'+','.join(fail_cn[k] for k in fails)}")
