"""
Interactive head-pose visualization tool for SemiUHPE.

Same model arguments as image.py:
    python pose_tool.py DAD-COCOHead-ResNet50-best --is_full_range \
        --config settings/DAD3DHeads_COCOHead.yml --network resnet50 --gpu_ids 1

Optional arguments:
    --img <file-or-folder>   input image file / folder (default: test_DetHPE/CrowdHuman)
    --headless               process images and dump angle JSON files without opening the GUI
    --demo                   draw a synthetic reference line on the first image, save a preview, then exit

Mouse / keyboard inside the window:
    left drag   draw the reference (direct-facing) line
    r           clear the reference line
    + / -       increase / decrease the turning threshold (5 deg steps)
    f           flip the reference-line sign convention
    s           save the annotated image and angle JSON
    n / p       next / previous image
    h           toggle the help text
    q / ESC     quit
"""

import json
import math
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as tfs
from PIL import Image
from scipy.spatial.transform import Rotation

from src.config import get_config
from src.agent import get_agent
from src.fisher.fisher_utils import batch_torch_A_to_R
from src.utils import limit_angle
from src.vis_plot import draw_axis_ypr
from utils.general import check_img_size, non_max_suppression
from utils.datasets import LoadImages
from models.experimental import attempt_load

from image import post_process_batch, colors_list


EDGES_SCALE = -0.05


def compute_head_pose(img_head_cv2, agent, config):
    """Estimate yaw / pitch / roll (degrees) of a cropped head image."""
    img_head_pil = Image.fromarray(img_head_cv2[:, :, ::-1]).resize([224, 224])
    img_tensor = tfs.ToTensor()(img_head_pil)
    img_tensor = tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_tensor)
    with torch.no_grad():
        agent.net.eval()
        fisher_out = agent.net(img_tensor.reshape((1, 3, 224, 224)).to(config.device))
        pd_m = batch_torch_A_to_R(fisher_out)
    rot_mat = pd_m.detach().cpu().numpy()[0]
    rot_mat_2 = np.transpose(rot_mat)
    angle = Rotation.from_matrix(rot_mat_2).as_euler("xyz", degrees=True)
    roll, pitch, yaw = list(map(limit_angle, [angle[2], angle[0] - 180, angle[1]]))
    return float(yaw), float(pitch), float(roll)


def detect_and_pose(image_path, agent, config, detector, imgsz, stride):
    """Run head detection + pose estimation on every image, return {path: page}."""
    dataset = LoadImages(image_path, img_size=imgsz, stride=stride, auto=True)
    dataset_iter = iter(dataset)
    pages = {}
    for index in range(len(dataset)):
        single_path, img, im0, _ = next(dataset_iter)
        if '_res' in single_path or '_vis' in single_path:
            continue

        img_t = torch.from_numpy(img).to(config.device) / 255.0
        if len(img_t.shape) == 3:
            img_t = img_t[None]

        with torch.no_grad():
            out_ori = detector(img_t, augment=True, scales=[1])[0]
        body_dets = non_max_suppression(out_ori, 0.5, 0.75, classes=[0], num_offsets=2)
        part_dets = non_max_suppression(out_ori, 0.5, 0.75, classes=[1], num_offsets=2)
        bboxes, points, scores = post_process_batch(
            img_t, [], [[im0.shape[:2]]], body_dets, part_dets, 2, 0.6)

        img_h, img_w, _ = im0.shape
        heads = []
        for bbox, point, score in zip(bboxes, points, scores):
            f_score, f_bbox = point[0][2], point[0][3:]
            if f_score == 0:
                continue
            px1, py1, px2, py2 = f_bbox
            pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
            head_size = max(px2 - px1, py2 - py1)
            nx1 = max(0, int(pcx - (0.5 - EDGES_SCALE) * head_size))
            nx2 = min(img_w - 1, int(pcx + (0.5 - EDGES_SCALE) * head_size))
            ny1 = max(0, int(pcy - (0.5 - EDGES_SCALE) * head_size))
            ny2 = min(img_h - 1, int(pcy + (0.5 - EDGES_SCALE) * head_size))

            yaw, pitch, roll = compute_head_pose(im0[ny1:ny2, nx1:nx2], agent, config)
            cx, cy = int((nx1 + nx2) / 2), int((ny1 + ny2) / 2)
            heads.append({
                "bbox": [nx1, ny1, nx2, ny2],
                "center": (cx, cy),
                "line_len": max(int(head_size * 0.9), 24),
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
            })

        pages[single_path] = {"image": im0, "heads": heads}
        print("[{}] {}: {} heads".format(index, single_path, len(heads)))
    return pages


def reference_yaw(ref_line, flip=False):
    """Map the drawn 2D reference line to a reference yaw (degrees).

    A horizontal line maps to 0 (facing the camera). The line is treated as
    undirected, so the result always lies in [-90, 90].
    """
    if ref_line is None or len(ref_line) < 2:
        return 0.0
    (x1, y1), (x2, y2) = ref_line
    dx, dy = x2 - x1, y2 - y1
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    ang = math.degrees(math.atan2(dy, dx))
    ang = limit_angle(ang)
    if ang > 90:
        ang -= 180
    elif ang < -90:
        ang += 180
    if flip:
        ang = -ang
    return ang


def judge_head(head, yaw_ref, threshold):
    yaw_diff = limit_angle(head["yaw"] - yaw_ref)
    return yaw_diff, abs(yaw_diff) > threshold


def draw_overlay(base, heads, ref_line, yaw_ref, threshold, show_help=True):
    img = base.copy()
    line_thick = max(base.shape[:2]) // 1000 + 2

    for i, hd in enumerate(heads):
        color = colors_list[i % len(colors_list)]
        nx1, ny1, nx2, ny2 = hd["bbox"]
        cv2.rectangle(img, (nx1, ny1), (nx2, ny2), color, line_thick)

        yaw_diff, turned = judge_head(hd, yaw_ref, threshold)
        cx, cy = hd["center"]
        line_len = hd["line_len"]
        ex = cx + line_len * math.sin(math.radians(hd["yaw"]))
        ey = cy - line_len * math.cos(math.radians(hd["yaw"])) * math.sin(math.radians(hd["pitch"]))
        cv2.arrowedLine(img, (cx, cy), (int(ex), int(ey)), (255, 0, 0), 2, cv2.LINE_AA, tipLength=0.25)

        label = 'yaw {:5.1f} pitch {:5.1f} roll {:5.1f}'.format(hd["yaw"], hd["pitch"], hd["roll"])
        status = 'TURNED' if turned else 'OK'
        status_color = (0, 0, 255) if turned else (0, 255, 0)
        cv2.putText(img, label, (cx - 70, cy - line_len - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, status, (cx - 25, cy - line_len - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)

    if ref_line is not None and len(ref_line) == 2:
        (x1, y1), (x2, y2) = ref_line
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, 'REF', (int((x1 + x2) / 2) - 20, int((y1 + y2) / 2) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if show_help:
        info = ('thres={:.0f}  yaw_ref={:+.1f}  heads={} | drag=ref line  r=clear  '
                '+/-=thres  f=flip  s=save  n/p=img  q=quit').format(
                    threshold, yaw_ref, len(heads))
        cv2.putText(img, info, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, info, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def heads_to_json(path, heads, yaw_ref, threshold):
    return {
        "image": path,
        "ref_yaw": round(float(yaw_ref), 2),
        "turn_thres": float(threshold),
        "heads": [
            {
                "bbox": [int(v) for v in hd["bbox"]],
                "yaw": round(hd["yaw"], 2),
                "pitch": round(hd["pitch"], 2),
                "roll": round(hd["roll"], 2),
                "yaw_diff_to_ref": round(float(judge_head(hd, yaw_ref, threshold)[0]), 2),
                "turned": bool(judge_head(hd, yaw_ref, threshold)[1]),
            }
            for hd in heads
        ],
    }


def save_output(path, page, ref_line, yaw_ref, threshold, show_help=False):
    annotated = draw_overlay(page["image"], page["heads"], ref_line, yaw_ref, threshold, show_help=show_help)
    out_img = path[:-4] + "_tool_res.jpg"
    out_json = path[:-4] + "_tool_angles.json"
    cv2.imwrite(out_img, annotated)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(heads_to_json(path, page["heads"], yaw_ref, threshold), f, indent=2, ensure_ascii=False)
    print("Saved:", out_img)
    print("Saved:", out_json)
    return out_img


def main():
    config = get_config('test')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config.device = device
    config.pretrain_backbone = False  # released checkpoints already contain full weights
    agent = get_agent(config)
    agent.load_ckpt(config.ckpt)

    image_path = config.img if config.img else "test_DetHPE/CrowdHuman"
    weights_path = "weights/ch_head_l_1536_e150_best_mMR.pt"
    print('Using device: {}'.format(device))
    detector = attempt_load(weights_path, map_location=device)
    stride = int(detector.stride.max())
    imgsz = check_img_size(1536, s=stride)

    print('Processing images under:', image_path)
    pages = detect_and_pose(image_path, agent, config, detector, imgsz, stride)
    if not pages:
        print("No images to process.")
        return
    paths = list(pages.keys())

    threshold = float(config.turn_thres)

    if config.headless:
        for path in paths:
            save_output(path, pages[path], None, 0.0, threshold)
        return

    if config.demo:
        path = paths[0]
        page = pages[path]
        h, w = page["image"].shape[:2]
        ref_line = [(int(w * 0.30), int(h * 0.62)), (int(w * 0.70), int(h * 0.50))]
        yaw_ref = reference_yaw(ref_line)
        print("Demo reference line: {} -> {}, yaw_ref = {:.1f} deg".format(
            ref_line[0], ref_line[1], yaw_ref))
        out_img = save_output(path, page, ref_line, yaw_ref, threshold, show_help=True)
        print("Demo preview saved. Open", out_img, "to inspect.")
        return

    # ---- interactive GUI ----
    idx = 0
    ref_line = None
    flip = False
    show_help = True
    drag_start = None
    drag_cur = None
    scale = 1.0
    quit_flag = False

    def disp_to_img(pt):
        return (pt[0] / scale, pt[1] / scale)

    def on_mouse(event, x, y, flags, param):
        nonlocal drag_start, drag_cur, ref_line
        if event == cv2.EVENT_LBUTTONDOWN:
            drag_start = (x, y)
            drag_cur = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drag_start is not None:
            drag_cur = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drag_start is not None:
            p1 = disp_to_img(drag_start)
            p2 = disp_to_img((x, y))
            drag_start = None
            drag_cur = None
            if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) >= 8:
                ref_line = [p1, p2]

    cv2.namedWindow("Pose Reference Tool", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Pose Reference Tool", on_mouse)

    while not quit_flag:
        path = paths[idx]
        page = pages[path]
        yaw_ref = reference_yaw(ref_line, flip)
        overlay = draw_overlay(page["image"], page["heads"], ref_line, yaw_ref, threshold, show_help=show_help)

        if drag_start is not None and drag_cur is not None:
            p1 = disp_to_img(drag_start)
            p2 = disp_to_img(drag_cur)
            cv2.line(overlay, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                     (0, 255, 255), 2, cv2.LINE_AA)

        h, w = overlay.shape[:2]
        scale = min(1.0, 1000.0 / h, 1500.0 / w)
        if scale < 1.0:
            disp = cv2.resize(overlay, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            disp = overlay
        cv2.imshow("Pose Reference Tool", disp)

        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord('q')):
            quit_flag = True
        elif key == ord('r'):
            ref_line = None
        elif key in (ord('+'), ord('=')):
            threshold = min(180.0, threshold + 5)
        elif key in (ord('-'), ord('_')):
            threshold = max(5.0, threshold - 5)
        elif key == ord('f'):
            flip = not flip
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('n'):
            idx = (idx + 1) % len(paths)
            ref_line = None
        elif key == ord('p'):
            idx = (idx - 1) % len(paths)
            ref_line = None
        elif key == ord('s'):
            save_output(path, page, ref_line, yaw_ref, threshold, show_help=False)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
