# -*- coding: utf-8 -*-
"""从主版本 video.py 构建 D:\\chatting\\video.py 精简版 v1"""
import os, re, sys, shutil

SRC = r"D:\deeplearning\ultralytics-8.3.163\pose_behavior_package_imitate\交头接耳\SemiUHPE-main\video.py"
DST = r"D:\chatting\SemiUHPE-main\video.py"
BACKUP = r"D:\chatting\SemiUHPE-main\video_official_demo.py"
report = []

# ---- 0. 备份旧文件 ----
if os.path.exists(DST) and not os.path.exists(BACKUP):
    shutil.copy2(DST, BACKUP)
    report.append(f"[备份] {DST} -> {BACKUP}")
else:
    report.append(f"[备份] 跳过 (备份已存在或源文件不存在)")

src = open(SRC, encoding="utf-8-sig").read()

def sub_once(text, old, new, tag):
    n = text.count(old)
    assert n == 1, f"[FAIL] {tag}: 期望1次匹配, 实际{n}次"
    report.append(f"[OK] {tag}")
    return text.replace(old, new, 1)

# ---- 1. 路径设置: 注入 PY_ROOT 并加入 sys.path ----
src = sub_once(src,
'''from src.config import get_config''',
'''# ============ 路径设置 ============
# 本脚本位于 D:\\chatting\\SemiUHPE-main (代码/权重同目录), PY_ROOT 显式指向该目录
PY_ROOT = r"D:\\chatting\\SemiUHPE-main"
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

from src.config import get_config''', "路径设置+sys.path")

# ---- 2. renderer 资源路径改为 PY_ROOT ----
src = sub_once(src,
'''renderer = Renderer(
    vertices_path=os.path.join(_BASE_DIR, "pose_references", "vertices_trans.npy"),
    triangles_path=os.path.join(_BASE_DIR, "pose_references", "triangles.npy")
)''',
'''renderer = Renderer(
    vertices_path=os.path.join(PY_ROOT, "pose_references", "vertices_trans.npy"),
    triangles_path=os.path.join(PY_ROOT, "pose_references", "triangles.npy")
)''', "renderer资源路径")

# ---- 3. 参数区替换 (删掉 v3 的一堆限制, 换成精简 v1) ----
START3 = "    # ============ 交头接耳检测参数 v3 (按流程图重构) ============"
END3 = "    # ============ 跨帧人脸追踪补漏参数 (解决漏帧问题) ============"
i3, j3 = src.index(START3), src.index(END3)
NEW_PARAMS = '''    # ============ 交头接耳检测参数 v1 (精简版: 限制少, 宁可漏检不误判) ============
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

'''
src = src[:i3] + NEW_PARAMS + src[j3:]
report.append("[OK] 参数区替换 (v3限制 -> v1精简)")

# ---- 4. 检测阈值调高 ----
src = sub_once(src, "weights_path = \"weights/ch_head_l_1536_e150_best_mMR.pt\"",
'''weights_path = os.path.join(PY_ROOT, "weights", "ch_head_l_1536_e150_best_mMR.pt")''', "权重绝对路径")
src = sub_once(src, "    HEAD_CONF_THRESH = 0.60  # 提高阈值过滤椅子靠背",
'''    HEAD_CONF_THRESH = 0.65  # 【v1调高】人头置信度, 防止椅子靠背被误认为人头''', "头置信度0.60->0.65")
src = sub_once(src, "    BODY_CONF_THRESH = 0.58  # 身体检测置信度阈值 (原0.62, 适当放宽避免漏检, 靠身头比等其他约束过滤椅子)",
'''    BODY_CONF_THRESH = 0.62  # 【v1调高】身体置信度''', "身体置信度0.58->0.62")

# ---- 5. 判定逻辑替换 (yaw时序信任块 + 加权评分块 -> 精简判定块) ----
START5 = "            # ============ yaw/pitch 时序一致性检查 v2 ============"
END5 = "            # --- 一对一贪心匹配 (避免一人配多人) ---"
i5, j5 = src.index(START5), src.index(END5)
NEW_LOGIC = '''            # ============ 交头接耳检测 v1 (精简版) ============
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

'''
src = src[:i5] + NEW_LOGIC + src[j5:]
report.append("[OK] 判定逻辑替换 (yaw时序+8道关卡+加权评分 -> 座位约束+钝角判定)")

# ---- 6. 输入视频目录 & 输出目录 ----
m = re.search(r"    video_files = \[\n(?:.*?\n)*?    \]\n", src)
assert m, "[FAIL] 未找到 video_files 列表"
src = src[:m.start()] + '''    _VIDEO_DIR = r"D:\\video\\1\\1\\right\\cutting_video2"
    video_files = sorted(glob.glob(os.path.join(_VIDEO_DIR, "*.mp4")))
    print(f"[输入] {_VIDEO_DIR}  共 {len(video_files)} 个视频")
''' + src[m.end():]
report.append("[OK] 输入视频目录 -> D:\\video\\1\\1\\right\\cutting_video2")

m = re.search(r"    output_folder = r\"[^\"]*\"\n", src)
assert m, "[FAIL] 未找到 output_folder"
src = src[:m.start()] + '    output_folder = r"D:\\chatting\\outputs"\n' + src[m.end():]
report.append("[OK] 输出目录 -> D:\\chatting\\outputs")

# ---- 7. chdir 到 PY_ROOT ----
sub_once(src, "os.chdir(_BASE_DIR)", "os.chdir(PY_ROOT)", "chdir -> PY_ROOT")

# ---- 8. 检查是否有残留的已删除标识符 ----
GONE = ["CHAT_SCORE_THRESH", "CHAT_SCORE_STRICT", "CHAT_MAX_SINGLE_YAW", "ROW_Y_HEAD_RATIO",
        "yaw_trusted", "_mutual_gaze", "CHAT_MIN_YAW", "CHAT_MAX_YAW", "CHAT_MAX_DIST_RATIO",
        "CHAT_MAX_DIST_BODY_W", "CHAT_MAX_YAW", "CHAT_CONFIRM_FRAMES =", "FACING_YAW_MIN",
        "BACK_YAW_THRESH", "CHAT_MAX_BODY_DIST_W", "CHAT_MAX_VERT_RATIO", "max_dist"]
leftovers = {}
for g in GONE:
    hits = [k + 1 for k, line in enumerate(src.split("\n")) if g in line]
    if hits:
        leftovers[g] = hits
report.append("[残留检查] " + ("无残留" if not leftovers else str(leftovers)))

open(DST, "w", encoding="utf-8").write(src)
report.append(f"[写出] {DST}  ({len(src.splitlines())} 行)")

open(os.path.join(os.environ.get("TEMP", "."), "build_report.txt"), "w", encoding="utf-8").write("\n".join(report))
print("\n".join(report))
