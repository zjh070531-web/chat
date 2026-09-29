# 带方向参考的转头识别

在本项目根目录运行。BPJDet 检测人体/头部，再用 SemiUHPE 判断所有人的转头。当前已暂停站立过滤，不加载或调用站立姿态模型，站立者也参与转头判断。初始正常范围使用30°全局参考轴；建立正常状态后，锁定该人的实际平滑头向作为本次有方向的参考。

```powershell
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

默认视频为用户指定的教室前中左视频；可用 `--video` 覆盖。直接运行默认处理完整视频，读到视频末尾才停止。需要短片测试时使用 `--seconds 10`，仅处理前10秒。`--device cpu` 可切换设备。每次使用新输出目录，已有结果不会被覆盖。

模型源码已复制到项目内 `BPJDet/`、`SemiUHPE/`；两份权重在 `weights/`，测试视频在 `videos/`，全部由 `config.yaml` 的相对路径加载。项目自身 `.venv` 已迁入经过验证的 CUDA 版 PyTorch 2.7.0+cu118 和 torchvision 0.22.0+cu118；原 CPU 包备份在 `dependency-backups/`。无需访问旧工程。

路径相对于配置文件解析，运行快照中的路径相对于快照文件解析，移动整个项目不需要修改模型路径。新电脑仍需安装 Python、兼容 NVIDIA 驱动并重建虚拟环境；Windows 虚拟环境本身不作为跨电脑可直接搬运的安装包。

新环境安装依赖时使用 `requirements.txt`，其中已固定本次验证的 CUDA 版本，避免重新安装成 CPU 版。

二维方向直接使用 `R @ [0,0,1]` 的 x/y 分量，单位向量角度为 `atan2(dy,dx)`：右 0°、下 90°、左 180°、上 270°。不转置 R，不翻转符号。旧模型诊断 yaw/pitch/roll 的 Euler 提取保持原样，它的矩阵转置不用于二维投影。

初次寻找正常状态时，轴夹角为 `abs(((head_angle - axis_angle + 90) % 180) - 90)`，范围0～90°，允许两侧相向就坐的人分别建立正常起点。进入NORMAL时，记录实际平滑朝向 `normal_reference_angle`；随后使用 `abs(((head_angle - normal_reference_angle + 180) % 360) - 180)`，范围0～180°。参考方向在整次转头中固定，继续转到其反面不会变回正常。只有返回该方向的正常范围并持续稳定后，才能开启下一周期并记录新的正常方向。无效帧和短时缺失不会把另一侧重新当成正常；站立暂停或人物轨迹过期才重置整个观察阶段。

每人独立维护 5 个有效方向的单位向量窗口，平均后重新归一化。不直接平均角度。二维投影长度不足配置门槛（当前0.08）、绝对 pitch 达到 90°、无头姿或平均向量互相抵消时，都不推进状态。CSV 保留可用的原始数据。

同时检查当前原始方向和候选平滑方向相对上一有效方向的圆周变化。任何一个超过 `max_frame_direction_jump_deg`，标记 `POSE_JUMP`，不纳入平滑、不推进状态。无效样本清除持续时间和转头路径证据；超过 0.2 秒有效样本间隔也断开路径。无效帧保持显示状态；恢复后未确认事件的人重新从 UNKNOWN 获取正常状态。这样不能用缺失或跳变两侧的数据拼接事件。

- UNKNOWN：首次用全局轴差寻找正常起点；已有锁定方向时，必须返回锁定方向附近。满足 `normal_threshold_deg` 和 `min_normal_stable_seconds` 后进入NORMAL。
- NORMAL：参考方向固定，不随逐帧头向漂移；偏离超过正常范围且趋势增大时进入TURNING。
- TURNING：相对锁定方向的偏离达到 `turn_threshold_deg`，持续 `min_turned_stable_seconds` 且路径样本足够后进入TURNED。
- TURNED：保持本次参考方向；偏离减小时进入RETURNING，继续转到相反方向不会因无向轴差折返而进入RETURNING或NORMAL。
- RETURNING：只有回到本次正常方向附近并稳定，才恢复NORMAL；再次转出仍属于同一事件，不重复编号。

当前配置：正常范围40°、转头阈值50°、正常/转头持续时间均0.1秒、最大方向跳变150°、5个有效样本平滑。参数以config.yaml为准。单帧方向质量和跳变检查仍然生效，历史报告采用的旧参数与无向判定不能直接代表当前结果。

站立过滤由本项目 `src/standing.py` 独立实现，参考站立发布包的六项几何打分、均值+EMA、S0/S1/S2因果窗口时序和精度门槛，不导入或调用该发布包。`weights/yolo26s-pose.pt` 已复制到本地，仅用于提取站立特征，不参与头向判定。站立结果一对一匹配到现有 BPJDet `person_id`，各人独立维护时序。

当前 `standing_filter.enabled: false`，所有人均参与转头判断并绘制头向；POST显示DISABLED，站立分数为空。保留站立代码和参数供后续调整，设为true才开启以下过滤逻辑。`min_standing_score: 0.80` 是站立过滤分数门槛，`enter_score: 0.60`、`release_score: 0.45` 控制时序状态。分数是几何规则评分，不是准确率概率。看不到腿时不凭上身猜站立；已确认站立者短时无证据保持最多0.6秒。

确认站立后跳过 SemiUHPE 推理和转头判定；当前诊断绘制用灰框显示该人并标记 `STANDING / TURN SKIPPED`，便于核对误过滤。CSV 中 `turn_excluded_reason=STANDING`、`transition_status=EXCLUDED_STANDING`，头姿/方向留空。立刻清除方向窗口、正常基准、未完成事件及当前事件锁；已保存的事件记录不删除，编号不重复。重新坐下后从 UNKNOWN 获取稳定 NORMAL，再允许新转头事件。

近处俯视压缩使用本地 `standing_perspective.json` 补偿，仅放大腿部跨度、全身伸展和框高宽比，系数≥1；上方区域不补偿。该曲线来自本测试视频前10秒的665个满足完整覆盖、膝角≥160°、躯干≤30°的模型样本，尚未经过独立人工标注评估。曲线限定视频文件名和分辨率；换视频/分辨率自动回退为无补偿。它不改变或学习转头参考轴。

如果需为另一视频生成曲线，先把 `standing_filter.perspective_calibration` 设为 `null`，运行该视频得到CSV，再执行：

```powershell
.\.venv\Scripts\python.exe calibrate_standing.py outputs/your_run/frames.csv --output standing_perspective_new.json
```

然后把配置中的 `perspective_calibration` 改为该相对文件名，重新运行。不同文件名即使同一机位也需要显式生成曲线，避免错误复用另一机位。

每个 TURNED 周期仅确认一次事件；回到 NORMAL 后重新允许下一事件。NORMAL建立时锁定实际平滑头向；active_reference_ray仅保留最接近的全局射线作为诊断。用累计有符号角度变化记录 CLOCKWISE/COUNTERCLOCKWISE。图像 y 向下，正角变化代表顺时针。事件CSV的 `max_axis_diff` 保留原无向轴差最大值；`max_reference_diff` 是相对本次正常方向的最大偏离，`normal_reference_angle` 记录正常方向，`state_sequence` 记录确认路径。

输出包括：

- `annotated.mp4`：当前诊断模式标注全部未确认站立的人，包括NORMAL/UNKNOWN；白色细箭头为当前原始头向，彩色粗箭头为有效平滑头向，未建立正常方向时显示蓝色双向轴，建立后显示蓝色单向参考箭头。H为原始角度、S为平滑角度、R为锁定的正常朝向、A为相对该方向的偏离（0～180°；R未建立时用轴差）。第二行POST为单帧姿态类别，S0/S1/S2为时序站立状态，ST为平滑站立分数，C为特征覆盖。被站立过滤的人用灰框显示TURN SKIPPED，无头向箭头。
- `frames.csv`：逐人逐帧诊断，包括矩阵、yaw/pitch/roll、原始方向、平滑方向、跳变与质量标记。
- `turn_events.csv`：事件记录；无事件仍有表头。
- `turn_report.html`：可直接用浏览器打开的单文件结果报告，包含每次确认转头的人员ID、视频内确认时间、转头起点和原分辨率完整截图。截图中仅红框突出该条事件对应的人；可按ID筛选、点击放大或保存图片。图片嵌入HTML，移动报告不依赖外部图片目录。同一次持续转头只记录一次，回正后再次确认会新增记录；无事件有空结果提示。每次确认后立即保存，中途Ctrl+C已保存的报告仍可查看。
- `summary.json`：处理帧数、状态计数和运行统计。
- `config.snapshot.yaml`：实际运行配置。
- `preview_*.jpg`：抽帧检查图。

绘制过滤只改变视频标注；检测、跟踪、状态判断以及全量CSV记录照常执行。旧输出视频不会自动重绘，需要重新运行生成新视频。

已有输出可运行 `.\.venv\Scripts\python.exe export_turn_report.py outputs/运行目录名` 补生成截图报告，依据旧CSV中的确认事件，不重新判断或调用模型，需要保留原视频与配置快照。已有报告不会被覆盖。

`display.draw_all_non_standing: true` 显示全部未站立者；设为false恢复只显示转头嫌疑状态。`display.show_standing_exclusions: true` 显示灰色排除框；设为false隐藏站立者。站立UNKNOWN不等于确认坐姿，保留绘制便于核对。异常/跳变头姿可能显示白色原始箭头，但不会因此推进转头判定；LOW_PROJECTION不画不可靠箭头。

IOU tracker 是临时人物编号，不是身份识别。人物交叉、遮挡可能造成 ID 切换。30° 是人工给定实验轴，当前不能保证该机位所有座位的自然头向都符合它；本项目不学习全局参考轴或每座位基线；只为每人的本次观察周期记录正常朝向。短片输出用于诊断，不能仅凭事件数量判断识别准确率。
