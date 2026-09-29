# 项目记忆

项目根目录：本项目根目录。项目主任务为固定二维参考轴转头识别，禁止小抄、手部、COMPUTER/HAND、作弊判断、每座位baseline、自动头向参考轴学习、训练或性能优化。后续用户新增“排除站立者”，因此仅为站立过滤使用本地YOLO26s姿态模型；头向仍用SemiUHPE。

当前运行默认：`.\.venv\Scripts\python.exe main.py` 处理完整视频直到EOF，不再默认只处理前10秒。`--seconds 10` 才限制前10秒。视频总帧数只用于进度显示，不作为完整模式提前截止条件。summary中完整模式requested_seconds为null、processing_scope=full_video，保留实际处理时长和最后一帧预览。

最新用户决定：站立算法效果不好，暂时停用。config.yaml的standing_filter.enabled=false，不初始化站立模型、不运行站立算法、不排除任何人；所有检测到的人参与转头判断，保持全员头向诊断绘制。POST为DISABLED；原站立实现及历史报告保留，历史过滤结果不代表当前运行策略。

最新快速转头修正：用户指出SemiUHPE存在反向误判，要求仅调参数试验。max_frame_direction_jump_deg已从180°改为120°，原始或平滑方向相对上一有效方向变化超过120°时拒绝，等于120°不触发跳变拒绝。正常阈值40°、转头阈值50°、5有效样本平滑、0.4秒持续确认及质量/间隔检查保持不变，实际效果待视频核对。

最新有方向参考需求优先于早期固定轴约束：用户要求转到相反方向仍持续判断为转头。建立NORMAL时锁定实际平滑头向normal_reference_angle，整次过程固定，不跟随逐帧漂移；之后用0～180°圆周差判定偏离和回正。只有回到原方向附近并稳定，才更新下一周期参考；无效帧保留方向，站立暂停/轨迹过期重置。CSV新增normal_reference_angle、smoothed_reference_diff_deg；事件新增max_reference_diff，原无向轴差字段保留诊断。绘图A为有向参考偏离，R为参考头向，蓝色单向箭头表示参考。当前用户数值：投影0.08、正常/转头40°/50°、正常/转头稳定时间0.1/0.1秒、跳变150°。

## 结构

最新输出需求：每次首次确认TURNED后，额外向turn_report.html写入人员ID、视频内确认/起点时间、原分辨率完整画面（只突出事件人员红框）。单HTML内嵌JPEG，可筛选ID、放大/保存，逐事件flush，中断时保留报告。主入口自动生成；export_turn_report.py可给历史输出补报告，不重新推理或改变旧事件。src/event_report.py负责报告。

- `main.py`：命令行入口。
- `config.yaml`：模型路径、人工参考轴、平滑、质量过滤、持续时间、跟踪和绘图参数。
- `src/detector.py`：从旧项目 `body_head_detector.py` 复用 BPJDet 推理及人体/头部关联。
- `src/head_pose.py`：从旧项目 `head_pose_detector.py` 复用 SemiUHPE、原图正方形裁头和批量头姿估计。
- `src/upstream.py`：从旧项目复用指定上游函数加载和 CPU SVD 适配。
- `src/tracker.py`：从旧项目 `person_tracker.py` 复用 IOU 临时编号。
- `src/head_direction.py`：独立二维投影、轴夹角、圆周差、向量平均。
- `src/turn_detector.py`：独立逐人状态机。
- `src/visualizer.py`：最新诊断模式绘制全部未确认站立的人（包括NORMAL/UNKNOWN），原始头向白细箭头、有效平滑头向彩粗箭头、参考轴蓝线；H原始角/S平滑角/A轴差；第二行POST姿态/S0-S2站立状态/ST平滑站立分数/C覆盖率。灰框显示被排除的站立者TURN SKIPPED，便于核对举手是否误判站立。display.draw_all_non_standing/show_standing_exclusions均为true；分别设false可恢复仅转头嫌疑绘制/隐藏站立排除框。推理与全量CSV不受绘制过滤影响。
- `src/pipeline.py`：逐帧检测、CSV、事件、视频及运行快照。
- `src/standing.py`：独立站立特征、均值EMA、三状态Viterbi、检测框一对一匹配与过滤。
- `src/standing_perspective.py`：同视频、同分辨率的站立透视补偿曲线拟合及查询。
- `standing_perspective.json`：当前测试视频本地补偿曲线。
- `calibrate_standing.py`：从本项目CSV生成站立曲线。
- `tests/test_head_direction.py`、`tests/test_turn_detector.py`：几何与状态机测试。
- `outputs/`：每次新目录，防止覆盖。
- `BPJDet/`、`SemiUHPE/`：完整模型源码已复制到本项目。
- `weights/`：两份模型权重已复制到本项目。
- `videos/`：指定测试视频已复制到本项目。
- 已存在 `turnhead/` 子目录里的空 `main.py`、IDE 和虚拟环境不属于本次实现，未改动。

## 模型和运行环境

模型最初来自旧项目，现在只使用本项目 `BPJDet/`、`SemiUHPE/` 和两份权重：

- `weights/ch_head_s_1536_e150_best_mMR.pt`
- `weights/DAD-WildHead-EffNetV2-S-best.pth`

解释器：`.venv/Scripts/python.exe`；PyTorch 2.7.0+cu118、torchvision 0.22.0+cu118，CUDA 可用，NVIDIA GeForce GTX 1060。本地原CPU包备份至 `dependency-backups/`。不需要下载或训练模型。SemiUHPE 保持旧工程 student 权重严格加载，EfficientNetV2-S 不下载 ImageNet 权重。配置及运行快照使用相对路径，不依赖旧项目。

## 几何约定

二维方向参考旧 `head_hand_geometry.py:head_pose_to_image_direction` 的纯矩阵投影部分，但新项目不导入该模块。`R @ [0,0,1]`，直接 x/y，不转置、不改符号、不除以 z。单位向量 `atan2(dy,dx)` 角度定义右0、下90、左180、上270。Euler 诊断角度继续保持旧模型约定。

轴差 `abs(((head_angle-axis_angle+90)%180)-90)`，范围0～90°，30/210等价。`reference.axis_degrees=30`。`active_reference_ray` 只在稳定 NORMAL 时更新；事件期间保持。

## 当前状态机

UNKNOWN → 稳定正常 0.4s → NORMAL → 离开10°且趋势增大 → TURNING → 连续有效路径达到转头阈值且稳定0.4s → TURNED → 趋势减小 → RETURNING → ≤10°稳定0.4s → NORMAL。用户已将转头阈值与最大跳变改成40°/40°，本次迁移保留；首轮历史数据使用35°/30°。

5帧单位向量平滑；原始和平滑相邻方向变化任一大于配置跳变阈值记POSE_JUMP。投影<0.12、abs(pitch)>=90、无姿态、平滑向量抵消不更新状态。无效帧冻结显示状态并清除路径及时间证据；未确认事件者恢复后重获NORMAL。有效间隔>0.2s断开连续性。已确认TURNED保持事件锁，直到回正才重启。

趋势窗口5帧，总体变化至少2°；至少3个连续路径样本。每个TURNED周期只触发一次，禁止初始大夹角直接确认转头。方向用累计圆周有符号变化，正=图像顺时针。所有参数配置化。

## 验证状态

13项单元测试已通过。真实视频为 `videos/教室前中_左_802监控_20260830130816_20260830133147_1530775.mp4`，3840×2160，20FPS，测试前10秒共200帧。完成时间2026-09-26。

输出 `outputs/first10s_axis30/`：annotated.mp4、frames.csv、turn_events.csv、summary.json、config.snapshot.yaml、audit.json、4张preview和2张事件对照图。4978条人物记录、34个临时ID、2个事件均来自P10，分别2.95→3.60s及5.80→6.50s；路径14/15个连续有效样本，≥35°均持续0.4s，无跳变/缺失。首轮回正后重新NORMAL，第二轮到片尾未稳定回正。投影角变化不能等同三维yaw转动幅度。

有效3591、异常pitch1061、无姿态272、跳变35、低投影19；UNKNOWN状态3620条，NORMAL965，TURNING299，TURNED32，RETURNING62。原始和平滑轴差均0～90，平滑范围0.0064～89.9492。有效NORMAL射线30°848条、210°仅7条，真实两排尚未充分验证。P10第二次确认投影长度约0.158，接近下限，需人工复核。事件对照图可见朝向变化，但无人工真值准确率结论。

`report.md` 包含用户要求的13项完成报告。`audit_run.py` 是本次20FPS/200帧的专用核验脚本，重新绘制可读标注并检查所有输出视频帧和事件连续证据，不重跑模型。输出MP4已完整解码验证200帧、1920×1080、20FPS。

主要限制：全局30°尚未经本机位人工核对；小头/遮挡/背面头姿可不稳定；IOU跟踪遮挡或交叉可能换ID；严格连续性会降低召回；不能把缺少事件当成无转头，更不能当成作弊判断。

## 本地化迁移验证

BPJDet、SemiUHPE、两份权重及完整指定视频已复制到项目内；权重和视频SHA256均与源文件一致。运行不再引用旧工程；所有应用配置/快照资产路径为相对路径。历史快照资产路径也迁移为本地副本，其原始阈值及CSV数据未改变。

本项目 `.venv` 的CPU版torch/torchvision相关包已先备份再迁入同Python3.12环境验证过的CUDA版；CUDA可用，pip check通过，requirements固定torch2.7.0+cu118/torchvision0.22.0+cu118。14项测试通过（新增相对资产路径与快照重载测试）。

用 `.\.venv\Scripts\python.exe main.py --seconds 10 --output outputs/local_models_axis30_turn40` 完整运行200帧通过，保留用户40°/40°设置。4978条记录、2事件：P10-E1确认3.65s，P10-E2确认7.10s。新结果与初轮35°/30°结果分开保存。默认不指定output时自动创建时间目录。

## 站立过滤（后续新增需求）

用户要求参考站立发布包原理/代码，禁止调用其接口。本项目所有站立代码独立实现，不导入发布包，也不启动外部进程。已复制其yolo26s-pose.pt至本地weights，ultralytics8.4.137及必要依赖已复制安装到本项目.venv，pip check通过。站立推理每帧进行，IOU一对一匹配到现有BPJDet人物ID；不新增第二套人物ID。

几何六项：躯干竖直、髋踝跨度/躯干、膝角、头踝伸展/躯干、框高宽比、髋踝跨度/框高。可见特征加权打分，缺腿证据默认UNKNOWN。15帧均值与EMA各半（alpha0.35），38帧因果三状态Viterbi，进入0.60/退出0.45，S2且平滑分数≥0.80才排除。无证据宽限0.6秒，间隔>0.2秒重置站立证据。YOLO缓存位于项目.cache，不自动安装或下载权重。

站立者不进SemiUHPE批次、不计算头向、不推进转头状态机。TurnDetector.suspend清空方向/正常起点/未完成事件和当前事件锁，保留既往事件编号；坐下后重获稳定NORMAL。CSV保存站立分数/特征/覆盖/状态/匹配框/透视系数，排除时transition_status=EXCLUDED_STANDING、turn_excluded_reason=STANDING，头姿方向空，转头state=UNKNOWN。最新诊断绘制再次显示站立者灰色排除框，帮助检查举手导致误站立的可能性，不对站立者恢复转头推理。

原图关键点复核发现此前两次事件的P10实际为站立者，所以新增过滤后应排除其转头。初版无透视补偿前10秒433条记录被排除，零事件；近处P1因投影压缩分数偏低存在漏过滤。随后用同视频665个完整膝角/躯干强证据样本拟合33个断点曲线，近区只放大三项压缩特征（k≥1，上限2），上方不变。曲线限同视频文件名与分辨率，否则回退不补偿。转头轴保持人工30°，无自动学习。

31项测试通过，覆盖站立/坐姿/遮挡、跳变、短时证据保持、退出、时序中断、人物匹配、跳过头姿、坐下重启、事件唯一、过滤开关、透视补偿和相对配置路径。最终真实测试结果见outputs/standing_filter_first10s_final。

最终200帧4978条记录中688条站立排除，8个临时ID曾排除，SemiUHPE调用4058次，转头事件0；688条排除记录全部核验头姿/方向/事件为空，转头UNKNOWN，视频完整解码200帧。详情见standing_filter_report.md。P1截断/低证据仍有漏过滤（77条P1被排除），不能宣称站立者全程完全识别；未移植发布包近距重检/补检/轨迹继承，缺失姿态超过预算会恢复UNKNOWN。曲线是同片段拟合复测，功能验证非准确率评估。

