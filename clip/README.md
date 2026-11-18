# 考场行为视频样本裁剪工具

用于人工观看考场 MP4 视频，采集“传递物品”或“交头接耳”的正样本和负样本。默认选择传递物品，可以在窗口顶部切换行为。两种行为各自保存视频、编号、CSV 和数量统计。

## Windows 安装与启动

1. 安装 Python 3.10 或更高版本（建议使用 python.org 的 Windows 版，包含 Tkinter）。
2. 从 [FFmpeg 官网](https://ffmpeg.org/download.html#build-windows)提供的 Windows 构建链接下载安装 FFmpeg，将包含 `ffmpeg.exe` 的 `bin` 目录加入 PATH。重新打开 VS Code / PyCharm 后运行 `ffmpeg -version` 验证。也可以把 `ffmpeg.exe` 放在 `main.py` 旁边，或在首次保存时通过文件选择窗口指定。
3. 在本目录打开 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

VS Code / PyCharm 选择 `.venv\Scripts\python.exe` 为解释器，直接运行 `main.py` 即可。不需要激活虚拟环境。Tkinter 属于 Python 安装组件，不通过 pip 安装。Pillow 仅用于在 Tkinter 中显示 OpenCV 视频帧。

## 使用

选择采集行为 → 打开 MP4 → 播放观看 → 暂停在行为核心时刻 → 鼠标框住目标区域 → 选择标签 → 保存样本。

采集**传递物品**时，正样本表示有传递物品，负样本表示没有传递物品。暂停在物品交接的核心时刻，选框应包含两人的手、物品和交接范围，并给动作留出余量。负样本可以选取同类场景中没有传递的片段。

切换采集行为后，正负样本标签和数量统计会更新，需要重新选择标签；当前视频时间和选框会保留。导出过程中暂时不能切换行为。

在视频画面上**按住鼠标左键拖动**，松开后出现绿色选框，保存时只导出框内区域。任意方向拖动都可以；开始框选会自动暂停播放。画面下方显示实际输出的宽、高和左上角坐标，均以原视频像素为单位。

重新拖动可以更换选框。“清除选框”恢复保存整幅画面；没有框选时也保存整幅画面。选框在同一个视频中持续保留，跳转进度、播放及调整窗口大小不会改变对应的原视频范围，打开新视频时自动清除。

框选区域在整个约 4 秒片段内保持固定，不自动跟随人物移动，框选时请给两人的动作留一点余量。选框会自动限制在视频范围内，并将坐标和尺寸对齐到偶数像素（边缘最多调整约 2 像素），以兼容 H.264 编码。

为兼容监控 HEVC/H.265 录像，导出时会先读取裁剪起点前最多 10 秒的画面，恢复参考帧后再截取目标区间。预读不会加入输出视频，片段时长仍是前后各 2 秒；保存可能多耗费几秒。

| 快捷键 | 操作 |
| --- | --- |
| Space | 播放 / 暂停 |
| A / D | 后退 / 前进 1 秒 |
| 1 | 正样本：有当前选定行为 |
| 2 | 负样本：没有当前选定行为 |
| S | 保存当前样本 |

拖动进度条松开后跳到对应位置；原来正在播放时会继续播放。界面提供 1、2、4 倍速切换。预览无音频，自动按窗口大小等比缩放。

保存时自动暂停并固定当前时刻，在后台通过 FFmpeg 裁剪。连续按 S 不会同时导出多个任务；保存完成后允许再次保存同一时刻。选择标签后会保留该选择，打开新视频或切换行为时重新选择。

## 输出

输出路径固定相对于 `main.py`，不受 IDE 工作目录影响：

```text
talking_clip_tool/
├── main.py
├── requirements.txt
├── README.md
└── dataset/
    ├── passing/
    │   ├── positive/
    │   │   └── passing_positive_0001.mp4
    │   ├── negative/
    │   │   └── passing_negative_0001.mp4
    │   └── labels.csv
    └── talking/
        ├── positive/
        │   └── talking_positive_0001.mp4
        ├── negative/
        │   └── talking_negative_0001.mp4
        └── labels.csv
```

每种行为的正负样本分别从 0001 编号，重新启动后接着已有最大编号保存，不覆盖旧文件。界面统计当前行为目录中已存在的非空样本文件；CSV 中已有编号也不会复用。请一次只运行一个工具实例，不要同时手工修改输出目录和 CSV。已有的交头接耳样本仍保存在 `dataset/talking/`。

裁剪范围为 `[max(0, 当前时间 - 2), min(视频时长, 当前时间 + 2)]`。例如 100 秒裁 98～102 秒，第 1 秒裁 0～3 秒；不足 4 秒的视频保留可用范围，不补帧。边界附近的样本可以短于 4 秒。

输出为 H.264 MP4、无音频。框选后使用 FFmpeg 的 `crop` 滤镜裁出所选区域，输出分辨率就是选框尺寸，不放大、不缩小；未框选时保留原视频分辨率。预览缩放不会影响实际导出，预览与导出均使用原始像素方向，不根据旋转标记自动旋转。使用 `libx264` 重新编码以精确定位，CRF 为 18。FFmpeg 需要包含该编码器。时间精度受原视频帧间隔限制，4 秒片段可能有一帧左右的时长差异。

CSV 使用 UTF-8 BOM 编码，便于 Excel 打开。字段如下，时间以**秒**记录、保留六位小数；界面显示为 `HH:MM:SS.mmm`：

也可读取 Excel 保存的 GBK/GB18030 或带 BOM 的 UTF-16 CSV。保存时统一转换为 UTF-8 BOM，并在首次转换旧编码前生成 `labels_before_utf8_*.csv.bak` 原文件备份，保留已有记录和编号。

```csv
clip_name,source_video,label,event_time,start_time,end_time
passing_positive_0001.mp4,D:\videos\exam.mp4,1,100.000000,98.000000,102.000000
```

`source_video` 是源视频完整路径，`positive=1`、`negative=0`。`event_time` 是保存时画面对应的时刻，`start_time/end_time` 是实际请求 FFmpeg 裁剪的区间，均相对于源视频开头。

## 常见情况

- **找不到 FFmpeg**：加入 PATH，放在程序旁边，设置环境变量 `FFMPEG_PATH` 指向完整 exe 路径，或保存时手动选择。
- **CSV 保存失败**：关闭占用 `labels.csv` 的 Excel 等程序后重试。此次导出会撤销，原有记录保留。
- **Excel 修改 CSV**：建议另存为“CSV UTF-8（逗号分隔）”，避免中文路径的编码变化。
- **无法打开视频**：检查视频是否损坏或编码是否受 OpenCV 支持。
- **倍速达不到预期**：高分辨率视频的解码速度受 CPU / 磁盘影响，工具会跳过部分预览帧，导出仍使用原视频。
- **旧样本出现灰屏、残影**：重新运行更新后的程序，从源视频重新导出该片段；已经编码进旧样本的残影不会随程序更新而消失。
- **可变帧率视频**：第一版按 OpenCV 报告的帧数、帧率计算总时长并定位进度条，适合常见恒定帧率监控 MP4。异常时间戳或可变帧率视频可能有定位偏差，建议先转换为恒定帧率再使用。

实现依据：[FFmpeg 裁剪参数](https://ffmpeg.org/ffmpeg.html)、[FFmpeg 画面裁剪滤镜](https://ffmpeg.org/ffmpeg-filters.html#crop)、[OpenCV 视频属性](https://docs.opencv.org/4.11.0/d4/d15/group__videoio__flags__base.html)、[Tkinter 文档](https://docs.python.org/3/library/tkinter.html)。
