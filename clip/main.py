"""人工视频样本裁剪工具。运行：python main.py（FFmpeg 需单独安装）。"""

import csv
import io
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk


APP_DIR = Path(__file__).resolve().parent
DATASET_DIR = APP_DIR / "dataset" / "talking"
BEHAVIORS = {"传递物品": "passing", "交头接耳": "talking"}
CSV_FIELDS = ["clip_name", "source_video", "label", "event_time", "start_time", "end_time"]
WINDOW_SECONDS = 2.0
SEEK_PREROLL_SECONDS = 10.0


def format_time(seconds):
    """以毫秒显示，允许超过 24 小时。"""
    milliseconds = max(0, round(seconds * 1000))
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def clip_bounds(event_time, duration):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("视频时长无效。")
    event_time = min(max(event_time, 0.0), duration)
    return max(0.0, event_time - WINDOW_SECONDS), min(duration, event_time + WINDOW_SECONDS)


def find_ffmpeg():
    candidates = [os.environ.get("FFMPEG_PATH"), shutil.which("ffmpeg"),
                  APP_DIR / "ffmpeg.exe", APP_DIR / "bin" / "ffmpeg.exe"]
    return next((str(p) for p in candidates if p and Path(p).is_file()), None)


def selection_to_roi(start, end, frame_width, frame_height):
    """把两角转换成原视频像素坐标，并对齐到偶数以兼容 H.264。"""
    left, right = sorted((start[0], end[0]))
    top, bottom = sorted((start[1], end[1]))
    x1 = int(min(frame_width, max(0, left))) // 2 * 2
    y1 = int(min(frame_height, max(0, top))) // 2 * 2
    x2 = int(min(frame_width, max(0, right))) // 2 * 2
    y2 = int(min(frame_height, max(0, bottom))) // 2 * 2
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2 - x1, y2 - y1


def decode_labels(data):
    """兼容程序生成的 UTF-8，以及 Excel 常用的中文/Unicode CSV。"""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16"), "utf-16"
    try:
        return data.decode("utf-8-sig"), "utf-8-sig"
    except UnicodeDecodeError:
        try:
            return data.decode("gb18030"), "gb18030"
        except UnicodeDecodeError as error:
            raise ValueError("无法识别 labels.csv 的编码，请将其另存为 CSV UTF-8 后重试。") from error


class SampleStore:
    """只在导出成功后写入 CSV；输出编号由磁盘上的文件决定。"""

    def __init__(self, root=DATASET_DIR, behavior="talking"):
        if behavior not in BEHAVIORS.values():
            raise ValueError("不支持的行为类型。")
        self.behavior = behavior
        self.root = Path(root)
        self.csv_path = self.root / "labels.csv"
        for category in ("positive", "negative"):
            (self.root / category).mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("x", encoding="utf-8-sig", newline="") as file:
                csv.writer(file).writerow(CSV_FIELDS)

    def samples(self, category):
        pattern = re.compile(rf"{self.behavior}_{category}_(\d+)\.mp4")
        return [(path, int(match.group(1)))
                for path in (self.root / category).iterdir()
                if path.is_file() and (match := pattern.fullmatch(path.name))]

    def counts(self):
        return {category: sum(path.stat().st_size > 0 for path, _ in self.samples(category))
                for category in ("positive", "negative")}

    def read_labels(self):
        text, _ = decode_labels(self.csv_path.read_bytes())
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError("labels.csv 表头不符合要求，请检查文件。")
        return list(reader)

    def write_labels(self, rows):
        # 使用临时文件替换，避免写入中断损坏原有 CSV。
        original = self.csv_path.read_bytes()
        _, encoding = decode_labels(original)
        if encoding != "utf-8-sig":
            # 首次转换旧编码前保留逐字节备份，不覆盖任何已有备份。
            backup = self.root / f"labels_before_utf8_{time.time_ns()}.csv.bak"
            with backup.open("xb") as file:
                file.write(original)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="",
                                             dir=self.root, prefix=".labels_", suffix=".tmp",
                                             delete=False) as file:
                temp_path = Path(file.name)
                writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_path, self.csv_path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def export(self, ffmpeg, source, category, event_time, duration, roi=None):
        if category not in ("positive", "negative"):
            raise ValueError("请先选择正样本或负样本。")
        if roi is not None:
            if len(roi) != 4 or any(not isinstance(value, int) or value % 2 for value in roi):
                raise ValueError("选框坐标和尺寸必须是偶数像素。")
            x, y, width, height = roi
            if x < 0 or y < 0 or width < 2 or height < 2:
                raise ValueError("选框无效，请重新框选。")
        start, end = clip_bounds(event_time, duration)
        rows = self.read_labels()
        pattern = re.compile(rf"{self.behavior}_{category}_(\d+)\.mp4")
        # CSV 中的编号也保留：即使手动删除了某个样本，也不复用它的编号。
        used = [number for _, number in self.samples(category)]
        used += [int(match.group(1)) for row in rows
                 if (match := pattern.fullmatch(row["clip_name"]))]
        number = max(used, default=0) + 1
        while True:
            output = self.root / category / f"{self.behavior}_{category}_{number:04d}.mp4"
            try:
                # 排他创建：已有文件绝不会被覆盖。
                with output.open("xb"):
                    pass
                break
            except FileExistsError:
                number += 1

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".clip_",
                                             suffix=".mp4", delete=False) as file:
                temp_path = Path(file.name)
            # 监控 HEVC 的跳转点可能依赖更早的参考帧；先解码一段，再丢弃到目标时间。
            seek_start = max(0.0, start - SEEK_PREROLL_SECONDS)
            command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                       "-ss", f"{seek_start:.6f}", "-noautorotate", "-i", str(source),
                       "-ss", f"{start - seek_start:.6f}", "-t", f"{end - start:.6f}",
                       "-map", "0:v:0", "-an"]
            if roi is not None:
                command += ["-vf", f"crop={width}:{height}:{x}:{y}"]
            command += ["-c:v", "libx264", "-preset", "fast", "-crf", "18",
                        "-movflags", "+faststart", str(temp_path)]
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if result.returncode != 0:
                error = result.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(error[-2500:] or "FFmpeg 导出失败。")
            if temp_path.stat().st_size == 0:
                raise RuntimeError("FFmpeg 未生成有效的视频文件。")
            os.replace(temp_path, output)
            rows.append({"clip_name": output.name, "source_video": str(Path(source).resolve()),
                         "label": 1 if category == "positive" else 0,
                         "event_time": f"{event_time:.6f}", "start_time": f"{start:.6f}",
                         "end_time": f"{end:.6f}"})
            self.write_labels(rows)
            return output, start, end
        except Exception:
            # CSV 写入失败（例如被 Excel 占用）时撤销此次样本，方便重试。
            output.unlink(missing_ok=True)
            raise
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


class TalkingClipTool:
    def __init__(self, root, dataset_root=None):
        self.root = root
        self.dataset_root = Path(dataset_root) if dataset_root is not None else APP_DIR / "dataset"
        self.current_behavior = "passing"
        self.store = SampleStore(self.dataset_root / self.current_behavior, behavior=self.current_behavior)
        self.ffmpeg = find_ffmpeg()
        self.capture = None
        self.source = None
        self.frame = None
        self.frame_index = 0
        self.current_time = 0.0
        self.duration = 0.0
        self.fps = 25.0
        self.frame_count = 0
        self.playing = False
        self.dragging = False
        self.selecting = False
        self.roi = None
        self.selection_start = None
        self.selection_end = None
        self.preview_bounds = None
        self.saving = False
        self.updating_slider = False
        self.results = queue.Queue()
        self.category = tk.StringVar(value="")
        self.behavior = tk.StringVar(value="传递物品")
        self.positive_text = tk.StringVar(value="正样本：有传递物品 (1)")
        self.negative_text = tk.StringVar(value="负样本：没有传递物品 (2)")
        self.output_text = tk.StringVar(value="保存到：dataset/passing/")
        self.speed = tk.StringVar(value="1")
        self.status = tk.StringVar(value="当前采集传递物品：框住两人的手、物品和交接范围，选择标签并保存。")
        self.crop_text = tk.StringVar(value="当前保存：整幅画面（在视频上按住鼠标左键拖动框选）")
        self.time_text = tk.StringVar(value="00:00:00.000 / 00:00:00.000")
        self.count_text = tk.StringVar()
        self.build_ui()
        self.update_counts()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(15, self.tick)

    def build_ui(self):
        self.root.title("考场行为视频样本裁剪工具")
        self.root.geometry("1100x800")
        self.root.minsize(880, 640)
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        toolbar = ttk.Frame(outer)
        toolbar.pack(fill="x", pady=(0, 8))
        self.open_button = ttk.Button(toolbar, text="打开 MP4 视频", command=self.open_video)
        self.open_button.pack(side="left")
        self.source_text = tk.StringVar(value="尚未打开视频")
        ttk.Label(toolbar, textvariable=self.source_text).pack(side="left", padx=12)
        ttk.Label(toolbar, textvariable=self.count_text).pack(side="right")

        behavior_controls = ttk.Frame(outer)
        behavior_controls.pack(fill="x", pady=(0, 8))
        ttk.Label(behavior_controls, text="采集行为：").pack(side="left")
        self.behavior_box = ttk.Combobox(behavior_controls, textvariable=self.behavior,
                                        values=list(BEHAVIORS), state="readonly", width=14,
                                        takefocus=False)
        self.behavior_box.pack(side="left", padx=(0, 16))
        self.behavior_box.bind("<<ComboboxSelected>>", self.change_behavior)
        ttk.Label(behavior_controls, textvariable=self.output_text).pack(side="left")

        self.canvas = tk.Canvas(outer, background="#171a21", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda event: self.draw_frame())
        self.canvas.configure(cursor="crosshair")
        self.canvas.bind("<ButtonPress-1>", self.begin_selection)
        self.canvas.bind("<B1-Motion>", self.move_selection)
        self.canvas.bind("<ButtonRelease-1>", self.end_selection)
        crop_controls = ttk.Frame(outer)
        crop_controls.pack(fill="x", pady=(6, 0))
        ttk.Label(crop_controls, textvariable=self.crop_text).pack(side="left")
        ttk.Button(crop_controls, text="清除选框（恢复整幅画面）",
                   command=self.clear_selection, takefocus=False).pack(side="right")
        self.slider = ttk.Scale(outer, from_=0, to=1, command=self.slider_changed, takefocus=False)
        self.slider.pack(fill="x", pady=(10, 4))
        self.slider.bind("<ButtonPress-1>", self.begin_drag)
        self.slider.bind("<ButtonRelease-1>", self.end_drag)
        ttk.Label(outer, textvariable=self.time_text, font=("Consolas", 14)).pack()

        playback = ttk.Frame(outer)
        playback.pack(pady=8)
        ttk.Button(playback, text="后退 1 秒 (A)", command=lambda: self.step(-1)).pack(side="left", padx=4)
        self.play_button = ttk.Button(playback, text="播放 (Space)", command=self.toggle_play)
        self.play_button.pack(side="left", padx=4)
        ttk.Button(playback, text="前进 1 秒 (D)", command=lambda: self.step(1)).pack(side="left", padx=4)
        ttk.Label(playback, text="播放速度：").pack(side="left", padx=(18, 0))
        for value in ("1", "2", "4"):
            ttk.Radiobutton(playback, text=f"{value} 倍", variable=self.speed, value=value,
                            command=self.reset_clock, takefocus=False).pack(side="left", padx=4)

        labels = ttk.LabelFrame(outer, text="样本标签（裁剪当前时刻前 2 秒 + 后 2 秒）", padding=10)
        labels.pack(fill="x", pady=6)
        ttk.Radiobutton(labels, textvariable=self.positive_text, value="positive",
                        variable=self.category, takefocus=False).pack(side="left", padx=10)
        ttk.Radiobutton(labels, textvariable=self.negative_text, value="negative",
                        variable=self.category, takefocus=False).pack(side="left", padx=10)
        self.save_button = ttk.Button(labels, text="保存样本 (S)", command=self.save_sample)
        self.save_button.pack(side="right", padx=10)
        ttk.Label(outer, textvariable=self.status, wraplength=850).pack(fill="x", pady=(6, 2))
        ttk.Label(outer, text="Space 播放/暂停  ·  A/D 后退/前进 1 秒  ·  1/2 选择正/负样本  ·  S 保存").pack()
        # 快捷键先于按钮的默认键盘行为执行，防止 Space 触发两次。
        self.root.bind_class("ClipShortcuts", "<KeyPress>", self.on_key)

        def bind_shortcuts(widget):
            widget.bindtags(("ClipShortcuts",) + widget.bindtags())
            for child in widget.winfo_children():
                bind_shortcuts(child)

        bind_shortcuts(self.root)

    def on_key(self, event):
        actions = {"space": self.toggle_play, "a": lambda: self.step(-1),
                   "d": lambda: self.step(1), "1": lambda: self.category.set("positive"),
                   "2": lambda: self.category.set("negative"), "s": self.save_sample}
        action = actions.get(event.keysym.lower())
        if action:
            action()
            return "break"  # 防止 Space 同时触发获得焦点的按钮。

    def update_counts(self):
        counts = self.store.counts()
        self.count_text.set(f"正样本：{counts['positive']}    负样本：{counts['negative']}")

    def change_behavior(self, event=None):
        previous_name = next(name for name, key in BEHAVIORS.items() if key == self.current_behavior)
        if self.saving:
            self.behavior.set(previous_name)
            return
        name = self.behavior.get()
        key = BEHAVIORS.get(name)
        if key is None or key == self.current_behavior:
            self.behavior.set(previous_name)
            return
        self.pause()
        try:
            store = SampleStore(self.dataset_root / key, behavior=key)
            store.read_labels()
        except (OSError, ValueError) as error:
            self.behavior.set(previous_name)
            messagebox.showerror("切换失败", f"无法打开该行为的数据目录：\n{error}")
            return
        self.store = store
        self.current_behavior = key
        self.category.set("")
        self.positive_text.set(f"正样本：有{name} (1)")
        self.negative_text.set(f"负样本：没有{name} (2)")
        self.output_text.set(f"保存到：dataset/{key}/")
        self.update_counts()
        self.status.set(f"已切换为{name}，请重新选择正样本或负样本。当前时刻和选框已保留。")

    def open_video(self):
        if self.saving:
            return
        self.pause()
        path = filedialog.askopenfilename(title="选择考场监控视频", filetypes=[("MP4 视频", "*.mp4")])
        if not path:
            return
        capture = cv2.VideoCapture(path)
        # 预览与 FFmpeg 的 -noautorotate 保持一致，选框对应原始像素。
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        ok, frame = capture.read()
        if not ok or not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count < 1:
            capture.release()
            messagebox.showerror("打开失败", "无法解码视频或读取帧率/时长，请检查 MP4 文件。")
            return
        if self.capture is not None:
            self.capture.release()
        self.capture, self.source = capture, Path(path).resolve()
        self.fps, self.frame_count = fps, int(count)
        self.duration = self.frame_count / self.fps
        self.slider.configure(to=self.duration)
        self.source_text.set(self.source.name)
        self.category.set("")
        self.clear_selection()
        self.accept_frame(frame)
        if self.current_behavior == "passing":
            self.status.set("暂停在物品交接的核心时刻，框住两人的手、物品和交接范围，再选择标签保存。")
        else:
            self.status.set("已打开视频。暂停在核心时刻，在画面上拖动框住两个人，再选择标签保存。")

    def accept_frame(self, frame):
        self.frame = frame
        self.frame_index = max(0, round(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        timestamp = self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if not math.isfinite(timestamp) or timestamp < 0 or (timestamp == 0 and self.frame_index > 0):
            timestamp = self.frame_index / self.fps
        self.current_time = min(timestamp, self.duration)
        self.time_text.set(f"{format_time(self.current_time)} / {format_time(self.duration)}")
        if not self.dragging:
            self.updating_slider = True
            self.slider.set(self.current_time)
            self.updating_slider = False
        self.draw_frame()

    def draw_frame(self):
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width < 2 or height < 2:
            return
        self.canvas.delete("all")
        if self.frame is None:
            self.preview_bounds = None
            self.canvas.create_text(width / 2, height / 2, text="请打开一个 MP4 视频",
                                    fill="white", font=("Microsoft YaHei UI", 18))
            return
        frame_height, frame_width = self.frame.shape[:2]
        scale = min(width / frame_width, height / frame_height)
        preview_width = max(1, round(frame_width * scale))
        preview_height = max(1, round(frame_height * scale))
        preview = cv2.resize(self.frame, (preview_width, preview_height))
        self.preview_bounds = ((width - preview_width) / 2, (height - preview_height) / 2,
                               preview_width, preview_height)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(width / 2, height / 2, image=self.photo)
        self.draw_selection()

    def video_point(self, canvas_x, canvas_y, clamp=False):
        """消除等比缩放和黑边偏移，得到原视频中的坐标。"""
        if self.frame is None or self.preview_bounds is None:
            return None
        left, top, width, height = self.preview_bounds
        if not clamp and not (left <= canvas_x <= left + width and top <= canvas_y <= top + height):
            return None
        frame_height, frame_width = self.frame.shape[:2]
        x = min(width, max(0, canvas_x - left)) / width * frame_width
        y = min(height, max(0, canvas_y - top)) / height * frame_height
        return x, y

    def draw_selection(self):
        self.canvas.delete("selection")
        if self.frame is None or self.preview_bounds is None:
            return
        if self.selecting:
            start, end = self.selection_start, self.selection_end
        elif self.roi is not None:
            x, y, width, height = self.roi
            start, end = (x, y), (x + width, y + height)
        else:
            return
        left, top, width, height = self.preview_bounds
        frame_height, frame_width = self.frame.shape[:2]
        self.canvas.create_rectangle(left + start[0] / frame_width * width,
                                     top + start[1] / frame_height * height,
                                     left + end[0] / frame_width * width,
                                     top + end[1] / frame_height * height,
                                     outline="#44ff88", width=3, tags="selection")

    def begin_selection(self, event):
        point = self.video_point(event.x, event.y)
        if point is None:
            return
        self.pause()
        self.selecting = True
        self.selection_start = self.selection_end = point
        self.draw_selection()

    def move_selection(self, event):
        if self.selecting:
            self.selection_end = self.video_point(event.x, event.y, clamp=True)
            self.draw_selection()

    def end_selection(self, event):
        if not self.selecting:
            return
        self.selection_end = self.video_point(event.x, event.y, clamp=True)
        frame_height, frame_width = self.frame.shape[:2]
        roi = selection_to_roi(self.selection_start, self.selection_end, frame_width, frame_height)
        self.selecting = False
        if roi is None:
            self.status.set("选框太小，原选框已保留；请按住鼠标左键拖出矩形。")
        else:
            self.roi = roi
            self.update_crop_text()
            self.status.set("选框已确定，保存时只裁出绿色框内画面。重新拖动可更换选框。")
        self.draw_selection()

    def update_crop_text(self):
        if self.roi is None:
            self.crop_text.set("当前保存：整幅画面（在视频上按住鼠标左键拖动框选）")
        else:
            x, y, width, height = self.roi
            self.crop_text.set(f"当前保存：选框内画面  {width} × {height} 像素  |  左上角 ({x}, {y})")

    def clear_selection(self):
        self.selecting = False
        self.roi = None
        self.selection_start = self.selection_end = None
        self.update_crop_text()
        self.draw_selection()

    def reset_clock(self):
        self.clock_wall = time.perf_counter()
        self.clock_frame = self.frame_index

    def pause(self):
        self.playing = False
        self.play_button.configure(text="播放 (Space)")

    def toggle_play(self):
        if self.capture is None or self.dragging or self.selecting:
            return
        if self.playing:
            self.pause()
        else:
            if self.frame_index >= self.frame_count - 1:
                self.seek(0)
            self.reset_clock()
            self.playing = True
            self.play_button.configure(text="暂停 (Space)")

    def seek(self, seconds):
        if self.capture is None or self.selecting:
            return
        index = min(self.frame_count - 1, max(0, round(float(seconds) * self.fps)))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.capture.read()
        if ok:
            self.accept_frame(frame)
        else:
            self.pause()
            self.status.set("该位置读取失败，请尝试附近时间。")
        self.reset_clock()

    def step(self, seconds):
        if self.capture is not None:
            self.seek(self.current_time + seconds)

    def begin_drag(self, event):
        self.resume_after_drag = self.playing
        self.pause()
        self.dragging = True

    def slider_changed(self, value):
        if not self.updating_slider and self.capture is not None:
            if self.dragging:
                self.time_text.set(f"{format_time(float(value))} / {format_time(self.duration)}")
            else:
                self.seek(float(value))

    def end_drag(self, event):
        if not self.dragging:
            return
        self.dragging = False
        self.seek(self.slider.get())
        if self.capture is not None and self.resume_after_drag:
            self.toggle_play()

    def tick(self):
        # Tk 控件仅由主线程更新；导出线程通过队列回传结果。
        try:
            result, error = self.results.get_nowait()
        except queue.Empty:
            pass
        else:
            self.saving = False
            self.save_button.configure(state="normal", text="保存样本 (S)")
            self.open_button.configure(state="normal")
            self.behavior_box.configure(state="readonly")
            self.update_counts()
            if error:
                self.status.set("保存失败，请处理错误后重试。")
                messagebox.showerror("保存失败", error)
            else:
                output, start, end = result
                self.status.set(f"已保存：{output.name}  |  {format_time(start)} ～ {format_time(end)}")

        if self.playing and not self.dragging:
            elapsed = (time.perf_counter() - self.clock_wall) * float(self.speed.get())
            target = min(self.frame_count - 1, self.clock_frame + int(elapsed * self.fps))
            if target > self.frame_index:
                # 跳过不需要预览的帧，避免倍速播放时仍逐帧绘制。
                ok = True
                for _ in range(min(target - self.frame_index - 1, 120)):
                    if not self.capture.grab():
                        ok = False
                        break
                ok, frame = self.capture.read() if ok else (False, None)
                if ok:
                    self.accept_frame(frame)
                else:
                    self.pause()
                    self.status.set("播放结束或视频帧读取失败。")
            if self.frame_index >= self.frame_count - 1:
                self.pause()
        self.root.after(15, self.tick)

    def save_sample(self):
        if self.saving:
            return
        if self.capture is None:
            messagebox.showinfo("提示", "请先打开视频。")
            return
        if self.dragging or self.selecting:
            return
        self.pause()
        category = self.category.get()
        if not category:
            messagebox.showinfo("提示", "请先选择正样本（1）或负样本（2）。")
            return
        if not self.ffmpeg:
            path = filedialog.askopenfilename(title="请选择 FFmpeg 的 ffmpeg.exe",
                                             filetypes=[("FFmpeg", "ffmpeg.exe")])
            if not path:
                self.status.set("尚未找到 FFmpeg。请加入 PATH，或保存时选择 ffmpeg.exe。")
                return
            self.ffmpeg = path
        source, event_time, duration = self.source, self.current_time, self.duration
        roi = self.roi  # 固定此次导出范围，后台保存时重新框选不会影响它。
        store = self.store
        self.saving = True
        self.save_button.configure(state="disabled", text="正在保存…")
        self.open_button.configure(state="disabled")
        self.behavior_box.configure(state="disabled")
        area = "整幅画面" if roi is None else f"选框内 {roi[2]} × {roi[3]} 像素画面"
        self.status.set(f"正在准备画面并导出 {format_time(event_time)} 附近的{area}，请稍候…")

        def worker():
            try:
                result = store.export(self.ffmpeg, source, category, event_time, duration, roi=roi)
                self.results.put((result, None))
            except Exception as error:
                self.results.put((None, str(error)))

        threading.Thread(target=worker, daemon=True).start()

    def close(self):
        if self.saving:
            messagebox.showinfo("正在保存", "请等待当前样本保存完成后再关闭窗口。")
            return
        if self.capture is not None:
            self.capture.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        TalkingClipTool(root)
    except (OSError, ValueError) as error:
        messagebox.showerror("启动失败", f"无法初始化数据目录：\n{error}")
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
