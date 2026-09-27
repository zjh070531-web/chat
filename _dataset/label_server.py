# -*- coding: utf-8 -*-
r"""
label_server.py —— 候选 clip 快速标注服务

在浏览器里逐条播放候选 clip, 按 1(交头接耳) / 0(正常) 打标,
每打一条立即落盘到 labels.csv, 支持断点续标。

用法:
    python label_server.py            # 默认 http://127.0.0.1:8765
    python label_server.py --port 9000
"""

import argparse
import csv
import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_IN = os.path.join(ROOT, "candidates.csv")
LABELS = os.path.join(ROOT, "labels.csv")


def load_candidates():
    if not os.path.exists(CSV_IN):
        return []
    with open(CSV_IN, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_labels():
    d = {}
    if os.path.exists(LABELS):
        with open(LABELS, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                k = (r.get("clip") or "").strip()
                v = (r.get("label") or "").strip()
                if k and v in ("0", "1"):
                    d[k] = int(v)
    return d


def save_labels(d):
    with open(LABELS, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "label"])
        for k in sorted(d):
            w.writerow([k, d[k]])


MEDIA_EXT = (".mp4", ".webm", ".m4v")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def guess_type(self, path):
        # Windows 注册表里的 .mp4 映射不可靠, 显式钉死成 video/mp4
        if os.path.splitext(str(path))[1].lower() in MEDIA_EXT:
            return "video/mp4"
        return super().guess_type(path)

    def _serve_media(self, path):
        """带 Range + no-store 的视频下发(浏览器 seek/循环播放需要)"""
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return
        with f:
            size = os.fstat(f.fileno()).st_size
            start, end, code = 0, size - 1, 200
            rng = self.headers.get("Range") or ""
            if rng.startswith("bytes="):
                try:
                    spec = rng.split("=", 1)[1].split(",")[0].strip()
                    a, _, b = spec.partition("-")
                    if a == "":
                        start = max(0, size - int(b))
                    else:
                        start = int(a)
                        if b:
                            end = min(int(b), size - 1)
                    if start > end or start >= size:
                        raise ValueError(spec)
                    code = 206
                except Exception:
                    start, end, code = 0, size - 1, 200
            n = end - start + 1
            self.send_response(code)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(n))
            self.send_header("Cache-Control", "no-store, must-revalidate")
            if code == 206:
                self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
            self.end_headers()
            f.seek(start)
            left = n
            while left > 0:
                chunk = f.read(min(65536, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/data"):
            self._json({"rows": load_candidates(), "labels": load_labels()})
            return
        if self.path == "/":
            self.path = "/label.html"
        path = self.translate_path(self.path)
        if os.path.isfile(path) and path.lower().endswith(MEDIA_EXT):
            self._serve_media(path)
            return
        return super().do_GET()

    def do_POST(self):
        if self.path == "/_note":          # 仅调试用: 把请求体原样落盘到 _note.json
            n = int(self.headers.get("Content-Length", 0))
            with open(os.path.join(ROOT, "_note.json"), "wb") as fh:
                fh.write(self.rfile.read(n))
            self._json({"ok": True})
            return
        if self.path != "/api/label":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as e:
            self._json({"ok": False, "err": str(e)}, 400)
            return
        d = load_labels()
        for k, v in data.items():
            if v is None:
                d.pop(k, None)
            else:
                d[k] = int(v)
        save_labels(d)
        self._json({"ok": True, "labeled": len(d)})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    rows = load_candidates()
    print(f"候选 {len(rows)} 条 | 已标注 {len(load_labels())} 条")
    print(f"打开: http://127.0.0.1:{args.port}/")
    print(f"标注结果写入: {LABELS}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
