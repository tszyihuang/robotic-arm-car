"""锁靶网页：共享摄像头预览和瞄准点，不在 HTTP 线程中操作电机。"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from urllib.parse import urlsplit

from arm.config import finite
from base.control import MotionCancelled, cleanup
from config import ROOT, VISION

DEFAULT_AIM_FILE = ROOT / "tasks" / "target_lock_aim.json"


def validate_aim(x, y):
    point = (finite(x, "瞄准点 x"), finite(y, "瞄准点 y"))
    if any(not 0 <= value <= 1 for value in point):
        raise ValueError("瞄准点坐标必须在 [0, 1] 内")
    return point


class AimPoint:
    """按画面宽高归一化保存瞄准点；磁盘写入不阻塞控制线程取值。"""

    def __init__(self, path=DEFAULT_AIM_FILE):
        self.path = Path(path).resolve()
        self._lock = threading.Lock()
        self._save_lock = threading.Lock()
        self._point = (0.5, 0.5)
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            pass
        else:
            if not isinstance(data, dict) or set(data) != {"x", "y"}:
                raise ValueError(f"瞄准点校正文件格式无效：{self.path}")
            self._point = validate_aim(data["x"], data["y"])
        self._saved = self._point

    def point(self):
        with self._lock:
            return self._point

    def set(self, x, y):
        point = validate_aim(x, y)
        with self._lock:
            self._point = point
        return self.status()

    def status(self):
        with self._lock:
            return {"x": self._point[0], "y": self._point[1],
                    "dirty": self._point != self._saved}

    def save(self):
        with self._save_lock:
            point = self.point()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                                 dir=self.path.parent, prefix=".target-lock-aim-",
                                                 suffix=".json", delete=False) as output:
                    temporary = Path(output.name)
                    json.dump(dict(zip(("x", "y"), point)), output, indent=2)
                    output.write("\n")
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, self.path)
                with self._lock:
                    self._saved = point
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return self.status()


class TargetLockHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(5.0)

    def log_message(self, format, *args):
        pass

    def reply(self, status, payload, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def json_reply(self, status, value):
        self.reply(status, json.dumps(value, ensure_ascii=False).encode(),
                   "application/json; charset=utf-8")

    def do_GET(self):
        path = urlsplit(self.path).path
        try:
            if path == "/":
                self.reply(200, Path(__file__).with_name("target_lock.html").read_bytes(),
                           "text/html; charset=utf-8")
            elif path == "/api/status":
                self.json_reply(200, self.server.view.status())
            elif path == "/frame.jpg":
                image, _ = self.server.view.snapshot()
                self.reply(200, image, "image/jpeg")
            elif path == "/stream.mjpg":
                self.stream()
            else:
                self.json_reply(404, {"error": "没有此页面"})
        except (RuntimeError, TimeoutError, OSError) as exc:
            self.json_reply(503, {"error": str(exc)})

    def stream(self):
        image, index = self.server.view.snapshot()
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while not self.server.view.closed.is_set():
                header = (f"--frame\r\nContent-Type: image/jpeg\r\n"
                          f"Content-Length: {len(image)}\r\n\r\n").encode("ascii")
                self.wfile.write(header + image + b"\r\n")
                image, index = self.server.view.snapshot(after=index)
        except (RuntimeError, TimeoutError, OSError):
            return

    def do_POST(self):
        path = urlsplit(self.path).path
        if path not in ("/api/aim", "/api/aim/reset", "/api/aim/save"):
            self.json_reply(404, {"error": "没有此操作"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= 4096:
                raise ValueError("请求长度无效")
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("请求必须为 JSON 对象")
            if path == "/api/aim":
                if set(body) != {"x", "y"}:
                    raise ValueError("需要瞄准点归一化坐标 x、y")
                result = self.server.view.aim.set(body["x"], body["y"])
            else:
                if body:
                    raise ValueError("此操作不接受额外参数")
                result = (self.server.view.aim.save() if path.endswith("/save")
                          else self.server.view.aim.set(0.5, 0.5))
            self.json_reply(200, {"ok": True, "aim": result})
        except (ValueError, TypeError, UnicodeError) as exc:
            self.json_reply(400, {"error": str(exc)})
        except OSError as exc:
            self.json_reply(409, {"error": str(exc)})


class TargetLockWeb:
    """最多 15 Hz 在独立线程编码预览；跟踪和网页共享同一个 CameraStream。"""

    def __init__(self, aim, host="0.0.0.0", port=8080):
        self.aim = aim
        self.closed = threading.Event()
        self._condition = threading.Condition()
        self._image = None
        self._index = 0
        self._stamp = 0.0
        self._size = None
        self._error = ""
        self._tracking = {"state": "等待跟踪启动", "target": None, "errors": None}
        self._threads = []
        # 先占用网页端口；端口冲突时不打开摄像头或机械臂。
        self.server = ThreadingHTTPServer((host, port), TargetLockHandler)
        self.server.daemon_threads = True
        self.server.view = self

    @property
    def port(self):
        return self.server.server_port

    def start(self, camera):
        workers = (("target-lock-preview", lambda: self._preview(camera)),
                   ("target-lock-http", lambda: self.server.serve_forever(poll_interval=0.1)))
        for name, work in workers:
            thread = threading.Thread(target=work, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def _preview(self, camera):
        import cv2
        index = 0
        while not self.closed.is_set():
            tick = time.monotonic()
            try:
                frame, index, stamp = camera.next_frame(after=index, timeout=0.2,
                                                       stop_event=self.closed)
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    raise RuntimeError("预览画面编码失败")
                with self._condition:
                    self._image, self._index, self._stamp = encoded.tobytes(), index, stamp
                    self._size = (frame.shape[1], frame.shape[0])
                    self._error = ""
                    self._condition.notify_all()
            except TimeoutError:
                continue
            except MotionCancelled:
                return
            except Exception as exc:
                with self._condition:
                    self._error = str(exc)
                    self._condition.notify_all()
                return
            self.closed.wait(max(0.0, 1 / 15 - (time.monotonic() - tick)))

    def update(self, state, result):
        target = None
        if result is not None:
            width, height = result["size"]
            x1, y1, x2, y2 = result["target"]["box"]
            target = {"color": result["target"]["value"],
                      "box": [x1 / width, y1 / height, x2 / width, y2 / height]}
        with self._condition:
            self._tracking = {"state": state, "target": target,
                              "errors": result["errors"] if result is not None else None}

    def status(self):
        with self._condition:
            status = dict(self._tracking, size=self._size, error=self._error,
                          image_age=time.time() - self._stamp if self._stamp else None)
        status["aim"] = self.aim.status()
        return status

    def snapshot(self, after=0, timeout=2.0):
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if self.closed.is_set():
                    raise RuntimeError("锁靶网页已关闭")
                if self._error:
                    raise RuntimeError(self._error)
                if (self._image is not None and self._index > after
                        and -0.1 <= time.time() - self._stamp < VISION["frame_stale"]):
                    return self._image, self._index
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("等待摄像头新画面")
                self._condition.wait(min(remaining, 0.05))

    def close(self):
        self.closed.set()
        with self._condition:
            self._condition.notify_all()
        actions = []
        if any(thread.name == "target-lock-http" for thread in self._threads):
            actions.append(("锁靶 HTTP 服务停止", self.server.shutdown))
        actions.append(("锁靶网页端口关闭", self.server.server_close))
        actions.extend((thread.name, lambda thread=thread: thread.join(timeout=3.0))
                       for thread in self._threads)
        cleanup(*actions)
