"""端点监视：复用项目的 MJPEG 协议，所有客户端共享一份标记图像。"""
from http.server import ThreadingHTTPServer
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

from base.control import cleanup
from config import VISION
from debug.camera_web import CameraHandler


class EndpointHandler(CameraHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(5.0)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/":
            self.reply(200, Path(__file__).with_name("runway_endpoint.html").read_bytes(),
                       "text/html; charset=utf-8")
        elif path == "/api/status":
            self.json_reply(200, self.server.vision.status())
        else:
            super().do_GET()

    def do_POST(self):
        self.json_reply(405, {"error": "端点监视仅提供查看接口"})


class EndpointWeb:
    def __init__(self, host="0.0.0.0", port=8081, *, stop_event=None):
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.condition = threading.Condition()
        self._image = None
        self._stamp = 0.0
        self._status = {"state": "searching", "point": None, "candidate": None,
                        "pending_count": 0, "required_count": 0, "valid": False,
                        "size": None, "frame_index": 0}
        self.server = ThreadingHTTPServer((host, port), EndpointHandler)
        self.server.daemon_threads = True
        self.server.vision = self
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.1),
                                        name="runway-endpoint-http", daemon=True)
        self._thread.start()

    def publish(self, frame, result, *, index, stamp):
        import cv2

        image = frame.copy()
        height, width = image.shape[:2]
        x = width // 2
        cv2.line(image, (x, height - 1), (x, 0), (255, 200, 0), 1)
        point = result["point"]
        # 即使新候选远离旧点，也只标记一个已确认的点。
        if point is not None:
            point = tuple(point)
            color = (0, 255, 0) if result["valid"] else (0, 165, 255)
            cv2.circle(image, point, 7, color, 2)
            cv2.putText(image, f"end ({point[0]}, {point[1]})", (point[0] + 12, max(22, point[1] - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        label = result["state"]
        if result["required_count"]:
            label += f" {result['pending_count']}/{result['required_count']}"
        cv2.putText(image, label, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError("端点监视画面编码失败")
        with self.condition:
            self._image, self._stamp = encoded.tobytes(), stamp
            self._status = dict(result, size=(width, height), frame_index=index)
            self.condition.notify_all()

    def status(self):
        with self.condition:
            age = time.time() - self._stamp if self._stamp else None
            status = dict(self._status, image_age=age)
            if self.stop_event.is_set() or (age is not None and age >= VISION["frame_stale"]):
                status.update(state="stale", point=None, valid=False)
            return status

    def snapshot(self, *, after_stamp=None, timeout=2.0):
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                if self.stop_event.is_set():
                    raise RuntimeError("端点监视已关闭")
                if (self._image is not None and (after_stamp is None or self._stamp > after_stamp)
                        and -0.1 <= time.time() - self._stamp < VISION["frame_stale"]):
                    return self._image, self._stamp
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("等待端点监视新画面")
                self.condition.wait(min(remaining, 0.05))

    def close(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        actions = []
        if self._thread is not None and self._thread.is_alive():
            actions.append(("端点 HTTP 服务停止", self.server.shutdown))
        actions.append(("端点网页端口关闭", self.server.server_close))
        if self._thread is not None:
            actions.append(("端点 HTTP 线程", lambda: self._thread.join(timeout=3.0)))
        cleanup(*actions)
