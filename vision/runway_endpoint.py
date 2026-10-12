"""中心竖线由下往上查找第一个灰→黄突变，连续新帧确认并推流。"""
import argparse
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import signal
import sys
import threading
import time

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base.control import MotionCancelled, cleanup
from config import RUNWAY_ENDPOINT, VISION


def endpoint_config(overrides=None):
    config = dict(RUNWAY_ENDPOINT)
    if overrides:
        unknown = set(overrides) - set(config)
        if unknown:
            raise ValueError(f"未知端点参数：{', '.join(sorted(unknown))}")
        config.update(overrides)
    for name in ("window", "stable_frames", "jump_frames", "lost_frames"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} 必须为正整数")
    for name in ("color_threshold", "yellow_threshold", "stable_tolerance_px", "max_jump_px"):
        value = config[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} 必须为有限正数")
    for name in ("yellow_support", "smoothing"):
        value = config[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} 必须在 (0, 1] 内")
    return config


@dataclass(frozen=True)
class EndpointCandidate:
    x: int
    y: int
    color_delta: float
    yellow_delta: float


def first_endpoint(frame, config=None):
    """每帧只返回从下往上遇到的第一个合格突变；命中即返回，不再搜索。"""
    import cv2
    import numpy as np

    config = endpoint_config(config)
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3
            or frame.shape[2] != 3 or min(frame.shape[:2]) < 1):
        raise ValueError("画面必须为非空的 uint8 BGR 三通道图像")
    height, width = frame.shape[:2]
    window = config["window"]
    if height < 2 * window:
        return None
    x = width // 2
    # 只转换中心一列。前缀和只用于窗口降噪，不搜索其他列或后续端点。
    column = cv2.cvtColor(frame[:, x:x + 1], cv2.COLOR_BGR2LAB)[:, 0].astype(np.float32)
    sums = np.vstack((np.zeros((1, 3), dtype=np.float32), column.cumsum(axis=0)))
    for y in range(height - window, window - 1, -1):
        upper = (sums[y] - sums[y - window]) / window
        lower = (sums[y + window] - sums[y]) / window
        yellow_delta = float(upper[2] - lower[2])
        if yellow_delta < config["yellow_threshold"]:
            continue
        color_delta = float(np.linalg.norm(upper[1:] - lower[1:]))
        if color_delta < config["color_threshold"]:
            continue
        support = np.count_nonzero(column[y - window:y, 2] - lower[2]
                                   >= config["yellow_threshold"]) / window
        if support >= config["yellow_support"]:
            return EndpointCandidate(x, y, color_delta, yellow_delta)
    return None


class EndpointDetector:
    """小幅移动平滑；大跳变保持旧点，连续在首个候选附近才接受新位置。"""

    def __init__(self, config=None):
        self.config = endpoint_config(config)
        self.point = None
        self._filtered_y = None
        self._pending = []
        self._misses = 0
        self._seen_stamp = 0.0
        self._index = 0
        self._stamp = 0.0
        self._size = None

    def _clear_point(self):
        self.point = self._filtered_y = None
        self._pending.clear()
        self._seen_stamp = 0.0
        self._misses = 0

    def reset_confirmation(self):
        """取帧中断时保留旧点，但后续新位置必须重新连续确认。"""
        self._pending.clear()

    def _result(self, state, candidate=None, required=0, valid=False, show_point=True):
        return {"state": state, "point": self.point if show_point else None,
                "candidate": asdict(candidate) if candidate is not None else None,
                "pending_count": len(self._pending), "required_count": required,
                "valid": valid}

    def process(self, frame, *, frame_index, capture_stamp, now=None):
        now = time.time() if now is None else now
        fresh = (math.isfinite(capture_stamp) and -0.1 <= now - capture_stamp < VISION["frame_stale"])
        if not fresh or frame_index <= self._index or capture_stamp <= self._stamp:
            self._pending.clear()
            if self.point is not None and now - self._seen_stamp >= VISION["frame_stale"]:
                self._clear_point()
            return self._result("stale", show_point=False)
        self._index, self._stamp = frame_index, capture_stamp
        size = frame.shape[:2]
        if size != self._size or (self.point is not None
                                 and now - self._seen_stamp >= VISION["frame_stale"]):
            self._clear_point()
        self._size = size
        candidate = first_endpoint(frame, self.config)
        if candidate is None:
            self._pending.clear()
            self._misses += 1
            if self._misses >= self.config["lost_frames"]:
                self._clear_point()
            return self._result("held" if self.point is not None else "searching")
        self._misses = 0
        if self.point is not None and abs(candidate.y - self.point[1]) <= self.config["max_jump_px"]:
            self._pending.clear()
            alpha = self.config["smoothing"]
            self._filtered_y += alpha * (candidate.y - self._filtered_y)
            self.point = (candidate.x, round(self._filtered_y))
            self._seen_stamp = capture_stamp
            return self._result("tracking", candidate, valid=True)
        required = self.config["stable_frames"] if self.point is None else self.config["jump_frames"]
        if self._pending and abs(candidate.y - self._pending[0]) > self.config["stable_tolerance_px"]:
            self._pending.clear()
        self._pending.append(candidate.y)
        if len(self._pending) >= required:
            self._filtered_y = sum(self._pending) / len(self._pending)
            self.point = (candidate.x, round(self._filtered_y))
            self._pending.clear()
            self._seen_stamp = capture_stamp
            return self._result("tracking", candidate, valid=True)
        return self._result("confirming" if self.point is None else "jump-pending", candidate, required)


class MjpegSource:
    """读取已有调试推流，只保留最新帧，不另开或打断摄像头。"""

    def __init__(self, url):
        from urllib.request import ProxyHandler, build_opener

        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.frame, self.index, self.capture_stamp = None, 0, 0.0
        self.error = ""
        self.response = build_opener(ProxyHandler({})).open(url, timeout=3.0)
        if "multipart/x-mixed-replace" not in self.response.headers.get("Content-Type", "").lower():
            self.response.close()
            raise RuntimeError("输入地址必须返回 MJPEG 推流，例如 /stream.mjpg")
        self.thread = threading.Thread(target=self._capture, name="endpoint-mjpeg-input", daemon=True)
        try:
            self.thread.start()
        except BaseException:
            self.response.close()
            raise

    def _capture(self):
        import cv2
        import numpy as np

        try:
            while not self.stop.is_set():
                line = self.response.readline(4097)
                if not line:
                    raise RuntimeError("输入 MJPEG 推流已断开")
                if len(line) > 4096:
                    raise RuntimeError("MJPEG 分隔行过长")
                if not line.startswith(b"--"):
                    continue
                headers = {}
                for _ in range(32):
                    line = self.response.readline(4097)
                    if line in (b"\r\n", b"\n"):
                        break
                    if not line or len(line) > 4096 or b":" not in line:
                        raise RuntimeError("MJPEG 图像头无效")
                    name, value = line.decode("ascii").split(":", 1)
                    headers[name.lower()] = value.strip()
                else:
                    raise RuntimeError("MJPEG 图像头过长")
                length = int(headers.get("content-length", "0"))
                if not 0 < length <= 16 * 1024 * 1024:
                    raise RuntimeError("MJPEG 图像长度无效，需要 Content-Length")
                data = self.response.read(length)
                stamp = time.time()  # 网络输入使用本机接收时间，不代表远端采集时间。
                if len(data) != length:
                    raise RuntimeError("MJPEG 图像不完整")
                frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError("MJPEG 图像解码失败")
                with self.condition:
                    self.frame = frame
                    self.index += 1
                    self.capture_stamp = stamp
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = str(exc)
                self.condition.notify_all()

    def next_frame(self, **kwargs):
        from vision.camera import CameraStream
        return CameraStream.next_frame(self, **kwargs)

    def close(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        cleanup(("输入推流连接", self.response.close),
                ("输入推流线程", lambda: self.thread.join(timeout=4.0)))


def main(args=None):
    from urllib.parse import urlsplit
    from debug.camera_web import local_addresses
    from vision.camera import CameraStream
    from vision.runway_endpoint_web import EndpointWeb

    parser = argparse.ArgumentParser(description="中心竖线跑道端点识别，并将标记画面推流到局域网")
    parser.add_argument("--camera", default=VISION["device"], help="摄像头编号或 /dev/video 路径")
    parser.add_argument("--stream-url", help="可选：读取已有原始 MJPEG 推流，替代直接打开摄像头")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--hz", type=float, default=20.0, help="最高识别和推流频率")
    parser.add_argument("--duration", type=float, help="可选运行秒数")
    for key in RUNWAY_ENDPOINT:
        parser.add_argument("--" + key.replace("_", "-"), dest=key,
                            type=int if type(RUNWAY_ENDPOINT[key]) is int else float,
                            default=None, help=f"默认 {RUNWAY_ENDPOINT[key]}；含义见 config.py")
    opts = parser.parse_args(args)
    if not 1 <= opts.port <= 65535:
        parser.error("--port 必须为 1..65535")
    for name in ("hz", "duration"):
        value = getattr(opts, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name} 必须为有限正数")
    if opts.stream_url:
        try:
            url = urlsplit(opts.stream_url)
            input_port = url.port or (443 if url.scheme == "https" else 80)
        except ValueError as exc:
            parser.error(f"--stream-url 地址无效：{exc}")
        if url.scheme not in ("http", "https") or not url.hostname:
            parser.error("--stream-url 必须为 HTTP(S) MJPEG 地址")
        if opts.host in ("0.0.0.0", "127.0.0.1", "localhost") and url.hostname in (
                "127.0.0.1", "localhost", *local_addresses()) and input_port == opts.port:
            parser.error("输入推流与输出监视画面不能使用同一个端口")
    try:
        detector = EndpointDetector({key: getattr(opts, key) for key in RUNWAY_ENDPOINT
                                     if getattr(opts, key) is not None})
    except ValueError as exc:
        parser.error(str(exc))
    stop = threading.Event()
    source = web = None

    def interrupt(signum, frame):
        stop.set()
        raise KeyboardInterrupt

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        web = EndpointWeb(opts.host, opts.port, stop_event=stop)
        source = MjpegSource(opts.stream_url) if opts.stream_url else CameraStream(opts.camera)
        web.start()
        print(f"本机：http://127.0.0.1:{opts.port}，按 Ctrl+C 退出。", flush=True)
        for address in local_addresses():
            print(f"局域网：http://{address}:{opts.port}", flush=True)
        index = 0
        start = time.monotonic()
        last_log = 0.0
        last_state = None
        while not stop.is_set():
            tick = time.monotonic()
            remaining = opts.duration - (tick - start) if opts.duration is not None else None
            if remaining is not None and remaining <= 0:
                break
            try:
                frame, index, stamp = source.next_frame(after=index, timeout=min(0.2, remaining)
                                                       if remaining is not None else 0.2, stop_event=stop)
            except TimeoutError:
                detector.reset_confirmation()
                continue
            result = detector.process(frame, frame_index=index, capture_stamp=stamp)
            web.publish(frame, result, index=index, stamp=stamp)
            if result["state"] != last_state or tick - last_log >= 1.0:
                candidate = result["candidate"]
                print(f"帧 {index}：{result['state']}，端点 {result['point']}，"
                      f"当前候选 y={candidate['y'] if candidate else '-'}，"
                      f"确认 {result['pending_count']}/{result['required_count']}", flush=True)
                last_state, last_log = result["state"], tick
            wait = max(0.0, 1 / opts.hz - (time.monotonic() - tick))
            if remaining is not None:
                wait = min(wait, max(0.0, opts.duration - (time.monotonic() - start)))
            stop.wait(wait)
    except (KeyboardInterrupt, MotionCancelled):
        pass
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"跑道端点识别失败：{exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        stop.set()
        actions = []
        if web is not None:
            actions.append(("端点监视网页", web.close))
        if source is not None:
            actions.append(("端点画面输入", source.close))
        cleanup(*actions, raise_errors=False)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
