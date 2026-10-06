"""共用摄像头采集；分辨率切换与读取串行，旧帧不会跨切换复用。"""
import threading
import time

from base.control import check_cancel, cleanup
from config import VISION


def normalize_device(device):
    text = str(device)
    return int(text) if text.isdecimal() else device


class CameraStream:
    def __init__(self, device=0):
        import cv2
        self.device = normalize_device(device)
        self.condition = threading.Condition()
        self._capture_lock = threading.Lock()
        self._resizing = threading.Event()
        self.stop = threading.Event()
        self.frame = None
        self.index = 0
        self.capture_stamp = 0.0
        self.error = ''
        self.resolution = None
        self.camera = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.camera.isOpened():
            self.camera.release()
            raise RuntimeError(f'打不开摄像头 {self.device}')
        try:
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.set_resolution(VISION['width'], VISION['height'])
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.thread = threading.Thread(target=self._capture, name='camera-capture', daemon=True)
            self.thread.start()
        except BaseException:
            cleanup(('摄像头连接', self.camera.release))
            raise

    def set_resolution(self, width, height):
        import cv2
        if any(type(value) is not int or value <= 0 for value in (width, height)):
            raise ValueError('摄像头分辨率必须为正整数')
        self._resizing.set()
        try:
            with self._capture_lock:
                if self.stop.is_set():
                    raise RuntimeError('摄像头已关闭')
                if self.resolution == (width, height):
                    return
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                actual = (round(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          round(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                with self.condition:
                    self.resolution = actual
                    self.frame, self.capture_stamp = None, 0.0
                if actual != (width, height):
                    raise RuntimeError(f'摄像头不支持 {width}×{height}，实际为 {actual[0]}×{actual[1]}')
        finally:
            with self.condition:
                self._resizing.clear()
                self.condition.notify_all()

    def _capture(self):
        try:
            while not self.stop.is_set():
                with self.condition:
                    # 让分辨率切换优先于下一次读取，避免采集线程反复抢锁。
                    while self._resizing.is_set() and not self.stop.is_set():
                        self.condition.wait()
                with self._capture_lock:
                    if self.stop.is_set():
                        return
                    ok, frame = self.camera.read()
                    if not ok or frame is None:
                        raise RuntimeError('摄像头读取失败')
                    with self.condition:
                        # 驱动切换后可能仍返回缓冲中的旧尺寸画面。
                        if (frame.shape[1], frame.shape[0]) != self.resolution:
                            continue
                        self.frame = frame
                        self.index += 1
                        self.capture_stamp = time.time()
                        self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = str(exc)
                self.condition.notify_all()

    def next_frame(self, after=0, timeout=3.0, stop_event=None, after_stamp=0.0):
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                check_cancel(stop_event)
                if self.error:
                    raise RuntimeError(self.error)
                if self.stop.is_set():
                    raise RuntimeError('摄像头已关闭')
                if (self.frame is not None and self.index > after and self.capture_stamp > after_stamp
                        and time.time() - self.capture_stamp < VISION['frame_stale']):
                    return self.frame, self.index, self.capture_stamp
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('摄像头未返回新画面')
                self.condition.wait(min(remaining, 0.02))

    def close(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()

        def release():
            # VideoCapture 的释放也必须与 read()/set() 串行；否则可能在
            # 采集线程仍访问 V4L2 缓冲区时销毁它，触发原生代码段错误。
            with self._capture_lock:
                self.camera.release()

        cleanup(('摄像头释放', release),
                ('摄像头采集线程', lambda: self.thread.join(timeout=3.0)))
