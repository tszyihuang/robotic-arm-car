"""One camera owner supplies the latest frame to QR and model inference."""
import threading
import time

from control import check_cancel
from config import VISION
from control import cleanup


def normalize_device(device):
    text = str(device)
    return int(text) if text.isdecimal() else device


class CameraStream:
    def __init__(self, device=0):
        import cv2
        self.device = normalize_device(device)
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.frame = None
        self.index = 0
        self.capture_stamp = 0.0
        self.error = ''
        self.camera = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.camera.isOpened():
            self.camera.release()
            raise RuntimeError(f'打不开摄像头 {self.device}')
        try:
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, VISION['width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, VISION['height'])
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.thread = threading.Thread(target=self._capture, name='camera-capture', daemon=True)
            self.thread.start()
        except BaseException:
            cleanup(('摄像头连接', self.camera.release))
            raise

    def _capture(self):
        try:
            while not self.stop.is_set():
                ok, frame = self.camera.read()
                if not ok or frame is None:
                    raise RuntimeError('摄像头读取失败')
                with self.condition:
                    self.frame = frame
                    self.index += 1
                    self.capture_stamp = time.time()
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = str(exc)
                self.condition.notify_all()

    def next_frame(self, after=0, timeout=3.0, stop_event=None):
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                check_cancel(stop_event)
                if self.error:
                    raise RuntimeError(self.error)
                if self.stop.is_set():
                    raise RuntimeError('摄像头已关闭')
                if self.index > after and time.time() - self.capture_stamp < VISION['frame_stale']:
                    return self.frame, self.index, self.capture_stamp
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('摄像头未返回新画面')
                self.condition.wait(min(remaining, 0.02))

    def close(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        cleanup(('摄像头释放', self.camera.release),
                ('摄像头采集线程', lambda: self.thread.join(timeout=3.0)))
