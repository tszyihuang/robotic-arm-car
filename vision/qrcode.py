"""Scan the central half of the captured frame at 2x magnification."""
import time
import re

from config import VISION
from control import check_cancel

COLORS = ("red", "green", "blue")
SHAPES = ("cylinder", "cone", "drum")


def mission_code(data):
    if not isinstance(data, str) or not re.fullmatch(r"[123]{3}", data.strip()):
        raise ValueError("任务二维码必须是三位 1、2、3 的组合，例如 211")
    code = data.strip()
    return {"ball": COLORS[int(code[0]) - 1], "target": COLORS[int(code[1]) - 1],
            "object": SHAPES[int(code[2]) - 1]}


def scan_qrcode(camera, timeout=VISION["scan_timeout"], stop_event=None):
    import cv2
    detector = cv2.QRCodeDetector()
    deadline = time.monotonic() + timeout
    index = 0
    while True:
        check_cancel(stop_event)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('二维码扫描超时，未识别到有效数据')
        frame, index, _ = camera.next_frame(after=index, timeout=min(3.0, remaining),
                                          stop_event=stop_event)
        height, width = frame.shape[:2]
        crop_height, crop_width = max(1, height // 2), max(1, width // 2)
        top, left = (height - crop_height) // 2, (width - crop_width) // 2
        center = frame[top:top + crop_height, left:left + crop_width]
        zoomed = cv2.resize(center, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        data, _, _ = detector.detectAndDecode(zoomed)
        check_cancel(stop_event)
        if data:
            return data
