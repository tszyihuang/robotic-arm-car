"""普通直走、视觉控制和倒车共用的左右轮里程与测速。"""
from collections import deque
import time

from config import BASE, STRAIGHT
from .control import check_cancel


class WheelOdometry:
    def __init__(self, origin):
        self.signs = STRAIGHT["FORWARD_SIGN"]
        self.mm_per_count = BASE["meters_per_count"] * 1000
        self.origin = self._sides(origin)
        self._speeds = [deque(maxlen=STRAIGHT["SPD_SAMPLES"]) for _ in range(2)]

    @classmethod
    def read_origin(cls, board, stop_event=None):
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            check_cancel(stop_event)
            totals, _ = board.feedback(0.1)
            check_cancel(stop_event)
            if totals is not None:
                return cls(totals)
        raise RuntimeError("读不到编码器 $MAll，请检查驱动板串口、接线和供电")

    def _sides(self, counts):
        signs = self.signs
        return ((counts[0] * signs[0] + counts[1] * signs[1]) / 2,
                (counts[2] * signs[2] + counts[3] * signs[3]) / 2)

    def travel(self, totals):
        return tuple((value - origin) * self.mm_per_count
                     for value, origin in zip(self._sides(totals), self.origin))

    def speed(self, increments):
        """只添加新增量；缺帧时保留最近窗口，过期由各控制器停车保护处理。"""
        if increments is not None:
            scale = self.mm_per_count / STRAIGHT["TEP_WINDOW"]
            for history, value in zip(self._speeds, self._sides(increments)):
                history.append(value * scale)
        return tuple(sum(history) / len(history) if history else 0.0
                     for history in self._speeds)

    @property
    def has_speed(self):
        return bool(self._speeds[0])
