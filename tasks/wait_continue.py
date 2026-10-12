"""暂停任务：回车或舵机 ID2 在 0.5 秒内转动至少 5° 后继续。"""
from collections import deque
import math
import os
import select
import sys
import time

from arm.errors import ProtocolError
from base.control import MotionCancelled, check_cancel


class ServoMotion:
    """只比较最近半秒内的角度位移，按单圈角度处理跨零点。"""

    def __init__(self):
        self.samples = deque()
        self.previous = None
        self.position = 0.0

    def update(self, angle, now, *, sampled_at=None):
        if not math.isfinite(angle):
            raise ValueError("舵机角度必须为有限数值")
        if self.previous is not None:
            self.position += (angle - self.previous + 180.0) % 360.0 - 180.0
        self.previous = angle
        while self.samples and now - self.samples[0][0] > 0.5:
            self.samples.popleft()
        moved = any(abs(self.position - position) >= 5.0 for _, position in self.samples)
        self.samples.append((now if sampled_at is None else sampled_at, self.position))
        return moved


def _enter_pressed(fd):
    if not select.select([fd], [], [], 0.05)[0]:
        return False
    # 逐字节读取，半行输入不会阻塞舵机检测，也不预读下一次 pause 的回车。
    char = os.read(fd, 1)
    if not char:
        raise EOFError("控制台输入已关闭")
    return char == b"\n"


def run(arm, *, stop_event=None):
    from serial import SerialTimeoutException

    print("  任务已暂停，按回车或让舵机 ID2 在 0.5 秒内转动至少 5° 继续（Ctrl+C 取消）。",
          flush=True)
    fd = sys.stdin.fileno()
    motion = ServoMotion()
    monitor_servo = True
    retry_after = None
    retrying = False
    while True:
        check_cancel(stop_event)
        entered = _enter_pressed(fd)
        check_cancel(stop_event)
        if entered:
            return
        if monitor_servo and (retry_after is None or time.monotonic() >= retry_after):
            try:
                sampled_at = time.monotonic()
                angle = arm.read_servo_angle(2, stop_event=stop_event)
                check_cancel(stop_event)
                moved = motion.update(angle, time.monotonic(), sampled_at=sampled_at)
            except MotionCancelled:
                raise
            except (SerialTimeoutException, ProtocolError) as exc:
                motion = ServoMotion()
                retry_after = time.monotonic() + 0.2
                if not retrying:
                    print(f"  舵机 ID2 查询暂时失败：{exc}；继续重试，回车仍可继续。", flush=True)
                retrying = True
            except Exception as exc:
                monitor_servo = False
                print(f"  舵机 ID2 检测不可用：{exc}；本次暂停仍可按回车继续。", flush=True)
            else:
                retry_after = None
                if retrying:
                    print("  舵机 ID2 通信已恢复，继续检测转动。", flush=True)
                    retrying = False
                if moved:
                    print("  舵机 ID2 在 0.5 秒内转动已达 5°，继续执行。", flush=True)
                    return
