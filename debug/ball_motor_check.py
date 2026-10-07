"""电机实测：真实编码器 + 模拟视觉目标；只打开指定电机串口。

python3 -m debug.ball_motor_check --port /dev/ttyUSB0 --distance-mm 20
不会连接摄像头、IMU 或机械臂；此检查不验证球识别或 mm/px 标定。
"""
import argparse
import json
from pathlib import Path
import signal
import threading
import time

from base.ball_position import calibrate_ball_position
from base.feedback import WheelOdometry
from base.motor import Motor
from config import BALL_POSITION


class EncoderTarget:
    """用真实编码器构造 20 Hz 目标帧，不实例化任何视觉设备。"""

    def __init__(self, motor, wheels, distance):
        self.motor, self.wheels, self.distance = motor, wheels, distance
        self.index = -1
        self.next_frame = 0.0
        self.latest = None

    def ball_layout_sample(self):
        if time.monotonic() >= self.next_frame:
            with self.motor.condition:
                totals = list(self.motor._totals)
                stamp = self.motor.capture_stamp
            distance = sum(self.wheels.travel(totals)) / 2
            error_px = (self.distance - distance) / BALL_POSITION["mm_per_px"]
            self.index += 1
            self.latest = {
                "size": [640, 480], "frame_index": self.index,
                "capture_stamp": time.time() - (time.monotonic() - stamp),
                "candidates": [{"center_x": 10, "value": "red"},
                               {"center_x": 320 + error_px, "value": "green"},
                               {"center_x": 630, "value": "blue"}],
            }
            self.next_frame = time.monotonic() + 0.05
        return self.latest

    def observe_ball_layout(self, **kwargs):
        return self.ball_layout_sample()

    def wait_ball_layout(self, *, after, timeout, stop_event=None):
        if stop_event is not None:
            stop_event.wait(timeout)
        else:
            time.sleep(timeout)
        frame = self.ball_layout_sample()
        return frame if frame["frame_index"] != after else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="仅使用这个已确认的电机串口，不扫描其他串口")
    parser.add_argument("--distance-mm", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=Path("/tmp/ball_motor_check.json"))
    args = parser.parse_args()
    if not args.port.strip():
        parser.error("必须明确指定已确认的电机串口，不能留空自动扫描")
    if not 0 < abs(args.distance_mm) <= 30:
        parser.error("测试位移必须在 ±30 mm 内且非零")
    stop_event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop_event.set())
    messages = []
    result = None
    motor = Motor(args.port, stop_event=stop_event)
    try:
        wheels = WheelOdometry.read_origin(motor, stop_event)
        vision = EncoderTarget(motor, wheels, args.distance_mm)

        def log(message):
            print(message, flush=True)
            messages.append(message)

        print("仅电机实测：编码器是真实反馈，视觉帧为模拟目标。", flush=True)
        result = calibrate_ball_position(motor, vision, stop_event=stop_event, log=log,
                                         config={"timeout": 12.0, "max_distance_m": 0.045})
        print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        try:
            motor.close(release_motors=False)
        finally:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"motor_port": args.port,
                                              "simulated_visual_target_mm": args.distance_mm,
                                              "result": result, "log": messages},
                                             ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
