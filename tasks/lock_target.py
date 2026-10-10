"""独立锁定彩色标靶：ID2 保持固定角度，横向、纵向 PID 分别控制 ID1、ID4。"""
import argparse
import math
from pathlib import Path
import sys
import time

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm.config import ArmConfig, finite
from arm.motor import ANGLE_SCALE, MotorBus, check_fault, position_payload
from base.control import MotionCancelled, check_cancel, clamp, cleanup, wait_cancelable
from config import ARM, TARGET_LOCK, VISION
from vision.camera import CameraStream
from vision.targets import colored_targets

TARGETS = ("middle", "left", "right", "red", "green", "blue")
AXES = (1, 4)


def validate_config(config=None):
    cfg = dict(TARGET_LOCK)
    if config is not None:
        unknown = config.keys() - cfg.keys()
        if unknown:
            raise ValueError(f"未知锁靶参数：{sorted(unknown)}")
        cfg.update(config)
    for key in cfg:
        cfg[key] = finite(cfg[key], key)
        if key == "motor2_angle_deg":
            continue
        if cfg[key] < 0 or cfg[key] == 0 and key not in ("ki", "kd", "derivative_tau", "tolerance_px"):
            raise ValueError(f"{key} 必须为有限正数（Ki、Kd、微分滤波和容差可为 0）")
    for key in ("min_area_ratio", "match_distance_ratio"):
        if not 0 < cfg[key] < 1:
            raise ValueError(f"{key} 必须在 (0, 1) 内")
    if cfg["max_step_deg"] < ANGLE_SCALE:
        raise ValueError(f"max_step_deg 不能小于电机编码器精度 {ANGLE_SCALE:g}°")
    position_payload(0.0, cfg["max_rate_deg_s"] / 6.0)
    position_payload(cfg["motor2_angle_deg"], ARM["speed_rpm"])
    return cfg


class PID:
    """带微分滤波、积分抗饱和和方向约束的独立单轴 PID。"""

    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.integral = self.derivative = 0.0
        self.previous = None

    def step(self, error, dt):
        cfg = self.config
        if self.previous is not None:
            if error * self.previous < 0:
                self.integral = 0.0
            raw_d = (error - self.previous) / dt
            self.derivative += dt / (cfg["derivative_tau"] + dt) * (raw_d - self.derivative)
        self.previous = error
        limit = cfg["max_rate_deg_s"]
        lower, upper = (-limit, 0.0) if error < 0 else (0.0, limit)
        delta_i = cfg["ki"] * error * dt
        integral = clamp(self.integral + delta_i, -limit, limit)
        pd = cfg["kp"] * error + cfg["kd"] * self.derivative
        raw = pd + integral
        if (lower <= raw <= upper or raw > upper and delta_i < 0
                or raw < lower and delta_i > 0):
            self.integral = integral
        # 积分和微分不会让偏左/偏上时发送增角指令，反方向同理。
        return clamp(pd + self.integral, lower, upper)


class TargetLock:
    def __init__(self, target="middle", config=None):
        if target not in TARGETS:
            raise ValueError(f"目标必须为 {', '.join(TARGETS)}")
        self.target = target
        self.config = validate_config(config)
        self.pids = {addr: PID(self.config) for addr in AXES}
        self.color = self.center = None

    def lost(self):
        # 保留所选目标身份；丢失后不重新挑选画面中央的其他颜色。
        for pid in self.pids.values():
            pid.reset()

    def step(self, candidates, size, dt):
        width, height = size
        if width <= 0 or height <= 0 or not math.isfinite(dt) or dt <= 0:
            raise ValueError("画面尺寸和 PID 时间间隔必须为正数")

        def center(row):
            x1, y1, x2, y2 = row["box"]
            return ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height))

        reference = self.center if self.center is not None else (0.5, 0.5)

        def distance(row):
            return math.dist(center(row), reference)

        if self.color is not None:
            choices = [row for row in candidates if row["value"] == self.color
                       and distance(row) <= self.config["match_distance_ratio"]]
        elif self.target in ("red", "green", "blue"):
            choices = [row for row in candidates if row["value"] == self.target]
        else:
            choices = candidates
        if not choices:
            self.lost()
            return None
        if self.color is None and self.target in ("left", "right"):
            ordered = sorted(choices, key=lambda row: center(row)[0])
            chosen = ordered[0 if self.target == "left" else -1]
        else:
            chosen = min(choices, key=distance)
        self.color, self.center = chosen["value"], center(chosen)
        errors = {1: (self.center[0] - 0.5) * width,
                  4: (self.center[1] - 0.5) * height}
        rates = {}
        for addr, half_size in ((1, width / 2), (4, height / 2)):
            if abs(errors[addr]) <= self.config["tolerance_px"]:
                self.pids[addr].reset()
                rates[addr] = 0.0
            else:
                rates[addr] = self.pids[addr].step(errors[addr] / half_size, dt)
        return {"target": chosen, "errors": errors, "rates": rates}


def read_angle(motor):
    status = motor.read_status()
    check_fault(status, motor.address)
    return finite(status["multi_turn_deg"], f"ID{motor.address} 编码器角度")


def hold(bus):
    """覆盖未完成的位置目标，让两轴保持当前反馈角；两轴分别尝试。"""
    def hold_axis(addr):
        motor = bus.motors[addr]
        motor.move(read_angle(motor), speed_rpm=0.01)

    cleanup(*[(f"ID{addr} 保持当前位置", lambda addr=addr: hold_axis(addr)) for addr in AXES])


def run(camera, bus, target="middle", *, config=None, duration=0.0, stop_event=None, log=print):
    """持续锁定直到取消或到达 duration；调用者负责关闭摄像头和总线。"""
    controller = TargetLock(target, config)
    cfg = controller.config
    duration = finite(duration, "运行时长")
    if duration < 0:
        raise ValueError("运行时长必须为非负数，0 表示持续运行")
    period = 1 / cfg["hz"]
    start = time.monotonic()
    index, previous_stamp, moving = 0, None, False
    previous_state, next_log = None, start
    try:
        check_cancel(stop_event)
        # 位置命令使能电机并持续保持目标；缺靶和退出时仍保留 ID2 的固定位置。
        bus.motors[2].move(cfg["motor2_angle_deg"], speed_rpm=ARM["speed_rpm"])
        while not duration or time.monotonic() - start < duration:
            check_cancel(stop_event)
            tick = time.monotonic()
            result = None
            try:
                frame, new_index, stamp = camera.next_frame(
                    after=index, timeout=max(period, 0.1), stop_event=stop_event)
            except TimeoutError:
                state = "等待摄像头新画面"
            else:
                # 只消费新帧，并在识别、读取电机反馈之后再次检查帧龄。
                fresh = new_index > index and -0.1 <= time.time() - stamp < VISION["frame_stale"]
                index = max(index, new_index)
                if fresh:
                    candidates = colored_targets(frame, cfg["min_area_ratio"])
                    fresh = -0.1 <= time.time() - stamp < VISION["frame_stale"]
                if fresh:
                    dt = period if previous_stamp is None else max(1e-4, stamp - previous_stamp)
                    result = controller.step(candidates, (frame.shape[1], frame.shape[0]), dt)
                    if result is not None:
                        current = {addr: read_angle(bus.motors[addr]) for addr in AXES}
                        check_cancel(stop_event)
                        if not -0.1 <= time.time() - stamp < VISION["frame_stale"]:
                            result = None
                state = "目标丢失，保持当前位置" if fresh else "画面重复或过期，保持当前位置"
            if result is None:
                controller.lost()
                previous_stamp = None
                if moving:
                    hold(bus)
                    moving = False
            else:
                commands = {}
                for addr in AXES:
                    rate = result["rates"][addr]
                    delta = clamp(rate * dt, -cfg["max_step_deg"], cfg["max_step_deg"])
                    angle = current[addr]
                    if rate:
                        # 直接按编码器步长生成可表示的目标，小修正不会被协议截成 0。
                        max_counts = math.floor(cfg["max_step_deg"] / ANGLE_SCALE)
                        counts = min(max_counts, max(1, math.ceil(abs(delta) / ANGLE_SCALE)))
                        sign = 1 if rate > 0 else -1
                        angle = (round(current[addr] / ANGLE_SCALE) + sign * counts) * ANGLE_SCALE
                    speed = max(0.01, abs(rate) / 6.0)
                    position_payload(angle, speed)
                    commands[addr] = (angle, speed)
                # 全部参数校验通过才发送；目标角基于真实反馈，避免累积未到位指令。
                for addr, command in commands.items():
                    check_cancel(stop_event)
                    bus.motors[addr].move(*command)
                    moving = True
                previous_stamp = stamp
                errors = result["errors"]
                state = (f"锁定 {controller.color} | 偏差 x={errors[1]:+.1f}px y={errors[4]:+.1f}px"
                         f" | ID1={commands[1][0]:.2f}° ID4={commands[4][0]:.2f}°")
            now = time.monotonic()
            # 状态切换立即打印，跟踪数值最多每 0.5 秒打印一次。
            status = "tracking" if result is not None else state
            if status != previous_state or now >= next_log:
                log(state)
                previous_state, next_log = status, now + 0.5
            wait_cancelable(max(0.0, period - (time.monotonic() - tick)), stop_event)
    finally:
        hold(bus)


def main(args=None):
    parser = argparse.ArgumentParser(description="使用两轴 PID 持续锁定摄像头中的红、绿、蓝标靶")
    parser.add_argument("target", nargs="?", choices=TARGETS, default="middle",
                        help="初次选靶的位置或颜色；默认选择最靠近画面中心的标靶")
    parser.add_argument("--camera", default=VISION["device"], help="摄像头编号或 /dev/video 路径")
    parser.add_argument("--port", default=None, help="机械臂 RS485 串口；默认读取 ARM 配置")
    for key, help_text in (("hz", "最高控制频率"), ("kp", "比例增益"), ("ki", "积分增益"),
                           ("kd", "微分增益"), ("tolerance_px", "中心容差（像素）"),
                           ("max_rate_deg_s", "最大角速度（度/秒）"), ("max_step_deg", "单次最大角度增量")):
        parser.add_argument("--" + key.replace("_", "-"), type=float, default=None, help=help_text)
    parser.add_argument("--duration", type=float, default=0.0, help="运行秒数；0 为持续锁定")
    opts = parser.parse_args(args)
    try:
        cfg = validate_config({key: getattr(opts, key) for key in TARGET_LOCK
                               if hasattr(opts, key) and getattr(opts, key) is not None})
        if not math.isfinite(opts.duration) or opts.duration < 0:
            raise ValueError("--duration 必须为有限非负数")
        arm_cfg = ArmConfig(**({"port": opts.port} if opts.port is not None else {}))
    except ValueError as exc:
        parser.error(str(exc))

    camera = bus = None
    try:
        camera = CameraStream(opts.camera)
        bus = MotorBus(arm_cfg.port, arm_cfg.baudrate, arm_cfg.serial_timeout,
                       latency_ms=arm_cfg.serial_latency_ms)
        print(f"锁靶选择：{opts.target}；PID Kp={cfg['kp']:g} Ki={cfg['ki']:g} Kd={cfg['kd']:g}。"
              f"ID2 保持 {cfg['motor2_angle_deg']:g}°。"
              "按 Ctrl+C 结束并保持当前角度。", flush=True)
        run(camera, bus, opts.target, config=cfg, duration=opts.duration,
            log=lambda message: print(message, flush=True))
    except (KeyboardInterrupt, MotionCancelled):
        print("锁靶已结束。", flush=True)
    except Exception as exc:
        print(f"锁靶失败：{exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        actions = []
        if bus is not None:
            actions.append(("机械臂串口关闭", lambda: bus.close(disable_motors=False)))
        if camera is not None:
            actions.append(("摄像头关闭", camera.close))
        cleanup(*actions, raise_errors=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
