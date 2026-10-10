"""YOLO 锁定彩色标靶：ID2 保持固定角度，横向、纵向 PID 分别控制 ID1、ID4。"""
import argparse
import math
from pathlib import Path
import signal
import sys
import threading
import time

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm.config import ArmConfig, JOINT_IDS, finite
from arm.motor import ANGLE_SCALE, MotorBus, check_fault, position_payload
from base.control import MotionCancelled, check_cancel, clamp, cleanup, wait_cancelable
from config import ARM, TARGET_LOCK, VISION
from debug.camera_web import local_addresses
from tasks.target_lock_web import AimPoint, DEFAULT_AIM_FILE, TargetLockWeb, validate_aim
from vision.camera import CameraStream
from vision.targets import candidates_from_detections

TARGETS = ("middle", "left", "right", "red", "green", "blue")
AXES = (1, 4)
TASK_TOLERANCE_PX = 8.0
TASK_STABLE_SECONDS = 2.0
WAIT_JOINTS = {1: -90.0, 2: 0.0, 3: 160.0, 4: 4.0}


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
    if not cfg["stable_frames"].is_integer():
        raise ValueError("stable_frames 必须为正整数")
    cfg["stable_frames"] = int(cfg["stable_frames"])
    for key in ("match_distance_ratio", "stable_distance_ratio"):
        if not 0 < cfg[key] < 1:
            raise ValueError(f"{key} 必须在 (0, 1) 内")
    if cfg["max_step_deg"] < ANGLE_SCALE:
        raise ValueError(f"max_step_deg 不能小于电机编码器精度 {ANGLE_SCALE:g}°")
    position_payload(0.0, cfg["max_rate_deg_s"] / 6.0)
    position_payload(cfg["motor2_angle_deg"], ARM["speed_rpm"])
    return cfg


def load_predictor(*, device=None, threshold=None):
    """独立入口按需加载项目物体 YOLO；任务入口直接复用视觉后台。"""
    from vision.yolo_objects import ObjectPredictor
    return ObjectPredictor(VISION["objects_weights"],
                           device=VISION["infer_device"] if device is None else device,
                           fp16=VISION["fp16"], threshold=threshold, args_path=VISION["objects_args"])


class TargetAcquisition:
    """只在三色唯一、排列和中心位置连续稳定时允许开始瞄准。"""

    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.count = 0
        self.signature = self.centers = self.size = self.stamp = None

    def step(self, candidates, size, stamp):
        if len(candidates) != 3 or {row["value"] for row in candidates} != {"red", "green", "blue"}:
            self.reset()
            return None
        width, height = size
        ordered = sorted(candidates, key=lambda row: row["box"][0] + row["box"][2])
        centers = tuple(((row["box"][0] + row["box"][2]) / (2 * width),
                         (row["box"][1] + row["box"][3]) / (2 * height)) for row in ordered)
        if any((right[0] - left[0]) * width < 1 for left, right in zip(centers, centers[1:])):
            self.reset()
            return None
        signature = tuple(row["value"] for row in ordered)
        stable = (signature == self.signature and size == self.size
                  and 0 < stamp - self.stamp < VISION["frame_stale"]
                  and all(math.dist(current, first) <= self.config["stable_distance_ratio"]
                          for current, first in zip(centers, self.centers)))
        if stable:
            self.count += 1
        else:
            self.count = 1
            self.signature, self.centers, self.size = signature, centers, size
        self.stamp = stamp
        return ordered if self.count >= self.config["stable_frames"] else None


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
        self.aim = (0.5, 0.5)

    def lost(self):
        # 保留所选目标身份；丢失后不重新挑选画面中央的其他颜色。
        for pid in self.pids.values():
            pid.reset()

    def step(self, candidates, size, dt, aim=(0.5, 0.5)):
        width, height = size
        if width <= 0 or height <= 0 or not math.isfinite(dt) or dt <= 0:
            raise ValueError("画面尺寸和 PID 时间间隔必须为正数")
        aim = validate_aim(*aim)
        if aim != self.aim:
            # 网页修改设定点时清除旧积分和微分，避免人为跳变产生瞬时尖峰。
            self.lost()
            self.aim = aim

        def center(row):
            x1, y1, x2, y2 = row["box"]
            return ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height))

        reference = self.center if self.center is not None else (0.5, 0.5)

        def distance(row):
            return math.dist(center(row), reference)

        if self.color is not None:
            choices = [row for row in candidates if row["value"] == self.color
                       and (self.center is None or distance(row) <= self.config["match_distance_ratio"])]
        elif self.target in ("red", "green", "blue"):
            choices = [row for row in candidates if row["value"] == self.target]
        else:
            choices = candidates
        if not choices:
            self.lost()
            return None
        if self.color is None and (self.target in ("left", "right")
                                   or self.target == "middle" and len(choices) == 3):
            ordered = sorted(choices, key=lambda row: center(row)[0])
            chosen = ordered[{"left": 0, "middle": 1, "right": -1}[self.target]]
        else:
            chosen = min(choices, key=distance)
        self.color, self.center = chosen["value"], center(chosen)
        x1, y1, x2, y2 = chosen["box"]
        errors = {1: (x1 + x2) / 2 - aim[0] * width,
                  4: (y1 + y2) / 2 - aim[1] * height}
        rates = {}
        for addr, half_size in ((1, width / 2), (4, height / 2)):
            if abs(errors[addr]) <= self.config["tolerance_px"]:
                self.pids[addr].reset()
                rates[addr] = 0.0
            else:
                rates[addr] = self.pids[addr].step(errors[addr] / half_size, dt)
        return {"target": chosen, "errors": errors, "rates": rates, "size": size}


def read_angle(motor):
    status = motor.read_status()
    check_fault(status, motor.address)
    return finite(status["multi_turn_deg"], f"ID{motor.address} 编码器角度")


def hold(bus, addresses=AXES):
    """覆盖未完成的位置目标，让指定各轴保持当前反馈角；各轴分别尝试。"""
    def hold_axis(addr):
        motor = bus.motors[addr]
        motor.move(read_angle(motor), speed_rpm=0.01)

    cleanup(*[(f"ID{addr} 保持当前位置", lambda addr=addr: hold_axis(addr)) for addr in addresses])


class RecoveryPose:
    """发送一次四轴复位命令，逐拍确认到位；等待期间不会被 PID 覆盖。"""

    def __init__(self, bus, config=None, encoder_zero=None, stop_event=None):
        self.bus = bus
        self.config = config or ArmConfig()
        zero = encoder_zero if encoder_zero is not None else self.config.encoder_zero_deg
        if zero is None:
            # 无已有校准时，启动姿态须是 WAIT_JOINTS，记录该姿态的真实编码器位置。
            self.targets = {}
            for addr in JOINT_IDS:
                check_cancel(stop_event)
                self.targets[addr] = read_angle(bus.motors[addr])
        else:
            self.targets = {addr: zero[addr] + self.config.joint_signs[addr] *
                            (WAIT_JOINTS[addr] - self.config.joint_offsets_deg[addr])
                            for addr in JOINT_IDS}
        for angle in self.targets.values():
            position_payload(angle, self.config.speed_rpm)
        self.active = False

    def start(self, stop_event):
        self.started = time.monotonic()
        self.active = True
        for addr, angle in self.targets.items():
            check_cancel(stop_event)
            self.bus.motors[addr].move(angle, speed_rpm=self.config.speed_rpm)

    def arrived(self, stop_event):
        current = {}
        for addr in JOINT_IDS:
            check_cancel(stop_event)
            current[addr] = read_angle(self.bus.motors[addr])
        check_cancel(stop_event)
        if all(abs(current[addr] - angle) <= self.config.arrival_tolerance_deg
               for addr, angle in self.targets.items()):
            self.active = False
            return True
        if time.monotonic() - self.started >= self.config.move_timeout:
            raise TimeoutError("机械臂复位超时：未到达 arm-move -90 0 160 4")
        return False


def run(camera, bus, target="middle", *, config=None, duration=0.0, stop_event=None, log=print,
        aim=None, on_update=None, complete_after=0.0, complete_tolerance_px=TASK_TOLERANCE_PX,
        arm_config=None, encoder_zero=None, predictor=None, candidate_source=None):
    """三靶稳定后瞄准，丢靶复位再确认；调用者负责关闭摄像头和总线。"""
    controller = TargetLock(target, config)
    cfg = controller.config
    duration = finite(duration, "运行时长")
    if duration < 0:
        raise ValueError("运行时长必须为非负数，0 表示持续运行")
    complete_after = finite(complete_after, "到位保持时间")
    complete_tolerance_px = finite(complete_tolerance_px, "完成误差容差")
    if complete_after < 0 or complete_tolerance_px < 0:
        raise ValueError("到位保持时间和完成误差容差必须为非负数")
    period = 1 / cfg["hz"]
    start = time.monotonic()
    capture_start = time.time()
    acquisition = TargetAcquisition(cfg)
    index, previous_stamp, tracking = 0, None, False
    previous_state, next_log = None, start
    aligned_since = aligned_stamp = aligned_aim = None
    recovery = None
    try:
        check_cancel(stop_event)
        if candidate_source is None and predictor is None:
            predictor = load_predictor()
        recovery = RecoveryPose(bus, arm_config, encoder_zero, stop_event)
        if encoder_zero is None and recovery.config.encoder_zero_deg is None:
            log("无已有编码器零点：记录启动姿态作为 arm-move -90 0 160 4 的复位位置")
        while not duration or time.monotonic() - start < duration:
            check_cancel(stop_event)
            tick = time.monotonic()
            result = None
            if recovery.active:
                if recovery.arrived(stop_event):
                    # 到位前采集的画面不能用于下一轮确认。
                    capture_start = time.time()
                    state = "复位完成，等待红绿蓝三个标靶稳定出现"
                else:
                    state = "正在复位到 arm-move -90 0 160 4"
            else:
                try:
                    if candidate_source is None:
                        frame, new_index, stamp = camera.next_frame(
                            after=index, timeout=max(period, 0.1), stop_event=stop_event)
                        size = (frame.shape[1], frame.shape[0])
                    else:
                        # YOLO 帧率可能低于控制频率；等待新推理至视觉新鲜度上限。
                        timeout = VISION["frame_stale"]
                        if duration:
                            timeout = min(timeout, max(0.0, duration - (time.monotonic() - start)))
                        sample = candidate_source(after=index, timeout=timeout, stop_event=stop_event)
                        if sample is None:
                            raise TimeoutError("等待 YOLO 标靶新结果")
                        candidates, size = sample["candidates"], sample["size"]
                        new_index, stamp = sample["frame_index"], sample["capture_stamp"]
                except TimeoutError:
                    acquisition.reset()
                    state = "等待 YOLO 标靶新结果"
                else:
                    # 只消费本轮开始后采集的新帧，检查采集时间递增及识别后的帧龄。
                    fresh = (new_index > index and stamp >= capture_start
                             and -0.1 <= time.time() - stamp < VISION["frame_stale"]
                             and (previous_stamp is None or stamp > previous_stamp))
                    index = max(index, new_index)
                    if fresh:
                        if candidate_source is None:
                            info = predictor.predict(frame)
                            candidates = candidates_from_detections(info["detections"], "target")
                        check_cancel(stop_event)
                        fresh = -0.1 <= time.time() - stamp < VISION["frame_stale"]
                    state = "目标丢失" if fresh else "画面重复或过期"
                    if fresh:
                        ready = candidates if tracking else acquisition.step(candidates, size, stamp)
                        if ready is None:
                            state = (f"等待红绿蓝三个标靶稳定出现 "
                                     f"({acquisition.count}/{cfg['stable_frames']})")
                        else:
                            dt = period if previous_stamp is None else max(1e-4, stamp - previous_stamp)
                            result = controller.step(ready, size, dt,
                                                     aim=aim.point() if aim is not None else (0.5, 0.5))
                            if result is not None:
                                if not tracking:
                                    check_cancel(stop_event)
                                    tracking = True
                                    bus.motors[2].move(cfg["motor2_angle_deg"],
                                                       speed_rpm=recovery.config.speed_rpm)
                                current = {addr: read_angle(bus.motors[addr]) for addr in AXES}
                                check_cancel(stop_event)
                                if not -0.1 <= time.time() - stamp < VISION["frame_stale"]:
                                    result = None
                                    state = "画面过期"
                    else:
                        acquisition.reset()
            if result is None:
                controller.lost()
                previous_stamp = None
                if tracking:
                    # 复位会改变画面位置；保留锁定颜色，清除旧中心和稳定计数。
                    controller.center = None
                    acquisition.reset()
                    recovery.start(stop_event)
                    tracking = False
                    state += "，复位到 arm-move -90 0 160 4 后重新等待"
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
                previous_stamp = stamp
                errors = result["errors"]
                state = (f"锁定 {controller.color} | 偏差 x={errors[1]:+.1f}px y={errors[4]:+.1f}px"
                         f" | ID1={commands[1][0]:.2f}° ID4={commands[4][0]:.2f}°")
            now = time.monotonic()
            if on_update is not None:
                on_update(state, result)
            # 状态切换立即打印，跟踪数值最多每 0.5 秒打印一次。
            status = "tracking" if result is not None else state
            if status != previous_state or now >= next_log:
                log(state)
                previous_state, next_log = status, now + 0.5
            if complete_after:
                in_band = (result is not None
                           and all(abs(error) <= complete_tolerance_px for error in result["errors"].values())
                           and stamp >= capture_start
                           and -0.1 <= time.time() - stamp < VISION["frame_stale"]
                           and (aim is None or aim.point() == controller.aim))
                if in_band:
                    # 用新帧采集时间确认连续到位；长时间无新帧或瞄准点变化均重新计时。
                    if (aligned_since is None or aligned_aim != controller.aim
                            or not 0 < stamp - aligned_stamp < VISION["frame_stale"]):
                        aligned_since = stamp
                    aligned_stamp, aligned_aim = stamp, controller.aim
                    stable_seconds = stamp - aligned_since
                    if stable_seconds >= complete_after:
                        check_cancel(stop_event)
                        log(f"锁靶完成：{controller.color}，横纵误差均 ≤ {complete_tolerance_px:g}px，"
                            f"连续保持 {stable_seconds:.2f}s")
                        return {"ok": True, "color": controller.color,
                                "errors": dict(result["errors"]), "stable_seconds": stable_seconds}
                else:
                    aligned_since = aligned_stamp = aligned_aim = None
            wait_cancelable(max(0.0, period - (time.monotonic() - tick)), stop_event)
        if complete_after:
            raise TimeoutError(f"锁靶超时：未在 {duration:g}s 内达到横纵误差均 ≤ "
                               f"{complete_tolerance_px:g}px 并连续保持 {complete_after:g}s")
    finally:
        hold(bus, JOINT_IDS if recovery is not None and recovery.active else AXES)


def run_task(arm, vision, target="middle", *, config=None, duration=0.0,
             stop_event=None, log=print, aim_file=DEFAULT_AIM_FILE):
    """任务表入口：复用机械臂和视觉会话，退出只关闭本次校正网页。"""
    if target not in TARGETS:
        raise ValueError(f"目标必须为 {', '.join(TARGETS)}")
    cfg = validate_config(config)
    duration = finite(duration, "运行时长")
    if duration < 0:
        raise ValueError("锁靶超时必须为非负数，0 表示不限时")
    stop_event = stop_event if stop_event is not None else getattr(arm, "stop_event", None)
    check_cancel(stop_event)
    if getattr(arm, "dry_run", False):
        end = (f"横纵误差均 ≤ {TASK_TOLERANCE_PX:g}px 连续保持 {TASK_STABLE_SECONDS:g} 秒后完成，"
               "继续下一条指令")
        if duration:
            end += f"；超时 {duration:g} 秒则停止任务"
        log(f"lock_target.run(target={target!r})  # {end}；"
            f"使用项目 YOLO，先连续 {cfg['stable_frames']} 帧确认三色标靶，"
            "丢靶复位 arm-move -90 0 160 4；"
            f"ID2 保持 {cfg['motor2_angle_deg']:g}°，ID1/ID4 PID 跟踪，开启校正网页")
        return None

    web = bus = None
    try:
        aim = AimPoint(aim_file)
        web = TargetLockWeb(aim)
        check_cancel(stop_event)
        camera = vision.start_camera()
        vision.enable_objects()
        bus = arm._ensure_bus()
        check_cancel(stop_event)
        web.start(camera)
        log(f"YOLO 锁靶选择：{target}；ID2 保持 {cfg['motor2_angle_deg']:g}°；"
            f"瞄准点矫正网页：http://127.0.0.1:{web.port}")
        return run(camera, bus, target, config=cfg, duration=duration,
                   stop_event=stop_event, aim=aim, on_update=web.update, log=log,
                   complete_after=TASK_STABLE_SECONDS, complete_tolerance_px=TASK_TOLERANCE_PX,
                   arm_config=arm.config,
                   encoder_zero=arm._arm.encoder_zero_deg if arm.calibrated else None,
                   candidate_source=vision.wait_target_layout)
    except (KeyboardInterrupt, MotionCancelled):
        if bus is not None:
            cleanup(("锁靶取消，机械臂 ID1–4 失能", bus.disable_all), raise_errors=False)
        raise
    finally:
        if web is not None:
            cleanup(("瞄准点矫正网页关闭", web.close), raise_errors=False)


def main(args=None):
    parser = argparse.ArgumentParser(description="使用项目 YOLO 识别红、绿、蓝标靶并以两轴 PID 瞄准")
    parser.add_argument("target", nargs="?", choices=TARGETS, default="middle",
                        help="三色标靶稳定后按左右位置或颜色选靶；默认选择中间标靶")
    parser.add_argument("--camera", default=VISION["device"], help="摄像头编号或 /dev/video 路径")
    parser.add_argument("--port", default=None, help="机械臂 RS485 串口；默认读取 ARM 配置")
    parser.add_argument("--device", default=VISION["infer_device"], help="YOLO 推理设备，如 cuda:0 或 cpu")
    parser.add_argument("--conf", type=float, default=None, help="YOLO 置信度阈值；默认读取模型配置")
    parser.add_argument("--web-host", default="0.0.0.0", help="瞄准点矫正网页监听地址")
    parser.add_argument("--web-port", type=int, default=8080, help="瞄准点矫正网页端口，默认 8080")
    parser.add_argument("--aim-file", type=Path, default=DEFAULT_AIM_FILE,
                        help="瞄准点校正文件；启动时读取，网页点击保存后写入")
    for key, help_text in (("hz", "最高控制频率"), ("kp", "比例增益"), ("ki", "积分增益"),
                           ("kd", "微分增益"), ("tolerance_px", "瞄准点容差（像素）"),
                           ("max_rate_deg_s", "最大角速度（度/秒）"), ("max_step_deg", "单次最大角度增量"),
                           ("stable_frames", "开始瞄准前连续确认三色标靶的帧数"),
                           ("stable_distance_ratio", "稳定确认时允许的归一化中心漂移")):
        parser.add_argument("--" + key.replace("_", "-"), type=float, default=None, help=help_text)
    parser.add_argument("--duration", type=float, default=0.0, help="运行秒数；0 为持续锁定")
    opts = parser.parse_args(args)
    try:
        cfg = validate_config({key: getattr(opts, key) for key in TARGET_LOCK
                               if hasattr(opts, key) and getattr(opts, key) is not None})
        if not math.isfinite(opts.duration) or opts.duration < 0:
            raise ValueError("--duration 必须为有限非负数")
        if opts.conf is not None and (not math.isfinite(opts.conf) or not 0 < opts.conf < 1):
            raise ValueError("--conf 必须在 (0, 1) 内")
        if not 1 <= opts.web_port <= 65535:
            raise ValueError("--web-port 必须为 1..65535")
        arm_cfg = ArmConfig(**({"port": opts.port} if opts.port is not None else {}))
    except ValueError as exc:
        parser.error(str(exc))

    camera = bus = web = None
    stop_event = threading.Event()
    disable_on_exit = False

    def interrupt(signum, frame):
        # 清理期间再次按 Ctrl+C，不中断正在发送的失能命令。
        if stop_event.is_set():
            return
        stop_event.set()
        raise KeyboardInterrupt

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        aim = AimPoint(opts.aim_file)
        web = TargetLockWeb(aim, opts.web_host, opts.web_port)
        predictor = load_predictor(device=opts.device, threshold=opts.conf)
        camera = CameraStream(opts.camera)
        bus = MotorBus(arm_cfg.port, arm_cfg.baudrate, arm_cfg.serial_timeout,
                       latency_ms=arm_cfg.serial_latency_ms)
        web.start(camera)
        print(f"YOLO 锁靶选择：{opts.target}；模型 {VISION['objects_weights']}；"
              f"PID Kp={cfg['kp']:g} Ki={cfg['ki']:g} Kd={cfg['kd']:g}。"
              f"ID2 保持 {cfg['motor2_angle_deg']:g}°。"
              "按 Ctrl+C 结束并失能机械臂 ID1–4。", flush=True)
        web_address = "127.0.0.1" if opts.web_host == "0.0.0.0" else opts.web_host
        print(f"瞄准点矫正网页：http://{web_address}:{web.port}", flush=True)
        if opts.web_host == "0.0.0.0":
            for address in local_addresses():
                print(f"局域网：http://{address}:{web.port}", flush=True)
        run(camera, bus, opts.target, config=cfg, duration=opts.duration,
            stop_event=stop_event, aim=aim, on_update=web.update,
            arm_config=arm_cfg, predictor=predictor,
            log=lambda message: print(message, flush=True))
    except (KeyboardInterrupt, MotionCancelled):
        disable_on_exit = True
        print("锁靶已结束，正在失能机械臂 ID1–4。", flush=True)
    except Exception as exc:
        print(f"锁靶失败：{exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        stop_event.set()
        actions = []
        if bus is not None:
            # 优先失能并关闭总线，然后停止网页和摄像头。
            actions.append(("机械臂串口关闭", lambda: bus.close(disable_motors=disable_on_exit)))
        if web is not None:
            actions.append(("瞄准点矫正网页关闭", web.close))
        if camera is not None:
            actions.append(("摄像头关闭", camera.close))
        cleanup(*actions, raise_errors=False)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
