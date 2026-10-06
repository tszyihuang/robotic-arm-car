"""底盘直接接口；一条电机串口、一条 IMU 串口供整趟比赛共用。"""
import math
import threading

from control import check_cancel, cleanup, wait_cancelable
from . import straight_pid, arc_turn, calibrate_position, vision_align, vision_straight
from config import BASE, SENSOR, ARM, POSITION, VISION


class Base:
    def __init__(self, *, stop_event=None):
        self.config = dict(BASE)
        self.stop_event = stop_event or threading.Event()
        self.motor = None
        self.imu = None
        self._imu_checked = False

    def _connect(self):
        check_cancel(self.stop_event)
        if self.motor is None:
            from .motor import Motor
            self.motor = Motor(self.config["motor_port"], stop_event=self.stop_event)
        if not self._imu_checked:
            from sensor.imu import ImuLink, find_imu_port
            skip = [self.motor.port, ARM["port"], ARM["gripper_port"]]
            port = SENSOR["imu_port"] or find_imu_port(skip=skip, stop_event=self.stop_event)
            if port:
                self.imu = ImuLink(port)
                self.imu.send_rate(SENSOR["hz"])
                self.imu.start()
            else:
                print("未找到 IMU；转弯使用编码器，倒车靠坎将拒绝执行。")
            self._imu_checked = True
        check_cancel(self.stop_event)

    @staticmethod
    def _require_success(result):
        if not result["ok"]:
            raise RuntimeError(result.get("fault") or result.get("reason")
                               or result.get("vision", {}).get("reason")
                               or result.get("summary") or f"底盘未到位：{result}")
        return result

    def straight(self, distance, speed=None):
        speed = self.config["speed"] if speed is None else speed
        if not math.isfinite(distance) or distance == 0 or not 0 < speed <= 1000:
            raise ValueError("直走距离必须为有限非零数，速度为 (0, 1000] mm/s")
        self._connect()
        measured, info = straight_pid.straight(self.motor, distance * 1000, speed,
                                             stop_event=self.stop_event)
        tolerance = max(0.03, 0.03 * abs(distance))
        return self._require_success({"ok": not info["reason"] and
                                      abs(measured / 1000 - abs(distance)) <= tolerance,
                                      "distance_m": measured / 1000 * (1 if distance > 0 else -1), **info})

    def turn(self, degrees, radius=None):
        radius = self.config["turn_radius"] if radius is None else radius
        speed = self.config["turn_speed"] / 1000
        limit = self.config["turn_spin_speed"] / 1000 if radius == 0 else None
        arc_turn.make_plan(radius, degrees, speed, limit, self.config["turn_track_width"])
        self._connect()
        imu = self.imu if self.imu is not None and self.imu.has_data() else None
        result = arc_turn.turn_with_radius(radius, degrees, board=self.motor, imu=imu,
                                          cruise=speed, max_speed=limit,
                                          timeout_s=self.config["turn_timeout"],
                                          track_width_m=self.config["turn_track_width"],
                                          stop_event=self.stop_event)
        return self._require_success(result)

    def calibrate_position(self):
        self._connect()
        if self.imu is None:
            raise RuntimeError("倒车靠坎需要 IMU 原始加速度反馈，请检查 config.py 的 SENSOR")
        self.imu.impact_threshold = POSITION["impact_threshold"]
        wait_cancelable(POSITION["boot_wait"], self.stop_event)
        old_timeout = self.motor.default_timeout
        self.motor.default_timeout = 0.0  # 倒车时每拍立即检查撞击，不等待编码器。
        try:
            result = calibrate_position.calibrate_position(
                self.motor, self.imu, speed=self.config["speed"],
                max_distance_mm=POSITION["max_distance_m"] * 1000,
                timeout=POSITION["timeout"], stop_event=self.stop_event)
        finally:
            self.motor.default_timeout = old_timeout
        return self._require_success(result)

    def _heading(self, vision):
        self._connect()
        vision.enable_boundary()
        if not vision.wait_ready(VISION["model_wait"], stop_event=self.stop_event):
            check_cancel(self.stop_event)
            raise RuntimeError("视觉模型或新边界未就绪")
        cfg = vision_straight.VisionCfg(speed=self.config["speed"], track_mm=straight_pid.TRACK_MM)
        return vision_straight.VisionHeading(cfg, vision)

    def align(self, vision):
        heading = self._heading(vision)
        source = self.config["align_rate_src"]
        if source not in ("auto", "gyro", "gap"):
            raise ValueError("align_rate_src 只能是 auto、gyro 或 gap")
        imu = self.imu if source != "gap" and self.imu is not None and self.imu.has_data() else None
        if source == "gyro" and imu is None:
            raise RuntimeError("视觉对正需要新鲜的 IMU 数据")
        heading.cfg.rate_src = "gyro" if imu else "gap"
        result = vision_align.vision_align(self.motor, heading, vision_align.AlignCfg(),
                                          imu=imu, stop_event=self.stop_event)
        return self._require_success(result)

    def vision_straight(self, distance, vision):
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError("视觉直走距离必须为有限正数")
        heading = self._heading(vision)
        if heading.wait_track(stop_event=self.stop_event) is None:
            raise RuntimeError("出发前未拿到可用跑道中心线")
        measured, info = vision_straight.vision_straight(
            self.motor, heading, goal_mm=distance * 1000, stop_event=self.stop_event)
        return self._require_success({"ok": not info["reason"] and
                                      abs(measured / 1000 - distance) <= max(0.03, 0.03 * distance),
                                      "distance_m": measured / 1000, "vision": info})

    def stop(self):
        if self.motor is not None:
            self.motor.stop()

    def close(self):
        motor, imu = self.motor, self.imu
        self.motor = self.imu = None
        actions = []
        if motor is not None:
            actions.append(("底盘连接", motor.close))
        if imu is not None:
            actions.append(("IMU 连接", imu.close))
        cleanup(*actions)
