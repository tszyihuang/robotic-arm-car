"""跑道尽头的位置 PID：确认的新画面 → 像素误差 → 前进目标速度。"""
import math

from config import GO_AHEAD_POSITION, VISION
from .control import clamp


class EndpointApproach:
    def __init__(self, endpoint, speed, *, config=None, log=print):
        self.config = {**GO_AHEAD_POSITION, **({} if config is None else config)}
        for name in ("kp", "decel"):
            if not math.isfinite(self.config[name]) or self.config[name] <= 0:
                raise ValueError(f"GO_AHEAD_POSITION[{name!r}] 必须为有限正数")
        for name in ("ki", "kd", "derivative_tau", "min_speed", "tolerance_px"):
            if not math.isfinite(self.config[name]) or self.config[name] < 0:
                raise ValueError(f"GO_AHEAD_POSITION[{name!r}] 必须为有限非负数")
        if not math.isfinite(speed) or not 0 < speed <= 1000:
            raise ValueError("跑道尽头巡航速度必须为 (0, 1000] mm/s")
        self.endpoint, self.speed, self.log = endpoint, speed, log
        self.target_speed = speed
        self.error_px = None
        self.reached = False
        self.integral = self.derivative = 0.0
        self._stamp = None

    def update(self, now):
        previous_stamp = self.endpoint.capture_stamp
        self.endpoint(now)  # 每轮检查摄像头和取消；达到停车行不直接结束运动。
        result = self.endpoint.result
        if self.endpoint.capture_stamp > previous_stamp and result["valid"]:
            stamp = self.endpoint.capture_stamp
            error = self.endpoint.stop_y - result["point"][1]
            dt = 0.0 if self._stamp is None else max(1e-4, stamp - self._stamp)
            if self.error_px is not None:
                raw_d = (error - self.error_px) / dt
                tau = self.config["derivative_tau"]
                self.derivative += dt / (tau + dt) * (raw_d - self.derivative)
            self.error_px, self._stamp = error, stamp
            if not self.reached and error <= self.config["tolerance_px"]:
                self.reached = True
                self.log(f"  跑道尽头 y={result['point'][1]}px，位置 PID 开始收尾减速")
            if self.reached:
                self.target_speed = self.integral = 0.0
                return
            pd = self.config["kp"] * error + self.config["kd"] * self.derivative
            delta_i = self.config["ki"] * error * dt
            integral = clamp(self.integral + delta_i, 0.0, self.speed)
            # 巡航限幅时不积累积分，避免接近终点仍维持高速。
            if 0.0 <= pd + integral <= self.speed:
                self.integral = integral
            self.target_speed = clamp(pd + self.integral,
                                      min(self.config["min_speed"], self.speed), self.speed)
        if not self.reached and self._stamp is not None and now - self._stamp >= VISION["frame_stale"]:
            raise RuntimeError("跑道尽头位置持续丢失，已停车")
