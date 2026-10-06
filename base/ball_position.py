"""小球位置 PID：图像横向误差控制前后速度，编码器保持直行。"""
import math
import time

from config import BALL_POSITION, BASE, VISION
from vision.targets import ordered_candidates
from . import straight_pid as straight
from .control import check_cancel, clamp, cleanup
from .feedback import WheelOdometry


def validate_config(config=None):
    cfg = {**BALL_POSITION, **({} if config is None else config)}
    for name in ("speed", "tolerance_px", "kp", "accel", "loop_hz", "speed_tolerance",
                 "lost_timeout", "max_distance_m", "timeout"):
        if not math.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(f"BALL_POSITION[{name!r}] 必须为有限正数")
    for name in ("ki", "kd", "derivative_tau", "settle"):
        if not math.isfinite(cfg[name]) or cfg[name] < 0:
            raise ValueError(f"BALL_POSITION[{name!r}] 必须为有限非负数")
    if cfg["speed"] > straight.SPEED_LIMIT:
        raise ValueError(f"小球校准速度不能超过 {straight.SPEED_LIMIT:g} mm/s")
    if type(cfg["stable_frames"]) is not int or cfg["stable_frames"] < 1:
        raise ValueError("小球校准 stable_frames 必须为正整数")
    return cfg


class PositionPID:
    """按图像采集间隔更新 PID；微分低通、积分限幅和条件积分。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.integral = self.derivative = 0.0
        self.error = self.stamp = None

    def step(self, error, stamp):
        cfg = self.cfg
        dt = 0.0 if self.stamp is None else max(1e-4, stamp - self.stamp)
        if self.error is not None:
            if error * self.error < 0:
                self.integral = 0.0
            raw_d = (error - self.error) / dt
            self.derivative += dt / (cfg["derivative_tau"] + dt) * (raw_d - self.derivative)
        self.error, self.stamp = error, stamp
        lower, upper = (-cfg["speed"], 0.0) if error < 0 else (0.0, cfg["speed"])
        delta_i = cfg["ki"] * error * dt
        integral = clamp(self.integral + delta_i, -cfg["speed"], cfg["speed"])
        pd = cfg["kp"] * error + cfg["kd"] * self.derivative
        raw = pd + integral
        # 输出饱和时不继续积累；允许反向积分把输出拉回有效范围。
        if (lower <= raw <= upper or raw > upper and delta_i < 0
                or raw < lower and delta_i > 0):
            self.integral = integral
        return clamp(pd + self.integral, lower, upper)


def middle_ball(layout):
    size = layout.get("size")
    if (not isinstance(size, (list, tuple)) or len(size) != 2
            or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in size)):
        raise ValueError("小球位置校准缺少有效的原图尺寸，已停车")
    balls = layout["candidates"]
    width = size[0]
    if any(not math.isfinite(ball["center_x"]) or not 0 <= ball["center_x"] < width for ball in balls):
        raise ValueError("小球横坐标超出画面，已停车")
    try:
        ordered = ordered_candidates(balls)
    except ValueError:
        return None, width
    return ordered[1], width


def calibrate_ball_position(board, vision, *, config=None, stop_event=None, log=print):
    """连续控制，不调用定距离 straight()；所有退出路径都发送零速度。"""
    try:
        check_cancel(stop_event)
        cfg = validate_config(config)
        board.stop()
        started = time.monotonic()
        deadline = started + cfg["timeout"]
        # 加载与首次观察期间保持静止；后续只读取后台缓存，不阻塞控制环。
        first = vision.observe_ball_layout(timeout=min(cfg["timeout"], VISION["observe_timeout"]))
        check_cancel(stop_event)
        wheels = WheelOdometry.read_origin(board, stop_event)
        position = PositionPID(cfg)
        head = straight.PID(straight.KP_GAP, straight.KI_GAP, straight.KD_GAP, straight.TRIM_MAX)
        speed_l = straight.PID(straight.KP_SPD, straight.KI_SPD, 0, straight.CORR_MAX)
        speed_r = straight.PID(straight.KP_SPD, straight.KI_SPD, 0, straight.CORR_MAX)
        last_tick = last_totals = last_steps = waiting_since = time.monotonic()
        last_index = last_stamp = None
        last_valid_stamp = None
        color = None
        centered = 0
        in_band = usable = False
        settled_since = None
        previous_l = previous_r = travel = distance = gap = 0.0
        command_l = command_r = target = error = 0.0
        center_x = center_line = 0.0
        last_visual_state = None
        log(f"开始小球位置 PID 校准：最大速度 {cfg['speed']:g}mm/s，"
            f"中心容差 ±{cfg['tolerance_px']:g}px")

        while True:
            check_cancel(stop_event)
            tick = time.monotonic()
            if tick >= deadline:
                raise TimeoutError("小球位置校准超时，已停车")
            dt = clamp(tick - last_tick, 1e-4, 0.05)
            last_tick = tick
            totals, increments = board.feedback(0.0)
            check_cancel(stop_event)
            if totals is not None:
                last_totals = tick
                left, right = wheels.travel(totals)
                distance, gap = (left + right) / 2, left - right
                travel += (abs(left - previous_l) + abs(right - previous_r)) / 2
                previous_l, previous_r = left, right
            if increments is not None:
                last_steps = tick
            v_l, v_r = wheels.speed(increments)
            if tick - last_totals >= BASE["feedback_stale"] or tick - last_steps >= BASE["feedback_stale"]:
                raise RuntimeError("小球位置校准编码器里程或轮速断流，已停车")
            if travel >= cfg["max_distance_m"] * 1000:
                raise RuntimeError("小球位置校准达到累计移动距离上限，已停车")
            if abs(gap) > straight.GAP_DEV_MAX:
                raise RuntimeError("小球位置校准左右里程差过大，已停车")

            new_layout = False
            layout = first if first is not None else vision.ball_layout_sample()
            first = None
            if layout is not None:
                index, stamp = layout["frame_index"], layout["capture_stamp"]
                age = time.time() - stamp
                if (index != last_index and (last_stamp is None or stamp > last_stamp)
                        and -0.1 <= age < min(cfg["lost_timeout"], VISION["frame_stale"])):
                    last_index, last_stamp = index, stamp
                    new_layout = True
                    middle, width = middle_ball(layout)
                    usable = middle is not None
                    if usable:
                        if color is not None and middle["value"] != color:
                            raise RuntimeError("中间小球身份发生变化，已停止位置校准")
                        color = middle["value"]
                        center_x, center_line = middle["center_x"], width / 2
                        error = center_x - center_line
                        last_valid_stamp = stamp
                        in_band = abs(error) <= cfg["tolerance_px"]
                        centered = centered + 1 if in_band else 0
                        if in_band:
                            target = 0.0
                            position.reset()
                        else:
                            target = position.step(error, stamp)
                    else:
                        centered = 0
                        position.reset()
            stale = last_valid_stamp is None or time.time() - last_valid_stamp >= min(
                cfg["lost_timeout"], VISION["frame_stale"])
            visual_state = "缺球" if not usable else "画面过期" if stale else "有效"
            if not usable or stale:
                # 新帧缺球立即断速；重复或过期帧不能延长视觉有效期。
                board.stop()
                target = command_l = command_r = 0.0
                centered = 0
                settled_since = None
                position.reset()
                head.i = speed_l.i = speed_r.i = 0.0
                lost_for = tick - waiting_since if last_valid_stamp is None else time.time() - last_valid_stamp
                if lost_for >= cfg["lost_timeout"]:
                    raise TimeoutError("小球位置校准视觉断流或持续缺球，已停车")
            else:
                if in_band or not wheels.has_speed or abs(target) < 1e-6:
                    want_l = want_r = 0.0
                    head.i = speed_l.i = speed_r.i = 0.0
                else:
                    trim = head.step(gap, dt)
                    lower, upper = (-cfg["speed"], 0.0) if target < 0 else (0.0, cfg["speed"])
                    set_l = clamp(target - trim, lower, upper)
                    set_r = clamp(target + trim, lower, upper)
                    want_l = clamp(set_l + speed_l.step(set_l - v_l, dt), lower, upper)
                    want_r = clamp(set_r + speed_r.step(set_r - v_r, dt), lower, upper)
                # 启动、减速和换向都限制最终轮速指令的加速度。
                step = cfg["accel"] * dt
                command_l += clamp(want_l - command_l, -step, step)
                command_r += clamp(want_r - command_r, -step, step)
                check_cancel(stop_event)
                if wheels.has_speed:
                    signs = wheels.signs
                    board.spd(command_l * signs[0], command_l * signs[1],
                              command_r * signs[2], command_r * signs[3])
                still = (in_band and wheels.has_speed and max(abs(v_l), abs(v_r)) <= cfg["speed_tolerance"]
                         and max(abs(command_l), abs(command_r)) < 1e-6)
                settled_since = (tick if settled_since is None else settled_since) if still else None
                if (centered >= cfg["stable_frames"] and settled_since is not None
                        and tick - settled_since >= cfg["settle"]):
                    log(f"小球位置校准完成：球心 x={center_x:.1f}px，"
                        f"中心线 x={center_line:.1f}px，偏差 {error:+.1f}px")
                    return {"ok": True, "color": color, "center_x": center_x,
                            "error_px": error, "distance_m": distance / 1000,
                            "travel_m": travel / 1000, "elapsed": tick - started}
            if new_layout or visual_state != last_visual_state:
                last_visual_state = visual_state
                frame_age = max(0.0, time.time() - last_stamp) if last_stamp is not None else math.inf
                log(f"  位置校准：偏差 {error:+.1f}px，目标速度 {target:+.1f}mm/s，"
                    f"轮速指令 {command_l:+.1f}/{command_r:+.1f}mm/s，"
                    f"视觉{visual_state}，帧 {last_index}，帧龄 {frame_age:.3f}s")
            first = vision.wait_ball_layout(
                after=last_index, stop_event=stop_event,
                timeout=max(0.0, 1 / cfg["loop_hz"] - (time.monotonic() - tick)))
    finally:
        cleanup(("小球校准停车", board.stop))
