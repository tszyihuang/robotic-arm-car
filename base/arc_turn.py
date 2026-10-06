#!/usr/bin/env python3
"""单段转弯：剩余转角减速 + 圆弧配速 + IMU 修正 → 驱动板速度接口。

IMU 决定何时停车，编码器修正圆弧半径并限制异常行程。程序不使用积分、
最低轮速或末尾原地补角；R=0 共用控制循环。每次调用最终停车，串口由底盘接口持有。
"""
from __future__ import annotations

from collections import deque
import csv
from dataclasses import dataclass
import math
import time

from control import cleanup, check_cancel, wait_cancelable
from config import BASE, TURN

COUNTS_PER_METER = TURN["COUNTS_PER_METER"]
LOOP_HZ = TURN["LOOP_HZ"]
MAX_WHEEL_SPEED = TURN["MAX_WHEEL_SPEED"]
ACCEL = TURN["ACCEL"]
ANGULAR_ACCEL = TURN["ANGULAR_ACCEL"]
ANGLE_KP = TURN["ANGLE_KP"]
PATH_KP = TURN["PATH_KP"]
PATH_KD = TURN["PATH_KD"]
RATE_KP = TURN["RATE_KP"]
WHEEL_SPEED_KP = TURN["WHEEL_SPEED_KP"]
BRAKE_LEAD_SECONDS = TURN["BRAKE_LEAD_SECONDS"]
BRAKE_KP = TURN["BRAKE_KP"]
STOP_ANGLE = math.radians(TURN["STOP_ANGLE_DEG"])
SETTLE_SECONDS = TURN["SETTLE_SECONDS"]


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


class WheelFeedback:
    """用累计编码器窗口拟合轮速，仅用于反馈、日志和停稳检查。"""
    def __init__(self):
        self.history = deque()

    def measure(self, now, counts):
        self.history.append((now, tuple(counts)))
        while len(self.history) > 2 and now - self.history[1][0] >= 0.08:
            self.history.popleft()
        t0 = self.history[0][0]
        ts = [t - t0 for t, _ in self.history]
        mean_t = sum(ts) / len(ts)
        denom = sum((t - mean_t)**2 for t in ts)
        if denom < 1e-9:
            return [0.0] * 4
        origin = self.history[0][1]
        return [sum((t - mean_t) * (c[i] - origin[i]) for t, (_, c) in zip(ts, self.history))
                / denom / COUNTS_PER_METER for i in range(4)]

    def brake_speeds(self):
        """停车力矩使用约 30ms 的计数差分，减少正常测速窗口带来的延迟。"""
        now, counts = self.history[-1]
        start, previous = self.history[0]
        for stamp, sample in self.history:
            if now - stamp >= .025:
                start, previous = stamp, sample
        span = now - start
        return [(n - n0) / COUNTS_PER_METER / span if span > 0 else 0.0
                for n, n0 in zip(counts, previous)]


@dataclass
class ArcPlan:
    """纯计算规划器；结束后保持零指令，不再倒车或原地补角。单位 m/rad/s。"""
    radius: float
    angle: float
    cruise: float
    wheel_limit: float
    track: float
    v: float = 0.0
    w: float = 0.0
    correction: float = 0.0
    braking: bool = False

    @property
    def length(self):
        return self.radius * abs(self.angle)

    @property
    def tolerance(self):
        return max(math.radians(1.0), 0.05 * abs(self.angle))

    def step(self, travel, yaw, dt, rate=0.0, speed=None):
        dt = clamp(dt, 0.0, 0.03)
        sign = math.copysign(1.0, self.angle)
        remaining = sign * (self.angle - yaw)
        # 根据实测角速度提前断速；达到刹车条件后不再起步或倒转补角。
        stop_window = STOP_ANGLE + BRAKE_LEAD_SECONDS * max(0.0, sign * rate)
        path_ready = self.radius == 0 or travel >= .97 * self.radius * abs(yaw)
        self.braking |= remaining <= 0 or (remaining <= stop_window and path_ready)
        if self.braking:
            self.v = self.w = 0.0
            return 0.0, 0.0

        w_cap = min(2.0, 2 * self.wheel_limit / self.track)
        if self.radius > 0:
            w_cap = min(w_cap, self.cruise / self.radius)
        turn_rate = min(w_cap, ANGLE_KP * max(remaining, 0.0))
        if self.radius == 0:
            self.v = 0.0
            w_ref = sign * turn_rate
            target_w = w_ref + RATE_KP * (w_ref - rate)
        else:
            # 转角提前而里程落后时保持前进，让半径修正有机会赶上；反之仍按转角减速。
            remaining_s = max(self.radius * remaining, self.length - travel, 0.0)
            target_v = min(self.radius * w_cap, ANGLE_KP * remaining_s)
            self.v += clamp(target_v - self.v, -ACCEL * dt, ACCEL * dt)
            yaw_ref = sign * clamp(travel / self.radius, 0, abs(self.angle))
            self.correction += dt / (0.12 + dt) * (PATH_KP * (yaw_ref - yaw) - self.correction)
            w_ref = sign * self.v / self.radius + self.correction
            # 圆弧的阻尼使用路径误差的变化率：实测前进速度/R - 实测角速度。
            forward_speed = self.v if speed is None else max(0.0, speed)
            target_w = w_ref + PATH_KD * (sign * forward_speed / self.radius - rate)
        # 不使用积分，转弯只沿目标方向推进。
        target_w = sign * max(0.0, sign * target_w)
        w_limit = min(2.0, 2 * max(self.wheel_limit - self.v, 0.0) / self.track)
        if self.radius > self.track / 2:
            w_limit = min(w_limit, 2 * self.v / self.track)
        target_w = clamp(target_w, -w_limit, w_limit)
        self.w += clamp(target_w - self.w, -ANGULAR_ACCEL * dt, ANGULAR_ACCEL * dt)
        self.w = clamp(self.w, -w_limit, w_limit)
        half = self.w * self.track / 2
        return self.v - half, self.v + half


def make_plan(radius, degrees, cruise, max_speed, track):
    values = (radius, degrees, cruise, track, max_speed if max_speed is not None else MAX_WHEEL_SPEED)
    if not all(math.isfinite(x) for x in values):
        raise ValueError('转弯参数必须为有限数字')
    if radius < 0 or degrees == 0 or abs(degrees) > 360 or cruise <= 0 or track <= 0:
        raise ValueError('半径≥0、0<|角度|≤360、速度和有效轮距必须为正')
    limit = min(MAX_WHEEL_SPEED, max_speed if max_speed is not None else MAX_WHEEL_SPEED)
    if limit <= 0:
        raise ValueError('转弯速度上限必须为正')
    if radius > 0:
        outer = 1 + track / (2 * radius)
        cap = min(limit / outer, 2.0 * radius)
        cruise = min(cruise, cap)
    return ArcPlan(radius, math.radians(degrees), cruise, limit, track)


def turn_with_radius(radius_m, degrees, *, board=None, imu=None, cruise=0.25,
                     max_speed=None, timeout_s=None, dry_run=False, track_width_m=0.41,
                     log_path=None, verbose=True, stop_event=None):
    """使用注入的电机和传感器接口执行转弯。干跑仅计算计划。"""
    plan = make_plan(radius_m, degrees, 0.25 if cruise is None else cruise, max_speed, track_width_m)
    check_cancel(stop_event)
    if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s <= 0):
        raise ValueError('转弯超时必须为正有限数值')
    source = 'IMU' if imu else '编码器'
    if verbose:
        print(f'  圆弧计划：R={radius_m:g}m，{degrees:+g}°，弧长 {plan.length:.3f}m，'
              f'巡航 {plan.cruise * 1000:.0f}mm/s，驱动板速度闭环 + IMU 修正（程序 Ki=0）')
    rows = []
    log_file = None
    travel = yaw = 0.0
    fault = None
    settled = False
    started = None
    elapsed = 0.0
    timeout = timeout_s or (plan.length / max(plan.cruise, 0.01) + abs(plan.angle) / 0.5 + 5)
    fields = ['t', 'dt', 'phase', 'done_s', 'yaw', 'v_ref', 'w_ref', 'v_meas', 'omega_meas',
              'ref1', 'ref2', 'ref3', 'ref4', 'meas1', 'meas2', 'meas3', 'meas4',
              'count1', 'count2', 'count3', 'count4', 'telemetry_age', 'imu_age', 'send_ms',
              'cmd1', 'cmd2', 'cmd3', 'cmd4',
              'brake_pwm1', 'brake_pwm2', 'brake_pwm3', 'brake_pwm4']
    try:
        if not dry_run:
            # 路径无效要在打开电机前报错；逐拍数据暂存，停车后再写。
            if log_path:
                log_file = open(log_path, 'w', newline='')
                writer = csv.writer(log_file)
                writer.writerow(fields)
            if board is None:
                raise ValueError('转弯需要底盘接口提供电机接口')
            check_cancel(stop_event)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                check_cancel(stop_event)
                board.poll()
                check_cancel(stop_event)
                if board.counts is not None and (imu is None or (imu.has_data() and imu.age() < 0.3)):
                    break
                wait_cancelable(0.01, stop_event)
            else:
                raise RuntimeError('转弯前未收到编码器或 IMU 数据')
            if imu:
                imu.zero()
            origin = board.counts
            wheels = WheelFeedback()
            started = previous = next_tick = time.monotonic()
            last_motion = started
            previous_counts = origin
            rate = 0.0
            still_since = None
            while True:
                check_cancel(stop_event)
                now = time.monotonic()
                if now < next_tick:
                    wait_cancelable(next_tick - now, stop_event)
                    now = time.monotonic()
                dt = now - previous
                previous = now
                next_tick = now + 1 / LOOP_HZ
                board.poll()
                check_cancel(stop_event)
                age = time.monotonic() - board.stamp
                if age > 0.3:
                    fault = '编码器数据中断'
                    break
                if imu and imu.age() > 0.3:
                    fault = 'IMU 数据中断'
                    break
                if board.battery is not None and board.battery < 9.8:
                    fault = f'电池欠压 {board.battery:.1f}V'
                    break
                if board.counts != previous_counts:
                    last_motion, previous_counts = now, board.counts
                elif not plan.braking and now - last_motion > 1.5:
                    fault = '运动指令下发后编码器持续无变化'
                    break
                distances = [(n - n0) / COUNTS_PER_METER for n, n0 in zip(board.counts, origin)]
                travel = sum(distances) / 4
                yaw = math.radians(imu.yaw) if imu else ((distances[2] + distances[3]) -
                                                       (distances[0] + distances[1])) / (2 * plan.track)
                speeds = wheels.measure(now, board.counts)
                span = now - wheels.history[0][0]
                # IMU 原始陀螺作快反馈；无 IMU 时使用编码器估计角速度。
                rate_sample = math.radians(imu.rate) if imu else (speeds[2] + speeds[3] -
                                                                speeds[0] - speeds[1]) / (2 * plan.track)
                rate += dt / (0.06 + dt) * (rate_sample - rate)
                left, right = plan.step(travel, yaw, dt, rate, sum(speeds) / 4)
                if abs(yaw) > abs(plan.angle) + math.radians(15):
                    fault = '转角超出诊断保护范围'
                    break
                if travel > plan.length + max(0.10, 0.25 * plan.length) and radius_m > 0:
                    fault = '转角不足且超过弧长保护范围'
                    break
                if math.copysign(1.0, plan.angle) * yaw < -math.radians(5):
                    fault = '转向与目标相反'
                    break
                send_start = time.monotonic()
                brake_pwm = (0, 0, 0, 0)
                commands = (0.0,) * 4
                if plan.braking:
                    brake_pwm = board.brake(wheels.brake_speeds())
                else:
                    commands = tuple(clamp(ref + WHEEL_SPEED_KP * (ref - speed),
                                           -plan.wheel_limit, plan.wheel_limit)
                                     for ref, speed in zip((left, left, right, right), speeds))
                    check_cancel(stop_event)
                    board.speed(commands)
                send_ms = (time.monotonic() - send_start) * 1000
                rows.append([now - started, dt, 'brake' if plan.braking else 'run', travel,
                             math.degrees(yaw), plan.v, plan.w, sum(speeds) / 4,
                             rate,
                             left, left, right, right, *speeds, *board.counts,
                             age, imu.age() if imu else None, send_ms,
                             *(round(v * 1000) for v in commands), *brake_pwm])
                if plan.braking:
                    stopped = (span >= 0.08 and max(abs(v) for v in speeds) < 0.01
                               and abs(rate) < math.radians(2))
                    still_since = (still_since if still_since is not None else now) if stopped else None
                    if still_since is not None and now - still_since >= SETTLE_SECONDS:
                        settled = True
                        break
                if now - started >= timeout:
                    fault = '转弯超时'
                    break
    finally:
        if started is not None:
            elapsed = time.monotonic() - started
        actions = []
        if board is not None:
            actions.extend((("转弯停车", board.stop), ("转弯释放", board.release)))
        if log_file is not None:
            actions.extend((("转弯日志写入", lambda: writer.writerows(rows)),
                            ("转弯日志关闭", log_file.close)))
        cleanup(*actions)
    check_cancel(stop_event)
    error = math.degrees(plan.angle - yaw)
    effective = abs(travel / yaw) if radius_m > 0 and abs(yaw) > 1e-6 else None
    radius_error = 100 * (effective / radius_m - 1) if effective is not None else None
    in_tolerance = abs(error) <= math.degrees(plan.tolerance)
    at_goal = settled and in_tolerance
    ok = at_goal and fault is None and (radius_error is None or abs(radius_error) <= 5)
    return dict(angle_deg=degrees, turned_deg=math.degrees(yaw), yaw_deg=math.degrees(yaw),
                angle_error_deg=error, distance_m=abs(travel), arc_length_m=abs(travel), radius_m=radius_m,
                effective_radius_m=effective, radius_error_pct=radius_error, elapsed_s=elapsed,
                ok=ok, at_goal=at_goal, in_tolerance=in_tolerance,
                tolerance_deg=math.degrees(plan.tolerance), fault=fault, heading_source=source,
                dry_run=dry_run, log_path=log_path,
                summary=f'{degrees:+g}° → {math.degrees(yaw):+.2f}°，弧长 {travel:.3f}m，'
                        f'{"OK" if ok else (fault or "未到位")}')
