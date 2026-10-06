"""倒车靠坎：逐帧检测加速度撞击，停车并释放电机。"""
import csv
import math
import time

from . import straight_pid as straight
from sensor.imu import ImpactDetector, AccelState
from config import BASE, POSITION
IMU_STALE = BASE["feedback_stale"]
from .control import cleanup, check_cancel, wait_cancelable


BOOT_WAIT = POSITION["boot_wait"]             # 倒车靠坎不额外等待 IMU 启动；仍检查原始加速度是否就绪
IMPACT_THRESHOLD = POSITION["impact_threshold"]     # 相邻三轴加速度变化的模长，单位：原始 int16 计数
MAX_DISTANCE_MM = POSITION["max_distance_m"] * 1000
TIMEOUT = POSITION["timeout"]
FEEDBACK_STALE = BASE["feedback_stale"]
POLL_INTERVAL = POSITION["poll_interval"]




def calibrate_position(board, imu, speed=straight.SPEED_CRUISE,
                       max_distance_mm=MAX_DISTANCE_MM, timeout=TIMEOUT,
                       accel=straight.ACCEL, kp_gap=straight.KP_GAP,
                       ki_gap=straight.KI_GAP, kd_gap=straight.KD_GAP,
                       log_path=None, log=print, stop_event=None):
    """调用前打开 board.upload() 并启动 IMU；返回 ok/reason/dist_mm 等诊断。

    board.feedback() 必须非阻塞，board.release() 必须提供零 PWM 释放操作，
    由底盘接口提供。所有退出路径（含 Ctrl-C 和串口异常）
    都会尝试停车并释放电机。
    """
    fp = None
    dist = gap = 0.0
    started = None
    released = False
    try:
        check_cancel(stop_event)
        for name, value in (("speed", speed), ("max_distance_mm", max_distance_mm),
                            ("timeout", timeout), ("accel", accel)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须是有限正数")
        if speed > straight.SPEED_LIMIT:
            raise ValueError(f"speed 不能超过 {straight.SPEED_LIMIT:g}mm/s")
        for value in (kp_gap, ki_gap, kd_gap):
            if not math.isfinite(value) or value < 0:
                raise ValueError("gap PID 参数必须是有限非负数")

        board.stop()
        if log_path:
            fp = open(log_path, 'w', newline='', encoding='utf-8')
            writer = csv.writer(fp)
            writer.writerow(['t', 'dist_mm', 'gap_mm', 'v_l', 'v_r', 'trim_mms',
                             'cmd_l', 'cmd_r', 'ax_raw', 'ay_raw', 'az_raw',
                             'delta_raw', 'peak_raw', 'hit'])

        # 静止取编码器起点；加速度原始帧必须独立就绪，不能以四元数帧代替。
        ready_deadline = time.monotonic() + 3.0
        base = None
        while time.monotonic() < ready_deadline:
            check_cancel(stop_event)
            totals, _ = board.feedback()
            check_cancel(stop_event)
            if totals is not None:
                base = totals
            if base is not None and imu.accel_state().age < IMU_STALE:
                break
            wait_cancelable(POLL_INTERVAL, stop_event)
        else:
            raise RuntimeError("编码器或 IMU 原始加速度未就绪，未开始倒车")

        signs = straight.FORWARD_SIGN
        base_l = (base[0] * signs[0] + base[1] * signs[1]) / 2.0
        base_r = (base[2] * signs[2] + base[3] * signs[3]) / 2.0
        head = straight.PID(kp_gap, ki_gap, kd_gap, straight.TRIM_MAX)
        spd_l = straight.PID(straight.KP_SPD, straight.KI_SPD, 0.0, straight.CORR_MAX)
        spd_r = straight.PID(straight.KP_SPD, straight.KI_SPD, 0.0, straight.CORR_MAX)
        hist_l, hist_r = [], []
        v_l = v_r = cruise_cmd = 0.0
        imu.arm_impact()
        started = last_control = last_totals = last_steps = time.monotonic()
        last_log = started
        ok = False
        log(f"开始倒车靠坎：速度 {speed:g}mm/s，最大距离 {max_distance_mm:g}mm")

        while True:
            check_cancel(stop_event)
            state = imu.accel_state()
            now = time.monotonic()
            # 撞击优先判断，即使驱动板此时没有反馈，也能立即退出并断速。
            if state.hit:
                ok, reason = True, f"检测到加速度突变（峰值 {state.peak:.1f} 原始计数），已停车并解锁电机"
                break
            if state.age >= IMU_STALE:
                reason = "IMU 原始加速度断流，已停车"
                break
            if now - started >= timeout:
                reason = "倒车超时，未检测到碰撞，已停车"
                break

            totals, steps = board.feedback()
            check_cancel(stop_event)
            now = time.monotonic()
            if steps is not None:
                last_steps = now
                scale = straight.MM_PER_COUNT / straight.TEP_WINDOW / 2.0
                hist_l.append((steps[0] * signs[0] + steps[1] * signs[1]) * scale)
                hist_r.append((steps[2] * signs[2] + steps[3] * signs[3]) * scale)
                del hist_l[:-straight.SPD_SAMPLES]
                del hist_r[:-straight.SPD_SAMPLES]
                v_l, v_r = sum(hist_l) / len(hist_l), sum(hist_r) / len(hist_r)
            if totals is not None:
                last_totals = now
                d_l = ((totals[0] * signs[0] + totals[1] * signs[1]) / 2.0 - base_l) * straight.MM_PER_COUNT
                d_r = ((totals[2] * signs[2] + totals[3] * signs[3]) / 2.0 - base_r) * straight.MM_PER_COUNT
                dist, gap = -(d_l + d_r) / 2.0, d_l - d_r
            if now - last_totals >= FEEDBACK_STALE or now - last_steps >= FEEDBACK_STALE:
                reason = "编码器里程或轮速断流，已停车"
                break
            if dist >= max_distance_mm:
                reason = "达到最大倒车距离，未检测到碰撞，已停车"
                break
            if abs(gap) > straight.GAP_DEV_MAX:
                reason = f"左右里程差 {gap:+.1f}mm 过大，已停车"
                break

            # 必须已收到轮速才启动 PI；没有新里程时仍持续检查加速度。
            if totals is not None and hist_l:
                dt = straight.clamp(now - last_control, 1e-4, 0.05)
                last_control = now
                cruise_cmd = min(speed, cruise_cmd + accel * dt)
                v = max(cruise_cmd, min(straight.SPEED_MIN, speed))
                # 与 straight(..., distance_mm<0, heading='gap') 同号：
                # trim 不随倒车取反，左=-v-trim，右=-v+trim。
                trim = head.step(gap, dt)
                set_l = straight.clamp(-v - trim, -straight.SPEED_LIMIT, 0.0)
                set_r = straight.clamp(-v + trim, -straight.SPEED_LIMIT, 0.0)
                cmd_l = straight.clamp(set_l + spd_l.step(set_l - v_l, dt), -straight.SPEED_LIMIT, 0.0)
                cmd_r = straight.clamp(set_r + spd_r.step(set_r - v_r, dt), -straight.SPEED_LIMIT, 0.0)
                # 发运动指令前再检查一次，避免在计算期间撞击后多发一拍。
                state = imu.accel_state()
                if state.hit or state.age >= IMU_STALE:
                    continue
                check_cancel(stop_event)
                board.spd(cmd_l * signs[0], cmd_l * signs[1],
                          cmd_r * signs[2], cmd_r * signs[3])
                if fp:
                    writer.writerow([now - started, dist, gap, v_l, v_r, trim,
                                     cmd_l, cmd_r, *state.acc, state.delta, state.peak, state.hit])
            if now - last_log >= 0.5:
                last_log = now
                log(f"  {now - started:.1f}s  倒车 {dist:.1f}mm  gap {gap:+.1f}mm"
                    f"  加速度突变峰值 {state.peak:.1f}")
            wait_cancelable(POLL_INTERVAL, stop_event)

        # 先断速并释放电机，再记录/返回；不再让零速闭环持续顶住坎。
        # 碰撞那一帧可能没有对应的新编码器读数。
        board.stop()
        board.release()
        released = True
        if fp:
            writer.writerow([time.monotonic() - started, dist, gap, v_l, v_r,
                             '', 0, 0, *state.acc, state.delta, state.peak, state.hit])
        check_cancel(stop_event)
        return {'ok': ok, 'reason': reason, 'dist_mm': dist, 'gap_mm': gap,
                'impact_peak_raw': state.peak, 'elapsed': time.monotonic() - started}
    finally:
        actions = []
        if not released:
            actions.extend((("倒车停车", board.stop), ("倒车释放", board.release)))
        actions.append(("碰撞检测解除", imu.disarm_impact))
        if fp is not None:
            actions.append(("倒车日志", fp.close))
        cleanup(*actions)
