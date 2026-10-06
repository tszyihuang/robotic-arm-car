"""普通直走：位置 P、左右里程差 PID、轮速 PI；支持前进和倒退。"""
import math
import time

from .control import CsvLog, clamp, cleanup, check_cancel
from .feedback import WheelOdometry
from config import BASE, STRAIGHT

__all__ = ["PID", "straight"]

FORWARD_SIGN = STRAIGHT["FORWARD_SIGN"]
TRACK_MM = STRAIGHT["TRACK_MM"]
SPEED_CRUISE = STRAIGHT["SPEED_CRUISE"]
SPEED_MIN = STRAIGHT["SPEED_MIN"]
SPEED_LIMIT = STRAIGHT["SPEED_LIMIT"]
ACCEL = STRAIGHT["ACCEL"]
MARGIN_MM = STRAIGHT["MARGIN_MM"]
KP_POS = STRAIGHT["KP_POS"]
TRIM_MAX = STRAIGHT["TRIM_MAX"]
KP_GAP = STRAIGHT["KP_GAP"]
KI_GAP = STRAIGHT["KI_GAP"]
KD_GAP = STRAIGHT["KD_GAP"]
GAP_DEV_MAX = STRAIGHT["GAP_DEV_MAX"]
KP_SPD = STRAIGHT["KP_SPD"]
KI_SPD = STRAIGHT["KI_SPD"]
CORR_MAX = STRAIGHT["CORR_MAX"]
BRAKE_TAU = STRAIGHT["BRAKE_TAU"]
FEEDBACK_STALE = BASE["feedback_stale"]


class PID:
    """位置式 PID：输出限幅，且只在没饱和时积分（抗积分饱和）。"""

    def __init__(self, kp, ki, kd, out_max):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_max = out_max
        self.i = 0.0
        self.last_e = None

    def step(self, e, dt):
        d = 0.0 if self.last_e is None else (e - self.last_e) / dt
        self.last_e = e
        out = self.kp * e + self.i + self.kd * d
        if abs(out) < self.out_max:
            self.i = clamp(self.i + self.ki * e * dt, -self.out_max, self.out_max)
        return clamp(out, -self.out_max, self.out_max)


def straight(board, distance_mm, v_cruise, kp_gap=KP_GAP, ki_gap=KI_GAP,
             kd_gap=KD_GAP, log_path=None, log=print, stop_event=None):
    """返回沿行进方向的距离（mm，恒为正）与到位／故障诊断。"""
    run_log = None
    try:
        check_cancel(stop_event)
        direction = -1.0 if distance_mm < 0 else 1.0
        goal = abs(distance_mm)
        wheels = WheelOdometry.read_origin(board, stop_event)

        head = PID(kp_gap, ki_gap, kd_gap, TRIM_MAX)
        spd_l, spd_r = PID(KP_SPD, KI_SPD, 0, CORR_MAX), PID(KP_SPD, KI_SPD, 0, CORR_MAX)
        pos_cmd = dist = gap = gap_max = 0.0
        t_start = t_prev = t_data = t_log = time.monotonic()
        deadline = t_start + max(10.0, 3.0 * goal / v_cruise + 5.0)
        reason = ""
        run_log = CsvLog(log_path, ["t", "dist_mm", "gap_mm", "trim_mms", "v_l", "v_r", "set_l", "set_r"])
        while True:
            check_cancel(stop_event)
            totals, steps = board.feedback()
            check_cancel(stop_event)
            now = time.monotonic()
            dt = clamp(now - t_prev, 1e-4, 0.05)
            t_prev = now
            if now > deadline:
                reason = "总时长超时，已停车"
                break
            if totals is None:
                if now - t_data > FEEDBACK_STALE:
                    reason = "编码器数据中断，已停车"
                    break
                continue
            t_data = now
            left, right = wheels.travel(totals)
            dist, gap = direction * (left + right) / 2, left - right
            gap_max = max(gap_max, abs(gap))
            v_l, v_r = wheels.speed(steps)
            if abs(gap) > GAP_DEV_MAX:
                reason = "左右里程差发散，已停车"
                break
            if goal - dist <= BRAKE_TAU * abs((v_l + v_r) / 2) + MARGIN_MM:
                break
            want = clamp(KP_POS * (goal - dist), 0, v_cruise)
            pos_cmd = clamp(want, max(0, pos_cmd - ACCEL * dt), pos_cmd + ACCEL * dt)
            speed = direction * max(pos_cmd, min(SPEED_MIN, v_cruise))
            trim = head.step(gap, dt)
            set_l = clamp(speed - trim, -SPEED_LIMIT, SPEED_LIMIT)
            set_r = clamp(speed + trim, -SPEED_LIMIT, SPEED_LIMIT)
            cmd_l = set_l + spd_l.step(set_l - v_l, dt)
            cmd_r = set_r + spd_r.step(set_r - v_r, dt)
            s = FORWARD_SIGN
            check_cancel(stop_event)
            board.spd(cmd_l * s[0], cmd_l * s[1], cmd_r * s[2], cmd_r * s[3])
            run_log.row(now - t_start, dist, gap, trim, v_l, v_r, set_l, set_r)
            if now - t_log > 0.5:
                t_log = now
                log(f"  剩余 {goal-dist:.0f}mm，里程差 {gap:+.1f}mm，修正 {trim:+.1f}mm/s")
        board.stop()
        # 保留原来的滑行后测距，避免把断速瞬间的里程当成最终位置。
        deadline = time.monotonic() + 1.5
        previous, still = dist, 0
        while still < 3 and time.monotonic() < deadline:
            check_cancel(stop_event)
            totals, _ = board.feedback(0.05)
            check_cancel(stop_event)
            if totals is None:
                continue
            left, right = wheels.travel(totals)
            dist, gap = direction * (left + right) / 2, left - right
            gap_max = max(gap_max, abs(gap))
            still = still + 1 if abs(dist - previous) < 0.2 else 0
            previous = dist
        check_cancel(stop_event)
        return dist, {"gap_max": gap_max, "gap_end": gap, "reason": reason,
                      "yaw_end": math.degrees(gap / TRACK_MM),
                      "elapsed": time.monotonic() - t_start}

    finally:
        actions = [("直走停车", board.stop)]
        if run_log is not None:
            actions.append(("直走日志", run_log.close))
        cleanup(*actions)
