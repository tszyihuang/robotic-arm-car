"""原地视觉对正：与视觉直走共享中心线和航向双环。"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from . import straight_pid
from config import BASE, ALIGN
from .control import CsvLog, clamp, cleanup, check_cancel, format_number
from .feedback import WheelOdometry
from .vision_straight import VisionHeading, prepare_heading

IMU_STALE = BASE["feedback_stale"]


class RefPicker:
    """把 ref 选的量折成'要往左修多少度'，并跟着 VisionHeading 的外推走。"""

    def __init__(self, ref: str = ALIGN["REF"], tau: float = ALIGN["REF_TAU"], gate: float = ALIGN["REF_GATE"]):
        self.ref = ref
        self.tau = max(float(tau), 0.0)
        self.gate = max(float(gate), 0.0)
        self.delta = 0.0          # head/mid 相对 e 的固定差（度）
        self._target = 0.0
        self.n_gated = 0
        self.n_fallback = 0       # head 拿不到（单侧）的次数

    def _raw(self, view) -> float | None:
        """本帧 head / mid 相对 e 的差；这个量本身已经按"往左修为正"折算过。"""
        if self.ref == "head":
            if view.head_deg is None:
                self.n_fallback += 1
                return None        # 只有一侧：灭点没有，退回 e（差值保持不动）
            return view.head_deg - view.e_deg
        if self.ref == "mid":
            return -view.midline_deg - view.e_deg
        return 0.0

    def update(self, view, dt: float, new_frame: bool) -> float:
        raw = self._raw(view)
        if raw is not None:
            if new_frame or self._target != raw:
                step = raw - self._target
                if self.gate > 0 and abs(step) > self.gate:
                    step = math.copysign(self.gate, step)
                    self.n_gated += 1
                self._target += step
        alpha = 1.0 if self.tau <= 0 else clamp(dt / (self.tau + dt), 0.0, 1.0)
        self.delta += alpha * (self._target - self.delta)
        return self.delta

    def err(self, hd) -> float:
        """本拍的参考角误差（+ = 中心线在左边 = 要往左修；未乘 dir_sign）。"""
        return hd.e + self.delta


@dataclass
class AlignCfg:
    """原地校准自己的参数（航向环增益那些在 VisionCfg 里，和走直线共用一份）。"""

    ref: str = ALIGN["REF"]
    ref_tau: float = ALIGN["REF_TAU"]
    ref_gate: float = ALIGN["REF_GATE"]
    bias: float = ALIGN["BIAS"]
    spin_speed: float = ALIGN["SPIN_SPEED"]
    min_u: float = ALIGN["MIN_U"]
    tol: float = ALIGN["TOL"]
    rate_tol: float = ALIGN["RATE_TOL"]
    settle: float = ALIGN["SETTLE"]
    timeout: float = ALIGN["TIMEOUT"]
    max_rot: float = ALIGN["MAX_ROT"]
    max_dev: float = ALIGN["MAX_DEV"]
    diverge: float = ALIGN["DIVERGE"]

    def __post_init__(self):
        if self.ref not in ("e", "head", "mid"):
            raise ValueError("ref 只能是 e / head / mid")
        for name in ("ref_tau", "ref_gate", "bias", "spin_speed", "min_u", "tol",
                     "rate_tol", "settle", "timeout", "max_rot", "max_dev", "diverge"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"align {name} 必须是有限数")
            if name != "bias" and value < 0:
                raise ValueError(f"align {name} 不能为负数")
            if name in ("spin_speed", "tol", "timeout", "max_rot", "max_dev") and value == 0:
                raise ValueError(f"align {name} 必须为正数")
        if self.spin_speed > 1000 or self.min_u > self.spin_speed:
            raise ValueError("对正轮速不能超过 1000 mm/s，min_u 不能超过轮速上限")


LOG_COLUMNS = ['t', 'err_deg', 'e_deg', 'head_deg', 'mid_deg', 'w_ref_dps', 'w_meas_dps', 'w_err_dps', 'trim_ff', 'trim_fb', 'trim_mms', 'cmd_l', 'cmd_r', 'gap_mm', 'rot_deg', 'src']


def vision_align(board, heading: VisionHeading, align: AlignCfg = None, imu=None,
                 log=print, log_path=None, stop_event=None):
    """原地把车头转到跑道中心线方向，返回诊断字典。"""
    rl = None
    stopped = False
    try:
        check_cancel(stop_event)
        align = align or AlignCfg()
        cfg = heading.cfg
        fwd = straight_pid.FORWARD_SIGN
        loop, gap_src = prepare_heading(heading, imu)
        picker = RefPicker(align.ref, align.ref_tau, align.ref_gate)
        wheels = WheelOdometry.read_origin(board, stop_event)

        settle_n = max(1, int(round(align.settle * ALIGN["LOOP_HZ"])))
        yaw0 = float(imu.yaw) if imu is not None else None
        rl = CsvLog(log_path, LOG_COLUMNS)
        t_start = t_prev = t_log = time.time()
        t_warn = 0.0
        t_data = t_prev
        dt = 0.0
        gap = rot = 0.0
        e_now = raw_now = head_now = mid_now = float("nan")
        e_max = trim_sum = 0.0
        w_ref_max = 0.0
        n_tick = n_ctrl = n_lost = 0
        lost_total = 0.0
        settle_cnt = 0
        first = True
        err0 = 0.0
        ok = False
        reason = "未完成"
        last_why = "模型还在加载 / 车不在跑道上 / 欠曝"
        n_frames0 = heading.n_frames

        if log and align.ref != "e":
            log(f"  参考量 {align.ref}（+ = 中心线在左边 = 要往左转）")
        while True:
            check_cancel(stop_event)
            totals, tep = board.feedback()          # 阻塞到收到一条新数据，约 100Hz
            check_cancel(stop_event)
            now = time.time()
            dt = clamp(now - t_prev, 1e-4, 0.05)
            t_prev = now
            n_tick += 1
            if now - t_start > align.timeout:
                if n_ctrl == 0:
                    reason = (f"{align.timeout:.0f}s 里一直没拿到可用边界（{last_why}），"
                              f"已停车：检查摄像头画面，别让车在没视觉的时候瞎转")
                else:
                    reason = (f"超过总时长 {align.timeout:.0f}s 还没锁住"
                              f"（误差 {e_now:+.1f}°，累计转了 {rot:+.0f}°），已停车")
                break
            if totals is None:
                if now - t_data > 0.5:
                    reason = "编码器数据中断 0.5s 以上，已停车"
                    break
                continue
            t_data = now

            # ---- 反馈：左右里程 / 里程差（内环要用的角速度从这来）----
            d_l, d_r = wheels.travel(totals)
            gap = d_l - d_r
            rot = -math.degrees(gap / max(cfg.track_mm, 1e-3))   # 左转为正
            if gap_src is not None:
                gap_src.update(now, gap)

            # ---- 视觉：中心线 -> 角度误差 ----
            hd = heading.step(now)
            view = heading.view
            usable, why = True, ""
            if hd is None or view is None:
                usable, why = False, "模型还在加载 / 车不在跑道上 / 欠曝"
            elif hd.age > cfg.lost_stop:
                usable, why = False, (f"连续 {hd.age:.1f}s 没有新边界（模型关了 / "
                                      f"车偏出跑道 / 欠曝）")
            elif align.ref != "mid" and not view.absolute:
                usable, why = False, ("只有一侧边界、也没有两侧都在时量到的半宽基准，"
                                      "算不出中心线：把车摆到两侧都进画面的位置")
            if usable and cfg.rate_src == "gyro" and imu.age() > IMU_STALE:
                usable, why = False, f"IMU 数据中断 {imu.age():.2f}s"
            if why:
                last_why = why

            if not usable:
                # 拿不到可信的角度就不许转：停车保持，等视觉回来（最长等到超时）
                board.spd(0, 0, 0, 0)
                n_lost += 1
                lost_total += dt
                settle_cnt = 0
                if now - t_warn > 1.0:
                    t_warn = now
                    log(f"  ! 保持不动：{why}")
                continue

            # ---- 参考角误差：ref 选量 + 低通/外推 + bias ----
            picker.update(view, dt, hd.new)
            raw_now = picker.err(hd) - align.bias       # + = 要往左修
            e_now = cfg.dir_sign * raw_now
            head_now = float("nan") if view.head_deg is None else view.head_deg
            mid_now = -view.midline_deg
            if first:
                err0, first = e_now, False
                log(f"  出发：误差 {e_now:+.2f}°（{align.ref}）"
                    f"  中线 {'两侧都有' if view.src == 'both' else view.src + ' 单侧'}"
                    f"  前视行 {view.look_row:.0f}px")
            if abs(e_now) > align.max_dev:
                reason = (f"角度误差 {e_now:+.1f}° 超过 {align.max_dev:.0f}°，已急停："
                          f"视觉在乱跳（检查摄像头画面），或者车根本不在跑道上")
                break
            if abs(e_now) > abs(err0) + align.diverge:
                fix = ("设置 config.py 的 dir_sign=-1" if cfg.dir_sign > 0
                       else "设置 config.py 的 dir_sign=+1")
                reason = (f"误差从 {err0:+.1f}° 发散到 {e_now:+.1f}°，已急停："
                          f"方向多半接反了：{fix}，或车被外力拨动")
                break
            if imu is not None and abs(imu.yaw - yaw0) > align.max_rot:
                reason = (f"已经转了 {abs(imu.yaw - yaw0):.0f}°，超过上限 "
                          f"{align.max_rot:.0f}° 还没锁住，已停车：检查 align ref / "
                          f"dir_sign，或车头是不是反着放的")
                break

            # ---- ② 航向环：角度误差 -> 目标角速度 -> 差动 ----
            trim = loop.step(e_now, heading.src.rate, gap_src is None or gap_src.valid, dt)
            n_ctrl += 1
            in_band = abs(e_now) <= align.tol and abs(loop.w_meas) <= align.rate_tol
            u = trim
            if not in_band and align.min_u > 0 and 0.0 < abs(u) < align.min_u:
                # 死区前馈：指令太小推不动车（原地自转摩擦大），会被卡在目标前几度。
                # 只在容差外补，容差内保持纯 PID，免得在目标附近来回蹭。
                u = math.copysign(align.min_u, u)
            u = clamp(u, -align.spin_speed, align.spin_speed)
            check_cancel(stop_event)
            board.spd(-u * fwd[0], -u * fwd[1], u * fwd[2], u * fwd[3])

            e_max = max(e_max, abs(e_now))
            if math.isfinite(loop.w_ref):
                w_ref_max = max(w_ref_max, abs(loop.w_ref))
            trim_sum += abs(u)
            if log_path:
                rl.row(f"{now - t_start:.3f}", f"{e_now:.3f}", f"{hd.e:.3f}",
                       format_number(head_now, ".3f"), format_number(mid_now, ".3f"),
                       format_number(loop.w_ref, ".2f"), format_number(loop.w_meas, ".2f"),
                       format_number(loop.w_err, ".2f"), format_number(loop.trim_ff, ".1f"),
                       format_number(loop.trim_fb, ".1f"), f"{u:.1f}", f"{-u:.0f}", f"{u:.0f}",
                       f"{gap:.1f}", f"{rot:.1f}", view.src)

            if now - t_log > ALIGN["PRINT_INTERVAL"]:
                t_log = now
                w_txt = (f"{loop.w_ref:+5.1f}/{loop.w_meas:+5.1f}"
                         if math.isfinite(loop.w_ref) else f"  —  /{loop.w_meas:+5.1f}")
                head_txt = "  —  " if view.head_deg is None else f"{view.head_deg:+6.2f}"
                log(f"  {now - t_start:5.1f}s  误差 {e_now:+6.2f}°"
                    f"  角速度 {w_txt}°/s（目标/实测）"
                    f"  差动 {u:+6.1f}mm/s（前馈 {loop.trim_ff if math.isfinite(loop.trim_ff) else 0:+5.1f}）"
                    f"  转了 {rot:+6.1f}°"
                    f"   [e {hd.e:+6.2f}°  head {head_txt}°  mid {mid_now:+6.2f}°]"
                    f"  {view.src}")

            settle_cnt = settle_cnt + 1 if in_band else 0
            if settle_cnt >= settle_n:
                ok, reason = True, "锁定到位"
                break
        board.spd(0, 0, 0, 0)
        stopped = True

        # 停车后等轮子真停（原地自转停下来的那点余转也算进去）
        end = time.time() + 0.8
        quiet = 0
        prev = rot
        while quiet < 3 and time.time() < end:
            check_cancel(stop_event)
            totals, _ = board.feedback(0.05)
            check_cancel(stop_event)
            if totals is None:
                continue
            d_l, d_r = wheels.travel(totals)
            gap = d_l - d_r
            rot = -math.degrees(gap / max(cfg.track_mm, 1e-3))
            quiet = quiet + 1 if abs(rot - prev) < 0.1 else 0
            prev = rot

        check_cancel(stop_event)
        rot_imu = None if imu is None else float(imu.yaw) - yaw0
        info = {
            "ok": ok, "reason": reason, "ref": align.ref, "err": e_now,
            "err0": err0, "tol": align.tol, "rot": rot, "rot_imu": rot_imu,
            "e_max": e_max, "elapsed": time.time() - t_start, "ticks": n_tick,
            "ctrl": n_ctrl, "lost": n_lost, "lost_total": lost_total,
            "frames": heading.n_frames - n_frames0, "single": heading.n_single,
            "degraded": heading.n_degraded, "gated": heading.n_gated,
            "ref_fallback": picker.n_fallback, "ref_gated": picker.n_gated,
            "trim_avg": trim_sum / n_ctrl if n_ctrl else 0.0, "w_ref_max": w_ref_max,
            "rate_src": cfg.rate_src, "src": None if heading.view is None else heading.view.src,
            "imu": imu is not None, "log_path": log_path}
        if log:
            tail = (f"实际转了 {rot:+.1f}°" if rot_imu is None
                    else f"实际转了 {rot_imu:+.1f}°（陀螺）")
            log(f"  {'已锁定' if ok else '结束'}：误差 {e_now:+.2f}°（出发 {err0:+.2f}°）"
                f"  {tail}  用时 {time.time() - t_start:.2f}s  {reason}")
            if log_path:
                log(f"  日志 {log_path}")
        return info

    finally:
        actions = []
        if not stopped:
            actions.append(("视觉停车", lambda: board.spd(0, 0, 0, 0)))
        if rl is not None:
            actions.append(("视觉日志", rl.close))
        cleanup(*actions)
