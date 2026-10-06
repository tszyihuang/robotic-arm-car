"""视觉直走：直接边界反馈、中心线几何与航向／轮速闭环。"""
from __future__ import annotations

import csv
import math
import time
from collections import deque
from dataclasses import dataclass

from . import straight_pid
from config import BASE, STRAIGHT, VISION_CONTROL
IMU_STALE = BASE["feedback_stale"]
from .control import cleanup, check_cancel, wait_cancelable

# ============================== 默认参数 ==============================

FRAME_W = VISION_CONTROL["FRAME_W"]
FRAME_H = VISION_CONTROL["FRAME_H"]

LOOKAHEAD = VISION_CONTROL["LOOKAHEAD"]
FOV_DEG = VISION_CONTROL["FOV_DEG"]
RATE_TAU = VISION_CONTROL["RATE_TAU"]
RATE_MAX = VISION_CONTROL["RATE_MAX"]
E_TAU = VISION_CONTROL["E_TAU"]
E_GATE = VISION_CONTROL["E_GATE"]
E_DEAD = VISION_CONTROL["E_DEAD"]
RATE_SRC = VISION_CONTROL["RATE_SRC"]
GAP_WINDOW = VISION_CONTROL["GAP_WINDOW"]
GAP_TAU = VISION_CONTROL["GAP_TAU"]
LOST_STOP = VISION_CONTROL["LOST_STOP"]
MAX_DEV = VISION_CONTROL["MAX_DEV"]
TRACK_WAIT = VISION_CONTROL["TRACK_WAIT"]

# ---- ② 航向环双环：外环（角度）把视觉误差变成目标角速度，内环（角速度）再把它
#      变成差动。内环反馈默认用电机 gap，所以外环不用再替视觉延时操心 ----
KP_YAW = VISION_CONTROL["KP_YAW"]
KI_YAW = VISION_CONTROL["KI_YAW"]
YAW_W_MAX = VISION_CONTROL["YAW_W_MAX"]
KP_RATE = VISION_CONTROL["KP_RATE"]
KI_RATE = VISION_CONTROL["KI_RATE"]
KFF = VISION_CONTROL["KFF"]
KD_RATE = VISION_CONTROL["KD_RATE"]


TRIM_MAX = VISION_CONTROL["TRIM_MAX"]
MIN_CONF = VISION_CONTROL["MIN_CONF"]


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class BoundarySample:
    """视觉对象最近一帧的状态和采集时间。"""

    info: dict | None       # 最近一次"检出了边界"的 info（没有就是 None）
    frames: object          # 采集帧号
    seq: int                # 有效帧计数：每换一帧 +1（控制侧看它判断"新帧"）
    state: str              # off / loading / ready / error
    error: str              # 模型或采集的报错文本
    t: float                # 最近一次收到视觉反馈的时间
    t_valid: float          # 最近一次拿到"有边界"的帧的时间（0 = 从没有过）
    frame_dt: float = 0.0   # 这一帧从"采集完"到"控制侧拿到"隔了多久（秒）


@dataclass
class VisionCfg:
    """视觉与控制参数（默认值来自 config.py；测试可直接构造）。"""

    lookahead: float | str = LOOKAHEAD  # 比例，或 "vanishing"（两侧边界延长线交点）
    fov_deg: float = FOV_DEG
    focal_px: float | None = None
    width: int = FRAME_W
    height: int = FRAME_H
    min_conf: float = MIN_CONF
    dir_sign: float = VISION_CONTROL["dir_sign"]
    kp_yaw: float = KP_YAW          # 外环（角度环）
    ki_yaw: float = KI_YAW
    w_max: float = YAW_W_MAX
    kp_rate: float = KP_RATE        # 内环（角速度环）
    ki_rate: float = KI_RATE
    kff: float = KFF                # 前馈：ω_ref -> 差动（按轮距换算）
    kd_rate: float = KD_RATE
    rate_tau: float = RATE_TAU
    e_tau: float = E_TAU
    e_gate: float = E_GATE
    e_dead: float = E_DEAD
    predict: bool = True
    rate_src: str = RATE_SRC          # 内环角速度来源：gap / gyro
    gap_window: float = GAP_WINDOW
    gap_tau: float = GAP_TAU
    track_mm: float = STRAIGHT["TRACK_MM"]           # 左右轮中心距：里程差 <-> 角度、前馈换算都用它
    trim_max: float = TRIM_MAX
    max_dev: float = MAX_DEV
    lost_stop: float = LOST_STOP
    speed: float = STRAIGHT["SPEED_CRUISE"]
    accel: float = STRAIGHT["ACCEL"]
    margin: float = STRAIGHT["MARGIN_MM"]
    kp_pos: float = STRAIGHT["KP_POS"]


    def size(self, info: dict | None) -> tuple[int, int]:
        """原帧尺寸：优先用 info 里模型报的 [宽,高]，没有就退回 720p 兜底值。"""
        sz = (info or {}).get("size")
        if isinstance(sz, (list, tuple)) and len(sz) == 2 and sz[0] and sz[1]:
            return int(sz[0]), int(sz[1])
        return self.width, self.height

    def focal(self, width: int) -> float:
        """焦距（像素）：--focal-px 优先，否则由水平视场角换算。"""
        if self.focal_px:
            return float(self.focal_px)
        return 0.5 * width / math.tan(math.radians(self.fov_deg) / 2.0)


@dataclass
class TrackView:
    """一帧中心线，以及航向环要用的几个量（原图像素 / 度）。"""

    a: float                # 中心线 x = a*y + b
    b: float
    y_lo: float             # 中心线可信的行范围（两侧线段范围的交集/并集）
    y_hi: float
    src: str                # both / left / right（哪几侧真正检出）
    absolute: bool          # 误差是不是绝对几何；False = 单侧相对角退化
    midline_deg: float      # 中心线方向角：0=竖直，+ = 远端偏向画面右
    look_row: float         # 目标点的 y；数值前视被线段范围夹过，灭点允许在画面外
    x_look: float           # 目标点的 x（中心线前视点或灭点）
    lat_px: float           # x_look - u0；灭点模式下仅表示目标像素偏移，不是横向位置
    e_deg: float            # 航向环误差：+ = 需要往左修
    head_deg: float | None  # 灭点算的纯航向项；两侧都在才是真值
    half_px: float | None   # 本帧前视行上的半宽（两侧都在时才有）
    angle_left: float | None
    angle_right: float | None


def _side_rec(rec: dict | None, cfg: VisionCfg, height: float):
    """一侧边界记录 -> (a, b, y_lo, y_hi, angle_deg)；缺字段/信心不足返回 None。"""
    if not rec:
        return None
    if cfg.min_conf and float(rec.get("confidence") or 0.0) < cfg.min_conf:
        return None
    try:
        a, b = float(rec["a"]), float(rec["b"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    y_near = rec.get("near_y")          # 近端行（大 y）；老版本没有就按画面底部算
    y_far = rec.get("far_y")
    if y_far is None:
        y_far = rec.get("stop_row")
    y_near = height - 1.0 if y_near is None else float(y_near)
    y_far = 0.0 if y_far is None else float(y_far)
    angle = rec.get("angle_deg")
    angle = math.degrees(math.atan(-a)) if angle is None else float(angle)
    return a, b, min(y_far, y_near), max(y_far, y_near), angle


def track_view(cfg: VisionCfg, info: dict, half_px: float | None = None,
               ref: dict | None = None) -> TrackView | None:
    """把一条 boundary.info 变成 TrackView；一帧都没有可用边界就返回 None。"""
    if not info:
        return None
    w, h = cfg.size(info)
    u0 = 0.5 * w
    left = _side_rec(info.get("left"), cfg, h)
    right = _side_rec(info.get("right"), cfg, h)
    if left is None and right is None:
        return None
    src = "both" if (left and right) else ("left" if left else "right")
    absolute = True
    angle_l = left[4] if left else None
    angle_r = right[4] if right else None
    half_now = None
    vp = None
    if src == "both":
        denom = left[0] - right[0]
        # 近乎平行时交点对关键点噪声极敏感，不把它交给航向控制。
        if abs(denom) > 1e-6:
            vp_y = (right[1] - left[1]) / denom
            vp_x = left[0] * vp_y + left[1]
            if math.isfinite(vp_x) and math.isfinite(vp_y):
                vp = (vp_x, vp_y)
    use_vp = cfg.lookahead == "vanishing"
    if use_vp and vp is None:
        return None  # 单侧边界或无稳定交点，不能估算灭点航向。

    if src == "both":
        a = 0.5 * (left[0] + right[0])
        b = 0.5 * (left[1] + right[1])
        y_lo = max(left[2], right[2])           # 两侧线段范围的交集：最可信的区间
        y_hi = min(left[3], right[3])
        if y_lo > y_hi:                         # 不重叠就退到并集（外推一段）
            y_lo = min(left[2], right[2])
            y_hi = max(left[3], right[3])
    elif src == "left":
        a, b = left[0], left[1]
        y_lo, y_hi = left[2], left[3]
        if half_px:
            b += half_px                        # 左边界 + 半宽 = 中心线
        else:
            absolute = False
    else:
        a, b = right[0], right[1]
        y_lo, y_hi = right[2], right[3]
        if half_px:
            b -= half_px                        # 右边界 - 半宽 = 中心线
        else:
            absolute = False

    if y_hi <= y_lo:                            # 兜底，别让前视行夹到非法区间
        y_hi = y_lo + 1.0
    look_row = vp[1] if use_vp else clamp(cfg.lookahead * h, y_lo, y_hi)
    x_look = vp[0] if use_vp else a * look_row + b
    f = cfg.focal(w)
    e_deg = math.degrees(math.atan2(u0 - x_look, f))
    head_deg = None
    if src == "both":
        if vp is not None:
            head_deg = math.degrees(math.atan2(u0 - vp[0], f))
        # 前视行上的半宽：只跟这一行和路宽有关，和车偏多少、偏角多少无关，
        # 所以两侧都在时顺手量一次，单侧时拿它把缺的那条线补出来。
        xl = left[0] * look_row + left[1]
        xr = right[0] * look_row + right[1]
        if 1.0 < xr - xl < w:
            half_now = 0.5 * (xr - xl)

    if not absolute:
        # 单侧退化：保留相对角用于诊断，控制侧发现中心线不可用时停车。
        # 相对参考角的变化量当误差。左右边界的角度都随"车往右偏"变大（角度是
        # 远端偏向画面右的倾角），所以两侧同号，不用再分左右。
        side_angle = angle_l if src == "left" else angle_r
        side_ref = (ref or {}).get(src)
        if side_angle is None:
            return None
        e_deg = side_angle if side_ref is None else (side_angle - side_ref)

    return TrackView(a=a, b=b, y_lo=y_lo, y_hi=y_hi, src=src, absolute=absolute,
                     midline_deg=math.degrees(math.atan(-a)), look_row=look_row,
                     x_look=x_look, lat_px=x_look - u0, e_deg=e_deg, head_deg=head_deg,
                     half_px=half_now, angle_left=angle_l, angle_right=angle_r)


@dataclass
class Heading:
    """一拍的航向环输入（VisionHeading.step 的结果）。"""

    e: float                # 相对画幅中心的误差（度）：+ = 需要往左修；已滤波和外推
    e_raw: float            # 帧时刻相对画幅中心的原始几何误差（诊断用）
    rate: float             # 视觉差分算出的"往左转"角速度（度/秒）
    lat_px: float           # 目标点相对画面中心的像素（前视点或灭点）
    head_deg: float | None  # 灭点算的纯航向项
    src: str                # both / left / right
    absolute: bool          # False = 单侧相对角退化（控制侧会因此停车）
    age: float              # 距最近一帧有效边界多久（秒）
    new: bool               # 本拍是不是拿到了一帧新的边界


class ImuSource:
    """陀螺当角速度来源：heading 左转为正（度），rate 度/秒。"""

    def __init__(self, imu):
        self.imu = imu

    @property
    def heading(self) -> float:
        return self.imu.yaw

    @property
    def rate(self) -> float:
        return self.imu.yaw_rate


class WheelGapSource:
    """左右轮里程差（gap）当角速度来源 —— 不用陀螺，就是电机自己的数据。"""

    def __init__(self, track_mm=190.0, window=GAP_WINDOW, tau=GAP_TAU):
        self.track = max(float(track_mm), 1e-3)
        self.window = max(float(window), 0.02)
        self.tau = max(float(tau), 0.0)
        self.hist = deque(maxlen=600)
        self.heading = 0.0
        self.rate = 0.0
        self.valid = False           # 攒够一个窗口、算出过角速度才为 True

    def update(self, now: float, gap: float) -> float:
        """喂一条新的累计里程差（mm），返回平滑后的角速度（度/秒，左转为正）。"""
        self.heading = -math.degrees(gap / self.track)
        self.hist.append((now, gap))
        while len(self.hist) > 2 and self.hist[1][0] <= now - self.window:
            self.hist.popleft()
        t0, g0 = self.hist[0]
        dt = now - t0
        if dt >= 0.5 * self.window:                  # 攒够一个窗口再算差分
            raw = -math.degrees((gap - g0) / dt / self.track)
            alpha = clamp(dt / (self.tau + dt), 0.0, 1.0)
            self.rate += alpha * (raw - self.rate)
            self.valid = True
        return self.rate


def _num(v: float | None, fmt: str = ".1f") -> str:
    """CSV 用：不是有限数就写空，别让 nan 混进日志。"""
    return "" if v is None or not math.isfinite(v) else format(v, fmt)


class HeadingCascade:
    """航向环双环 PID：外环角度环（视觉）+ 内环角速度环（电机 gap / 陀螺）。"""

    kind = "cascade"
    yaw_i_frac = 0.5             # 外环积分贡献最多给到 w_max 的这个比例
    rate_i_frac = 0.6            # 内环积分贡献最多给到 trim_max 的这个比例

    def __init__(self, cfg: VisionCfg, src: str = "gap"):
        self.cfg = cfg
        self.src = src
        self.track = max(float(cfg.track_mm), 1e-3)
        # 1 度/秒的目标角速度，按差速运动学需要多少差动（mm/s）
        self.gain_ff = math.radians(1.0) * 0.5 * self.track
        self.i_yaw_max = (self.yaw_i_frac * cfg.w_max / cfg.ki_yaw
                          if cfg.ki_yaw > 0 else 0.0)
        self.i_rate_max = (self.rate_i_frac * cfg.trim_max / cfg.ki_rate
                           if cfg.ki_rate > 0 else 0.0)
        self.i_yaw = self.i_rate = 0.0
        self.reset()

    def reset(self) -> None:
        """每趟出发前清积分（换一趟不要带着上一趟的尾巴）。"""
        self.i_yaw = self.i_rate = 0.0
        self.w_ref = self.w_meas = self.w_err = 0.0
        self.trim_ff = self.trim_fb = self.trim = 0.0
        self.sat = False

    def step(self, e: float, w_meas: float, valid: bool, dt: float) -> float:
        """一拍：e = 角度误差（度，+ = 要往左修），w_meas = 实测角速度（度/秒）。"""
        cfg = self.cfg
        # 外环（角度环）：角度误差 -> 目标角速度
        w_ref = clamp(cfg.kp_yaw * e + cfg.ki_yaw * self.i_yaw,
                      -cfg.w_max, cfg.w_max)
        # 前馈：按轮距把目标角速度折算成差动
        trim_ff = cfg.kff * self.gain_ff * w_ref
        # 内环（角速度环）：角速度残差 -> 差动修正
        w_meas = float(w_meas)
        w_used = w_meas if valid else w_ref       # 还没有可信角速度：当残差 0
        w_err = w_ref - w_used
        trim_fb = (cfg.kp_rate * w_err + cfg.ki_rate * self.i_rate
                   - cfg.kd_rate * w_used)
        trim_raw = trim_ff + trim_fb
        trim = clamp(trim_raw, -cfg.trim_max, cfg.trim_max)
        self.sat = abs(trim_raw) >= cfg.trim_max
        # 抗积分饱和：外环顶到限幅、或内环整体贴限幅，就都别积分了
        if abs(w_ref) < cfg.w_max and not self.sat:
            self.i_yaw = clamp(self.i_yaw + e * dt, -self.i_yaw_max, self.i_yaw_max)
        if valid and not self.sat:
            self.i_rate = clamp(self.i_rate + w_err * dt,
                                -self.i_rate_max, self.i_rate_max)
        self.w_ref, self.w_meas, self.w_err = w_ref, w_meas, w_err
        self.trim_ff, self.trim_fb, self.trim = trim_ff, trim_fb, trim
        return trim


def make_heading_loop(cfg: VisionCfg):
    return HeadingCascade(cfg, src=cfg.rate_src)


class VisionHeading:
    """把摄像头最新的边界变成航向环的输入：角度误差 e（外环）+ 角速度（内环）。"""

    def __init__(self, cfg: VisionCfg, cam):
        self.cfg, self.cam = cfg, cam
        self.half_px = None         # 最近一次两侧都在时量到的前视行半宽
        self.ref = {}               # 单侧退化用的参考角 {'left': 度, 'right': 度}
        self.e = self.e_raw = 0.0
        self.rate = 0.0
        self.e_filt = None          # 低通之后的误差（帧时刻的值）
        self.src = None             # 角速度来源（gap / 陀螺），用它外推误差
        self.n_gated = 0            # 被跳变保护截断的帧数
        self.t_frame = 0.0          # 最近一帧的采集时刻（估）
        self._yaw_frame = None      # 那一时刻的 yaw（查不到就是 None，不做外推）
        self._yaw_hist = deque(maxlen=500)   # (t, yaw)：查"这帧拍的时候车头朝哪"
        self.view: TrackView | None = None
        self.n_frames = 0           # 一共用了多少帧新边界
        self.n_single = 0           # 其中只有单侧边界的帧数（用半宽补齐中心线）
        self.n_degraded = 0         # 其中连半宽都没有、只能按相对角算的帧数
        self._seq = 0
        self._prev = None           # (e, t)：视觉差分用

    # ---- 出发前检查跑道几何 ----
    def age(self, now=None) -> float:
        """距最近一帧有效边界多久（秒）。"""
        return self.cam.age(now)

    def wait_track(self, seconds: float = TRACK_WAIT, log=print, stop_event=None) -> TrackView | None:
        """等待可信的跑道几何并记住半宽；误差始终以画幅中心为基准。"""
        end = time.monotonic() + max(seconds, 0.0)
        saw_relative = False
        while True:
            check_cancel(stop_event)
            s = self.cam.sample()
            if s.info is not None and self.age() <= self.cfg.lost_stop:
                v = track_view(self.cfg, s.info, self.half_px, self.ref)
                if v is not None:
                    self._remember(v)
                    if v.absolute:
                        w, _ = self.cfg.size(s.info)
                        log(f"  视觉参考固定为画幅中心 x={0.5 * w:g}px，"
                            f"当前误差 {v.e_deg:+.2f}°")
                        return v
                    saw_relative = True
            if time.monotonic() >= end:
                if saw_relative:
                    log("  只看到一侧边界，也没有两侧都在时量到的半宽，"
                        "算不出跑道中心线")
                return None
            wait_cancelable(0.03, stop_event)

    def _remember(self, v: TrackView) -> None:
        """记住半宽和单侧参考角，供单侧/退化时用。"""
        if v.half_px:
            self.half_px = v.half_px
        if v.angle_left is not None:
            self.ref.setdefault("left", v.angle_left)
        if v.angle_right is not None:
            self.ref.setdefault("right", v.angle_right)

    def attach_imu(self, imu) -> None:
        """挂上陀螺当角速度来源（等价于 --rate-src gyro）。"""
        self.attach_source(ImuSource(imu))

    def attach_source(self, src) -> None:
        """挂一路"快"的角速度来源（要有 .heading 属性，左转为正，单位度）。

        谁挂上就用谁把视觉误差外推到当前时刻：补视觉的 0.1~0.2s 延时，
        这是治左右摆关键的一步。传 None 就是不外推（纯视觉，增益要保守）。
        """
        self.src = src
        self._yaw_hist.clear()
        self._yaw_frame = None

    def reset_run(self) -> None:
        """每趟出发前清一下差分状态，别把上一趟的尾巴带进来。"""
        self._prev = None
        self.rate = 0.0
        self.e_filt = None
        self.t_frame = 0.0
        self._yaw_frame = None
        # 减一：让第一拍就把当前这帧用上（车在静止等出发，这帧就是最新的）
        self._seq = self.cam.sample().seq - 1

    # ---- 内部：误差的"现在值" ----
    def _yaw_at(self, t: float) -> float | None:
        """t 时刻的 yaw；历史里没有够老的数据（刚挂上陀螺）就返回 None。"""
        for ts, yaw in reversed(self._yaw_hist):
            if ts <= t:
                return yaw if t - ts < 0.25 else None
        return None

    def _predict(self, now: float, yaw_now: float) -> float:
        """帧时刻的误差 + 陀螺从那时到现在的转动量 = 这一刻的误差。

        e 的定义是"要往左修多少度"，而车往左转（yaw 增大）会让 e 变小，所以要
        减去 dir_sign×(yaw_now − yaw_frame)。
        """
        e = self.e_filt if self.e_filt is not None else 0.0
        if self.src is not None and self.cfg.predict and self._yaw_frame is not None:
            e -= self.cfg.dir_sign * (yaw_now - self._yaw_frame)
        return e

    # ---- 每拍读一次 ----
    def step(self, now: float | None = None) -> Heading | None:
        """读最新的一帧边界；挂了角速度来源（gap/陀螺）时每拍都外推到"现在"。"""
        now = time.time() if now is None else now
        yaw_now = 0.0
        if self.src is not None:
            yaw_now = self.src.heading
            self._yaw_hist.append((now, yaw_now))
        s = self.cam.sample()
        new = False
        if s.info is not None and s.seq != self._seq and self.age(now) <= self.cfg.lost_stop:
            self._seq = s.seq
            v = track_view(self.cfg, s.info, self.half_px, self.ref)
            if v is None and self.cfg.lookahead == "vanishing":
                # HTTP 仍有新帧不等于灭点仍有效；清掉旧结果，让控制侧停车。
                self.view = None
                return None
            if v is not None:
                new = True
                self.n_frames += 1
                self.n_single += 1 if v.src != "both" else 0
                self.n_degraded += 1 if not v.absolute else 0
                self._remember(v)
                e_raw = v.e_deg
                e_frame = e_raw  # 固定以画幅中心为目标，保留起步时已有的偏移。
                # 坏帧保护：一帧之差超过 --e-gate 度就截断。关键点预测偶尔会跳一下，
                # 单帧跳变直接进 PID 就是把方向盘掰一下，是"抖/摆"的常见来源。
                if self.e_filt is not None and self.cfg.e_gate > 0:
                    pred = self._predict(now, yaw_now)
                    jump = e_frame - pred
                    if abs(jump) > self.cfg.e_gate:
                        e_frame = pred + math.copysign(self.cfg.e_gate, jump)
                        self.n_gated += 1
                # 低通：压掉逐帧预测噪声（有陀螺外推时这点滞后会被补回来）
                dt_v = clamp(now - self.t_frame, 1e-3, 1.0) if self.t_frame else 0.0
                if self.e_filt is None or self.cfg.e_tau <= 0 or not dt_v:
                    self.e_filt = e_frame
                else:
                    alpha = dt_v / (self.cfg.e_tau + dt_v)
                    self.e_filt += alpha * (e_frame - self.e_filt)
                # 视觉角速度：车往左转时误差 e 变小，所以 rate = -de/dt
                if self._prev is not None:
                    dt = now - self._prev[1]
                    if 0.02 <= dt <= 0.6:
                        raw = clamp(-(self.e_filt - self._prev[0]) / dt,
                                    -RATE_MAX, RATE_MAX)
                        alpha = dt / (self.cfg.rate_tau + dt)
                        self.rate += alpha * (raw - self.rate)
                self._prev = (self.e_filt, now)
                # 记下这帧的采集时刻和当时的车头朝向，供后面每拍外推
                lag = float(getattr(s, "frame_dt", 0.0) or 0.0)
                t_cap = max(now - lag, self.t_frame)
                self._yaw_frame = self._yaw_at(t_cap) if self.src is not None else None
                self.t_frame = t_cap
                self.e_raw, self.view = e_raw, v
        if self.view is None:
            return None
        self.e = self._predict(now, yaw_now)
        return Heading(e=self.e, e_raw=self.e_raw, rate=self.rate,
                       lat_px=self.view.lat_px, head_deg=self.view.head_deg,
                       src=self.view.src, absolute=self.view.absolute,
                       age=self.age(now), new=new)


LOG_COLUMNS = ['t', 'dist_mm', 'e_deg', 'e_raw_deg', 'lat_px', 'head_deg', 'w_ref_dps', 'w_meas_dps', 'w_err_dps', 'trim_ff', 'trim_fb', 'trim_mms', 'v_l', 'v_r', 'set_l', 'set_r', 'gap_mm', 'src']


class _RunLog:
    """逐拍写 CSV（不给路径就是空操作），列见 LOG_COLUMNS。"""

    def __init__(self, path):
        self.fp = open(path, "w", newline="") if path else None
        self.wr = csv.writer(self.fp) if self.fp else None
        if self.wr:
            self.wr.writerow(LOG_COLUMNS)

    def row(self, *vals) -> None:
        if self.wr:
            self.wr.writerow(vals)

    def close(self) -> None:
        if self.fp:
            self.fp.close()
            self.fp = None
            self.wr = None


def vision_straight(board, heading: VisionHeading, imu=None, goal_mm: float | None = None,
                    log=print, log_path=None, stop_event=None):
    """按跑道中心线前视点或灭点控制直走（只支持前进）。"""
    rl = None
    stopped = False
    try:
        check_cancel(stop_event)
        if goal_mm is not None and goal_mm <= 0:
            raise ValueError("视觉走直线只支持前进（goal_mm > 0），倒车请用 straight_pid")

        cfg = heading.cfg
        s_sign = straight_pid.FORWARD_SIGN
        mmc = straight_pid.MM_PER_COUNT
        limit = straight_pid.SPEED_LIMIT

        # 内环反馈来自电机里程差或 IMU 融合角速度。
        gap_src = None
        if cfg.rate_src == "gyro":
            if imu is None:
                raise RuntimeError("--rate-src gyro 需要 IMU；改用 --rate-src gap 或接上陀螺")
            heading.attach_source(ImuSource(imu))
        elif cfg.rate_src == "gap":
            gap_src = WheelGapSource(cfg.track_mm, cfg.gap_window, cfg.gap_tau)
            heading.attach_source(gap_src)
        else:
            raise ValueError("rate_src 只能是 gap / gyro")
        loop = make_heading_loop(cfg)
        loop.reset()
        heading.reset_run()

        # ---- 取基准计数（上电后 $MAll 是累计值，先读一次做零点）----
        base, end = None, time.time() + 3.0
        while base is None and time.time() < end:
            check_cancel(stop_event)
            totals, _ = board.feedback(0.1)
            check_cancel(stop_event)
            if totals is not None:
                base = [totals[i] * s_sign[i] for i in range(4)]
        if base is None:
            raise RuntimeError("读不到编码器 $MAll：检查接线/供电，以及串口是不是驱动板")
        base_l = (base[0] + base[1]) / 2.0
        base_r = (base[2] + base[3]) / 2.0

        def travel(totals):
            return (((totals[0] * s_sign[0] + totals[1] * s_sign[1]) / 2.0 - base_l) * mmc,
                    ((totals[2] * s_sign[2] + totals[3] * s_sign[3]) / 2.0 - base_r) * mmc)

        spd_l = straight_pid.PID(straight_pid.KP_SPD, straight_pid.KI_SPD, 0.0,
                                 straight_pid.CORR_MAX)
        spd_r = straight_pid.PID(straight_pid.KP_SPD, straight_pid.KI_SPD, 0.0,
                                 straight_pid.CORR_MAX)
        v_hist_l, v_hist_r = [], []
        pos_cmd = 0.0
        dist = gap = v_meas = trim = 0.0
        e_now = rate_now = e_max = trim_sum = 0.0
        w_ref_max = w_err_max = trim_ff_sum = 0.0
        w_err_n = trim_n = trim_sat = 0
        lost_max = 0.0
        reason = ""
        t_start = t_prev = time.time()
        t_log = t_prev
        t_data = t_prev                     # 最近一次收到编码器数据
        deadline = (t_start + max(10.0, 3.0 * goal_mm / max(cfg.speed, 1.0) + 5.0)
                    if goal_mm else float("inf"))
        rl = _RunLog(log_path)

        while True:
            check_cancel(stop_event)
            totals, tep = board.feedback()          # 阻塞到收到一条新数据，约 100Hz
            check_cancel(stop_event)
            now = time.time()
            dt = clamp(now - t_prev, 1e-4, 0.05)
            t_prev = now
            if now > deadline:
                reason = "总时长超时，已强制停车"
                break
            if totals is None:
                if now - t_data > 0.5:
                    reason = "编码器数据中断 0.5s 以上，已停车"
                    break
                continue
            t_data = now

            # ---- 反馈：左右里程 / 进度 / 左右轮速（都来自编码器）----
            d_l, d_r = travel(totals)
            dist = (d_l + d_r) / 2.0
            gap = d_l - d_r
            if tep is not None:
                v_hist_l.append((tep[0] * s_sign[0] + tep[1] * s_sign[1]) / 2.0
                                / straight_pid.TEP_WINDOW * mmc)
                v_hist_r.append((tep[2] * s_sign[2] + tep[3] * s_sign[3]) / 2.0
                                / straight_pid.TEP_WINDOW * mmc)
                del v_hist_l[:-straight_pid.SPD_SAMPLES]
                del v_hist_r[:-straight_pid.SPD_SAMPLES]
            v_l = sum(v_hist_l) / len(v_hist_l) if v_hist_l else 0.0
            v_r = sum(v_hist_r) / len(v_hist_r) if v_hist_r else 0.0
            v_meas = (v_l + v_r) / 2.0
            if gap_src is not None:                  # --rate-src gap 才用电机的里程差
                gap_rate = gap_src.update(now, gap)

            # ---- 航向反馈：视觉中心线 -> 角度误差 ----
            hd = heading.step(now)
            if hd is None:
                reason = ("视觉没有可用灭点：需要两侧边界且延长线有稳定交点，已停车"
                          if cfg.lookahead == "vanishing" else
                          "视觉一直没有可用边界：确认边界模型开着、车在跑道上")
                break
            if not heading.view.absolute:
                reason = ("只有一侧边界、也没有两侧都在时量到的半宽基准，算不出跑道"
                          "中心线，已停车：把车摆到两侧都进画面的位置再出发")
                break
            lost_max = max(lost_max, hd.age)
            if hd.age > cfg.lost_stop:
                reason = (f"连续 {hd.age:.1f}s 没有新边界（模型关了/车偏出跑道/欠曝），"
                          f"已停车")
                break
            if abs(hd.e) > cfg.max_dev:
                reason = (f"视觉角度误差 {hd.e:+.1f}° 超过 {cfg.max_dev:.0f}°，已急停："
                          f"方向可能接反（用 --dir-sign -1 反过来）或车被拨偏了")
                break
            if cfg.rate_src == "gyro" and imu.age() > IMU_STALE:
                reason = f"IMU 数据中断 {imu.age():.2f}s，已急停"
                break

            # ---- ② 航向环：角度误差 -> 目标角速度 -> 差动修正 ----
            # e 按 dir_sign 折算到物理转向（e 正 = 需要往左修）。双环的内环工作在
            # 物理量上：ω_meas 正 = 正在往左转、trim 正 = 右轮快 = 往左转，所以
            # 内环反馈已经是物理转向，不再乘方向符号。
            e_now = cfg.dir_sign * hd.e
            if cfg.e_dead > 0:                      # 画幅中心附近别较劲（差动死区/摩擦）
                if abs(e_now) <= cfg.e_dead:
                    e_now = 0.0
                else:
                    e_now -= math.copysign(cfg.e_dead, e_now)
            if cfg.rate_src == "gyro":
                rate_raw, rate_ok = imu.yaw_rate, True
            elif cfg.rate_src == "gap":
                rate_raw, rate_ok = gap_rate, gap_src.valid
            trim = loop.step(e_now, rate_raw, rate_ok, dt)
            rate_now = loop.w_meas                  # 真正进了控制的那一路角速度

            # ---- ① 位置环（可选）：剩余距离 -> 目标线速度 ----
            if goal_mm is not None:
                if goal_mm - dist <= straight_pid.BRAKE_TAU * abs(v_meas) + cfg.margin:
                    break                            # 进入断速滑行区
                want = clamp(cfg.kp_pos * (goal_mm - dist), 0.0, cfg.speed)
                pos_cmd = clamp(want, max(0.0, pos_cmd - cfg.accel * dt),
                                pos_cmd + cfg.accel * dt)
                v = max(pos_cmd, min(straight_pid.SPEED_MIN, cfg.speed))
            else:
                pos_cmd = clamp(cfg.speed, max(0.0, pos_cmd - cfg.accel * dt),
                                pos_cmd + cfg.accel * dt)
                v = pos_cmd
            set_l = clamp(v - trim, -limit, limit)
            set_r = clamp(v + trim, -limit, limit)

            # ---- ③ 速度环：按单侧目标速度闭环 ----
            cmd_l = set_l + spd_l.step(set_l - v_l, dt)
            cmd_r = set_r + spd_r.step(set_r - v_r, dt)
            check_cancel(stop_event)
            board.spd(cmd_l * s_sign[0], cmd_l * s_sign[1],
                      cmd_r * s_sign[2], cmd_r * s_sign[3])

            e_max = max(e_max, abs(e_now))
            if math.isfinite(loop.w_ref):
                w_ref_max = max(w_ref_max, abs(loop.w_ref))
            if math.isfinite(loop.w_err):
                w_err_max, w_err_n = max(w_err_max, abs(loop.w_err)), w_err_n + 1
            trim_sum += abs(trim)
            trim_ff_sum += abs(loop.trim_ff) if math.isfinite(loop.trim_ff) else 0.0
            trim_n += 1
            trim_sat += 1 if abs(trim) >= 0.98 * cfg.trim_max else 0
            rl.row(f"{now - t_start:.3f}", f"{dist:.1f}", f"{e_now:.3f}", f"{hd.e_raw:.3f}",
                   f"{hd.lat_px:.1f}", "" if hd.head_deg is None else f"{hd.head_deg:.2f}",
                   _num(loop.w_ref, ".2f"), _num(loop.w_meas, ".2f"),
                   _num(loop.w_err, ".2f"), _num(loop.trim_ff, ".1f"),
                   _num(loop.trim_fb, ".1f"), f"{trim:.1f}",
                   f"{v_l:.1f}", f"{v_r:.1f}", f"{set_l:.0f}", f"{set_r:.0f}",
                   f"{gap:.1f}", hd.src)

            if now - t_log > 0.5:                    # 半秒一行，盯直线性和视觉状态
                t_log = now
                w_txt = (f"{loop.w_ref:+5.1f}/{rate_now:+5.1f}"
                         if math.isfinite(loop.w_ref) else f"  —  /{rate_now:+5.1f}")
                ff_txt = (f"{loop.trim_ff:+5.1f}"
                          if math.isfinite(loop.trim_ff) else "  —  ")
                log(f"  {now - t_start:5.1f}s"
                    + (f"  剩余 {goal_mm - dist:6.0f}mm" if goal_mm else
                       f"  里程 {dist / 1000:5.2f}m")
                    + f"  速度 {v_meas:5.0f}mm/s"
                    f"  误差 {e_now:+6.2f}°"
                    f"  角速度 {w_txt}°/s（目标/实测）"
                    f"  修正 {trim:+6.1f}mm/s（前馈 {ff_txt} 反馈 {loop.trim_fb:+5.1f}）"
                    f"  {hd.src}"
                    f"  {'灭点' if cfg.lookahead == 'vanishing' else '中线'}偏画面 {hd.lat_px:+5.0f}px")
        board.spd(0, 0, 0, 0)
        stopped = True

        # 断速后车还会滑一段，等轮子真不转了再报最终位置
        end = time.time() + 1.5
        prev, still = dist, 0
        while still < 3 and time.time() < end:
            check_cancel(stop_event)
            totals, _ = board.feedback(0.05)
            check_cancel(stop_event)
            if totals is None:
                continue
            d_l, d_r = travel(totals)
            dist = (d_l + d_r) / 2.0
            gap = d_l - d_r
            still = still + 1 if abs(dist - prev) < 0.2 else 0
            prev = dist

        check_cancel(stop_event)
        info = {"reason": reason, "elapsed": time.time() - t_start, "dist": dist,
                "e_max": e_max, "frames": heading.n_frames,
                "single": heading.n_single, "degraded": heading.n_degraded,
                "gated": heading.n_gated,
                "lost_max": lost_max, "gap_max": abs(gap),
                "trim_avg": trim_sum / trim_n if trim_n else 0.0,
                "trim_sat": 100.0 * trim_sat / trim_n if trim_n else 0.0,
                "loop": loop.kind, "w_ref_max": w_ref_max,
                "w_err_max": w_err_max if w_err_n else None,
                "trim_ff_avg": trim_ff_sum / trim_n if trim_n else 0.0,
                "log_path": log_path, "rate_src": cfg.rate_src, "imu": imu is not None}
        return dist, info

    finally:
        actions = []
        if not stopped:
            actions.append(("视觉停车", lambda: board.spd(0, 0, 0, 0)))
        if rl is not None:
            actions.append(("视觉日志", rl.close))
        cleanup(*actions)


def describe(info: dict) -> str:
    """把诊断字典格式化成一行中文（和 straight_pid.describe 一个风格）。"""
    if info.get("loop") == "cascade":
        src_name = "电机 gap" if info.get("rate_src") == "gap" else "陀螺"
        w_err = info.get("w_err_max")
        extra = (f"；航向环双环：外环 ω 目标最大 {info['w_ref_max']:.1f}°/s，"
                 f"内环（{src_name}）残差最大 "
                 + ("—" if w_err is None else f"{w_err:.1f}°/s")
                 + f"，前馈均值 {info['trim_ff_avg']:.0f}mm/s")
    else:
        extra = ""
    if info.get("single"):
        extra += f"；其中 {info['single']} 帧只有单侧边界（用半宽补齐中心线）"
    if info.get("degraded"):
        extra += f"；{info['degraded']} 帧只能按单侧相对角估计（诊断用）"
    if info.get("lost_max", 0.0) > 0.01:
        extra += f"；最长一次 {info['lost_max']:.2f}s 没有新边界"
    if info.get("gated"):
        extra += f"；{info['gated']} 帧的误差跳变被截断（关键点预测偶发跳变，已挡掉）"
    return (f"最大角度误差 {info['e_max']:.2f}°，视觉基准为画幅中心，"
            f"用了 {info['frames']} 帧边界；航向修正均值 {info['trim_avg']:.0f}mm/s、"
            f"贴限幅 {info['trim_sat']:.0f}% 的时间{extra}")
