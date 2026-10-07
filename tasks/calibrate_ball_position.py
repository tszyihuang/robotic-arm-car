"""任务表的小球位置校准入口；底盘执行编码器位置 PID 与轮速 PI。"""
from base.ball_position import validate_config
from base.control import check_cancel, cleanup


def run(base, vision, *, stop_event=None, config=None, log=print):
    stop_event = stop_event if stop_event is not None else getattr(base, "stop_event", None)
    try:
        check_cancel(stop_event)
        cfg = validate_config(config)
        if getattr(base, "dry_run", False):
            log(f"位置校准 PID：中间小球偏左 → 后退，偏右 → 前进；"
                f"视觉换算 {cfg['mm_per_px']:g} mm/px → 编码器目标位置 → 位置 PID → 轮速 PI；"
                f"Kp={cfg['kp']:g}(1/s)，Ki={cfg['ki']:g}(1/s²)，Kd={cfg['kd']:g}，"
                f"最大速度 {cfg['speed']:g} mm/s，容差 ±{cfg['tolerance_px']:g}px，"
                f"位置容差 ±{cfg['position_tolerance_mm']:g}mm，"
                f"起步速度下限 {min(cfg['min_speed'], cfg['speed']):g} mm/s，"
                f"主机轮速 Kp={cfg['speed_kp']:g}，Ki={cfg['speed_ki']:g}；"
                f"提前制动，停稳后确认连续 {cfg['stable_frames']} 个新帧到位")
            return None
        return base.calibrate_ball_position(vision, config=cfg, stop_event=stop_event, log=log)
    finally:
        cleanup(("小球校准停车", base.stop))
