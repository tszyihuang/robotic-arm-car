"""任务表的小球位置校准入口；连续 PID 控制由底盘执行。"""
from base.ball_position import validate_config
from base.control import check_cancel, cleanup


def run(base, vision, *, stop_event=None, config=None, log=print):
    stop_event = stop_event if stop_event is not None else getattr(base, "stop_event", None)
    try:
        check_cancel(stop_event)
        cfg = validate_config(config)
        if getattr(base, "dry_run", False):
            log(f"位置校准 PID：中间小球偏左 → 后退，偏右 → 前进；"
                f"Kp={cfg['kp']:g}，Ki={cfg['ki']:g}，Kd={cfg['kd']:g}，"
                f"最大速度 {cfg['speed']:g} mm/s，容差 ±{cfg['tolerance_px']:g}px，"
                f"连续 {cfg['stable_frames']} 个新帧到位后停车")
            return None
        return base.calibrate_ball_position(vision, config=cfg, stop_event=stop_event, log=log)
    finally:
        cleanup(("小球校准停车", base.stop))
