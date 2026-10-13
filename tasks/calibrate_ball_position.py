"""任务表的小球位置校准入口；底盘执行编码器位置 PID 与轮速 PI。"""
from base.ball_position import validate_config
from base.control import check_cancel, cleanup
from vision.targets import validate_target


def run(base, vision, *, stop_event=None, config=None, log=print):
    return _run(base, vision, stop_event=stop_event, config=config, log=log,
                kind="ball", label="小球", validator=validate_config)


def _run(base, vision, *, stop_event, config, log, kind, label, validator, target_value=None):
    stop_event = stop_event if stop_event is not None else getattr(base, "stop_event", None)
    try:
        check_cancel(stop_event)
        cfg = validator(config)
        if target_value is not None:
            validate_target(kind, target_value)
        target_label = f"指定{label}（{target_value}）" if target_value is not None else f"中间{label}"
        if getattr(base, "dry_run", False):
            log(f"位置校准 PID：{target_label}偏左 → 后退，偏右 → 前进；"
                f"视觉换算 {cfg['mm_per_px']:g} mm/px → 编码器目标位置 → 位置 PID → 轮速 PI；"
                f"Kp={cfg['kp']:g}(1/s)，Ki={cfg['ki']:g}(1/s²)，Kd={cfg['kd']:g}，"
                f"最大速度 {cfg['speed']:g} mm/s，容差 ±{cfg['tolerance_px']:g}px，"
                f"位置容差 ±{cfg['position_tolerance_mm']:g}mm，"
                f"起步速度下限 {min(cfg['min_speed'], cfg['speed']):g} mm/s，"
                f"主机轮速 Kp={cfg['speed_kp']:g}，Ki={cfg['speed_ki']:g}；"
                f"提前制动，停稳后确认连续 {cfg['stable_frames']} 个新帧到位；"
                f"连续缺失 {cfg['missing_frames']} 帧先暂停，恢复确认 {cfg['recovery_frames']} 帧，"
                f"持续 {cfg['lost_timeout']:g}s 无有效完整画面才终止")
            return None
        kwargs = {} if target_value is None else {"target_shape": target_value}
        return getattr(base, f"calibrate_{kind}_position")(
            vision, config=cfg, stop_event=stop_event, log=log, **kwargs)
    finally:
        cleanup((f"{label}校准停车", base.stop))
