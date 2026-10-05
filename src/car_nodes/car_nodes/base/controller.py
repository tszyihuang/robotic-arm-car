"""Dispatch a base target to the original controllers using node-owned devices."""
from ..common.control import check_cancel, wait_cancelable
from . import straight_pid as straight
from . import arc_turn, calibrate_position, vision_align, vision_straight
from .settings import DEFAULTS, _ALIGN_FIELDS


def execute(step, board, sensors, vision, stop_event, log=print):
    params = {**DEFAULTS, **step.kwargs}
    check_cancel(stop_event)
    straight.MM_PER_COUNT = params['meters_per_count'] * 1000
    board.stop()
    try:
        if step.command == 'straight':
            distance = step.args[0]
            gains = {key: params['straight_' + key] if params['straight_' + key] is not None
                     else getattr(straight, key.upper()) for key in ('kp_gap', 'ki_gap', 'kd_gap')}
            measured, info = straight.straight(board, distance * 1000, params['speed'],
                                               **gains, log=log, stop_event=stop_event)
            tolerance = max(0.03, 0.03 * abs(distance))
            return {'ok': not info['reason'] and abs(measured / 1000 - abs(distance)) <= tolerance,
                    'distance_m': measured / 1000 * (1 if distance > 0 else -1), **info}
        if step.command == 'turn':
            # Preserve the old encoder fallback when no fresh IMU is available.
            imu = sensors if sensors.has_data() else None
            limit = params['turn_max_speed']
            if step.args[1] == 0 and params['turn_spin_speed'] is not None:
                spin_limit = params['turn_spin_speed'] / 1000
                limit = min(limit, spin_limit) if limit is not None else spin_limit
            return arc_turn.turn_with_radius(step.args[1], step.args[0], board=board, imu=imu,
                cruise=params['speed'] / 1000, max_speed=limit, timeout_s=params['turn_timeout'],
                track_width_m=params['turn_track_width'], verbose=params['verbose'],
                stop_event=stop_event)
        if step.command == 'calibrate-position':
            sensors.impact_threshold = params.get('impact_threshold', calibrate_position.IMPACT_THRESHOLD)
            wait_cancelable(params.get('boot_wait', 0.0), stop_event)
            return calibrate_position.calibrate_position(board, sensors,
                speed=params['speed'], max_distance_mm=params.get('max_distance_m', 1.0) * 1000,
                timeout=params.get('timeout', 20.0), accel=params.get('accel', straight.ACCEL),
                **{key: params.get(key, getattr(straight, key.upper()))
                   for key in ('kp_gap', 'ki_gap', 'kd_gap')},
                log=log, stop_event=stop_event)
        if not vision.wait_ready(params['vision_model_wait'], log=log, stop_event=stop_event):
            check_cancel(stop_event)
            raise RuntimeError('视觉模型或新边界未就绪')
        cfg = vision_straight.VisionCfg(speed=params['speed'], track_mm=straight.TRACK_MM)
        if params['vision_lookahead'] is not None:
            value = params['vision_lookahead']
            cfg.lookahead = value if value == 'vanishing' else float(value)
        if params['vision_dir_sign'] is not None:
            cfg.dir_sign = params['vision_dir_sign']
        heading = vision_straight.VisionHeading(cfg, vision)
        if step.command == 'align':
            rate_source = params['align_rate_src']
            if rate_source == 'gyro' and not sensors.has_data():
                raise RuntimeError('视觉对正需要新鲜的 IMU 数据')
            imu = sensors if rate_source != 'gap' and sensors.has_data() else None
            cfg.rate_src = 'gyro' if imu else 'gap'
            align = vision_align.AlignCfg(**{
                key: params['align_' + key] for key in _ALIGN_FIELDS
                if params.get('align_' + key) is not None})
            info = vision_align.vision_align(board, heading, align, imu=imu,
                                            log=log, stop_event=stop_event)
            return {'ok': info['ok'], 'vision': info}
        if step.command == 'vision-straight':
            if heading.wait_track(log=log, stop_event=stop_event) is None:
                raise RuntimeError('出发前未拿到可用跑道中心线')
            distance = step.args[0]
            measured, info = vision_straight.vision_straight(board, heading, goal_mm=distance * 1000,
                                                          log=log, stop_event=stop_event)
            tolerance = max(0.03, 0.03 * distance)
            return {'ok': not info['reason'] and abs(measured / 1000 - distance) <= tolerance,
                    'distance_m': measured / 1000, 'vision': info}
        raise ValueError(f'未知底盘动作：{step.command}')
    finally:
        # End a task without closing the motor port or sensor subscriptions.
        try:
            board.stop()
        finally:
            if step.command == 'calibrate-position':
                board.release()
