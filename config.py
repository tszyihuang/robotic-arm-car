"""现场调参入口。距离 m，底盘速度 mm/s，关节与转弯角度 °。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEP_GAP = 0.5  # 沿用任务表每步间隔；干跑不等待。

BASE = {
    "motor_port": "",  # 留空自动识别；实机建议填稳定串口路径。
    "baudrate": 115200,
    "speed": 150.0,
    "turn_speed": 150.0,
    "turn_spin_speed": 150.0,
    "turn_radius": 0.3,
    "turn_track_width": 0.41,
    "turn_timeout": None,
    "meters_per_count": 0.00016029,
    "feedback_stale": 0.3,
    "feedback_timeout": 0.08,
    "align_rate_src": "auto",
}

STRAIGHT = {
    "FORWARD_SIGN": (1, 1, 1, 1), "TRACK_MM": 190.0,
    "SPEED_CRUISE": 100.0, "SPEED_MIN": 60.0, "SPEED_LIMIT": 1000.0,
    "ACCEL": 300.0, "MARGIN_MM": 1.0, "KP_POS": 2.0, "TRIM_MAX": 100.0,
    "KP_GAP": 0.30, "KI_GAP": 0.20, "KD_GAP": 0.45, "GAP_DEV_MAX": 300.0,
    "KP_SPD": 0.60, "KI_SPD": 0.80, "CORR_MAX": 40.0,
    "BRAKE_TAU": 0.12, "TEP_WINDOW": 0.010, "SPD_SAMPLES": 3,
}

TURN = {
    # 圆弧原算法独立使用 6379 count/m；本次不重新标定。
    "COUNTS_PER_METER": 6379.0, "LOOP_HZ": 100.0,
    "MAX_WHEEL_SPEED": 0.45, "ACCEL": 0.3, "ANGULAR_ACCEL": 3.0,
    "ANGLE_KP": 1.5, "PATH_KP": 8.0, "PATH_KD": 1.5, "RATE_KP": 0.5,
    "WHEEL_SPEED_KP": 0.6, "BRAKE_LEAD_SECONDS": 0.02,
    "BRAKE_KP": 3000.0, "STOP_ANGLE_DEG": 0.6, "SETTLE_SECONDS": 0.25,
}

POSITION = {
    "impact_threshold": 150.0, "max_distance_m": 1.0, "timeout": 20.0,
    "boot_wait": 0.0, "poll_interval": 0.005,
}

VISION_CONTROL = {
    "FRAME_W": 1280, "FRAME_H": 720, "LOOKAHEAD": 0.5, "FOV_DEG": 70.0,
    "E_TAU": 0.12, "E_GATE": 10.0,
    "E_DEAD": 0.0, "RATE_SRC": "gap", "GAP_WINDOW": 0.08, "GAP_TAU": 0.05,
    "LOST_STOP": 0.6, "MAX_DEV": 45.0, "TRACK_WAIT": 0.5,
    "KP_YAW": 10.0, "KI_YAW": 0.0, "YAW_W_MAX": 40.0,
    "KP_RATE": 10.0, "KI_RATE": 0.0, "KFF": 1.0, "KD_RATE": 0.0,
    "TRIM_MAX": 100.0, "MIN_CONF": 0.0, "dir_sign": 1.0,
}

ALIGN = {
    # 航向环增益共用 VISION_CONTROL；这里只配置对正参考、速度和停车条件。
    "REF": "e", "REF_TAU": 0.35, "REF_GATE": 8.0, "BIAS": 0.0,
    "SPIN_SPEED": 150.0, "MIN_U": 0.0, "TOL": 2.0, "RATE_TOL": 8.0,
    "SETTLE": 0.4, "TIMEOUT": 20.0, "MAX_ROT": 90.0, "MAX_DEV": 60.0,
    "DIVERGE": 12.0, "LOOP_HZ": 100.0, "PRINT_INTERVAL": 0.5,
}

ARM = {
    "port": "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBVQPDL-if00-port0",
    "baudrate": 921600, "serial_timeout": 0.1, "serial_latency_ms": 1,
    "joint_limits": {1: None, 2: (-5.0, 180.0), 3: (0.0, 160.0), 4: (-120.0, 120.0)},
    "joint_signs": {1: 1, 2: 1, 3: -1, 4: 1},
    "joint_offsets_deg": {1: 0.0, 2: 0.0, 3: 160.0, 4: 24.0},
    "encoder_zero_deg": None, "speed_rpm": 10.0,
    "arrival_tolerance_deg": 2.0, "move_timeout": 30.0,
    "gripper_port": None, "gripper_baudrate": None, "gripper_servo_id": 2,
    "gripper_speed": 1000, "gripper_timeout": 0.1,
    "gripper_open_angle_deg": 291.0, "gripper_close_angle_deg": 243.0,
    "gripper_release_on_close": False,
}

SERVO = {
    "default_id": 1, "baudrate": 1000000, "speed": 1000, "timeout": 0.1,
    "fixed_positions": {1: 340},  # 工具旋转舵机保持原位置，夹爪 ID2 自由运动。
    "probe_baudrates": (1000000, 115200, 500000, 250000, 128000, 76800, 57600, 38400),
}

SENSOR = {
    "imu_port": "", "baudrate": 115200, "hz": 100, "gyro_axis": 2,
    "stale": 0.3, "MAX_YAW_RATE": 1200.0, "MIN_YAW_STEP": 8.0,
    "MAX_YAW_STEP": 60.0, "GLITCH_HOLD": 0.15, "YAW_RATE_WINDOW": 0.02,
}

VISION = {
    "device": 0, "infer_device": "cpu", "fp16": True,
    "width": 1920, "height": 1080, "frame_stale": 0.6,
    "scan_timeout": 30.0, "observe_timeout": 10.0,
    "observe_stable_frames": 3, "target_min_area_ratio": 0.001,
    "model_wait": 60.0,
    "boundary_weights": ROOT / "vision/models/boundary/weights.pt",
    "boundary_args": ROOT / "vision/models/boundary/args.yaml",
    "objects_weights": ROOT / "vision/models/objects/weights.pt",
    "objects_args": ROOT / "vision/models/objects/args.yaml",
}
