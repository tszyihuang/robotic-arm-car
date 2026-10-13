"""现场调参入口。距离 m，底盘速度 mm/s，关节与转弯角度 °。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEP_GAP = 0.05  # 沿用任务表每步间隔；干跑不等待。

BASE = {
    "motor_port": "",  # 留空自动识别；实机建议填稳定串口路径。
    "baudrate": 115200,
    "write_timeout": 1.0,  # 底盘串口单次写入最长等待秒数；原为固定 0.2 s。
    "speed": 300.0,  # 普通直走／视觉直走共用速度，mm/s；倒车校准独立配置。
    "turn_speed": 250.0,
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
    # ACCEL 为加减速参考限值（mm/s²）；KP_POS 越大，越晚开始按剩余距离降速。
    "ACCEL": 500.0, "MARGIN_MM": 5.0, "KP_POS": 3.0, "TRIM_MAX": 100.0,
    "KP_GAP": 0.30, "KI_GAP": 0.0, "KD_GAP": 0.45, "GAP_DEV_MAX": 300.0,
    "KP_SPD": 0.60, "KI_SPD": 0.0, "CORR_MAX": 40.0,
    "BRAKE_TAU": 0.12, "TEP_WINDOW": 0.010, "SPD_SAMPLES": 3,
}

TURN = {
    # 圆弧原算法独立使用 6379 count/m；本次不重新标定。
    "COUNTS_PER_METER": 6379.0, "LOOP_HZ": 100.0,
    # 线加减速 m/s²、角加减速 rad/s²；ANGLE_KP 控制接近目标时的减速速度曲线。
    "MAX_WHEEL_SPEED": 0.45, "ACCEL": 0.6, "ANGULAR_ACCEL": 6.0,
    "ANGLE_KP": 3.0, "PATH_KP": 8.0, "PATH_KD": 1.5, "RATE_KP": 0.5,
    "WHEEL_SPEED_KP": 0.6, "BRAKE_LEAD_SECONDS": 0.02,
    "BRAKE_KP": 3000.0, "STOP_ANGLE_DEG": 2, "SETTLE_SECONDS": 0.25,
    # 仅原地转向使用：轮速单位 m/s；50 mm/s 为待实车确认的收尾下限。
    "SPIN_MIN_WHEEL_SPEED": 0.05, "SPIN_STOP_ANGLE_DEG": 2.0,
    "SPIN_BRAKE_LEAD_SECONDS": 0.04,
    "LOG_DIR": ROOT / "logs" / "turn",  # 每次转弯保存 CSV；None 关闭。
}

POSITION = {
    "speed": 200.0,  # calibrate-position 倒车靠坎速度，mm/s，填写正数。
    "impact_threshold": 180.0, "max_distance_m": 1.0, "timeout": 20.0,
    "boot_wait": 0.0, "poll_interval": 0.005,
}

BALL_POSITION = {
    # 横向误差 px → 位移 mm → 编码器目标位置 → 位置 PID → 轮速 PI。
    # 中间球偏左后退、偏右前进；现场估计 110 px ≈ 40 mm，后续可实测修正。
    # mm_per_px = 前后移动距离(mm) / 球心横坐标变化绝对值(px)。
    "mm_per_px": 40.0 / 110.0, "position_tolerance_mm": 2.0,
    # kp 单位 1/s；110 px 时位置修正约 40 mm，P 输出 24 mm/s，再应用起步下限。
    "speed": 100.0, "tolerance_px": 8.0, "stable_frames": 3,
    "kp": 0.6, "ki": 0.0, "kd": 0.0, "derivative_tau": 0.15,
    # 驱动板已有速度 PID，主机轮速环默认只用 P，避免两层积分共同积累。
    "speed_kp": 0.6, "speed_ki": 0.0,
    # 实测 20 mm/s 几乎不动；接近目标时采用可启动速度并提前制动。
    "min_speed": 35.0, "brake_tau": 0.04, "speed_window": 0.08,
    "accel": 100.0, "loop_hz": 100.0,
    "settle": 0.2, "speed_tolerance": 5.0,
    "missing_frames": 3, "recovery_frames": 3,  # 连续缺球先暂停，恢复需完整的新帧确认。
    "lost_timeout": 2.0,  # s，无有效完整画面持续这么久才终止；过期画面仍停止运动。
    "max_distance_m": 0.3, "timeout": 30.0,
}

# 物品校准独立调参，初值沿用小球；实际物品观察姿态下需重新标定 mm_per_px。
OBJECT_POSITION = dict(BALL_POSITION, missing_frames=1, recovery_frames=1, lost_timeout=0.6)

VISION_CONTROL = {
    "FRAME_W": 1280, "FRAME_H": 720,
    "LOOKAHEAD": 0.7,  # 前视行占画面高度的比例，同时修正航向与横向偏移。
    "LANE_WIDTH_MM": 300.0,  # 左右跑道边界实际间距，mm；用于横向补偿的像素换算。
    "LATERAL_OFFSET_MM": 15.0,  # vision-straight / go-ahead：正值向左偏，负值向右偏，0 = 画面居中。
    "FOV_DEG": 70.0,
    "E_TAU": 0.12, "E_GATE": 10.0,
    "E_DEAD": 0.0, "RATE_SRC": "gap", "GAP_WINDOW": 0.08, "GAP_TAU": 0.05,
    "LOST_STOP": 0.6, "MAX_DEV": 45.0, "TRACK_WAIT": 0.5,  # 直走视觉过期后切换 gap；align 仍停车。
    "GAP_RECOVERY_FRAMES": 2,  # gap 盲走期间，连续有效新帧达到此数量后恢复视觉纠偏。
    "KP_YAW": 10.0, "KI_YAW": 0.5, "YAW_W_MAX": 40.0,
    "KP_RATE": 10.0, "KI_RATE": 0.5, "KFF": 1.0, "KD_RATE": 0.0,
    "TRIM_MAX": 100.0, "MIN_CONF": 0.0, "dir_sign": 1.0,
    "GAP_FINISH_Y_RATIO": 0.15,  # 两侧完整线段均在 y ≥ 此比例 × 原图高度时，关闭本趟视觉纠偏；0 关闭判断。
    "GAP_FINISH_MM": 0.0,  # 可选的剩余距离切换：0 关闭；正值 = 最后这段距离只用 gap 保持航向。
    "LOG_DIR": ROOT / "logs" / "vision-straight",  # 每次行驶保存控制 CSV；None 关闭。
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
    "serial_write_timeout": 1.0,  # RS485 写入上限秒数；应答等待由 serial_timeout 单独控制。
    "joint_signs": {1: 1, 2: 1, 3: -1, 4: 1},
    "joint_offsets_deg": {1: 0.0, 2: 0.0, 3: 160.0, 4: 24.0},
    "encoder_zero_deg": None, "speed_rpm": 10.0,
    "arrival_tolerance_deg": 5.0,  # 恢复旧版到位容差；实际角度与目标相差不超过 5°。
    "move_timeout": 30.0,
    "arrival_settle_seconds": 0.2,
    "arrival_speed_rpm": None,  # 默认按角度连续到位确认；填正数时额外检查转速（rpm）。
    "gripper_port": None, "gripper_baudrate": None, "gripper_servo_id": 2,
    "gripper_speed": 1000, "gripper_timeout": 0.1,
    "gripper_write_timeout": 1.0,  # TTL 写入上限；应答等待仍用 gripper_timeout。
    "gripper_open_angle_deg": 291.0, "gripper_close_angle_deg": 243.0,
    "gripper_release_on_close": False,
}

SERVO = {
    "default_id": 1, "baudrate": 1000000, "speed": 1000, "timeout": 0.1,
    "write_timeout": 1.0,
    "move_timeout": 8.0, "arrival_tolerance": 6, "settle_seconds": 0.2,
    "fixed_positions": {1: 345},  # 仅作为 arm-calibrate 的目标位置，不限制后续运动。
    "probe_baudrates": (1000000, 115200, 500000, 250000, 128000, 76800, 57600, 38400),
}

SENSOR = {
    "imu_port": "", "baudrate": 115200, "hz": 100, "gyro_axis": 2,
    "stale": 0.3, "MAX_YAW_RATE": 1200.0, "MIN_YAW_STEP": 8.0,
    "MAX_YAW_STEP": 60.0, "GLITCH_HOLD": 0.15, "YAW_RATE_WINDOW": 0.02,
}

VISION = {
    "device": 0, "infer_device": "cuda:0", "fp16": True,
    "width": 1280, "height": 720, "scan_width": 1920, "scan_height": 1080,
    "frame_stale": 0.6, "screenshot_dir": ROOT / "screenshots",
    "scan_timeout": 30.0, "observe_timeout": 10.0,
    "observe_stable_frames": 3,
    "model_wait": 60.0,
    "boundary_weights": ROOT / "vision/models/boundary/weights.pt",
    "boundary_args": ROOT / "vision/models/boundary/args.yaml",
    "objects_weights": ROOT / "vision/models/objects/weights.pt",
    "objects_args": ROOT / "vision/models/objects/args.yaml",
}

RUNWAY_ENDPOINT = {
    # 只分析中心一列；上下各取 window 个像素的 Lab 均值。
    "window": 8,
    "color_threshold": 6.0,  # Lab a、b 色度差，排除只有亮度变化的阴影。
    "yellow_threshold": 4.0,  # 从下往上 b 分量至少增加此值。
    "yellow_support": 0.75,  # 上窗口至少此比例的像素确实变黄，排除孤立噪点。
    "stable_frames": 2, "jump_frames": 3,
    "stable_tolerance_px": 60.0,  # 连续候选相对本轮首点的允许偏差。
    "max_jump_px": 100.0,  # 超限先保持旧点，连续 jump_frames 个新帧后才切换。
    "smoothing": 0.7,  # 正常移动的指数平滑系数；1 表示不平滑。
    "lost_frames": 10,  # 缺失这么多新帧，或旧点超过 frame_stale，清除旧点。
}

TARGET_LOCK = {
    "motor2_angle_deg": 12.0,  # 跟踪期间由电机位置环保持的多圈编码器角度。
    # 瞄准前连续确认三色标靶；中心相对本轮首帧的漂移按画面宽、高归一化。
    "stable_frames": 3, "stable_distance_ratio": 0.03,
    # 两轴各自使用 PID；误差除以对应画面半宽/半高，输出为 °/s。
    "kp": 100.0, "ki": 0, "kd": 0, "derivative_tau": 0,
    "hz": 100.0, "tolerance_px": 0.0,
    "max_rate_deg_s": 150.0, "max_step_deg": 10.0,
    # 同色候选按前一帧的位置匹配，距离以画面宽、高归一化。
    "match_distance_ratio": 0.25,
}
