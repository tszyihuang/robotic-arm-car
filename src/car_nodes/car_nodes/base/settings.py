"""任务表使用的控制默认值；无设备副作用。"""

DEFAULTS = {
    "speed": 150.0,
    "log_path": None, "verbose": True, "meters_per_count": 0.00016029,
    "straight_kp_gap": None, "straight_ki_gap": None, "straight_kd_gap": None,
    "turn_radius": 0.3, "turn_max_speed": None, "turn_spin_speed": None,
    "turn_timeout": None, "turn_track_width": 0.41,
    "vision_lookahead": None,
    "vision_dir_sign": None, "vision_model_wait": 60.0,
    "align_ref": None, "align_ref_tau": None, "align_ref_gate": None,
    "align_bias": None, "align_spin_speed": None, "align_min_u": None,
    "align_tol": None, "align_rate_tol": None, "align_settle": None,
    "align_timeout": None, "align_max_rot": None, "align_max_dev": None,
    "align_diverge": None, "align_lost_stop": None, "align_rate_src": "auto",
}

_ALIGN_FIELDS = ("ref", "ref_tau", "ref_gate", "bias", "spin_speed", "min_u",
                 "tol", "rate_tol", "settle", "timeout", "max_rot", "max_dev", "diverge")
