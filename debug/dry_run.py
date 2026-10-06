"""只打印动作的设备；与实机共用 tasks.txt 的 [主线] 指令。"""
from config import BASE, ARM, VISION, POSITION, ALIGN, VISION_CONTROL


class DryBase:
    dry_run = True

    def straight(self, distance, speed=None):
        print(f"base.straight({distance:g} m, {BASE['speed'] if speed is None else speed:g} mm/s)")

    def turn(self, degrees, radius=None):
        radius = BASE["turn_radius"] if radius is None else radius
        speed = BASE['turn_spin_speed'] if radius == 0 else BASE['turn_speed']
        print(f"base.turn({degrees:g}°, R={radius:g} m, {speed:g} mm/s)")

    def calibrate_position(self):
        print(f"base.calibrate_position(speed={BASE['speed']:g} mm/s, "
              f"max_distance={POSITION['max_distance_m']:g} m, timeout={POSITION['timeout']:g} s, "
              f"impact_threshold={POSITION['impact_threshold']:g})")

    def align(self, vision):
        print(f"base.align(vision, spin_speed={ALIGN['SPIN_SPEED']:g} mm/s, "
              f"rate_source={BASE['align_rate_src']!r})")

    def vision_straight(self, distance, vision):
        finish = VISION_CONTROL['GAP_FINISH_MM'] / 1000
        detail = ("全程 gap 保持起步航向" if distance <= finish else
                  f"剩余 ≤ {finish:g} m 切换 gap 保持当时航向")
        print(f"base.vision_straight({distance:g} m, {BASE['speed']:g} mm/s)  # {detail}")

    def stop(self):
        pass

    def close(self):
        pass


class DryArm:
    dry_run = True

    def calibrate(self):
        print("arm.calibrate()  # 读取软件基准，不运动")

    def move_joints(self, q1, q2, q3, q4):
        from arm.config import ArmConfig
        from arm.joints import validate_joints
        validate_joints(dict(enumerate((q1, q2, q3, q4), 1)), ArmConfig())
        print(f"arm.move_joints({q1:g}, {q2:g}, {q3:g}, {q4:g}) °")

    def home(self):
        print(f"arm.home()  # 本次初始姿态 {tuple(ARM['joint_offsets_deg'].values())}")

    def open_gripper(self):
        print(f"arm.open_gripper()  # ID{ARM['gripper_servo_id']}，{ARM['gripper_open_angle_deg']:g}°")

    def close_gripper(self):
        print(f"arm.close_gripper()  # ID{ARM['gripper_servo_id']}，{ARM['gripper_close_angle_deg']:g}°，不等待到位")

    def cancel(self):
        pass

    def close(self):
        pass


class DryVision:
    dry_run = True

    def start(self):
        pass

    def scan_qrcode(self):
        print(f"vision.scan_qrcode(timeout={VISION['scan_timeout']:g} s)  # 三位 1..3；以下以 211 演示映射")
        return "211"

    def observe_target(self, kind, value):
        print(f"vision.observe_target({kind!r}, {value!r})  # 等待三个完整候选连续稳定，不猜位置")
        return None

    def close(self):
        pass
