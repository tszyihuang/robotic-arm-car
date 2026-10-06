"""比赛所需的四轴等时运动、校准与夹爪控制。"""

import time
import warnings

from .config import ArmConfig, JOINT_IDS, finite
from .errors import ArmError, MotionTimeout, ProtocolError
from .joints import validate_joints
from .motor import MotorBus, SimulatedBus, check_fault, position_payload
from .servo import FeetechSTSServo, MODE
from base.control import check_cancel, cleanup, wait_cancelable


class Arm:
    def __init__(self, config=None, *, simulate=False, bus=None, with_gripper=False,
                 gripper=None, stop_event=None):
        self.config = config or ArmConfig()
        self._closed = False
        self._target = None
        self._simulate = simulate
        self._gripper = gripper
        self._bus = bus if bus is not None else (
            SimulatedBus(self.config) if simulate else MotorBus(
                self.config.port, self.config.baudrate, self.config.serial_timeout,
                latency_ms=self.config.serial_latency_ms))
        try:
            if set(self._bus.motors) != set(JOINT_IDS):
                raise ValueError("总线必须包含且仅包含电机 ID1-4")
            raw = self._read_raw(stop_event)
            self.encoder_zero_deg = dict(self.config.encoder_zero_deg or raw)
            validate_joints(self._logical(raw), self.config)
            if with_gripper:
                self._connect_gripper()
            if self._gripper is not None and self._gripper.servo_id != self.config.gripper_servo_id:
                raise ValueError("外部夹爪舵机 ID 与 gripper_servo_id 不一致")
        except BaseException:
            cleanup(("机械臂连接", self.close))
            raise

    def _connect_gripper(self):
        if self._closed:
            raise ArmError("机械臂已关闭")
        if self._gripper is None:
            self._gripper = FeetechSTSServo(
                self.config.gripper_port, baudrate=self.config.gripper_baudrate,
                servo_id=self.config.gripper_servo_id, speed=self.config.gripper_speed,
                timeout=self.config.gripper_timeout, simulate=self._simulate,
                release_on_close=self.config.gripper_release_on_close)
        return self._gripper

    @staticmethod
    def _opening(value):
        if isinstance(value, str) and value.lower() in ("open", "close"):
            return 1.0 if value.lower() == "open" else 0.0
        opening = finite(value, "夹爪开度")
        if not 0 <= opening <= 1:
            raise ValueError("夹爪开度必须在 [0, 1] 内：0=闭合，1=张开")
        return opening

    def _gripper_angle(self, opening):
        opening = self._opening(opening)
        return self.config.gripper_close_angle_deg + opening * (
            self.config.gripper_open_angle_deg - self.config.gripper_close_angle_deg)


    def _read_raw(self, stop_event=None):
        if self._closed:
            raise ArmError("机械臂已关闭")
        raw = {}
        for addr in JOINT_IDS:
            check_cancel(stop_event)
            status = self._bus.motors[addr].read_status()
            check_cancel(stop_event)
            check_fault(status, addr)
            try:
                raw[addr] = finite(status["multi_turn_deg"], f"ID{addr} 编码器读数")
            except (KeyError, ValueError) as exc:
                raise ProtocolError(f"ID{addr} 编码器读数无效") from exc
        return raw

    def _logical(self, raw):
        return {a: self.config.joint_offsets_deg[a] + self.config.joint_signs[a] *
                (raw[a] - self.encoder_zero_deg[a]) for a in JOINT_IDS}

    def _motor_angle(self, addr, logical):
        return self.encoder_zero_deg[addr] + self.config.joint_signs[addr] * (
            logical - self.config.joint_offsets_deg[addr])

    def get_joints(self, *, stop_event=None):
        """读取逻辑角度，单位 °；通信失败直接抛异常。"""
        try:
            return self._logical(self._read_raw(stop_event))
        except BaseException:
            self._abort_on_error(include_gripper=False)
            raise

    def calibrate_zero(self, *, stop_event=None):
        """取当前四轴编码器为新软件零点；保留安装偏置，不发送移动指令。"""
        validate_joints(self.config.joint_offsets_deg, self.config)
        try:
            raw = self._read_raw(stop_event)
        except BaseException:
            self._abort_on_error(include_gripper=False)
            raise
        self.encoder_zero_deg = dict(raw)
        self._target = None
        return dict(self.encoder_zero_deg)


    def _speed(self, speed_rpm):
        speed = self.config.speed_rpm if speed_rpm is None else speed_rpm
        position_payload(0.0, speed)
        return speed

    def _motion_profile(self, current, target, addresses, speed, accel):
        """按最大位移缩放同一段的速度及加减速度，共用运动进度。

        1 rpm = 6 °/s；匀速时 T = max(|Δq|) / (6 * speed)。
        梯形/三角形曲线同时缩放速度与加速度，保持各轴加减速时长一致。
        协议只有 0.01 的分辨率，因此量化后实际到位时间存在误差。
        """
        distances = {a: abs(target[a] - current[a]) for a in addresses}
        longest = max(distances.values())

        def scaled(value, ratio):
            # 四舍五入到协议单位，不超过用户给定上限；最小非零值为 0.01。
            _, payload = position_payload(0.0, value)
            limit = int.from_bytes(payload[-4:], "little")
            return max(1, min(limit, round(limit * ratio))) / 100.0

        profile = {}
        for addr in addresses:
            # 零位移轴仍发送保持指令，以覆盖上一次未完成的目标；避免除零。
            ratio = distances[addr] / longest if longest and distances[addr] else 1.0
            axis_speed = scaled(speed, ratio)
            axis_accel = None if accel is None else scaled(accel, ratio)
            angle = self._motor_angle(addr, target[addr])
            position_payload(angle, axis_speed, axis_accel)
            profile[addr] = (angle, axis_speed, axis_accel)
        return profile

    def _send(self, profile, stop_event=None):
        # 整个运动的协议参数已经预检查；保持总线顺序逐帧收发。
        for addr, parameters in profile.items():
            check_cancel(stop_event)
            self._bus.motors[addr].move(*parameters)
        check_cancel(stop_event)

    def _abort(self, *, include_gripper=True):
        self._target = None
        try:
            self._bus.disable_all()
        finally:
            if include_gripper and self._gripper is not None:
                self._gripper.disable_torque()

    def _abort_on_error(self, *, include_gripper=True):
        if self._closed:
            return
        try:
            self._abort(include_gripper=include_gripper)
        except Exception as exc:
            # 通信中断时停止指令也可能失败，保留最初的故障供调用者处理。
            warnings.warn(f"异常后失能失败：{exc}", RuntimeWarning)

    def disable(self, *, include_gripper=True):
        """失能四轴；include_gripper=False 保持夹爪，下一条位置命令重新使能。"""
        self._abort(include_gripper=include_gripper)

    def wait_for_arrival(self, target=None, timeout=None, *, stop_event=None):
        target = self._target if target is None else target
        if target is None:
            raise ValueError("没有待等待的目标")
        if not set(target) or not set(target) <= set(JOINT_IDS):
            raise ValueError("等待目标的关节号无效")
        target = {a: finite(v, f"ID{a} 等待目标") for a, v in target.items()}
        timeout = self.config.move_timeout if timeout is None else finite(timeout, "到位超时")
        if timeout <= 0:
            raise ValueError("到位超时必须大于 0")
        try:
            return self._wait(target, timeout, stop_event)
        except BaseException:
            self._abort_on_error(include_gripper=False)
            raise

    def _wait(self, target, timeout, stop_event=None):
        deadline = time.monotonic() + timeout
        while True:
            check_cancel(stop_event)
            joints = self.get_joints(stop_event=stop_event)
            check_cancel(stop_event)
            if all(abs(joints[a] - v) <= self.config.arrival_tolerance_deg for a, v in target.items()):
                return True
            if time.monotonic() >= deadline:
                raise MotionTimeout(f"{timeout:g} 秒内未到位，停止本次运动并尝试失能四轴")
            wait_cancelable(0.05, stop_event)

    def move_joints(self, joints, *, wait=True, speed_rpm=None, accel_rpm_s=None,
                    order="together", gripper=None, stop_event=None):
        """四轴等时移动；可选夹爪开度 0..1，只等待四轴到位。

        order: together / plane-first / base-first；分段时每段内同步。
        所有目标与协议参数在首条运动指令前校验。
        """
        target = validate_joints(joints, self.config)
        if order not in ("together", "plane-first", "base-first"):
            raise ValueError("order 必须是 together、plane-first 或 base-first")
        speed = self._speed(speed_rpm)
        for addr, angle in target.items():
            position_payload(self._motor_angle(addr, angle), speed, accel_rpm_s)
        gripper_angle = None if gripper is None else self._gripper_angle(gripper)
        try:
            check_cancel(stop_event)
            current = self.get_joints(stop_event=stop_event)
            servo = None
            if gripper_angle is not None:
                servo = self._connect_gripper()
                if servo.read_register(MODE, 1)[0] != 0:
                    raise ValueError("夹爪只支持单圈位置模式（mode=0）")
            segmented = order != "together" and abs(current[1] - target[1]) > 0.5
            if segmented:
                groups = ((2, 3, 4), (1,)) if order == "plane-first" else ((1,), (2, 3, 4))
            else:
                groups = (JOINT_IDS,)
            profiles = [self._motion_profile(current, target, group, speed, accel_rpm_s)
                        for group in groups]
            self._send(profiles[0], stop_event)
            if segmented:
                self._wait({addr: target[addr] for addr in groups[0]},
                           self.config.move_timeout, stop_event)
                # 前段等待可能改变后段位置，按最新反馈重新规划。
                current = self.get_joints(stop_event=stop_event)
                self._send(self._motion_profile(current, target, groups[1], speed, accel_rpm_s), stop_event)
            self._target = dict(target)
            if servo is not None:
                check_cancel(stop_event)
                servo.move_angle(gripper_angle, wait=False)
            if wait:
                self._wait(target, self.config.move_timeout, stop_event)
            check_cancel(stop_event)
        except BaseException:
            self._abort_on_error(include_gripper=False)
            raise
        return dict(target)

    def close(self, *, disable_motors=True):
        """关闭连接；disable_motors=False 时保留四轴的使能及最后位置目标。"""
        if not self._closed:
            self._closed = True
            self._target = None
            actions = [("四轴总线", lambda: self._bus.close(disable_motors=disable_motors))]
            if self._gripper is not None:
                actions.append(("夹爪串口", self._gripper.close))
            cleanup(*actions)
