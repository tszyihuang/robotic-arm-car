"""比赛所需的四轴等时运动、校准与夹爪控制。"""

import time
import warnings

from .config import ArmConfig, JOINT_IDS, finite
from .errors import ArmError, MotionTimeout, ProtocolError
from .joints import validate_joints
from .motor import MotorBus, SimulatedBus, check_fault, position_payload
from .servo import FeetechSTSServo, MODE
from base.control import MotionCancelled, check_cancel as _check_cancel
from base.control import cleanup


class Arm:
    def __init__(self, config=None, *, simulate=False, bus=None, with_gripper=False,
                 gripper=None):
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
            raw = self._read_raw()
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
            _check_cancel(stop_event)
            status = self._bus.motors[addr].read_status()
            _check_cancel(stop_event)
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
        except BaseException as exc:
            self._abort_motion_error(exc, stop_event)
            raise

    def calibrate_zero(self, *, stop_event=None):
        """取当前四轴编码器为新软件零点；保留安装偏置，不发送移动指令。"""
        validate_joints(self.config.joint_offsets_deg, self.config)
        try:
            raw = self._read_raw(stop_event)
        except BaseException as exc:
            self._abort_motion_error(exc, stop_event)
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
            _check_cancel(stop_event)
            self._bus.motors[addr].move(*parameters)
        _check_cancel(stop_event)

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

    def _abort_motion_error(self, error, stop_event):
        self._abort_on_error(include_gripper=False)

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
            return self._wait(target, timeout, *(() if stop_event is None else (stop_event,)))
        except BaseException as exc:
            self._abort_motion_error(exc, stop_event)
            raise

    def _wait(self, target, timeout, stop_event=None):
        deadline = time.monotonic() + timeout
        while True:
            _check_cancel(stop_event)
            joints = self.get_joints(**({} if stop_event is None else {"stop_event": stop_event}))
            _check_cancel(stop_event)
            if all(abs(joints[a] - v) <= self.config.arrival_tolerance_deg for a, v in target.items()):
                return True
            if time.monotonic() >= deadline:
                raise MotionTimeout(f"{timeout:g} 秒内未到位，停止本次运动并尝试失能四轴")
            if stop_event is None:
                time.sleep(0.05)
            else:
                stop_event.wait(0.05)

    def move_joints(self, joints, *, wait=True, speed_rpm=None, accel_rpm_s=None, order="together",
                    gripper=None, stop_event=None):
        """四轴等时位置移动；速度、加速度参数为同一段的上限。

        order: together / plane-first / base-first；分段时每段内同步。
        gripper: 可选夹爪开度 0..1；随四轴发送目标，不等待夹爪到位。
        """
        if gripper is not None:
            self._opening(gripper)
            target = validate_joints(joints, self.config)
            current = self.get_joints(**({} if stop_event is None else {"stop_event": stop_event}))
            return self._move_pose(target, current, gripper, wait=wait, speed_rpm=speed_rpm,
                                   accel_rpm_s=accel_rpm_s, order=order, stop_event=stop_event)
        return self._move_joints(joints, wait=wait, speed_rpm=speed_rpm,
                                 accel_rpm_s=accel_rpm_s, order=order, stop_event=stop_event)

    def _move_joints(self, joints, *, current=None, wait=True, speed_rpm=None,
                     accel_rpm_s=None, order="together", stop_event=None):
        # 复用同一轮读取的四轴状态，检查目标后再发送。
        target = validate_joints(joints, self.config)
        if order not in ("together", "plane-first", "base-first"):
            raise ValueError("order 必须是 together、plane-first 或 base-first")
        speed = self._speed(speed_rpm)
        # 预检查包含原始编码器零点的所有协议参数。
        for a, angle in target.items():
            position_payload(self._motor_angle(a, angle), speed, accel_rpm_s)
        try:
            _check_cancel(stop_event)
            if current is None:
                current = self.get_joints(**({} if stop_event is None else {"stop_event": stop_event}))
            segmented = order != "together" and abs(current[1] - target[1]) > 0.5
            if segmented:
                first = (2, 3, 4) if order == "plane-first" else (1,)
                second = (1,) if order == "plane-first" else (2, 3, 4)
                groups = (first, second)
            else:
                groups = (JOINT_IDS,)
            # 两段也在第一条位置命令前完成检查，避免后段无效导致部分运动。
            profiles = [self._motion_profile(current, target, group, speed, accel_rpm_s)
                        for group in groups]
            self._send(profiles[0], stop_event)
            if segmented:
                self._wait({a: target[a] for a in first}, self.config.move_timeout,
                           *(() if stop_event is None else (stop_event,)))
                # 前段等待期间后段关节可能产生跟随误差，按最新反馈重新规划。
                current = self.get_joints(**({} if stop_event is None else {"stop_event": stop_event}))
                self._send(self._motion_profile(current, target, second, speed, accel_rpm_s), stop_event)
            self._target = dict(target)
            if wait:
                self._wait(target, self.config.move_timeout,
                           *(() if stop_event is None else (stop_event,)))
            _check_cancel(stop_event)
        except BaseException as exc:
            self._abort_motion_error(exc, stop_event)
            raise
        return dict(target)

    def _move_pose(self, target, current, gripper, *, wait, speed_rpm, accel_rpm_s, order,
                   wait_gripper=False, stop_event=None):
        if gripper is None:
            return self._move_joints(target, current=current, wait=wait, speed_rpm=speed_rpm,
                                     accel_rpm_s=accel_rpm_s, order=order, stop_event=stop_event)
        angle = self._gripper_angle(gripper)
        try:
            _check_cancel(stop_event)
            servo = self._connect_gripper()
            # 舵机也先检查模式，避免四轴已经启动后才发现夹爪不支持位置控制。
            if servo.read_register(MODE, 1)[0] != 0:
                raise ValueError("夹爪只支持单圈位置模式（mode=0）")
        except ValueError:
            raise
        except BaseException as exc:
            self._abort_motion_error(exc, stop_event)
            raise
        result = self._move_joints(target, current=current, wait=False, speed_rpm=speed_rpm,
                                   accel_rpm_s=accel_rpm_s, order=order, stop_event=stop_event)
        try:
            _check_cancel(stop_event)
            servo.move_angle(angle, wait=False)
            _check_cancel(stop_event)
            if wait:
                self._wait(result, self.config.move_timeout,
                           *(() if stop_event is None else (stop_event,)))
            # 夹爪指令不等待物理到位；夹住物体时可能无法达到闭合角。
            if wait_gripper:
                servo.wait_for_arrival()
        except BaseException as exc:
            self._abort_motion_error(exc, stop_event)
            raise
        return result


    def close(self, *, disable_motors=True):
        """关闭连接；disable_motors=False 时保留四轴的使能及最后位置目标。"""
        if not self._closed:
            self._closed = True
            self._target = None
            actions = [("四轴总线", lambda: self._bus.close(disable_motors=disable_motors))]
            if self._gripper is not None:
                actions.append(("夹爪串口", self._gripper.close))
            cleanup(*actions)
