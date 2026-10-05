"""机械臂与夹爪的持久会话，供命令行和 ROS 2 共用。"""

import warnings

from ..common.control import MotionCancelled, check_cancel
from .config import ArmConfig


class ArmSession:
    """按需连接设备，保留校准零点；动作由一个控制线程串行调用。"""

    def __init__(self, *, port=None, speed_rpm=None, gripper_port=None,
                 simulate=False, stop_event=None):
        self.config = ArmConfig(**{key: value for key, value in
                                   (("port", port), ("speed_rpm", speed_rpm),
                                    ("gripper_port", gripper_port))
                                   if value is not None})
        self._simulate = simulate
        self._stop_event = stop_event
        self._arm = None
        self._servo = None
        self._initial_joints = None

    def _event(self, stop_event):
        return self._stop_event if stop_event is None else stop_event

    @property
    def calibrated(self):
        return self._arm is not None and self._initial_joints is not None

    def _stop_after_error(self):
        # 停止失败不能覆盖原始运动异常，零点保持有效，后续动作可重新使能。
        if self._arm is not None:
            try:
                self._arm.disable(include_gripper=False)
            except Exception as exc:
                warnings.warn(f"机械臂停止失败：{exc}", RuntimeWarning)

    def calibrate(self, *, stop_event=None):
        event = self._event(stop_event)
        previous_initial = self._initial_joints
        previous_zero = None if self._arm is None else dict(self._arm.encoder_zero_deg)
        try:
            check_cancel(event)
            if self._arm is None:
                from .controller import Arm

                options = {"simulate": True} if self._simulate else {}
                # 若此前单独操作夹爪，统一驱动共用该连接，避免重复打开 TTL 串口。
                if self._servo is not None:
                    options["gripper"] = self._servo
                self._arm = Arm(self.config, **options)
            else:
                self._initial_joints = None
                options = {} if event is None else {"stop_event": event}
                self._arm.calibrate_zero(**options)
            check_cancel(event)
            self._initial_joints = dict(self._arm.config.joint_offsets_deg)
        except MotionCancelled:
            if self._arm is not None and previous_zero is not None:
                self._arm.encoder_zero_deg = previous_zero
                self._initial_joints = previous_initial
            self._stop_after_error()
            raise
        except BaseException:
            self._stop_after_error()
            raise
        print(f"  机械臂已校准，当前编码器初始位置（°）：{self._arm.encoder_zero_deg}")
        return {"ok": True}

    def _require_calibrated(self):
        if not self.calibrated:
            raise RuntimeError("机械臂未校准，请先执行 arm-calibrate")
        return self._arm

    def move_joints(self, q1, q2, q3, q4, gripper=None, *, stop_event=None):
        arm = self._require_calibrated()
        event = self._event(stop_event)
        options = {"wait": True, "order": "together"}
        if gripper is not None:
            options["gripper"] = gripper
        if event is not None:
            options["stop_event"] = event
        try:
            check_cancel(event)
            arm.move_joints(dict(enumerate((q1, q2, q3, q4), start=1)), **options)
            check_cancel(event)
        except BaseException:
            self._stop_after_error()
            raise
        return {"ok": True}

    def home(self, *, stop_event=None):
        arm = self._require_calibrated()
        event = self._event(stop_event)
        options = {"wait": True, "order": "together"}
        if event is not None:
            options["stop_event"] = event
        try:
            check_cancel(event)
            arm.move_joints(self._initial_joints, **options)
            check_cancel(event)
        except BaseException:
            self._stop_after_error()
            raise
        return {"ok": True}

    def disable(self):
        if self._arm is not None:
            self._arm.disable(include_gripper=False)
        else:
            # 独立失能只发送 RS485 停止指令，不依赖校准或编码器反馈。
            from .motor import MotorBus, SimulatedBus

            config = self.config
            bus = SimulatedBus(config) if self._simulate else MotorBus(
                config.port, config.baudrate, config.serial_timeout,
                latency_ms=config.serial_latency_ms)
            try:
                bus.disable_all()
            finally:
                bus.close(disable_motors=False)
        print("  机械臂 ID1-4 已失能")
        return {"ok": True}

    def cancel(self):
        """由控制线程停止当前或保持中的四轴；保留校准和夹爪，不打开新设备。"""
        if self._arm is not None:
            self._arm.disable(include_gripper=False)
        return {"ok": True}

    def move_gripper(self, angle, *, stop_event=None):
        """复用 STS 驱动和连接；夹持物体时不等待闭合角到位。"""
        event = self._event(stop_event)
        check_cancel(event)
        try:
            if self._arm is not None:
                servo = self._arm._connect_gripper()
            else:
                if self._servo is None:
                    from .servo import FeetechSTSServo

                    config = self.config
                    self._servo = FeetechSTSServo(
                        port=config.gripper_port, baudrate=config.gripper_baudrate,
                        servo_id=config.gripper_servo_id, speed=config.gripper_speed,
                        timeout=config.gripper_timeout, simulate=self._simulate,
                        release_on_close=config.gripper_release_on_close)
                servo = self._servo
            check_cancel(event)
            servo.move_angle(angle, wait=False)
            check_cancel(event)
        except BaseException:
            self._stop_after_error()
            raise
        return {"ok": True}

    def open_gripper(self, *, stop_event=None):
        return self.move_gripper(self.config.gripper_open_angle_deg,
                                 stop_event=stop_event)

    def close_gripper(self, *, stop_event=None):
        return self.move_gripper(self.config.gripper_close_angle_deg,
                                 stop_event=stop_event)

    def close(self):
        arm, servo = self._arm, self._servo
        self._arm = self._servo = None
        self._initial_joints = None
        try:
            if arm is not None:
                arm.close(disable_motors=False)
        finally:
            # STS.close 幂等；若已由统一驱动关闭，共用的夹爪也只释放一次串口。
            if servo is not None:
                servo.close()
