"""机械臂与夹爪的持久会话，供比赛和角度调试直接调用。"""

import warnings
import threading
import time

from control import MotionCancelled, check_cancel, cleanup
from .config import ArmConfig


class Arm:
    """按需连接设备，保留校准零点；动作由一个控制线程串行调用。"""

    def __init__(self, *, port=None, speed_rpm=None, gripper_port=None,
                 simulate=False, stop_event=None):
        self.config = ArmConfig(**{key: value for key, value in
                                   (("port", port), ("speed_rpm", speed_rpm),
                                    ("gripper_port", gripper_port))
                                   if value is not None})
        self._simulate = simulate
        self.stop_event = stop_event
        self._arm = None
        self._servo = None
        self._initial_joints = None
        self._bus = None
        self._connection_lock = threading.RLock()

    def _ensure_bus(self):
        with self._connection_lock:
            if self._bus is None:
                from .motor import MotorBus, SimulatedBus
                config = self.config
                self._bus = (SimulatedBus(config) if self._simulate else MotorBus(
                    config.port, config.baudrate, config.serial_timeout, latency_ms=config.serial_latency_ms))
            return self._bus

    def _ensure_servo(self):
        with self._connection_lock:
            if self._servo is None and self._arm is not None:
                self._servo = self._arm._gripper
            if self._servo is None:
                from .servo import FeetechSTSServo
                config = self.config
                self._servo = FeetechSTSServo(
                    port=config.gripper_port, baudrate=config.gripper_baudrate,
                    servo_id=config.gripper_servo_id, speed=config.gripper_speed,
                    timeout=config.gripper_timeout, simulate=self._simulate,
                    release_on_close=config.gripper_release_on_close)
            return self._servo

    def angles(self):
        """只读反馈；未校准时报告编码器读数，不建立软件零点或发送运动命令。"""
        motors, servos = {}, {}
        try:
            bus = self._ensure_bus()
        except Exception as exc:
            motors = {str(addr): {'error': str(exc)} for addr in (1, 2, 3, 4)}
        else:
            for addr, motor in bus.motors.items():
                try:
                    angle = motor.read_status()['multi_turn_deg']
                    row = {'encoder_deg': angle}
                    arm = self._arm
                    if self.calibrated:
                        row['joint_deg'] = (self.config.joint_offsets_deg[addr] + self.config.joint_signs[addr]
                                            * (angle - arm.encoder_zero_deg[addr]))
                    motors[str(addr)] = row
                except Exception as exc:
                    motors[str(addr)] = {'error': str(exc)}
        try:
            servo = self._ensure_servo()
        except Exception as exc:
            servos = {str(addr): {'error': str(exc)} for addr in (1, 2)}
        else:
            for addr in (1, 2):
                try:
                    servos[str(addr)] = {'angle_deg': servo.status(servo_id=addr)['angle_deg']}
                except Exception as exc:
                    servos[str(addr)] = {'error': str(exc)}
        return {'stamp': time.time(), 'calibrated': self.calibrated, 'motors': motors, 'servos': servos}

    def _event(self, stop_event):
        return self.stop_event if stop_event is None else stop_event

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
                from .controller import Arm as JointController

                options = {'simulate': self._simulate, 'bus': self._ensure_bus()}
                # 若此前单独操作夹爪，统一驱动共用该连接，避免重复打开 TTL 串口。
                with self._connection_lock:
                    if self._servo is not None:
                        options["gripper"] = self._servo
                    self._arm = JointController(self.config, **options)
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
            arm._gripper = self._ensure_servo()
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
                with self._connection_lock:
                    if self._servo is not None:
                        self._arm._gripper = self._servo
                    servo = self._arm._connect_gripper()
                    self._servo = servo
            else:
                servo = self._ensure_servo()
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
        arm, servo, bus = self._arm, self._servo, self._bus
        self._arm = self._servo = None
        self._bus = None
        self._initial_joints = None
        actions = []
        if arm is not None:
            actions.append(("四轴串口", lambda: arm.close(disable_motors=False)))
        elif bus is not None:
            actions.append(("四轴串口", lambda: bus.close(disable_motors=False)))
        if servo is not None:
            actions.append(("舵机串口", servo.close))
        cleanup(*actions)
