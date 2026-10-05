"""IMU serial owner; relay the motor board's encoders with their capture stamp."""
import math
import struct
import time

from rclpy.node import Node
from car_interfaces.msg import Encoders, ImuFeedback

from ..common.action_node import run_node
from ..common.messages import set_stamp
from ..common.safety import SafetyGate
from .imu import ImuLink, FUNC_RAW, IMU_HZ, find_imu_port


class TelemetryImu(ImuLink):
    """Publish every received IMU frame so a one-frame impact is retained."""
    def __init__(self, port, publish):
        self.acceleration = (0, 0, 0)
        self.accel_stamp = 0.0
        self.accel_delta = 0.0
        self.accel_sequence = 0
        self.publish = publish
        super().__init__(port)

    def _handle(self, func, payload):
        super()._handle(func, payload)
        if func == FUNC_RAW and len(payload) == 18:
            acceleration = struct.unpack('<9h', payload)[:3]
            now = time.time()
            with self.lock:
                self.accel_delta = (math.dist(acceleration, self.acceleration)
                                    if self.accel_stamp and now - self.accel_stamp < 0.3 else 0.0)
                self.acceleration = acceleration
                self.accel_stamp = now
                self.accel_sequence += 1
        with self.lock:
            message = ImuFeedback(
                yaw_deg=self.cont, yaw_rate_dps=self._yaw_rate,
                gyro_rate_dps=self._rate_raw, yaw_stamp=self.last_t,
                accel_stamp=self.accel_stamp, acceleration=list(self.acceleration),
                accel_delta=self.accel_delta, accel_sequence=self.accel_sequence)
        set_stamp(message.header, time.time(), 'imu_link')
        self.publish(message)


class SensorNode(Node):
    def __init__(self):
        super().__init__('sensor_node')
        self.gate = SafetyGate()
        self.imu = None
        self.declare_parameter('dry_run', True)
        self.declare_parameter('imu_port', '')
        self.declare_parameter('motor_port', '')
        self.encoder_pub = self.create_publisher(Encoders, 'sensors/encoders', 10)
        self.imu_pub = self.create_publisher(ImuFeedback, 'sensors/imu', 100)
        self.create_subscription(Encoders, 'base/raw_encoders', self.encoder_pub.publish, 10)
        if self.get_parameter('dry_run').value:
            self._sequence = 0
            self.create_timer(0.01, self._virtual)
        else:
            port = self.get_parameter('imu_port').value or find_imu_port(
                skip=[self.get_parameter('motor_port').value], stop_event=self.gate.stop_event)
            if not port:
                self.get_logger().warning('未找到 IMU；普通直走可用，需要 IMU 的动作会拒绝执行')
            else:
                try:
                    self.imu = TelemetryImu(port, self.imu_pub.publish)
                    self.imu.send_rate(IMU_HZ)
                    self.imu.start()
                except Exception:
                    self.close()
                    raise

    def _virtual(self):
        self._sequence += 1
        now = time.time()
        encoders = Encoders(sequence=self._sequence, totals=[0] * 4,
                            increments=[0] * 4, has_increments=True)
        set_stamp(encoders.header, now, 'base_link')
        self.encoder_pub.publish(encoders)
        imu = ImuFeedback(yaw_stamp=now, accel_stamp=now, accel_sequence=self._sequence)
        set_stamp(imu.header, now, 'imu_link')
        self.imu_pub.publish(imu)

    def request_stop(self):
        self.gate.trip()

    def close(self):
        if self.imu is not None:
            self.imu.close()
            self.imu = None


def main(args=None):
    run_node(SensorNode, args)
