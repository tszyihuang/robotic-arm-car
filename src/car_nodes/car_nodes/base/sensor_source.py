"""Thread-safe controller adapters for sensor feedback; stale frames stay stale."""
import math
import threading
import time

from ..common.messages import stamp_seconds
from .calibrate_position import AccelState


class SensorSource:
    def __init__(self, node=None, clock=time.time):
        self.clock = clock
        self.condition = threading.Condition()
        self.encoder_stamp = self.yaw_stamp = self.accel_stamp = 0.0
        self.encoder_sequence = None
        self.totals = self.increments = None
        self._yaw = self._yaw_rate = self._rate = self.yaw_ref = 0.0
        self.acceleration = None
        self.accel_sequence = 0
        self.accel_delta = self.impact_peak = 0.0
        self.impact_threshold = 150.0
        self.impact_armed = self.impact_hit = False
        if node is not None:
            from car_interfaces.msg import Encoders, ImuFeedback
            node.create_subscription(Encoders, 'sensors/encoders', self.ingest_encoders, 10)
            node.create_subscription(ImuFeedback, 'sensors/imu', self.ingest_imu, 100)

    def _valid_stamp(self, stamp):
        return math.isfinite(stamp) and 0 < stamp <= self.clock() + 0.1

    def ingest_encoders(self, message):
        stamp = stamp_seconds(message.header)
        with self.condition:
            if not self._valid_stamp(stamp) or stamp <= self.encoder_stamp:
                return
            if len(message.totals) != 4 or len(message.increments) != 4:
                return
            self.encoder_stamp = stamp
            self.encoder_sequence = (stamp, message.sequence)
            self.totals = list(message.totals)
            self.increments = list(message.increments) if message.has_increments else None
            self.condition.notify_all()

    def ingest_imu(self, message):
        with self.condition:
            numbers = (message.yaw_deg, message.yaw_rate_dps, message.gyro_rate_dps)
            if (self._valid_stamp(message.yaw_stamp) and message.yaw_stamp > self.yaw_stamp
                    and all(math.isfinite(value) for value in numbers)):
                self.yaw_stamp = message.yaw_stamp
                self._yaw, self._yaw_rate, self._rate = numbers
            if (self._valid_stamp(message.accel_stamp) and message.accel_stamp > self.accel_stamp
                    and math.isfinite(message.accel_delta) and message.accel_delta >= 0):
                self.accel_stamp = message.accel_stamp
                self.acceleration = tuple(message.acceleration)
                self.accel_sequence = message.accel_sequence
                self.accel_delta = message.accel_delta
                if self.impact_armed:
                    self.impact_peak = max(self.impact_peak, self.accel_delta)
                    if self.accel_delta >= self.impact_threshold:
                        self.impact_hit = True
            self.condition.notify_all()

    def encoder_snapshot(self):
        with self.condition:
            return self.encoder_sequence, self.encoder_stamp, self.totals, self.increments

    @property
    def yaw(self):
        with self.condition:
            return self._yaw - self.yaw_ref

    @property
    def yaw_rate(self):
        with self.condition:
            return self._yaw_rate

    @property
    def rate(self):
        with self.condition:
            return self._rate

    def age(self):
        with self.condition:
            return max(0.0, self.clock() - self.yaw_stamp) if self.yaw_stamp else 1e9

    def has_data(self):
        return self.age() < 0.3

    def zero(self):
        with self.condition:
            self.yaw_ref = self._yaw

    def accel_state(self):
        with self.condition:
            age = max(0.0, self.clock() - self.accel_stamp) if self.accel_stamp else 1e9
            return AccelState(age, self.acceleration, self.accel_delta,
                              self.impact_peak, self.impact_hit, self.accel_sequence)

    def arm_impact(self):
        with self.condition:
            if self.accel_state().age >= 0.3:
                raise RuntimeError('没有新鲜的 IMU 原始加速度帧，不能倒车')
            self.impact_peak = 0.0
            self.impact_hit = False
            self.impact_armed = True

    def disarm_impact(self):
        with self.condition:
            self.impact_armed = False

    def close(self):
        """Controllers end a step; the sensor node continues collecting."""
