"""Base node computes motion and sends motor commands; feedback arrives by ROS."""
import math

from car_interfaces.action import ExecuteCommand
from car_interfaces.msg import Encoders

from ..common.action_node import SerialActionNode, run_node
from ..planner.tasks import BASE_COMMANDS, prepare_tasks
from .board import MotorOwner, ControllerBoard
from .controller import execute
from .sensor_source import SensorSource
from .vision_source import RosVisionSource


class BaseNode(SerialActionNode):
    def __init__(self):
        super().__init__('base_node', ExecuteCommand, 'base/execute')
        defaults = {'motor_port': '', 'speed': 150.0, 'turn_speed': 150.0,
                    'turn_radius': 0.3, 'turn_track_width': 0.41, 'turn_spin_speed': 150.0,
                    'align_spin_speed': 150.0, 'align_rate_src': 'auto',
                    'vision_lookahead': '0.5', 'vision_model_wait': 60.0,
                    'vision_dir_sign': 1.0, 'meters_per_count': 0.00016029,
                    'straight_kp_gap': 0.30, 'straight_ki_gap': 0.20,
                    'straight_kd_gap': 0.45, 'verbose': True}
        self._keys = tuple(defaults)
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.sensors = SensorSource(self)
        self.vision = RosVisionSource(self)
        self.encoder_pub = self.create_publisher(Encoders, 'base/raw_encoders', 10)
        self.motor = None

    def parameters(self):
        return {key: self.get_parameter(key).value for key in self._keys}

    def prepare(self, request):
        params = self.parameters()
        for key in ('speed', 'turn_speed', 'turn_spin_speed', 'align_spin_speed',
                    'turn_track_width', 'meters_per_count', 'vision_model_wait'):
            if not math.isfinite(params[key]) or params[key] <= 0:
                raise ValueError(f'{key} 必须是有限正数')
        steps = prepare_tasks(request.command, params, require_calibration=False)
        if len(steps) != 1 or steps[0].command not in BASE_COMMANDS:
            raise ValueError('底盘 Action 只接受一条移动目标')
        return steps[0]

    def run(self, step, handle, stop_event):
        if self.get_parameter('dry_run').value:
            return {'ok': True, 'dry_run': True, 'command': step.text}
        if self.motor is None:
            self.motor = MotorOwner(self.get_parameter('motor_port').value,
                                    self.encoder_pub.publish, stop_event)
        board = ControllerBoard(self.motor, self.sensors,
                                default_timeout=0.0 if step.command == 'calibrate-position' else 0.08)
        return execute(step, board, self.sensors, self.vision, stop_event,
                       log=self.get_logger().info)

    def on_stop(self):
        if getattr(self, 'motor', None) is not None:
            self.motor.spd(0, 0, 0, 0)

    def interrupted_cleanup(self):
        self.on_stop()

    def close(self):
        self.vision.close()
        if self.motor is not None:
            self.motor.close()
            self.motor = None


def main(args=None):
    run_node(BaseNode, args)
