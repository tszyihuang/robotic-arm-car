"""Persistent mechanical arm calibration and gripper control through ROS actions."""
import math
import shlex
from std_msgs.msg import String
from .session import ArmSession
from ..planner.tasks import ARM_COMMANDS, prepare_tasks, PreparedTask
from car_interfaces.action import ExecuteCommand
from ..common.action_node import SerialActionNode, run_node, json_text


class ArmNode(SerialActionNode):
    def __init__(self):
        super().__init__('arm_node', ExecuteCommand, 'arm/execute')
        self.declare_parameter('arm_port', '')
        self.declare_parameter('arm_speed', 10.0)
        self.declare_parameter('servo_port', '')
        self.session = ArmSession(
            port=self.get_parameter('arm_port').value or None,
            speed_rpm=self.get_parameter('arm_speed').value,
            gripper_port=self.get_parameter('servo_port').value or None)
        self._dry_calibrated = False
        self._telemetry_timer = None
        self.angles_pub = self.create_publisher(String, 'arm/angles', 1)

    def prepare(self, request):
        parts = shlex.split(request.command)
        if parts and parts[0] in ('start-telemetry', 'stop-telemetry'):
            if parts[0] == 'stop-telemetry':
                if len(parts) != 1:
                    raise ValueError('stop-telemetry 不接受参数')
                return PreparedTask(0, parts[0], (), {}, request.command)
            if len(parts) > 2 or len(parts) == 2 and not parts[1].startswith('hz='):
                raise ValueError('start-telemetry [hz=频率]')
            hz = float(parts[1][3:]) if len(parts) == 2 else 5.0
            if not math.isfinite(hz) or not 0.2 <= hz <= 20:
                raise ValueError('角度反馈频率须为 0.2..20 Hz')
            return PreparedTask(0, parts[0], (), {'hz': hz}, request.command)
        steps = prepare_tasks(request.command, require_calibration=False)
        if len(steps) != 1 or steps[0].command not in ARM_COMMANDS:
            raise ValueError('机械臂 Action 只接受一条机械臂或夹爪命令')
        return steps[0]

    def run(self, step, handle, stop_event):
        if step.command in ('start-telemetry', 'stop-telemetry'):
            if self._telemetry_timer is not None:
                self.destroy_timer(self._telemetry_timer)
                self._telemetry_timer = None
            if step.command == 'start-telemetry':
                self._telemetry_timer = self.create_timer(1 / step.kwargs['hz'], self._publish_angles)
            return {'ok': True, 'command': step.command}
        if self.get_parameter('dry_run').value:
            if step.command == 'arm-calibrate':
                self._dry_calibrated = True
            elif step.command in ('arm-move', 'arm-home', 'home') and not self._dry_calibrated:
                raise RuntimeError('机械臂未校准，请先执行 arm-calibrate')
            return {'ok': True, 'dry_run': True, 'command': step.text}
        method = {'arm-calibrate': 'calibrate', 'arm-move': 'move_joints',
                  'arm-home': 'home', 'home': 'home', 'arm-disable': 'disable',
                  'gripper-open': 'open_gripper', 'gripper-close': 'close_gripper'}[step.command]
        options = {} if step.command == 'arm-disable' else {'stop_event': stop_event}
        return getattr(self.session, method)(*step.args, **options)

    def _publish_angles(self):
        if self.get_parameter('dry_run').value:
            data = {'dry_run': True, 'calibrated': self._dry_calibrated,
                    'motors': {str(addr): {'error': 'dry_run：无硬件读数'} for addr in (1, 2, 3, 4)},
                    'servos': {str(addr): {'error': 'dry_run：无硬件读数'} for addr in (1, 2)}}
        else:
            data = self.session.angles()
        self.angles_pub.publish(String(data=json_text(data)))

    def on_stop(self):
        # An active worker observes its event; idle stop also disables held axes.
        if hasattr(self, 'session') and not self.gate.busy and not self.get_parameter('dry_run').value:
            self.session.cancel()

    def close(self):
        try:
            if not self.get_parameter('dry_run').value:
                self.session.cancel()
        finally:
            self.session.close()

    def interrupted_cleanup(self):
        if not self.get_parameter('dry_run').value:
            self.session.cancel()


def main(args=None):
    run_node(ArmNode, args)
