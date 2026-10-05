"""Persistent mechanical arm calibration and gripper control through ROS actions."""
from .session import ArmSession
from ..planner.tasks import ARM_COMMANDS, prepare_tasks
from car_interfaces.action import ExecuteCommand
from ..common.action_node import SerialActionNode, run_node


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

    def prepare(self, request):
        steps = prepare_tasks(request.command, require_calibration=False)
        if len(steps) != 1 or steps[0].command not in ARM_COMMANDS:
            raise ValueError('机械臂 Action 只接受一条机械臂或夹爪命令')
        return steps[0]

    def run(self, step, handle, stop_event):
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
