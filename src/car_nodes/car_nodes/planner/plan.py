"""整表预校验，管理任务状态，向底盘、机械臂和视觉节点分派目标。"""
import json
import math
import time

from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Bool

from ..common.control import MotionCancelled, check_cancel, wait_cancelable
from .tasks import prepare_program, PreparedTask, DECISIONS
from .decision import mission_code, selected_branch
from car_interfaces.action import ExecuteCommand, RunTasks
from ..common.action_node import SerialActionNode, json_text, run_node
from ..base.vision_source import RosVisionSource


class PlanNode(SerialActionNode):
    def __init__(self):
        super().__init__('plan_node', RunTasks, 'tasks/run')
        self.declare_parameter('server_wait', 10.0)
        self.declare_parameter('step_timeout', 180.0)
        self.declare_parameter('cancel_wait', 10.0)
        self.declare_parameter('vision_model_wait', 60.0)
        for key in ('server_wait', 'step_timeout', 'cancel_wait', 'vision_model_wait'):
            value = self.get_parameter(key).value
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} 必须是有限正数')
        self._action_clients = {target: ActionClient(
            self, ExecuteCommand, target + '/execute', callback_group=self._group)
            for target in ('base', 'arm', 'vision')}
        self.vision = RosVisionSource(self, topic='vision/boundary')
        self.stop_pub = self.create_publisher(Bool, 'emergency_stop', QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._progress = (0, 0, '', 'idle')
        self._mission_results = []
        self.qr_data = None

    def _reset_service(self, request, response):
        response = super()._reset_service(request, response)
        if response.success:
            self.stop_pub.publish(Bool(data=False))
        return response

    def prepare(self, request):
        if not math.isfinite(request.gap_s) or request.gap_s < 0:
            raise ValueError('gap_s 必须是有限非负数')
        program = prepare_program(request.tasks)
        self._mission_results = []
        self.qr_data = None
        return program, request

    def _wait_future(self, future, stop_event, timeout):
        deadline = time.monotonic() + timeout
        while not future.done():
            check_cancel(stop_event)
            if time.monotonic() >= deadline:
                raise TimeoutError('等待 ROS Action 响应超时')
            if not self.context.ok():
                raise MotionCancelled('ROS 正在关闭')
            time.sleep(0.02)
        return future.result()

    def _stop_child(self, child):
        future = child.get_result_async()
        child.cancel_goal_async()
        deadline = time.monotonic() + self.get_parameter('cancel_wait').value
        while not future.done() and self.context.ok() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done():
            self.gate.trip()
            if self.context.ok():
                self.stop_pub.publish(Bool(data=True))
            raise RuntimeError('子动作取消未确认，已锁存全局停止；检查节点后复位')

    def _execute_step(self, step, stop_event, target=None):
        target = target or step.target
        client = self._action_clients[target]
        sent = client.send_goal_async(ExecuteCommand.Goal(command=step.text))
        child = None
        try:
            child = self._wait_future(sent, stop_event, self.get_parameter('server_wait').value)
            if child is None or not child.accepted:
                raise RuntimeError(f'{target} 拒绝动作：可能正忙、停止未复位或参数错误')
            result = self._wait_future(child.get_result_async(), stop_event,
                                       self.get_parameter('step_timeout').value)
            return {'command': step.text, 'success': bool(
                result.status == GoalStatus.STATUS_SUCCEEDED and result.result.success),
                'message': result.result.message, 'details_json': result.result.details_json}
        except Exception:
            # If cancellation arrived during goal acceptance, consume the response
            # and cancel an accepted child before returning the parent result.
            if child is None:
                deadline = time.monotonic() + self.get_parameter('cancel_wait').value
                while not sent.done() and self.context.ok() and time.monotonic() < deadline:
                    time.sleep(0.02)
                if sent.done():
                    child = sent.result()
                else:
                    self.gate.trip()
                    if self.context.ok():
                        self.stop_pub.publish(Bool(data=True))
            if child is not None and child.accepted:
                self._stop_child(child)
            raise

    def run(self, prepared, handle, stop_event):
        program, request = prepared
        steps = program.main
        dry_run = self.get_parameter('dry_run').value or request.dry_run
        if not dry_run:
            # Check all required servers before the first physical action.
            targets = {step.target for step in program.all_steps} - {'plan'}
            if any(step.needs_vision or step.command in DECISIONS for step in program.all_steps):
                targets.add('vision')
            for target in targets:
                deadline = time.monotonic() + self.get_parameter('server_wait').value
                while not self._action_clients[target].wait_for_server(timeout_sec=0.1):
                    check_cancel(stop_event)
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f'{target} Action 服务未启动')
            if any(step.needs_vision for step in program.all_steps):
                self._progress = (0, len(steps), '', 'vision_warmup')
                result = self._execute_step(PreparedTask(0, 'start-boundary', (), {}, 'start-boundary'),
                                            stop_event, target='vision')
                if not result['success']:
                    raise RuntimeError(f"视觉启动失败：{result['message']}")
                if not self.vision.wait_ready(self.get_parameter('vision_model_wait').value,
                                              log=self.get_logger().info, stop_event=stop_event):
                    raise RuntimeError('视觉模型或摄像头未就绪，任务未开始')
        results = self._mission_results
        pending = [(step, None) for step in steps]
        total, index = len(steps), 0
        while pending:
            step, branch = pending.pop(0)
            index += 1
            check_cancel(stop_event)
            self._progress = (index, total, step.text, 'dry_run' if dry_run else 'running')
            if dry_run:
                results.append({'command': step.text, 'success': True, 'dry_run': True})
            elif step.command in DECISIONS:
                kind = DECISIONS[step.command]
                try:
                    value = mission_code(self.qr_data)[kind]
                    command = f'observe-target kind={kind} value={value}'
                    self._progress = (index, total, step.text, 'selecting')
                    observed = self._execute_step(PreparedTask(step.lineno, 'observe-target', (), {}, command),
                                                  stop_event, target='vision')
                    if not observed['success']:
                        raise ValueError(observed['message'])
                    info = json.loads(observed['details_json'])
                    name = selected_branch(kind, info.get('position'))
                    branch_steps = program.branches.get(name)
                    if not branch_steps:
                        raise ValueError(f'选中的子任务 [{name}] 未定义或为空，请补充任务指令')
                    total += len(branch_steps)
                    pending[0:0] = [(child, name) for child in branch_steps]
                    result = {'command': step.text, 'success': True, 'selected_branch': name,
                              'target_value': value, 'observation': info}
                    self.get_logger().info(f'{step.command}：扫码目标 {value}，执行 [{name}]')
                except (ValueError, TypeError, KeyError) as exc:
                    result = {'command': step.text, 'success': False, 'message': str(exc)}
                results.append(result)
                if not result['success'] and not request.keep_going:
                    break
            else:
                if step.command == 'scan-qrcode':
                    self.qr_data = None
                result = self._execute_step(step, stop_event)
                if step.command == 'scan-qrcode' and result['success']:
                    self.qr_data = json.loads(result['details_json']).get('qr_data')
                    if any(s.command in DECISIONS for s in steps):
                        try:
                            mission_code(self.qr_data)
                        except ValueError as exc:
                            result.update(success=False, message=str(exc))
                if branch:
                    result['branch'] = branch
                results.append(result)
                if not result['success'] and not request.keep_going:
                    break
            if not dry_run and pending:
                wait_cancelable(request.gap_s, stop_event)
        return {'ok': all(result['success'] for result in results) and not pending,
                'completed_steps': sum(result['success'] for result in results),
                'dry_run': dry_run, 'qr_data': self.qr_data, 'results': results}

    def feedback(self, handle, prepared, elapsed_s):
        index, total, command, phase = self._progress
        handle.publish_feedback(RunTasks.Feedback(
            step_index=index, total_steps=total, command=command,
            phase='stopping' if self.gate.stop_event.is_set() else phase))

    def make_result(self, value, error):
        if value is None:
            value = {'ok': False, 'completed_steps': sum(
                row['success'] for row in self._mission_results),
                'qr_data': self.qr_data, 'results': self._mission_results}
        result = RunTasks.Result()
        result.success = error is None and bool(value and value.get('ok'))
        result.completed_steps = (value or {}).get('completed_steps', 0)
        result.message = str(error) if error else ('任务完成' if result.success else '任务失败')
        result.details_json = json_text(value or {})
        return result


def main(args=None):
    run_node(PlanNode, args)
