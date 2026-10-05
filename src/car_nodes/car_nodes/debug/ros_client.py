"""调试客户端复用节点的 Action，不直接打开串口或摄像头。"""
import json
import threading
import time

from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.node import Node
from car_interfaces.action import ExecuteCommand


class DebugClient(Node):
    def __init__(self, name, namespace, target):
        super().__init__(name, namespace=namespace)
        self.client = ActionClient(self, ExecuteCommand, target + '/execute')
        self.command_lock = threading.Lock()
        self.active = None

    def wait(self, future, timeout):
        ready = threading.Event()
        future.add_done_callback(lambda _: ready.set())
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() >= deadline:
                raise TimeoutError('等待节点响应超时')
            if not self.context.ok():
                raise RuntimeError('ROS 已关闭')
            ready.wait(0.1)
        return future.result()

    def command(self, text, timeout=180.0):
        if not self.command_lock.acquire(blocking=False):
            raise RuntimeError('调试请求正在执行，请等待完成')
        sent = None
        try:
            if not self.client.wait_for_server(timeout_sec=5.0):
                raise RuntimeError('执行节点未启动，请先启动 ROS 节点')
            sent = self.client.send_goal_async(ExecuteCommand.Goal(command=text))
            self.active = self.wait(sent, 10.0)
            if not self.active.accepted:
                raise RuntimeError('节点拒绝请求：节点忙、停止未复位或参数错误')
            wrapped = self.wait(self.active.get_result_async(), timeout)
            result = wrapped.result
            if wrapped.status != GoalStatus.STATUS_SUCCEEDED or not result.success:
                raise RuntimeError(result.message)
            data = json.loads(result.details_json)
            return data
        except BaseException:
            if self.active is None and sent is not None:
                try:
                    self.active = self.wait(sent, 10.0)
                except Exception:
                    pass
            if self.active is not None and self.active.accepted:
                self.cancel_active()
            raise
        finally:
            self.active = None
            self.command_lock.release()

    def cancel_active(self):
        if self.active is not None and self.active.accepted:
            self.active.cancel_goal_async()
            try:
                self.wait(self.active.get_result_async(), 10.0)
            except Exception as exc:
                self.get_logger().error(f'取消调试请求未确认：{exc}')
