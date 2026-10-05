"""Shared action lifecycle: serialization, cancel, stop latch and cleanup."""
import json
import signal
import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from .control import MotionCancelled, check_cancel
from .safety import SafetyGate


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)


class SerialActionNode(Node):
    """Exactly one goal per hardware owner; cancellation waits for cleanup."""
    def __init__(self, node_name, action_type, action_name):
        super().__init__(node_name)
        self.gate = SafetyGate()
        self._group = ReentrantCallbackGroup()
        self._prepared = None
        self.declare_parameter('dry_run', True)
        self.action_type = action_type
        self.action_name = action_name
        self.status_pub = self.create_publisher(
            String, action_name.rsplit('/', 1)[0] + '/status',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.stop_sub = self.create_subscription(
            Bool, 'emergency_stop', self._stop_message,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
            callback_group=self._group)
        prefix = action_name.rsplit('/', 1)[0]
        self.create_service(Trigger, prefix + '/stop', self._stop_service,
                            callback_group=self._group)
        self.create_service(Trigger, prefix + '/reset_stop', self._reset_service,
                            callback_group=self._group)
        self.server = ActionServer(
            self, action_type, action_name, execute_callback=self._execute,
            goal_callback=self._goal, cancel_callback=self._cancel,
            callback_group=self._group)
        self._publish_status('idle')

    def _publish_status(self, phase, **details):
        self.status_pub.publish(String(data=json_text({
            'phase': phase, 'stopped': self.gate.stopped,
            'dry_run': self.get_parameter('dry_run').value, **details})))

    def _goal(self, request):
        # Reserve before preparing: no two callbacks can mutate shared backend state.
        if not self.gate.reserve():
            return GoalResponse.REJECT
        try:
            self._prepared = self.prepare(request)
        except Exception as exc:
            self.gate.release()
            self.get_logger().warning(f'任务校验失败：{exc}')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel(self, goal_handle):
        # The execution monitor observes CANCELING after this callback returns.
        return CancelResponse.ACCEPT

    def _stop_message(self, message):
        if message.data:
            self.request_stop()

    def request_stop(self):
        self.gate.trip()
        self.on_stop()
        self._publish_status('stopped')

    def on_stop(self):
        pass

    def interrupted_cleanup(self):
        """Called after the worker exits, before releasing its hardware slot."""
        pass

    def _stop_service(self, request, response):
        self.request_stop()
        response.success = True
        response.message = '停止已锁存；活动动作完成停车清理后才返回结果'
        return response

    def _reset_service(self, request, response):
        response.success = self.gate.reset()
        response.message = '停止已复位' if response.success else '仍有动作清理中，暂不能复位'
        if response.success:
            self._publish_status('idle')
        return response

    def _execute(self, handle):
        prepared = self._prepared
        box = {}
        done = threading.Event()

        def work():
            try:
                check_cancel(self.gate.stop_event)
                box['value'] = self.run(prepared, handle, self.gate.stop_event)
                check_cancel(self.gate.stop_event)
            except Exception as exc:
                box['error'] = exc
            finally:
                done.set()

        worker = threading.Thread(target=work, name=self.get_name() + '-control')
        worker.start()
        started = time.monotonic()
        next_feedback = started
        self._publish_status('running')
        try:
            while not done.wait(0.02):
                if handle.is_cancel_requested or not self.context.ok():
                    self.gate.stop_event.set()
                if self.context.ok() and time.monotonic() >= next_feedback:
                    self.feedback(handle, prepared, time.monotonic() - started)
                    next_feedback = time.monotonic() + 0.2
            worker.join()
            if self.gate.stop_event.is_set() or handle.is_cancel_requested:
                try:
                    self.interrupted_cleanup()
                except Exception as exc:
                    box['error'] = exc
            result = self.make_result(box.get('value'), box.get('error'))
            if handle.is_cancel_requested:
                result.success = False
                result.message = '任务已取消'
                handle.canceled()
            elif result.success and not self.gate.stopped:
                handle.succeed()
            else:
                result.success = False
                handle.abort()
            self._publish_status('stopped' if self.gate.stopped else 'idle',
                                 success=result.success, message=result.message)
            return result
        finally:
            # Admission reopens only after the hardware worker has really exited.
            if worker.is_alive():
                self.gate.stop_event.set()
            worker.join()
            self.gate.finish(self.interrupted_cleanup)

    def prepare(self, request):
        raise NotImplementedError

    def run(self, prepared, handle, stop_event):
        raise NotImplementedError

    def feedback(self, handle, prepared, elapsed_s):
        feedback = self.action_type.Feedback()
        feedback.phase = 'stopping' if self.gate.stop_event.is_set() else 'running'
        feedback.elapsed_s = elapsed_s
        handle.publish_feedback(feedback)

    def make_result(self, value, error):
        result = self.action_type.Result()
        result.success = error is None and bool(value and value.get('ok'))
        result.message = str(error) if error else ('完成' if result.success else str(
            (value or {}).get('fault') or (value or {}).get('reason') or '动作未到位'))
        result.details_json = json_text(value or {})
        return result

    def close(self):
        pass


def run_node(node_class, args=None):
    from rclpy.signals import SignalHandlerOptions
    # Keep DDS alive during SIGINT cleanup so missions can cancel child goals.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    shutdown_requested = threading.Event()
    def terminate(signum, frame):
        # Finish the current executor dispatch before starting cleanup. Raising
        # mid-dispatch can leave a queued callback without its future tracked.
        shutdown_requested.set()
    previous_int = signal.signal(signal.SIGINT, terminate)
    previous_term = signal.signal(signal.SIGTERM, terminate)
    node = None
    executor = MultiThreadedExecutor(num_threads=4)
    try:
        node = node_class()
        executor.add_node(node)
        # A finite DDS wait lets Python process SIGINT/SIGTERM while preserving
        # the context for cooperative child-action cancellation.
        while node.context.ok() and not shutdown_requested.is_set():
            executor.spin_once(timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.request_stop()
            while node.gate.busy and node.context.ok():
                executor.spin_once(timeout_sec=0.1)
        # Jazzy's shutdown destroys its guard before joining the callback pool.
        # Drain queued callbacks while that guard is still valid; no further
        # spin occurs here and all hardware workers have already completed.
        executor._executor.shutdown(wait=True)
        executor.shutdown(wait_for_threads=True)
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)
