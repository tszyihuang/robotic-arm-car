"""Real ROS actions/topics; all hardware nodes stay in dry-run mode."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

try:
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from car_interfaces.action import ExecuteCommand, RunTasks
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


@unittest.skipUnless(ROS_AVAILABLE, '需要已构建的 ROS 工作空间')
class RosLaunchTests(unittest.TestCase):
    def test_launch_starts_all_nodes_runs_cli_and_exits_cleanly(self):
        namespace = 'car_launch_test_' + str(os.getpid())
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryFile(mode='w+t') as log:
            launched = subprocess.Popen(['ros2', 'launch', 'car_nodes', 'car.launch.py',
                                         'namespace:=' + namespace], stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
            inspector = client = None
            rclpy.init()
            try:
                inspector = Node('launch_inspector')
                client = ActionClient(inspector, RunTasks, '/' + namespace + '/tasks/run')
                self.assertTrue(client.wait_for_server(timeout_sec=10))
                expected = ['arm_node', 'base_node', 'plan_node', 'sensor_node', 'vision_node']
                deadline = time.monotonic() + 5
                observed = []
                while observed != expected and time.monotonic() < deadline:
                    observed = sorted(name for name, ns in inspector.get_node_names_and_namespaces()
                                      if ns == '/' + namespace)
                    rclpy.spin_once(inspector, timeout_sec=0.05)
                self.assertEqual(observed, expected)
                result = subprocess.run([sys.executable, '-B', str(root / 'plan.py'),
                                         '--ros-dry-run', '--namespace', '/' + namespace],
                                        capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('任务完成', result.stdout)
            finally:
                if client is not None:
                    client.destroy()
                if inspector is not None:
                    inspector.destroy_node()
                rclpy.shutdown()
                if launched.poll() is None:
                    os.killpg(launched.pid, signal.SIGINT)
                try:
                    code = launched.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(launched.pid, signal.SIGKILL)
                    launched.wait()
                    raise
            log.seek(0)
            output = log.read()
            self.assertEqual(code, 0, output)
            self.assertEqual(output.count('process has finished cleanly'), 5, output)


@unittest.skipUnless(ROS_AVAILABLE, '需要已构建的 ROS 工作空间')
class RosPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from car_nodes.arm.node import ArmNode
        from car_nodes.base.node import BaseNode
        from car_nodes.planner.plan import PlanNode
        from car_nodes.sensor.node import SensorNode
        from car_nodes.vision.node import VisionNode
        rclpy.init(args=['--ros-args', '-r', '__ns:=/car_test'])
        cls.base, cls.arm, cls.vision = BaseNode(), ArmNode(), VisionNode()
        cls.sensor, cls.plan = SensorNode(), PlanNode()
        # Route missions through real child actions; children still use no hardware.
        cls.plan.set_parameters([Parameter('dry_run', value=False)])
        cls.client_node = Node('test_client')
        cls.nodes = (cls.base, cls.arm, cls.vision, cls.sensor, cls.plan, cls.client_node)
        cls.executor = MultiThreadedExecutor(num_threads=12)
        for node in cls.nodes:
            cls.executor.add_node(node)
        cls.thread = threading.Thread(target=cls.executor.spin)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        for node in cls.nodes:
            if hasattr(node, 'request_stop'):
                node.request_stop()
        cls.executor.shutdown()
        cls.thread.join(timeout=10)
        for node in cls.nodes:
            if hasattr(node, 'close'):
                node.close()
            node.destroy_node()
        rclpy.shutdown()

    def tearDown(self):
        for node in self.nodes:
            if hasattr(node, 'gate'):
                deadline = time.monotonic() + 3
                while node.gate.busy and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(node.gate.reset())

    def wait(self, future, timeout=10):
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(future.done(), 'ROS 响应超时')
        return future.result()

    def client(self, action_type, target):
        client = ActionClient(self.client_node, action_type, target)
        self.addCleanup(client.destroy)
        self.assertTrue(client.wait_for_server(timeout_sec=5))
        return client

    def mission(self, text):
        client = self.client(RunTasks, 'tasks/run')
        handle = self.wait(client.send_goal_async(RunTasks.Goal(tasks=text, gap_s=0.0)))
        self.assertTrue(handle.accepted)
        return self.wait(handle.get_result_async()).result

    def test_feedback_from_vision_and_sensor_reaches_base(self):
        deadline = time.monotonic() + 5
        while (self.base.vision.age() > 0.6 or not self.base.sensors.has_data()) and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertLess(self.base.vision.age(), 0.6)
        self.assertTrue(self.base.sensors.has_data())
        self.assertIsNotNone(self.base.sensors.encoder_snapshot()[2])
        self.assertIsNone(self.base.motor)
        self.assertIsNone(self.vision.camera)
        self.assertIsNone(self.vision._models)

    def test_mixed_mission_and_qr_data_return_to_plan(self):
        trace = []
        originals = [(node, node.run) for node in (self.base, self.arm, self.vision)]
        for node, original in originals:
            def traced(step, handle, stop_event, original=original):
                trace.append(step.target)
                if step.command == 'scan-qrcode':
                    return {'ok': True, 'qr_data': '123+321'}
                return original(step, handle, stop_event)
            node.run = traced
        try:
            result = self.mission('arm-calibrate\nstraight 0.3\narm-move 6 0 160 1\nscan-qrcode\nhome')
            self.assertTrue(result.success, result.message)
            self.assertEqual(result.completed_steps, 5)
            self.assertEqual(trace, ['arm', 'base', 'arm', 'vision', 'arm'])
            self.assertEqual(json.loads(result.details_json)['qr_data'], '123+321')
            self.assertEqual(self.plan.qr_data, '123+321')
        finally:
            for node, original in originals:
                node.run = original

    def test_scan_failure_stops_remaining_tasks(self):
        original = self.vision.run
        self.vision.run = lambda *args: {'ok': False, 'reason': '二维码超时'}
        try:
            result = self.mission('scan-qrcode\nstraight 0.3')
            self.assertFalse(result.success)
            details = json.loads(result.details_json)
            self.assertEqual(len(details['results']), 1)
            self.assertIn('二维码超时', details['results'][0]['message'])
            self.assertIsNone(details['qr_data'])
        finally:
            self.vision.run = original

    def test_invalid_task_is_rejected_before_any_child_action(self):
        client = self.client(RunTasks, 'tasks/run')
        handle = self.wait(client.send_goal_async(RunTasks.Goal(tasks='straight 0.3\nturn nan')))
        self.assertFalse(handle.accepted)

    def test_base_rejects_scan_and_vision_rejects_motion(self):
        for target, command in (('base', 'scan-qrcode'), ('vision', 'straight 0.3')):
            client = self.client(ExecuteCommand, target + '/execute')
            handle = self.wait(client.send_goal_async(ExecuteCommand.Goal(command=command)))
            self.assertFalse(handle.accepted)

    def test_cancel_waits_for_child_cleanup_and_rejects_busy_goal(self):
        from car_nodes.common.control import check_cancel
        entered, cleaned = threading.Event(), threading.Event()
        original = self.base.run
        def waiting(step, handle, stop_event):
            entered.set()
            try:
                stop_event.wait(5)
                check_cancel(stop_event)
            finally:
                time.sleep(0.1)
                cleaned.set()
        self.base.run = waiting
        try:
            client = self.client(RunTasks, 'tasks/run')
            handle = self.wait(client.send_goal_async(RunTasks.Goal(tasks='straight 0.3')))
            self.assertTrue(entered.wait(3))
            rejected = self.wait(client.send_goal_async(RunTasks.Goal(tasks='straight 0.3')))
            self.assertFalse(rejected.accepted)
            self.wait(handle.cancel_goal_async())
            result = self.wait(handle.get_result_async()).result
            self.assertFalse(result.success)
            self.assertTrue(cleaned.is_set())
            self.assertFalse(self.base.gate.busy)
        finally:
            self.base.run = original
