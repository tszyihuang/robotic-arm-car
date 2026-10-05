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
from unittest.mock import Mock

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

    def test_all_decision_kinds_execute_selected_branch_then_resume_main(self):
        from car_nodes.planner.tasks import BRANCH_NAMES
        for marker, kind, value in (('抓球任务', 'ball', 'green'), ('打靶任务', 'target', 'red'),
                                    ('抓物体任务', 'object', 'cylinder')):
            for position in (0, 1, 2):
                with self.subTest(marker=marker, position=position):
                    trace = []
                    original_vision, original_arm, original_base = self.vision.run, self.arm.run, self.base.run
                    def vision(step, handle, stop_event):
                        trace.append(step.command)
                        if step.command == 'scan-qrcode':
                            return {'ok': True, 'qr_data': '211'}
                        self.assertEqual(step.command, 'observe-target')
                        self.assertEqual(step.kwargs, {'kind': kind, 'value': value})
                        return {'ok': True, 'position': position}
                    def arm(step, handle, stop_event):
                        trace.append(step.command)
                        return original_arm(step, handle, stop_event)
                    def base(step, handle, stop_event):
                        trace.append(step.text)
                        return original_base(step, handle, stop_event)
                    self.vision.run, self.arm.run, self.base.run = vision, arm, base
                    try:
                        text = f'[主线]\nscan-qrcode\n{marker}\nstraight 0.5\n'
                        for i, name in enumerate(BRANCH_NAMES[kind]):
                            text += f'[{name}]\nstraight {0.1 * (i+1):.1f}\n'
                        result = self.mission(text)
                        self.assertTrue(result.success, result.message)
                        self.assertEqual(trace, ['scan-qrcode', 'observe-target', f'straight {0.1*(position+1):.1f}', 'straight 0.5'])
                        details = json.loads(result.details_json)
                        self.assertEqual(details['results'][1]['selected_branch'], BRANCH_NAMES[kind][position])
                        self.assertEqual(result.completed_steps, 4)
                    finally:
                        self.vision.run, self.arm.run, self.base.run = original_vision, original_arm, original_base

    def test_empty_selected_branch_and_bad_qr_stop_before_next_motion(self):
        original = self.vision.run
        for qr, error in (('211', '为空'), ('123+321', '三位')):
            with self.subTest(qr=qr):
                def vision(step, handle, stop_event):
                    return {'ok': True, 'qr_data': qr} if step.command == 'scan-qrcode' else {'ok': True, 'position': 0}
                self.vision.run = vision
                try:
                    result = self.mission('[主线]\nscan-qrcode\n抓球任务\nstraight 0.5\n[抓左边的小球]\n')
                    self.assertFalse(result.success)
                    rows = json.loads(result.details_json)['results']
                    self.assertFalse(any(row['command'] == 'straight 0.5' for row in rows))
                    self.assertIn(error, rows[-1]['message'])
                finally:
                    self.vision.run = original

    def test_angle_debug_program_prints_all_six_devices_without_hardware(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, '-B', str(root / 'debug/angles.py'),
                                 '--namespace', '/car_test', '--duration', '1.0', '--hz', '5'],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for label in ('电机1', '电机2', '电机3', '电机4', '舵机1', '舵机2'):
            self.assertIn(label, result.stdout)
        self.assertIn('dry_run', result.stdout)
        self.assertIsNone(self.arm.session._arm)
        self.assertIsNone(self.arm.session._bus)
        self.assertIsNone(self.arm._telemetry_timer)

    def test_camera_debug_receives_images_switches_models_and_scans_shared_camera(self):
        import cv2
        import numpy as np
        from car_nodes.common.control import check_cancel
        from car_nodes.debug.camera_web import CameraDebug
        from car_nodes.vision.node import virtual_geometry
        frame = np.full((1080, 1920, 3), 255, dtype=np.uint8)
        qr = cv2.resize(cv2.QRCodeEncoder_create().encode('211'), (400, 400), interpolation=cv2.INTER_NEAREST)
        frame[340:740, 760:1160] = cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)
        class Camera:
            device, index, error = 0, 0, ''
            def __init__(self):
                self.condition = threading.Condition()
                self.frame = frame
                self.capture_stamp = time.time()
                self.stop = threading.Event()
                self.thread = threading.Thread(target=self.capture)
                self.thread.start()
            def capture(self):
                while not self.stop.wait(0.03):
                    with self.condition:
                        self.index += 1
                        self.capture_stamp = time.time()
                        self.condition.notify_all()
            def next_frame(self, after=0, timeout=3, stop_event=None):
                deadline = time.monotonic() + timeout
                with self.condition:
                    while self.index <= after:
                        check_cancel(stop_event)
                        if time.monotonic() >= deadline:
                            raise TimeoutError('模拟摄像头超时')
                        self.condition.wait(0.02)
                    return self.frame, self.index, self.capture_stamp
            def close(self):
                self.stop.set()
                self.thread.join()
        camera = Camera()
        geometry = virtual_geometry()
        geometry['timing_ms'] = {'total': 0.0}
        boundary = Mock()
        boundary.predict.return_value = geometry
        objects = Mock()
        objects.predict.return_value = {'detections': [], 'timing_ms': {'total': 0.0}}
        old_camera, old_models = self.vision.camera, self.vision._models
        debug = CameraDebug('/car_test')
        self.executor.add_node(debug)
        try:
            self.vision.set_parameters([Parameter('dry_run', value=False)])
            self.vision.camera, self.vision._models = camera, (boundary, objects)
            self.assertTrue(debug.command('start-camera')['ok'])
            self.assertTrue(debug.command('set-models boundary=false objects=true')['ok'])
            deadline = time.monotonic() + 3
            while debug.image is None and time.monotonic() < deadline:
                time.sleep(0.02)
            decoded = cv2.imdecode(np.frombuffer(debug.snapshot(), dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape, frame.shape)
            self.assertEqual(boundary.predict.call_count, 0)
            self.assertGreater(objects.predict.call_count, 0)
            self.assertEqual(debug.command('scan-qrcode')['qr_data'], '211')
            self.assertTrue(debug.command('set-models boundary=false objects=false')['ok'])
            count = objects.predict.call_count
            time.sleep(0.15)
            self.assertLessEqual(objects.predict.call_count, count + 1)
            self.assertIs(self.vision.camera, camera)
            self.assertIsNotNone(debug.snapshot())
        finally:
            self.vision._inference_stop.set()
            if self.vision._inference_thread is not None:
                self.vision._inference_thread.join(timeout=3)
            camera.close()
            self.vision.camera, self.vision._models = old_camera, old_models
            self.vision._preview = False
            self.vision._boundary = self.vision._objects = None
            self.vision.set_parameters([Parameter('dry_run', value=True)])
            self.executor.remove_node(debug)
            debug.destroy_node()

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
