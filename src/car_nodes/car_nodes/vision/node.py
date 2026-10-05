"""Vision owns the camera, publishes geometry/objects and answers QR actions."""
import math
import threading
import time

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from car_interfaces.action import ExecuteCommand
from car_interfaces.msg import Boundary, Objects

from ..common.action_node import SerialActionNode, run_node, json_text
from ..common.control import check_cancel
from ..common.messages import set_stamp
from ..planner.tasks import prepare_tasks, PreparedTask
from .commands import internal_command
from .targets import observe
from .camera import CameraStream, normalize_device
from .paths import MODEL_DIR, OBJECT_MODEL_DIR
from .qrcode import scan_qrcode


def virtual_geometry():
    return {'size': [1920, 1080],
            'left': {'a': -0.3, 'b': 704.0, 'near_y': 1079.0, 'far_y': 80.0, 'confidence': 1.0},
            'right': {'a': 0.3, 'b': 1216.0, 'near_y': 1079.0, 'far_y': 80.0, 'confidence': 1.0}}


class VisionNode(SerialActionNode):
    def __init__(self):
        super().__init__('vision_node', ExecuteCommand, 'vision/execute')
        defaults = {'device': '0', 'infer_device': 'cpu', 'enable_objects': True,
                    'weights': str(MODEL_DIR / 'weights.pt'),
                    'args_path': str(MODEL_DIR / 'args.yaml'),
                    'objects_weights': str(OBJECT_MODEL_DIR / 'weights.pt'),
                    'objects_args_path': str(OBJECT_MODEL_DIR / 'args.yaml'),
                    'scan_timeout': 30.0, 'publish_hz': 25.0,
                    'observe_timeout': 10.0, 'observe_stable_frames': 3,
                    'target_min_area_ratio': 0.001}
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        for key in ('scan_timeout', 'publish_hz', 'observe_timeout'):
            value = self.get_parameter(key).value
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} 必须是有限正数')
        if self.get_parameter('observe_stable_frames').value < 1:
            raise ValueError('observe_stable_frames 必须大于零')
        area = self.get_parameter('target_min_area_ratio').value
        if not math.isfinite(area) or not 0 < area < 1:
            raise ValueError('target_min_area_ratio 必须在 (0, 1) 内')
        self.camera = None
        self._models = None
        self._inference_thread = None
        self._inference_stop = threading.Event()
        self._data_lock = threading.Lock()
        self._model_lock = threading.RLock()
        self._enabled = {'boundary': False, 'objects': False}
        self._preview = False
        self._preview_index = 0
        self._state, self._error = 'off', ''
        self._boundary = self._objects = None
        self.boundary_pub = self.create_publisher(Boundary, 'vision/boundary', 1)
        self.objects_pub = self.create_publisher(Objects, 'vision/objects', 1)
        self.image_pub = self.create_publisher(CompressedImage, 'vision/image/compressed', 1)
        self.debug_pub = self.create_publisher(String, 'vision/debug_status', 1)
        self._virtual_index = 0
        self.create_timer(1 / self.get_parameter('publish_hz').value, self._publish)
        self.create_timer(0.1, self._publish_preview)

    def prepare(self, request):
        internal = internal_command(request.command)
        if internal is not None:
            name, kwargs = internal
            return PreparedTask(0, name, (), kwargs, request.command)
        steps = prepare_tasks(request.command, require_calibration=False)
        if len(steps) != 1 or steps[0].command != 'scan-qrcode':
            raise ValueError('视觉 Action 只接受扫码、目标观察或摄像头/模型调试请求')
        return steps[0]

    def _ensure_camera(self, device=None):
        device = normalize_device(self.get_parameter('device').value if device is None else device)
        if self.camera is not None and self.camera.device != device:
            if self._inference_thread is not None and self._inference_thread.is_alive():
                raise ValueError('边界推理正在运行，扫码设备须与视觉节点的 device 相同')
            self.camera.close()
            self.camera = None
        if self.camera is None:
            self.camera = CameraStream(device)
        return self.camera

    def run(self, step, handle, stop_event):
        if self.get_parameter('dry_run').value:
            if step.command == 'observe-target':
                raise RuntimeError('视觉节点为 dry_run，不能从虚拟画面选择实际目标')
            if step.command == 'set-models':
                self._enabled = dict(step.kwargs)
            if step.command == 'start-camera':
                self._preview = True
            return {'ok': True, 'dry_run': True, 'command': step.text}
        if step.command == 'scan-qrcode':
            camera = self._ensure_camera(step.kwargs.get('device'))
            data = scan_qrcode(camera, step.kwargs.get('timeout', self.get_parameter('scan_timeout').value),
                               stop_event=stop_event)
            return {'ok': True, 'qr_data': data}
        self._ensure_camera()
        if step.command == 'start-camera':
            self._preview = True
            return {'ok': True}
        if step.command == 'observe-target':
            kind, value = step.kwargs['kind'], step.kwargs['value']
            if kind != 'target':
                self._load_models(False, True, stop_event)
            return observe(self.camera, kind, value, None if kind == 'target' else self._models[1],
                           timeout=self.get_parameter('observe_timeout').value,
                           stable_frames=self.get_parameter('observe_stable_frames').value,
                           min_area_ratio=self.get_parameter('target_min_area_ratio').value,
                           stop_event=stop_event, predict_lock=self._model_lock)
        enabled = (step.kwargs if step.command == 'set-models' else
                   {'boundary': True, 'objects': self.get_parameter('enable_objects').value})
        self._load_models(enabled['boundary'], enabled['objects'], stop_event)
        if self._inference_thread is not None and self._inference_stop.is_set():
            self._inference_thread.join()
        with self._data_lock:
            self._enabled = dict(enabled)
            if not enabled['boundary']:
                self._boundary = None
            if not enabled['objects']:
                self._objects = None
            self._state, self._error = 'loading', ''
        if self._inference_thread is None or not self._inference_thread.is_alive():
            self._inference_stop.clear()
            self._inference_thread = threading.Thread(target=self._infer, name='vision-inference', daemon=True)
            self._inference_thread.start()
        return {'ok': True, 'models': enabled}

    def _load_models(self, want_boundary, want_objects, stop_event):
        with self._model_lock:
            boundary, objects = self._models or (None, None)
            device = self.get_parameter('infer_device').value
            if want_boundary and boundary is None:
                from .yolo_boundary import BoundaryPredictor
                boundary = BoundaryPredictor(self.get_parameter('weights').value, device=device,
                                             args_path=self.get_parameter('args_path').value)
            check_cancel(stop_event)
            if want_objects and objects is None:
                from .yolo_objects import ObjectPredictor
                objects = ObjectPredictor(self.get_parameter('objects_weights').value, device=device,
                                          args_path=self.get_parameter('objects_args_path').value)
            check_cancel(stop_event)
            self._models = boundary, objects

    def _infer(self):
        index = 0
        try:
            while not self._inference_stop.is_set():
                with self._data_lock:
                    enabled = dict(self._enabled)
                if not any(enabled.values()):
                    with self._data_lock:
                        self._state = 'off'
                    self._inference_stop.wait(0.05)
                    continue
                frame, index, stamp = self.camera.next_frame(after=index, stop_event=self._inference_stop)
                geometry = detections = None
                with self._model_lock:
                    boundary, objects = self._models
                    if enabled['boundary']:
                        geometry = boundary.predict(frame)
                    if enabled['objects']:
                        detections = objects.predict(frame)
                with self._data_lock:
                    if geometry is not None and self._enabled['boundary']:
                        self._boundary = (geometry, index, stamp)
                    if detections is not None and self._enabled['objects']:
                        self._objects = (detections, index, stamp)
                    self._state, self._error = 'ready', ''
        except Exception as exc:
            with self._data_lock:
                self._state, self._error = ('off', '') if self._inference_stop.is_set() else ('error', str(exc))

    def _publish(self):
        if self.get_parameter('dry_run').value:
            self._virtual_index += 1
            stamp = time.time()
            boundary = Boundary(state='ready', frame_index=self._virtual_index,
                                geometry_json=json_text(virtual_geometry()))
            objects = Objects(state='ready', frame_index=self._virtual_index, detections_json='[]')
            set_stamp(boundary.header, stamp, 'camera_optical_frame')
            set_stamp(objects.header, stamp, 'camera_optical_frame')
        else:
            with self._data_lock:
                state, error = self._state, self._error
                geometry, detections = self._boundary, self._objects
                enabled = dict(self._enabled)
            boundary = Boundary(state=state if enabled['boundary'] else 'off', error=error, geometry_json='null')
            objects = Objects(state=state if enabled['objects'] else 'off',
                              error=error, detections_json='[]')
            if geometry is not None:
                info, index, stamp = geometry
                boundary.frame_index = index
                boundary.geometry_json = json_text(info)
                boundary.inference_ms = float(info['timing_ms']['total'])
                set_stamp(boundary.header, stamp, 'camera_optical_frame')
            if detections is not None:
                info, index, stamp = detections
                objects.frame_index = index
                objects.detections_json = json_text(info['detections'])
                objects.inference_ms = float(info['timing_ms']['total'])
                set_stamp(objects.header, stamp, 'camera_optical_frame')
        self.boundary_pub.publish(boundary)
        self.objects_pub.publish(objects)

    def _publish_preview(self):
        if not self._preview:
            return
        with self._data_lock:
            status = {'state': self._state, 'error': self._error, 'models': dict(self._enabled),
                      'dry_run': self.get_parameter('dry_run').value}
            detections, boundary = self._objects, self._boundary
        self.debug_pub.publish(String(data=json_text(status)))
        if self.camera is None:
            return
        with self.camera.condition:
            if (self.camera.frame is None or self.camera.index == self._preview_index
                    or time.time() - self.camera.capture_stamp > 0.6):
                return
            frame = self.camera.frame.copy()
            self._preview_index, stamp = self.camera.index, self.camera.capture_stamp
        import cv2
        from .targets import LABELS
        if detections is not None and time.time() - detections[2] < 0.6:
            for row in detections[0]['detections']:
                x1, y1, x2, y2 = map(int, row['box'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                label = LABELS.get(row['name'])
                text = ' '.join(label) if label else str(row['class_id'])
                cv2.putText(frame, text, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if boundary is not None and time.time() - boundary[2] < 0.6:
            for side in ('left', 'right'):
                line = boundary[0].get(side)
                if line:
                    points = [(int(line['a'] * y + line['b']), int(y)) for y in (line['near_y'], line['far_y'])]
                    cv2.line(frame, points[0], points[1], (0, 255, 0), 2)
        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            message = CompressedImage(format='jpeg', data=encoded.tobytes())
            set_stamp(message.header, stamp, 'camera_optical_frame')
            self.image_pub.publish(message)

    def on_stop(self):
        if hasattr(self, '_inference_stop'):
            self._inference_stop.set()

    def interrupted_cleanup(self):
        self.on_stop()
        if self._inference_thread is not None:
            self._inference_thread.join()

    def close(self):
        self._inference_stop.set()
        if self.camera is not None:
            self.camera.close()
        if self._inference_thread is not None:
            self._inference_thread.join()
        self.camera = None


def main(args=None):
    run_node(VisionNode, args)
