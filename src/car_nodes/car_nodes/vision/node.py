"""Vision owns the camera, publishes geometry/objects and answers QR actions."""
import math
import threading
import time

from car_interfaces.action import ExecuteCommand
from car_interfaces.msg import Boundary, Objects

from ..common.action_node import SerialActionNode, run_node, json_text
from ..common.control import check_cancel
from ..common.messages import set_stamp
from ..planner.tasks import prepare_tasks
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
                    'scan_timeout': 30.0, 'publish_hz': 25.0}
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        for key in ('scan_timeout', 'publish_hz'):
            value = self.get_parameter(key).value
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} 必须是有限正数')
        self.camera = None
        self._models = None
        self._inference_thread = None
        self._inference_stop = threading.Event()
        self._data_lock = threading.Lock()
        self._state, self._error = 'off', ''
        self._boundary = self._objects = None
        self.boundary_pub = self.create_publisher(Boundary, 'vision/boundary', 1)
        self.objects_pub = self.create_publisher(Objects, 'vision/objects', 1)
        self._virtual_index = 0
        self.create_timer(1 / self.get_parameter('publish_hz').value, self._publish)

    def prepare(self, request):
        if request.command == 'start-boundary':
            return None
        steps = prepare_tasks(request.command, require_calibration=False)
        if len(steps) != 1 or steps[0].command != 'scan-qrcode':
            raise ValueError('视觉 Action 只接受扫码请求或 start-boundary')
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
            return {'ok': True, 'dry_run': True, 'command': step.text if step else 'start-boundary'}
        if step is not None:
            camera = self._ensure_camera(step.kwargs.get('device'))
            data = scan_qrcode(camera, step.kwargs.get('timeout', self.get_parameter('scan_timeout').value),
                               stop_event=stop_event)
            return {'ok': True, 'qr_data': data}
        self._ensure_camera()
        if self._inference_thread is not None and self._inference_stop.is_set():
            self._inference_thread.join()
        with self._data_lock:
            self._state, self._error = 'loading', ''
        try:
            if self._models is None:
                from .yolo_boundary import BoundaryPredictor
                from .yolo_objects import ObjectPredictor
                device = self.get_parameter('infer_device').value
                boundary = BoundaryPredictor(self.get_parameter('weights').value, device=device,
                                             args_path=self.get_parameter('args_path').value)
                check_cancel(stop_event)
                objects = (ObjectPredictor(self.get_parameter('objects_weights').value, device=device,
                           args_path=self.get_parameter('objects_args_path').value)
                           if self.get_parameter('enable_objects').value else None)
                check_cancel(stop_event)
                self._models = boundary, objects
            if self._inference_thread is None or not self._inference_thread.is_alive():
                self._inference_stop.clear()
                self._inference_thread = threading.Thread(target=self._infer, name='vision-inference', daemon=True)
                self._inference_thread.start()
            return {'ok': True, 'state': 'loading'}
        except Exception as exc:
            with self._data_lock:
                self._state, self._error = 'error', str(exc)
            raise

    def _infer(self):
        index = 0
        try:
            while not self._inference_stop.is_set():
                frame, index, stamp = self.camera.next_frame(after=index, stop_event=self._inference_stop)
                boundary, objects = self._models
                geometry = boundary.predict(frame)
                with self._data_lock:
                    self._boundary = (geometry, index, stamp)
                    self._state, self._error = 'ready', ''
                if objects is not None:
                    detections = objects.predict(frame)
                    with self._data_lock:
                        self._objects = (detections, index, stamp)
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
            boundary = Boundary(state=state, error=error, geometry_json='null')
            objects = Objects(state=state if self.get_parameter('enable_objects').value else 'off',
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
