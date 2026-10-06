"""共用摄像头和模型；扫码、目标观察及后台跑道边界直接读取。"""
import math
import threading
import time

from config import VISION
from base.control import check_cancel, cleanup, MotionCancelled
from .boundary import BoundarySample, valid_geometry
from .camera import CameraStream
from .qrcode import scan_qrcode
from .targets import LABELS, observe


class Vision:
    def __init__(self, *, stop_event=None, clock=time.time):
        self.config = dict(VISION)
        self.stop_event = stop_event or threading.Event()
        self._clock = clock
        self.condition = threading.Condition()
        self._connection_lock = threading.RLock()
        self._model_lock = threading.RLock()
        self._scan_lock = threading.Lock()
        self._scanning = threading.Event()
        self._frame_cutoff = 0.0
        self._inference_stop = threading.Event()
        self._thread = None
        self.camera = None
        self._boundary_model = self._object_model = None
        self._enabled = {"boundary": False, "objects": False}
        self._model_generation = 0
        self._preview = None
        self._objects = None
        self._state, self._error = "off", ""
        self._sample = BoundarySample(None, None, 0, "off", "", 0.0, 0.0)
        self._last_index = None
        self._latest_stamp = 0.0
        self._closed = False
        for key in ("scan_timeout", "observe_timeout", "frame_stale", "model_wait"):
            if not math.isfinite(self.config[key]) or self.config[key] <= 0:
                raise ValueError(f"{key} 必须为有限正数")
        if type(self.config["observe_stable_frames"]) is not int or self.config["observe_stable_frames"] < 1:
            raise ValueError("observe_stable_frames 必须为正整数")
        if not 0 < self.config["target_min_area_ratio"] < 1:
            raise ValueError("target_min_area_ratio 必须在 (0, 1) 内")

    def start_camera(self):
        with self._connection_lock:
            check_cancel(self.stop_event)
            if self._closed:
                raise RuntimeError("视觉已关闭")
            if self.camera is None:
                self.camera = CameraStream(self.config["device"])
            check_cancel(self.stop_event)
            return self.camera

    def _load_models(self, boundary=False, objects=False):
        with self._model_lock:
            check_cancel(self.stop_event)
            if boundary and self._boundary_model is None:
                from .yolo_boundary import BoundaryPredictor
                self._boundary_model = BoundaryPredictor(
                    self.config["boundary_weights"], device=self.config["infer_device"],
                    fp16=self.config["fp16"], args_path=self.config["boundary_args"])
            check_cancel(self.stop_event)
            if objects and self._object_model is None:
                from .yolo_objects import ObjectPredictor
                self._object_model = ObjectPredictor(
                    self.config["objects_weights"], device=self.config["infer_device"],
                    fp16=self.config["fp16"], args_path=self.config["objects_args"])
            check_cancel(self.stop_event)

    def scan_qrcode(self, timeout=None):
        with self._scan_lock:
            camera = self.start_camera()
            self._scanning.set()
            try:
                self._set_resolution(camera, self.config["scan_width"], self.config["scan_height"])
                return scan_qrcode(camera, self.config["scan_timeout"] if timeout is None else timeout,
                                   stop_event=self.stop_event)
            finally:
                try:
                    cleanup(("恢复摄像头分辨率", lambda: self._set_resolution(
                        camera, self.config["width"], self.config["height"])))
                finally:
                    with self.condition:
                        self._scanning.clear()
                        self.condition.notify_all()

    def _set_resolution(self, camera, width, height):
        with self._model_lock:
            camera.set_resolution(width, height)
            self._frame_cutoff = self._clock()
            with self.condition:
                self._objects = None
                self._preview = None
                self._model_generation += 1
                state = "loading" if self._enabled["boundary"] else "off"
                self._sample = BoundarySample(None, None, self._sample.seq, state, "", 0.0, 0.0)
                self._state = "loading" if any(self._enabled.values()) else "off"
                self.condition.notify_all()

    def observe_target(self, kind, value):
        camera = self.start_camera()
        if kind != "target":
            self._load_models(objects=True)
        result = observe(camera, kind, value, self._object_model,
                         timeout=self.config["observe_timeout"],
                         stable_frames=self.config["observe_stable_frames"],
                         min_area_ratio=self.config["target_min_area_ratio"],
                         stop_event=self.stop_event, predict_lock=self._model_lock)
        return ("left", "middle", "right")[result["position"]]

    def set_models(self, *, boundary, objects):
        if type(boundary) is not bool or type(objects) is not bool:
            raise ValueError("模型开关必须为布尔值")
        with self._connection_lock:
            self.start_camera()
            self._load_models(boundary, objects)
            with self.condition:
                self._enabled = {"boundary": boundary, "objects": objects}
                self._model_generation += 1
                self._preview = None
                self._state, self._error = ("loading" if boundary or objects else "off"), ""
                if not boundary:
                    self._sample = BoundarySample(None, None, self._sample.seq, "off", "", 0.0, 0.0)
                if not objects:
                    self._objects = None
                self.condition.notify_all()
            if self._thread is None:
                self._thread = threading.Thread(target=self._infer, name="vision-inference", daemon=True)
                self._thread.start()
        return {"ok": True, "models": dict(self._enabled)}

    def enable_boundary(self):
        with self.condition:
            enabled = dict(self._enabled)
        if not enabled["boundary"]:
            self.set_models(boundary=True, objects=enabled["objects"])

    def ingest_boundary(self, info, index, stamp, *, state="ready", error="", inference_seconds=0.0):
        """采集时间随帧保存；旧帧、重复帧、无效几何不能延长有效期。"""
        now = self._clock()
        with self.condition:
            if self._closed:
                return
            old = self._sample
            valid_stamp = (type(stamp) in (int, float) and math.isfinite(stamp)
                           and 0 < stamp <= now + 0.1)
            valid_index = type(index) is int and index >= 0
            valid_duration = (type(inference_seconds) in (int, float)
                              and math.isfinite(inference_seconds) and inference_seconds >= 0)
            new = valid_index and index != self._last_index
            ordered = valid_stamp and stamp >= self._latest_stamp
            frame_dt = max(0.0, now - stamp, inference_seconds) if valid_stamp and valid_duration else math.inf
            if new and ordered:
                self._last_index, self._latest_stamp = index, stamp
            if (new and ordered and frame_dt <= self.config["frame_stale"]
                    and state == "ready" and not error and valid_geometry(info)):
                self._sample = BoundarySample(info, index, old.seq + 1, state, error,
                                              now, now - frame_dt, frame_dt)
            else:
                self._sample = BoundarySample(old.info, old.frames, old.seq, state, error,
                                              now, old.t_valid, old.frame_dt)
            self.condition.notify_all()

    def _infer(self):
        index = 0
        try:
            while not self._inference_stop.is_set():
                check_cancel(self.stop_event)
                with self.condition:
                    enabled = dict(self._enabled)
                    generation = self._model_generation
                if not any(enabled.values()) or self._scanning.is_set():
                    self._inference_stop.wait(0.05)
                    continue
                frame, index, stamp = self.camera.next_frame(after=index, stop_event=self._inference_stop)
                geometry = detections = None
                with self._model_lock:
                    check_cancel(self.stop_event)
                    if self._scanning.is_set() or stamp < self._frame_cutoff:
                        continue
                    if enabled["boundary"]:
                        geometry = self._boundary_model.predict(frame)
                    if enabled["objects"]:
                        detections = self._object_model.predict(frame)
                    with self.condition:
                        if (generation != self._model_generation or self._scanning.is_set()
                                or self._closed):
                            continue
                        if geometry is not None and self._enabled["boundary"]:
                            self.ingest_boundary(geometry, index, stamp,
                                                 inference_seconds=geometry["timing_ms"]["total"] / 1000)
                        if detections is not None and self._enabled["objects"]:
                            self._objects = (detections, index, stamp)
                        # 同一帧的全部模型执行完才发布；只保留最新完成帧，不排队。
                        self._preview = (frame, stamp, geometry, detections)
                        self._state = "ready" if any(self._enabled.values()) else "off"
                        self._error = ""
                        self.condition.notify_all()
        except Exception as exc:
            state = "off" if self._inference_stop.is_set() or isinstance(exc, MotionCancelled) else "error"
            with self.condition:
                self._state, self._error = state, str(exc) if state == "error" else ""
            self.ingest_boundary(None, index, 0.0, state=state,
                                 error=str(exc) if state == "error" else "")

    def sample(self):
        with self.condition:
            return self._sample

    def age(self, now=None):
        sample = self.sample()
        return max(0.0, (self._clock() if now is None else now) - sample.t_valid) if sample.t_valid else 1e9

    def wait_ready(self, seconds=None, log=print, stop_event=None):
        seconds = self.config["model_wait"] if seconds is None else seconds
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("等待时间必须为有限非负数")
        deadline = time.monotonic() + seconds
        while True:
            if self.stop_event.is_set() or (stop_event is not None and stop_event.is_set()) or self._closed:
                return False
            sample = self.sample()
            if sample.state == "error":
                log(f"视觉报错：{sample.error}")
                return False
            if sample.state == "ready" and not sample.error and self.age() <= self.config["frame_stale"]:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                log(f"等待新边界超时（{seconds:g}s，状态 {sample.state}）")
                return False
            with self.condition:
                self.condition.wait(min(remaining, 0.05))

    def status(self):
        with self.condition:
            camera_error = self.camera.error if self.camera is not None else ""
            return {"state": "error" if camera_error else self._state,
                    "error": camera_error or self._error,
                    "models": dict(self._enabled)}

    def _snapshot_frame(self, after_stamp):
        """模型开启时等完成帧；关闭模型或扫码时读取实时原图。"""
        camera = self.start_camera()
        deadline = time.monotonic() + 3.0
        while True:
            with self.condition:
                check_cancel(self.stop_event)
                if self._closed:
                    raise RuntimeError("视觉已关闭")
                if camera.error:
                    raise RuntimeError(camera.error)
                if any(self._enabled.values()) and not self._scanning.is_set():
                    if self._state == "error":
                        raise RuntimeError(self._error)
                    if self._preview is not None:
                        frame, stamp, geometry, objects = self._preview
                        if after_stamp is None or stamp > after_stamp:
                            return frame.copy(), stamp, geometry, objects
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("YOLO 未返回新的推理画面")
                    self.condition.wait(min(remaining, 0.05))
                    continue
                generation = self._model_generation
                scanning = self._scanning.is_set()
            if after_stamp is None:
                with camera.condition:
                    if camera.frame is None or self._clock() - camera.capture_stamp > self.config["frame_stale"]:
                        raise RuntimeError("当前没有新鲜画面")
                    frame, stamp = camera.frame.copy(), camera.capture_stamp
            else:
                frame, _, stamp = camera.next_frame(after_stamp=after_stamp, stop_event=self.stop_event,
                                                  timeout=max(0.0, deadline - time.monotonic()))
                frame = frame.copy()
            with self.condition:
                if generation == self._model_generation and scanning == self._scanning.is_set():
                    return frame, stamp, None, None

    def snapshot(self, *, after_stamp=None):
        """调试 JPEG；检测框和边界只绘制在产生这些结果的原图上。"""
        import cv2
        frame, stamp, geometry, objects = self._snapshot_frame(after_stamp)
        if objects is not None:
            for row in objects["detections"]:
                x1, y1, x2, y2 = map(int, row["box"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                label = LABELS.get(row.get("name"))
                text = " ".join(label) if label else str(row["class_id"])
                cv2.putText(frame, text, (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if geometry is not None:
            height = frame.shape[0]
            for side in ("left", "right"):
                line = geometry.get(side)
                if line:
                    points = line.get("points") or [[line["a"] * y + line["b"], y]
                                                   for y in (line.get("near_y", height - 1), line.get("far_y", 0))]
                    cv2.line(frame, tuple(map(int, points[0])), tuple(map(int, points[1])), (0, 255, 0), 2)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError("JPEG 编码失败")
        return encoded.tobytes(), stamp

    def close(self):
        self._inference_stop.set()
        with self._connection_lock:
            with self.condition:
                self._closed = True
                self._preview = None
                self._sample = BoundarySample(None, None, self._sample.seq, "off", "", 0.0, 0.0)
                self.condition.notify_all()
            actions = []
            if self.camera is not None:
                actions.append(("摄像头", self.camera.close))
            if self._thread is not None:
                actions.append(("视觉推理线程", self._thread.join))
            cleanup(*actions)
            self.camera = None
