"""共用摄像头和模型；扫码、目标观察及后台跑道边界直接读取。"""
import math
import threading
import time

from config import VISION
from base.control import check_cancel, cleanup, MotionCancelled
from .boundary import BoundarySample, valid_geometry
from .camera import CameraStream
from .qrcode import scan_qrcode
from .targets import LABELS, candidates_from_detections, colored_targets, observe, validate_target


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
        self._target_requested = threading.Event()
        self._frame_cutoff = 0.0
        self._inference_stop = threading.Event()
        self._thread = None
        self.camera = None
        self._boundary_model = self._object_model = None
        self._enabled = {"boundary": False, "objects": False}
        self._model_generation = 0
        self._preview = None
        self._objects = None
        self._targets = None
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
            self._check_open()
            if boundary and self._boundary_model is None:
                from .yolo_boundary import BoundaryPredictor
                self._boundary_model = BoundaryPredictor(
                    self.config["boundary_weights"], device=self.config["infer_device"],
                    fp16=self.config["fp16"], args_path=self.config["boundary_args"])
            self._check_open()
            if objects and self._object_model is None:
                from .yolo_objects import ObjectPredictor
                self._object_model = ObjectPredictor(
                    self.config["objects_weights"], device=self.config["infer_device"],
                    fp16=self.config["fp16"], args_path=self.config["objects_args"])
            self._check_open()
            with self.condition:
                self.condition.notify_all()

    def _check_open(self):
        check_cancel(self.stop_event)
        if self._closed or self._inference_stop.is_set():
            raise MotionCancelled("视觉已关闭")

    def start(self):
        """立即启动后台采集、加载及推理；整趟任务复用两套模型。"""
        return self.set_models(boundary=True, objects=True)

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
                self._targets = None
                self._preview = None
                self._model_generation += 1
                if self._state == "error":
                    self.condition.notify_all()
                    return
                state = "loading" if self._enabled["boundary"] else "off"
                self._sample = BoundarySample(None, None, self._sample.seq, state, "", 0.0, 0.0)
                self._state = "loading" if any(self._enabled.values()) else "off"
                self.condition.notify_all()

    def observe_target(self, kind, value):
        validate_target(kind, value)
        result = self._observe(kind, value)
        return ("left", "middle", "right")[result["position"]]

    def observe_balls(self):
        """读取后台 YOLO 小球结果，返回连续稳定的左、中、右颜色。"""
        result = self._observe("ball", None)
        return [{"position": position, "color": row["value"]}
                for position, row in zip(("left", "middle", "right"), result["candidates"])]

    def observe_ball_layout(self, *, timeout=None):
        """读取调用后新帧的三个球坐标及同一原图尺寸，供位置校准使用。"""
        return self._observe("ball", None, timeout=timeout, stable_frames=1, include_size=True)

    def ball_layout_sample(self):
        """非阻塞读取最新物体推理；保留缺球帧，让连续控制及时停车。"""
        with self.condition:
            self._check_open()
            if self.camera is not None and self.camera.error:
                raise RuntimeError(self.camera.error)
            if self._state == "error":
                raise RuntimeError(self._error)
            if self._objects is None or not self._enabled["objects"] or self._scanning.is_set():
                return None
            info, index, stamp = self._objects
            if not -0.1 <= self._clock() - stamp < self.config["frame_stale"]:
                return None
            return {"size": info.get("size"), "candidates": candidates_from_detections(info["detections"], "ball"),
                    "frame_index": index, "capture_stamp": stamp}

    def wait_ball_layout(self, *, after, timeout, stop_event=None):
        """新物体结果发布即唤醒；到控制周期截止仍无新结果时返回 None。"""
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("等待小球结果的时间必须为有限非负数")
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                check_cancel(stop_event)
                sample = self.ball_layout_sample()
                if sample is not None and sample["frame_index"] != after:
                    return sample
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(min(remaining, 0.05))

    def _observe(self, kind, value, *, timeout=None, stable_frames=None, include_size=False):
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("观察超时必须为有限正数")
        deadline = None if timeout is None else time.monotonic() + timeout
        # 扫码与观察共用相机，串行请求避免分辨率切换污染稳定帧。
        with self._scan_lock:
            camera = self.start_camera()
            with self._connection_lock:
                with self.condition:
                    enabled = dict(self._enabled)
                self.set_models(boundary=enabled["boundary"],
                                objects=enabled["objects"] or kind != "target")
            if kind != "target":
                model_deadline = time.monotonic() + self.config["model_wait"]
                if deadline is not None:
                    model_deadline = min(model_deadline, deadline)
                with self.condition:
                    while self._object_model is None:
                        self._check_open()
                        if self._state == "error":
                            raise RuntimeError(self._error)
                        remaining = model_deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("等待目标模型加载超时")
                        self.condition.wait(min(remaining, 0.05))
            else:
                self._target_requested.set()
            try:
                remaining = self.config["observe_timeout"] if deadline is None else deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("目标观察超时")
                result = observe(camera, kind, value,
                                 timeout=remaining,
                                 stable_frames=self.config["observe_stable_frames"] if stable_frames is None else stable_frames,
                                 stop_event=self.stop_event,
                                 candidate_source=lambda **kwargs: self._wait_candidates(
                                     kind, include_size=include_size, **kwargs))
                return result
            finally:
                self._target_requested.clear()

    def _wait_candidates(self, kind, *, after, after_stamp, timeout, include_size=False):
        """读取调用之后的新帧，稳定计数不能重复消费同一份结果。"""
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                self._check_open()
                if self.camera.error:
                    raise RuntimeError(self.camera.error)
                if self._state == "error":
                    raise RuntimeError(self._error)
                cached = self._targets if kind == "target" else self._objects
                if cached is not None:
                    info, index, stamp = cached
                    if (index > after and stamp > after_stamp
                            and self._clock() - stamp <= self.config["frame_stale"]):
                        candidates = info if kind == "target" else candidates_from_detections(info["detections"], kind)
                        if include_size:
                            return candidates, index, stamp, info.get("size")
                        return candidates, index, stamp
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = "后台尚未完成目标推理"
                    if cached is not None:
                        age = max(0.0, self._clock() - stamp)
                        names = ([row['name'] for row in info] if kind == 'target' else
                                 [row['name'] for row in info['detections']])
                        detail = (f"最新帧距采集 {age:.3f}s，新鲜度上限 {self.config['frame_stale']:g}s；"
                                  f"最新识别类别：{names}")
                    raise TimeoutError(f"后台未返回新的有效识别结果（{detail}）")
                self.condition.wait(min(remaining, 0.05))

    def set_models(self, *, boundary, objects):
        if type(boundary) is not bool or type(objects) is not bool:
            raise ValueError("模型开关必须为布尔值")
        with self._connection_lock:
            self._check_open()
            with self.condition:
                if self._state == "error":
                    raise RuntimeError(self._error)
                if self._thread is not None and self._enabled == {"boundary": boundary, "objects": objects}:
                    return {"ok": True, "models": dict(self._enabled)}
                self._enabled = {"boundary": boundary, "objects": objects}
                self._model_generation += 1
                self._preview = None
                self._state, self._error = ("loading" if boundary or objects else "off"), ""
                self._sample = BoundarySample(None, None, self._sample.seq,
                                              "loading" if boundary else "off", "", 0.0, 0.0)
                if not objects:
                    self._objects = None
                self.condition.notify_all()
            if self._thread is None:
                self._thread = threading.Thread(target=self._infer, name="vision-inference", daemon=True)
                self._thread.start()
        return {"ok": True, "models": dict(self._enabled)}

    def enable_boundary(self):
        with self._connection_lock:
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
            camera = self.start_camera()
            while not self._inference_stop.is_set():
                check_cancel(self.stop_event)
                with self.condition:
                    enabled = dict(self._enabled)
                    generation = self._model_generation
                self._load_models(**enabled)
                if (not any(enabled.values()) and not self._target_requested.is_set()) or self._scanning.is_set():
                    self._inference_stop.wait(0.05)
                    continue
                frame, index, stamp = camera.next_frame(after=index, stop_event=self._inference_stop)
                geometry = detections = targets = None
                with self._model_lock:
                    self._check_open()
                    if self._scanning.is_set() or stamp < self._frame_cutoff:
                        continue
                    if enabled["objects"]:
                        detections = self._object_model.predict(frame)
                        with self.condition:
                            if (generation != self._model_generation or self._scanning.is_set()
                                    or self._closed):
                                continue
                            # 物体推理完成即交给控制端，后续边界推理单独执行。
                            self._objects = (detections, index, stamp)
                            self.condition.notify_all()
                    if enabled["boundary"]:
                        geometry = self._boundary_model.predict(frame)
                    if self._target_requested.is_set():
                        targets = colored_targets(frame, self.config["target_min_area_ratio"])
                    with self.condition:
                        if (generation != self._model_generation or self._scanning.is_set()
                                or self._closed):
                            continue
                        if geometry is not None and self._enabled["boundary"]:
                            self.ingest_boundary(geometry, index, stamp,
                                                 inference_seconds=geometry["timing_ms"]["total"] / 1000)
                        if targets is not None:
                            self._targets = (targets, index, stamp)
                        # 网页预览等本帧全部模型完成再更新，保证检测框与边界对应原图。
                        self._preview = (frame, stamp, geometry, detections)
                        self._state = "ready" if any(self._enabled.values()) else "off"
                        self._error = ""
                        self.condition.notify_all()
        except Exception as exc:
            state = "off" if self._inference_stop.is_set() or isinstance(exc, MotionCancelled) else "error"
            with self.condition:
                self._state, self._error = state, str(exc) if state == "error" else ""
                self.condition.notify_all()
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
                self._state, self._error = "off", ""
                self._preview = None
                self._objects = self._targets = None
                self._sample = BoundarySample(None, None, self._sample.seq, "off", "", 0.0, 0.0)
                self.condition.notify_all()
            camera, thread = self.camera, self._thread
        # 不持有连接锁等待线程：后台可能正在等待同一把锁打开相机。
        actions = []
        if camera is not None:
            actions.append(("摄像头", camera.close))
        if thread is not None:
            actions.append(("视觉推理线程", thread.join))
        try:
            cleanup(*actions)
        finally:
            self.camera = None
