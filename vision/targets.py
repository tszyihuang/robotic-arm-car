"""在当前观察姿态识别三个候选，并按画面从左到右确定目标位置。"""
import math
import time

from config import VISION
from base.control import check_cancel
from .qrcode import COLORS, SHAPES

LABELS = {"红球": ("ball", "red"), "绿球": ("ball", "green"), "蓝球": ("ball", "blue"),
          "red ball": ("ball", "red"), "green ball": ("ball", "green"), "blue ball": ("ball", "blue"),
          "圆柱": ("object", "cylinder"), "圆柱形": ("object", "cylinder"),
          "圆锥": ("object", "cone"), "圆锥形": ("object", "cone"),
          "腰鼓": ("object", "drum"), "腰鼓形": ("object", "drum"),
          "cylinder": ("object", "cylinder"), "cone": ("object", "cone"),
          "waist drum": ("object", "drum")}


def candidates_from_detections(detections, kind):
    candidates = []
    for detection in detections:
        label = LABELS.get(str(detection.get('name', '')).lower().replace('_', ' '))
        if label is None or label[0] != kind:
            continue
        box = detection.get('box', [])
        if (len(box) != 4 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)
                or box[2] <= box[0] or box[3] <= box[1]):
            continue
        candidates.append({'value': label[1], 'name': detection['name'],
                           'box': box, 'center_x': (box[0] + box[2]) / 2})
    return candidates


def colored_targets(frame, min_area_ratio=VISION["target_min_area_ratio"]):
    """现有 YOLO 未训练靶类别；以 HSV 色块轮廓识别三块彩色靶。"""
    import cv2
    import numpy as np
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ranges = {'red': [((0, 70, 50), (10, 255, 255)), ((170, 70, 50), (179, 255, 255))],
              'green': [((35, 70, 50), (85, 255, 255))],
              'blue': [((90, 70, 50), (135, 255, 255))]}
    candidates = []
    height, width = frame.shape[:2]
    for color, bounds in ranges.items():
        mask = np.zeros((height, width), dtype=np.uint8)
        for lower, upper in bounds:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < height * width * min_area_ratio:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            candidates.append({'value': color, 'name': color, 'box': [x, y, x + w, y + h],
                               'center_x': x + w / 2})
    return candidates


def choose_position(candidates, value):
    ordered = sorted(candidates, key=lambda row: row['center_x'])
    if len(ordered) != 3:
        raise ValueError(f'需要完整看见三个候选，当前识别到 {len(ordered)} 个')
    centers = [row['center_x'] for row in ordered]
    if centers[1] - centers[0] < 1 or centers[2] - centers[1] < 1:
        raise ValueError('候选位置重叠，无法确定左右顺序')
    matches = [i for i, row in enumerate(ordered) if row['value'] == value]
    if len(matches) != 1:
        raise ValueError(f'目标 {value} 必须唯一，当前匹配 {len(matches)} 个')
    return matches[0], ordered


def observe(camera, kind, value, predictor=None, *, timeout=VISION["observe_timeout"], stable_frames=VISION["observe_stable_frames"],
            min_area_ratio=VISION["target_min_area_ratio"], stop_event=None, predict_lock=None):
    if kind not in ('ball', 'target', 'object') or value not in (SHAPES if kind == 'object' else COLORS):
        raise ValueError('无效的目标类型或颜色/形状')
    # 不复用移动途中或上一个任务留下的画面。
    with camera.condition:
        index = camera.index
    deadline, previous, count, reason = time.monotonic() + timeout, None, 0, '等待新画面'
    while True:
        check_cancel(stop_event)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f'目标观察超时：{reason}')
        frame, index, stamp = camera.next_frame(after=index, timeout=min(3.0, remaining), stop_event=stop_event)
        if kind == 'target':
            candidates = colored_targets(frame, min_area_ratio)
        else:
            if predict_lock is None:
                info = predictor.predict(frame)
            else:
                with predict_lock:
                    check_cancel(stop_event)
                    info = predictor.predict(frame)
            candidates = candidates_from_detections(info['detections'], kind)
        check_cancel(stop_event)
        if time.time() - stamp > VISION["frame_stale"]:
            previous, count, reason = None, 0, '推理结果对应画面已过期'
            continue
        try:
            position, ordered = choose_position(candidates, value)
        except ValueError as exc:
            previous, count, reason = None, 0, str(exc)
            continue
        signature = (position, tuple(row['value'] for row in ordered))
        count = count + 1 if signature == previous else 1
        previous = signature
        if count >= stable_frames:
            return {'ok': True, 'kind': kind, 'value': value, 'position': position,
                    'candidates': ordered, 'frame_index': index, 'capture_stamp': stamp}
