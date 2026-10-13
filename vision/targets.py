"""在当前观察姿态识别三个候选，并按画面从左到右确定目标位置。"""
import math
import time

from config import VISION
from base.control import check_cancel
from .qrcode import COLORS, SHAPES

LABELS = {"红球": ("ball", "red"), "绿球": ("ball", "green"), "蓝球": ("ball", "blue"),
          "red ball": ("ball", "red"), "green ball": ("ball", "green"), "blue ball": ("ball", "blue"),
          "red target": ("target", "red"), "green target": ("target", "green"),
          "blue target": ("target", "blue"),
          "红靶": ("target", "red"), "绿靶": ("target", "green"), "蓝靶": ("target", "blue"),
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


def ordered_candidates(candidates):
    """三个候选必须完整可见，按检测框中心从左到右排序。"""
    ordered = sorted(candidates, key=lambda row: row['center_x'])
    if len(ordered) != 3:
        raise ValueError(f'需要完整看见三个候选，当前识别到 {len(ordered)} 个')
    centers = [row['center_x'] for row in ordered]
    if centers[1] - centers[0] < 1 or centers[2] - centers[1] < 1:
        raise ValueError('候选位置重叠，无法确定左右顺序')
    return ordered


def choose_position(candidates, value, *, require_all=True):
    ordered = (ordered_candidates(candidates) if require_all
               else sorted(candidates, key=lambda row: row['center_x']))
    matches = [i for i, row in enumerate(ordered) if row['value'] == value]
    if len(matches) != 1:
        raise ValueError(f'目标 {value} 必须唯一，当前匹配 {len(matches)} 个')
    return matches[0], ordered


def validate_target(kind, value):
    if kind not in ('ball', 'target', 'object') or value not in (SHAPES if kind == 'object' else COLORS):
        raise ValueError('无效的目标类型或颜色/形状')


def observe(camera, kind, value, predictor=None, *, timeout=VISION["observe_timeout"], stable_frames=VISION["observe_stable_frames"],
            stop_event=None, predict_lock=None, candidate_source=None, require_all=True):
    # 不指定颜色/形状时观察三个候选的左右顺序。
    layout_only = kind in ('ball', 'target', 'object') and value is None
    if not layout_only:
        validate_target(kind, value)
    if candidate_source is None and predictor is None:
        raise ValueError("目标观察需要 YOLO 预测器或后台 YOLO 结果源")
    # 不复用移动途中或上一个任务留下的画面。
    with camera.condition:
        index = camera.index
    after_stamp = time.time()
    deadline, previous, count, reason = time.monotonic() + timeout, None, 0, '等待新画面'
    while True:
        check_cancel(stop_event)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f'目标观察超时：{reason}')
        if candidate_source is not None:
            try:
                sample = candidate_source(after=index, after_stamp=after_stamp, timeout=remaining)
                candidates, index, stamp = sample[:3]
                size = sample[3] if len(sample) == 4 else None
            except TimeoutError as exc:
                raise TimeoutError(f'目标观察超时：{reason}；{exc}') from exc
        else:
            frame, index, stamp = camera.next_frame(after=index, timeout=min(3.0, remaining), stop_event=stop_event)
            size = None
            if predict_lock is None:
                info = predictor.predict(frame)
            else:
                with predict_lock:
                    check_cancel(stop_event)
                    info = predictor.predict(frame)
            candidates = candidates_from_detections(info['detections'], kind)
            size = info.get('size')
        check_cancel(stop_event)
        if time.time() - stamp > VISION["frame_stale"]:
            previous, count, reason = None, 0, '推理结果对应画面已过期'
            continue
        try:
            if layout_only:
                position, ordered = None, ordered_candidates(candidates)
                if kind == 'target' and {row['value'] for row in ordered} != set(COLORS):
                    raise ValueError('需要完整看见红、绿、蓝三个标靶，且每种颜色唯一')
            else:
                position, ordered = choose_position(candidates, value, require_all=require_all)
        except ValueError as exc:
            previous, count, reason = None, 0, str(exc)
            continue
        signature = (position, tuple(row['value'] for row in ordered))
        count = count + 1 if signature == previous else 1
        previous = signature
        if count >= stable_frames:
            return {'ok': True, 'kind': kind, 'value': value, 'position': position,
                    'candidates': ordered, 'frame_index': index, 'capture_stamp': stamp,
                    'size': size}
