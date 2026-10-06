"""读取 YOLO 小球识别结果，在终端打印左、中、右及对应颜色。"""
import argparse
import math
from pathlib import Path
import sys
import time

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import VISION

POSITIONS = {"left": "左", "middle": "中", "right": "右"}
COLORS = {"red": "红色", "green": "绿色", "blue": "蓝色"}


def run(vision):
    balls = vision.observe_balls()
    if balls is None and getattr(vision, "dry_run", False):
        return None
    for ball in balls:
        print(f"小球位置：{POSITIONS[ball['position']]}，颜色：{COLORS[ball['color']]}", flush=True)
    return balls


def print_debug(info, *, frame_index=None, capture_stamp=None):
    """立即展示当前帧，包括非小球类别和不完整的小球检测。"""
    from vision.targets import candidates_from_detections, ordered_candidates

    detections = info["detections"]
    balls = candidates_from_detections(detections, "ball")
    header = f"[{time.strftime('%H:%M:%S')}]"
    if frame_index is not None:
        header += f" 帧 {frame_index}"
    if capture_stamp is not None:
        header += f" 距采集 {max(0.0, time.time() - capture_stamp):.3f}s"
    header += f" | 推理 {info['timing_ms']['total']:.1f}ms | 小球数量：{len(balls)}"
    print(header, flush=True)
    if not detections:
        print("  YOLO 未检测到物体", flush=True)
    for row in detections:
        x1, y1, x2, y2 = row["box"]
        print(f"  YOLO：{row['name']}，置信度 {row['confidence']:.3f}，"
              f"中心 ({(x1 + x2) / 2:.1f}, {(y1 + y2) / 2:.1f})", flush=True)
    try:
        ordered = ordered_candidates(balls)
    except ValueError as exc:
        print(f"  位置待定：{exc}", flush=True)
        return
    for position, ball in zip(POSITIONS.values(), ordered):
        print(f"  小球位置：{position}，颜色：{COLORS[ball['value']]}", flush=True)


def main(args=None):
    parser = argparse.ArgumentParser(description="独立调试 YOLO 小球检测，持续打印当前帧识别结果")
    parser.add_argument("--camera", default=VISION["device"], help="摄像头编号或 /dev/video 路径")
    parser.add_argument("--device", default=VISION["infer_device"], help="推理设备，例如 cuda:0 或 cpu")
    parser.add_argument("--conf", type=float, default=None, help="检测置信度阈值，默认读取模型配置")
    parser.add_argument("--hz", type=float, default=2.0, help="最高检测与打印频率，默认 2 Hz")
    parser.add_argument("--once", action="store_true", help="检测一帧后退出")
    parser.add_argument("--image", type=Path, help="检测已有图片一次，替代摄像头")
    opts = parser.parse_args(args)
    if not math.isfinite(opts.hz) or opts.hz <= 0:
        parser.error("--hz 必须为有限正数")
    if opts.conf is not None and (not math.isfinite(opts.conf) or not 0 < opts.conf < 1):
        parser.error("--conf 必须在 (0, 1) 内")

    from base.control import cleanup
    from vision.camera import CameraStream
    from vision.yolo_objects import ObjectPredictor

    camera = None
    try:
        print(f"加载物体模型：{VISION['objects_weights']}，设备：{opts.device}", flush=True)
        predictor = ObjectPredictor(device=opts.device, threshold=opts.conf)
        print(f"模型类别：{predictor.names}，置信度阈值：{predictor.threshold:g}", flush=True)
        if opts.image is not None:
            import cv2
            frame = cv2.imread(str(opts.image))
            if frame is None:
                raise ValueError(f"无法读取图片：{opts.image}")
            print_debug(predictor.predict(frame))
            return 0
        camera = CameraStream(opts.camera)
        print(f"摄像头：{camera.device}，分辨率：{camera.resolution}。按 Ctrl+C 退出。", flush=True)
        index = 0
        while True:
            start = time.monotonic()
            try:
                frame, index, stamp = camera.next_frame(after=index)
            except TimeoutError as exc:
                print(f"等待画面：{exc}", flush=True)
                if opts.once:
                    return 1
                continue
            print_debug(predictor.predict(frame), frame_index=index, capture_stamp=stamp)
            if opts.once:
                return 0
            time.sleep(max(0.0, 1 / opts.hz - (time.monotonic() - start)))
    except KeyboardInterrupt:
        print("检测调试已结束。", flush=True)
    except Exception as exc:
        print(f"检测调试失败：{exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if camera is not None:
            cleanup(("摄像头关闭", camera.close), raise_errors=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
