"""保存本趟扫码及现场排列，选择任务表中的位置动作段。"""
from dataclasses import dataclass, field

from base.control import check_cancel
from vision.qrcode import COLORS, mission_code
from .detect_ball import COLORS as COLOR_NAMES

POSITION_NAMES = {"left": "左边", "middle": "中间", "right": "右边"}
SHAPE_NAMES = {"cylinder": "圆柱形", "cone": "圆锥形", "drum": "腰鼓形"}
TASKS = {"小球任务": ("ball", "排爆物", "抓{}的小球"),
         "打靶任务": ("target", "反恐靶", "打{}的靶"),
         "人质任务": ("object", "救援目标", "抓{}的物体")}
TASK_SECTIONS = {command: {position: template.format(name)
                           for position, name in POSITION_NAMES.items()}
                 for command, (_, _, template) in TASKS.items()}
TASK_SECTIONS["人质任务"] = {"middle": "抓中间的物体"}
DETECTIONS = {"detect-balls": "ball", "detect-targets": "target"}


@dataclass
class MissionState:
    goals: dict | None = None
    layouts: dict = field(default_factory=dict)
    object_shape: str | None = None
    object_initial_position: str | None = None
    rescued_object_position: str | None = None

    def record(self, command, result):
        if command == "scan-qrcode":
            self.goals = mission_code(result)
            # 新任务不能沿用上一次扫码收集的排列。
            self.layouts.clear()
            self.object_shape = None
            self.object_initial_position = self.rescued_object_position = None
            print(f"  任务目标：排爆物={COLOR_NAMES[self.goals['ball']]}，"
                  f"反恐靶={COLOR_NAMES[self.goals['target']]}，"
                  f"救援目标={SHAPE_NAMES[self.goals['object']]}", flush=True)
        elif command in DETECTIONS:
            self.layouts[DETECTIONS[command]] = result
        elif command == "calibrate-object-position":
            self.object_shape = (result.get("shape") if isinstance(result, dict)
                                 and result.get("ok") is True else None)
            if self.object_shape is None:
                self.object_initial_position = self.rescued_object_position = None

    def observe_object_initial_position(self, vision, *, stop_event=None):
        """在底盘移动校准之前确认目标原位置，供救援后的终点路程使用。"""
        check_cancel(stop_event)
        self.object_initial_position = self.rescued_object_position = None
        if self.goals is None:
            raise ValueError("确认救援物体原位置前需要先执行 scan-qrcode")
        shape = self.goals["object"]
        position = vision.observe_target("object", shape)
        check_cancel(stop_event)
        if position is None and getattr(vision, "dry_run", False):
            return
        if position not in POSITION_NAMES:
            raise ValueError(f"视觉未明确救援目标 {SHAPE_NAMES[shape]} 校准前的位置：{position!r}")
        self.object_initial_position = position
        print(f"  救援目标原位置：{SHAPE_NAMES[shape]}，{POSITION_NAMES[position]}。", flush=True)

    def select_section(self, command, vision, *, stop_event=None):
        check_cancel(stop_event)
        if self.goals is None:
            raise ValueError(f"{command} 前需要先执行 scan-qrcode，取得有效的三位任务码")
        kind, label, _ = TASKS[command]
        value = self.goals[kind]
        value_name = SHAPE_NAMES[value] if kind == "object" else COLOR_NAMES[value]
        if kind == "object":
            if self.object_shape is None and getattr(vision, "dry_run", False):
                print(f"  {command}：{label}={value_name}；校准到指定物体前方后执行中间抓取。", flush=True)
                return None
            if self.object_shape != value:
                raise ValueError(f"{command} 前需要先执行 calibrate-object-position，"
                                 f"将二维码指定的{value_name}校准到画面中心")
            position = "middle"
        else:
            detection = next(name for name, detected_kind in DETECTIONS.items() if detected_kind == kind)
            if kind not in self.layouts:
                raise ValueError(f"{command} 前需要先执行 {detection}，收集当前任务点的颜色排列")
            layout = self.layouts[kind]
            if layout is None and getattr(vision, "dry_run", False):
                position = None
            else:
                if (not isinstance(layout, (list, tuple)) or len(layout) != 3
                        or any(not isinstance(row, dict) for row in layout)
                        or {row.get('position') for row in layout} != set(POSITION_NAMES)
                        or {row.get('color') for row in layout} != set(COLORS)):
                    raise ValueError(f"{command} 的现场排列必须包含左、中、右三个位置，且红、绿、蓝各一个")
                position = next(row['position'] for row in layout if row['color'] == value)
        check_cancel(stop_event)
        if position is None and getattr(vision, "dry_run", False):
            print(f"  {command}：{label}={value_name}；实际位置等待现场识别。", flush=True)
            return None
        if position not in POSITION_NAMES:
            raise ValueError(f"视觉未明确{label} {value_name} 的位置：{position!r}")
        section = TASK_SECTIONS[command][position]
        print(f"  {command}：{label}={value_name}，位置={POSITION_NAMES[position]} → [{section}]", flush=True)
        return section
