"""识别前方三个彩色标靶，记录左右顺序及画面坐标。"""
import json
from pathlib import Path
import tempfile

from base.control import check_cancel
from config import ROOT
from .detect_ball import COLORS, POSITIONS

DEFAULT_POSITIONS_FILE = ROOT / "tasks" / "target_positions.json"


def run(vision, *, output_path=None, stop_event=None, log=print):
    stop_event = stop_event if stop_event is not None else getattr(vision, "stop_event", None)
    check_cancel(stop_event)
    targets = vision.observe_targets()
    check_cancel(stop_event)
    if targets is None and getattr(vision, "dry_run", False):
        return None

    path = DEFAULT_POSITIONS_FILE if output_path is None else Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".target-positions-", suffix=".json",
                                         delete=False) as output:
            temporary = Path(output.name)
            json.dump({"targets": targets}, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
        check_cancel(stop_event)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    for target in targets:
        log(f"标靶位置：{POSITIONS[target['position']]}，颜色：{COLORS[target['color']]}，"
            f"中心 ({target['center_x']:.1f}, {target['center_y']:.1f}) px")
    log(f"标靶位置已记录：{path}")
    return targets
