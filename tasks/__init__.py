"""任务间沿用原来的 0.5 秒等待；干跑无需等待。"""
from config import STEP_GAP
from base.control import wait_cancelable


def pause(device):
    if not getattr(device, "dry_run", False):
        wait_cancelable(STEP_GAP, getattr(device, "stop_event", None))
