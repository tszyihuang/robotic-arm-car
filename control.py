"""动作取消、可中断等待，以及逐项清理。"""
import sys
import time


class MotionCancelled(RuntimeError):
    """用户取消了正在执行的动作。"""


def check_cancel(stop_event):
    if stop_event is not None and stop_event.is_set():
        raise MotionCancelled("动作已取消，已停车")


def wait_cancelable(seconds, stop_event=None):
    check_cancel(stop_event)
    if stop_event is None:
        time.sleep(seconds)
    elif stop_event.wait(max(0.0, seconds)):
        check_cancel(stop_event)
    check_cancel(stop_event)


def cleanup(*actions, raise_errors=True):
    """每个 (名称, 函数) 都尝试；打印失败，保留正在传播的原始异常。"""
    original_error = sys.exc_info()[1]
    errors = []
    for name, action in actions:
        try:
            action()
        except (Exception, KeyboardInterrupt) as exc:
            print(f"清理失败（{name}）：{exc}", file=sys.stderr, flush=True)
            errors.append(exc)
    if errors and original_error is None and raise_errors:
        raise errors[0]
