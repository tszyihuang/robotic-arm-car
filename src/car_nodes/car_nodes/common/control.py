"""共享的协作取消：由控制线程停车，不在 ROS 回调中并发写串口。"""
import time


class MotionCancelled(RuntimeError):
    """动作因用户取消、急停或节点退出而终止。"""


def check_cancel(stop_event):
    if stop_event is not None and stop_event.is_set():
        raise MotionCancelled("动作已取消，已停车")


def wait_cancelable(seconds, stop_event=None):
    """保留普通调用的 sleep 行为；取消时立即唤醒启动与控制等待。"""
    check_cancel(stop_event)
    if stop_event is None:
        time.sleep(seconds)
    elif stop_event.wait(max(0.0, seconds)):
        check_cancel(stop_event)
    check_cancel(stop_event)
