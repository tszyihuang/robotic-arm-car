"""Thread-safe admission and latched stop state, independent of ROS."""
import threading


class SafetyGate:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.busy = False
        self.stopped = False

    def reserve(self):
        with self.lock:
            if self.busy or self.stopped:
                return False
            self.busy = True
            self.stop_event.clear()
            return True

    def release(self):
        with self.lock:
            self.busy = False

    def finish(self, interrupted_cleanup):
        """Serialize the final stop check with a late emergency-stop callback."""
        with self.lock:
            try:
                if self.stopped or self.stop_event.is_set():
                    interrupted_cleanup()
            finally:
                self.busy = False

    def trip(self):
        with self.lock:
            self.stopped = True
            self.stop_event.set()

    def reset(self):
        with self.lock:
            if self.busy:
                return False
            self.stopped = False
            self.stop_event.clear()
            return True
