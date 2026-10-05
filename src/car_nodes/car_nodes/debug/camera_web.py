"""局域网摄像头调试：画面、截图、模型开关和二维码。"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import socket
import threading
import time
from urllib.parse import urlsplit

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from ..common.messages import stamp_seconds
from .ros_client import DebugClient


class CameraDebug(DebugClient):
    def __init__(self, namespace):
        super().__init__('camera_debug', namespace, 'vision')
        self.lock = threading.Lock()
        self.image, self.image_stamp = None, 0.0
        self.status = {'state': 'waiting', 'models': {'boundary': False, 'objects': False}}
        self.create_subscription(CompressedImage, 'vision/image/compressed', self._image, 1)
        self.create_subscription(String, 'vision/debug_status', self._status, 1)

    def _image(self, message):
        with self.lock:
            self.image, self.image_stamp = bytes(message.data), stamp_seconds(message.header)

    def _status(self, message):
        try:
            data = json.loads(message.data)
        except (ValueError, TypeError):
            return
        with self.lock:
            self.status = data

    def snapshot(self):
        with self.lock:
            if self.image is None or time.time() - self.image_stamp > 1.0:
                raise RuntimeError('当前没有新鲜画面，请检查摄像头或 dry_run 设置')
            return self.image


class CameraHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def reply(self, status, payload, content_type, filename=None):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(payload)))
        if filename:
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def json_reply(self, status, value):
        self.reply(status, json.dumps(value, ensure_ascii=False).encode(), 'application/json; charset=utf-8')

    def do_GET(self):
        path = urlsplit(self.path).path
        try:
            if path == '/':
                self.reply(200, Path(__file__).with_name('camera.html').read_bytes(), 'text/html; charset=utf-8')
            elif path in ('/frame.jpg', '/snapshot.jpg'):
                image = self.server.bridge.snapshot()
                filename = time.strftime('camera-%Y%m%d-%H%M%S.jpg') if path == '/snapshot.jpg' else None
                self.reply(200, image, 'image/jpeg', filename)
            elif path == '/api/status':
                with self.server.bridge.lock:
                    status = dict(self.server.bridge.status)
                    status['image_age'] = (time.time() - self.server.bridge.image_stamp
                                           if self.server.bridge.image_stamp else None)
                self.json_reply(200, status)
            else:
                self.json_reply(404, {'error': '没有此页面'})
        except RuntimeError as exc:
            self.json_reply(503, {'error': str(exc)})

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 <= length <= 4096:
                raise ValueError('请求长度无效')
            body = json.loads(self.rfile.read(length) or b'{}')
            path = urlsplit(self.path).path
            if path == '/api/models':
                if (not isinstance(body, dict) or set(body) != {'boundary', 'objects'}
                        or any(type(v) is not bool for v in body.values())):
                    raise ValueError('模型开关必须为布尔值')
                command = 'set-models ' + ' '.join(f'{key}={str(value).lower()}' for key, value in body.items())
            elif path == '/api/scan':
                command = 'scan-qrcode'
            else:
                self.json_reply(404, {'error': '没有此操作'})
                return
            result = self.server.bridge.command(command)
            self.json_reply(200, result)
        except (ValueError, TypeError) as exc:
            self.json_reply(400, {'error': str(exc)})
        except (RuntimeError, TimeoutError) as exc:
            self.json_reply(409, {'error': str(exc)})


def local_addresses():
    addresses = set()
    try:
        addresses.update(socket.gethostbyname_ex(socket.gethostname())[2])
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
            connection.connect(('192.0.2.1', 9))
            addresses.add(connection.getsockname()[0])
    except OSError:
        pass
    return sorted(addresses - {'127.0.0.1'})


def main(args=None):
    parser = argparse.ArgumentParser(description='局域网摄像头调试网页')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--namespace', default='/car')
    opts = parser.parse_args(args)
    if not 1 <= opts.port <= 65535:
        parser.error('--port 必须为 1..65535')
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    bridge = CameraDebug(opts.namespace)
    executor = SingleThreadedExecutor()
    executor.add_node(bridge)
    worker = threading.Thread(target=executor.spin, daemon=True)
    worker.start()
    server = None
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        server = ThreadingHTTPServer((opts.host, opts.port), CameraHandler)
        server.daemon_threads = True
        server.bridge = bridge
        bridge.command('start-camera')
        print(f'本机：http://127.0.0.1:{opts.port}', flush=True)
        for address in local_addresses():
            print(f'局域网：http://{address}:{opts.port}', flush=True)
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, TimeoutError) as exc:
        print(f'摄像头调试启动失败：{exc}', flush=True)
        return 2
    finally:
        if server:
            server.server_close()
        bridge.cancel_active()
        executor.shutdown()
        worker.join(timeout=3)
        bridge.destroy_node()
        rclpy.shutdown()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0
