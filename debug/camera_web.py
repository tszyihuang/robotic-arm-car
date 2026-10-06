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

from vision.api import Vision
from base.control import cleanup


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
                image, _ = self.server.vision.snapshot()
                filename = time.strftime('camera-%Y%m%d-%H%M%S.jpg') if path == '/snapshot.jpg' else None
                self.reply(200, image, 'image/jpeg', filename)
            elif path == '/api/status':
                status = self.server.vision.status()
                camera = self.server.vision.camera
                stamp = camera.capture_stamp if camera is not None else 0.0
                status['image_age'] = time.time() - stamp if stamp else None
                self.json_reply(200, status)
            else:
                self.json_reply(404, {'error': '没有此页面'})
        except (RuntimeError, OSError) as exc:
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
                with self.server.action_lock:
                    result = self.server.vision.set_models(**body)
            elif path == '/api/scan':
                if not isinstance(body, dict) or body:
                    raise ValueError('扫码不接受额外参数')
                with self.server.action_lock:
                    result = {'ok': True, 'qr_data': self.server.vision.scan_qrcode()}
            else:
                self.json_reply(404, {'error': '没有此操作'})
                return
            self.json_reply(200, result)
        except (ValueError, TypeError) as exc:
            self.json_reply(400, {'error': str(exc)})
        except (RuntimeError, TimeoutError, OSError) as exc:
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
    parser = argparse.ArgumentParser(description='局域网摄像头调试网页；请勿同时运行比赛主线')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    opts = parser.parse_args(args)
    if not 1 <= opts.port <= 65535:
        parser.error('--port 必须为 1..65535')
    stop_event = threading.Event()
    vision = Vision(stop_event=stop_event)
    server = None

    def interrupt(signum, frame):
        stop_event.set()
        raise KeyboardInterrupt

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        server = ThreadingHTTPServer((opts.host, opts.port), CameraHandler)
        server.vision = vision
        server.action_lock = threading.Lock()
        vision.start_camera()
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
        stop_event.set()
        actions = []
        if server is not None:
            actions.append(('调试网页', server.server_close))
        actions.append(('视觉设备', vision.close))
        cleanup(*actions, raise_errors=False)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
