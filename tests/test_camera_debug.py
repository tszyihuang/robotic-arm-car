"""通过实际 HTTP 请求验证网页调试接口，不接硬件。"""
import json
import threading
import unittest
from unittest.mock import Mock
from urllib.request import Request, build_opener, ProxyHandler
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer

try:
    from car_nodes.debug.camera_web import CameraHandler
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

# 本地回归直接连接测试端口，不走机器上配置的 HTTP 代理。
urlopen = build_opener(ProxyHandler({})).open


@unittest.skipUnless(AVAILABLE, '需要已加载 ROS 环境')
class CameraDebugTests(unittest.TestCase):
    def setUp(self):
        self.bridge = Mock(lock=threading.Lock(), status={'state': 'ready'}, image_stamp=0.0)
        self.bridge.snapshot.return_value = b'\xff\xd8camera-jpeg\xff\xd9'
        self.bridge.command.return_value = {'ok': True, 'qr_data': '211'}
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), CameraHandler)
        self.server.daemon_threads = True
        self.server.bridge = self.bridge
        self.worker = threading.Thread(target=self.server.serve_forever)
        self.worker.start()
        self.url = 'http://127.0.0.1:' + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join()

    def post(self, path, body):
        return urlopen(Request(self.url + path, json.dumps(body).encode(),
                               headers={'Content-Type': 'application/json'}), timeout=3)

    def test_page_frame_and_download_snapshot(self):
        with urlopen(self.url, timeout=3) as response:
            page = response.read().decode()
        self.assertIn('截图下载', page)
        self.assertIn('物体 YOLO', page)
        for path in ('/frame.jpg', '/snapshot.jpg'):
            with urlopen(self.url + path, timeout=3) as response:
                self.assertEqual(response.headers['Content-Type'], 'image/jpeg')
                self.assertEqual(response.read(), self.bridge.snapshot.return_value)
                if path == '/snapshot.jpg':
                    self.assertIn('attachment', response.headers['Content-Disposition'])

    def test_model_switches_and_scan_route_to_vision(self):
        with self.post('/api/models', {'boundary': True, 'objects': False}) as response:
            self.assertTrue(json.load(response)['ok'])
        self.bridge.command.assert_called_with('set-models boundary=true objects=false')
        with self.post('/api/scan', {}) as response:
            self.assertEqual(json.load(response)['qr_data'], '211')
        self.bridge.command.assert_called_with('scan-qrcode')

    def test_stale_frame_and_malformed_requests_return_errors(self):
        self.bridge.snapshot.side_effect = RuntimeError('画面已过期')
        with self.assertRaises(HTTPError) as failed:
            urlopen(self.url + '/snapshot.jpg', timeout=3)
        self.assertEqual(failed.exception.code, 503)
        failed.exception.close()
        with self.assertRaises(HTTPError) as failed:
            self.post('/api/models', {'boundary': 'true', 'objects': False})
        self.assertEqual(failed.exception.code, 400)
        failed.exception.close()
        self.bridge.command.assert_not_called()
