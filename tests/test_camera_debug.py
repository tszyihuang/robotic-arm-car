"""通过实际 HTTP 请求验证网页调试接口，不接硬件。"""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock
from urllib.request import Request, build_opener, ProxyHandler
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer

from debug.camera_web import CameraHandler

# 本地回归直接连接测试端口，不走机器上配置的 HTTP 代理。
urlopen = build_opener(ProxyHandler({})).open


class CameraDebugTests(unittest.TestCase):
    def setUp(self):
        self.vision = Mock(camera=None)
        self.vision.stop_event = threading.Event()
        self.vision.status.return_value = {'state': 'ready'}
        self.vision.snapshot.return_value = (b'\xff\xd8camera-jpeg\xff\xd9', 0.0)
        self.vision.scan_qrcode.return_value = '211'
        self.vision.set_models.return_value = {'ok': True}
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), CameraHandler)
        self.server.daemon_threads = True
        self.server.vision = self.vision
        self.server.action_lock = threading.Lock()
        self.directory = tempfile.TemporaryDirectory()
        self.server.screenshot_dir = Path(self.directory.name) / 'screenshots'
        self.worker = threading.Thread(target=self.server.serve_forever)
        self.worker.start()
        self.url = 'http://127.0.0.1:' + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join()
        self.directory.cleanup()

    def post(self, path, body):
        return urlopen(Request(self.url + path, json.dumps(body).encode(),
                               headers={'Content-Type': 'application/json'}), timeout=3)

    def test_page_and_frame(self):
        with urlopen(self.url, timeout=3) as response:
            page = response.read().decode()
        self.assertIn('>截图</button>', page)
        self.assertIn('物体 YOLO', page)
        with urlopen(self.url + '/frame.jpg', timeout=3) as response:
            self.assertEqual(response.headers['Content-Type'], 'image/jpeg')
            self.assertEqual(response.read(), self.vision.snapshot.return_value[0])

    def test_screenshot_is_saved_locally_without_overwriting_previous_image(self):
        saved = []
        for _ in range(2):
            with self.post('/api/screenshot', {}) as response:
                result = json.load(response)
            self.assertTrue(result['ok'])
            path = Path(result['path'])
            self.assertEqual(path.parent, self.server.screenshot_dir)
            self.assertEqual(path.read_bytes(), self.vision.snapshot.return_value[0])
            saved.append(path)
        self.assertNotEqual(*saved)
        self.assertEqual(len(list(self.server.screenshot_dir.glob('*.jpg'))), 2)

    def test_stream_sends_successive_frames_and_waits_for_new_capture_stamps(self):
        self.vision.snapshot.side_effect = [(b'first-jpeg', 100.0), (b'second-jpeg', 101.0),
                                           RuntimeError('结束测试流')]
        with urlopen(self.url + '/stream.mjpg', timeout=3) as response:
            self.assertEqual(response.headers['Content-Type'],
                             'multipart/x-mixed-replace; boundary=frame')
            for image in (b'first-jpeg', b'second-jpeg'):
                self.assertEqual(response.readline(), b'--frame\r\n')
                self.assertEqual(response.readline(), b'Content-Type: image/jpeg\r\n')
                self.assertEqual(response.readline(), f'Content-Length: {len(image)}\r\n'.encode())
                self.assertEqual(response.readline(), b'\r\n')
                self.assertEqual(response.read(len(image) + 2), image + b'\r\n')
        self.assertEqual([call.kwargs['after_stamp'] for call in self.vision.snapshot.call_args_list],
                         [0.0, 100.0, 101.0])

    def test_screenshot_rejects_client_path_and_camera_failure_creates_no_file(self):
        with self.assertRaises(HTTPError) as failed:
            self.post('/api/screenshot', {'path': '/tmp/unexpected.jpg'})
        self.assertEqual(failed.exception.code, 400)
        failed.exception.close()
        self.vision.snapshot.assert_not_called()
        self.vision.snapshot.side_effect = RuntimeError('画面已过期')
        with self.assertRaises(HTTPError) as failed:
            self.post('/api/screenshot', {})
        self.assertEqual(failed.exception.code, 409)
        failed.exception.close()
        self.assertFalse(self.server.screenshot_dir.exists())

    def test_model_switches_and_scan_route_to_vision(self):
        with self.post('/api/models', {'boundary': True, 'objects': False}) as response:
            self.assertTrue(json.load(response)['ok'])
        self.vision.set_models.assert_called_once_with(boundary=True, objects=False)
        with self.post('/api/scan', {}) as response:
            self.assertEqual(json.load(response)['qr_data'], '211')
        self.vision.scan_qrcode.assert_called_once_with()

    def test_stale_frame_and_malformed_requests_return_errors(self):
        self.vision.snapshot.side_effect = RuntimeError('画面已过期')
        with self.assertRaises(HTTPError) as failed:
            urlopen(self.url + '/frame.jpg', timeout=3)
        self.assertEqual(failed.exception.code, 503)
        failed.exception.close()
        with self.assertRaises(HTTPError) as failed:
            self.post('/api/models', {'boundary': 'true', 'objects': False})
        self.assertEqual(failed.exception.code, 400)
        failed.exception.close()
        self.vision.set_models.assert_not_called()
        self.vision.scan_qrcode.assert_not_called()
