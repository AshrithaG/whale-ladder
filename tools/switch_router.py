"""Local non-streaming solver router with drainable, bounded backends.

No retries: an ambiguous upstream failure must not silently duplicate a rollout.
Backend 0 can serve requests while backend 1 wakes. Request counters are HTTP
requests, not completed puzzles; puzzle counts come from WHALE's val.json.
"""
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class BackendPool:
    def __init__(self, urls, capacity=128):
        self.urls = urls
        self.capacity = capacity
        self.condition = threading.Condition()
        self.enabled = {0}
        self.running = [0] * len(urls)
        self.completed = [0] * len(urls)
        self.failed = [0] * len(urls)
        self.closed = False

    def enable(self, indices):
        with self.condition:
            self.enabled = set(indices)
            self.condition.notify_all()

    def acquire(self):
        with self.condition:
            while not self.closed:
                available = [i for i in self.enabled if self.running[i] < self.capacity]
                if available:
                    i = min(available, key=lambda j: (self.running[j], self.completed[j], j))
                    self.running[i] += 1
                    return i
                self.condition.wait(1)
            raise RuntimeError('Router closed')

    def release(self, i, success):
        with self.condition:
            self.running[i] -= 1
            (self.completed if success else self.failed)[i] += 1
            self.condition.notify_all()

    def drain(self, index, timeout=900):
        with self.condition:
            self.enabled.discard(index)
            if not self.condition.wait_for(lambda: self.running[index] == 0, timeout):
                raise TimeoutError(f'Backend {index} failed to drain')

    def snapshot(self):
        with self.condition:
            return dict(enabled=sorted(self.enabled), running=self.running[:],
                        completed=self.completed[:], failed=self.failed[:])


def start_router(port, pool):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.0'

        def log_message(self, *_):
            pass

        def respond(self, status, body, content_type='application/json'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/router/stats':
                self.respond(200, json.dumps(pool.snapshot()).encode())
            elif self.path == '/v1/models':
                try:
                    with urllib.request.urlopen(pool.urls[0] + self.path, timeout=15) as r:
                        self.respond(r.status, r.read())
                except Exception as exc:
                    self.respond(502, json.dumps({'error': str(exc)}).encode())
            else:
                self.respond(404, b'{}')

        def do_POST(self):
            if self.path != '/v1/chat/completions':
                self.respond(404, b'{}')
                return
            index = None
            success = False
            try:
                body = self.rfile.read(int(self.headers['Content-Length']))
                if json.loads(body).get('stream'):
                    self.respond(400, b'{"error":"Solver router requires stream=false"}')
                    return
                index = pool.acquire()
                request = urllib.request.Request(pool.urls[index] + self.path, data=body,
                    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer EMPTY'})
                with urllib.request.urlopen(request, timeout=900) as result:
                    response = result.read()
                    success = result.status == 200
                    self.respond(result.status, response)
            except urllib.error.HTTPError as exc:
                try:
                    self.respond(exc.code, exc.read())
                finally:
                    exc.close()
            except Exception as exc:
                self.respond(502, json.dumps({'error': str(exc)}).encode())
            finally:
                if index is not None:
                    pool.release(index, success)

    class Server(ThreadingHTTPServer):
        request_queue_size = 512
        daemon_threads = True
    server = Server(('127.0.0.1', port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
