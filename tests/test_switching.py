import json
from pathlib import Path
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from switch_router import BackendPool, start_router
from two_gpu_search import Experiment


class PoolTests(unittest.TestCase):
    def test_capacity_drain_and_enable(self):
        pool = BackendPool(['a','b'], capacity=1)
        self.assertEqual(pool.acquire(), 0)
        with ThreadPoolExecutor(max_workers=2) as executor:
            waiting = executor.submit(pool.acquire)
            time.sleep(.03)
            self.assertFalse(waiting.done())
            pool.enable([0,1])
            self.assertEqual(waiting.result(timeout=1),1)
            drain = executor.submit(pool.drain,1,1)
            time.sleep(.03)
            self.assertFalse(drain.done())
            pool.release(1,True)
            drain.result(timeout=1)
        self.assertEqual(pool.snapshot()['enabled'],[0])
        pool.release(0,True)
        self.assertEqual(pool.snapshot()['completed'],[1,1])

    def test_routing_preserves_requests_and_propagates_errors_without_retry(self):
        received = []
        lock = threading.Lock()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self):
                value=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                with lock: received.append((self.server.server_port,value))
                time.sleep(.01)
                body=json.dumps(value).encode()
                self.send_response(503 if value.get('fail') else 200)
                self.send_header('Content-Length',str(len(body)))
                self.end_headers(); self.wfile.write(body)
        backends = [ThreadingHTTPServer(('127.0.0.1',0),Handler) for _ in range(2)]
        for server in backends:
            threading.Thread(target=server.serve_forever,daemon=True).start()
        pool=BackendPool([f'http://127.0.0.1:{s.server_port}' for s in backends],capacity=4)
        pool.enable([0,1]);router=start_router(0,pool)
        def post(value):
            with urllib.request.urlopen(urllib.request.Request(
                f'http://127.0.0.1:{router.server_port}/v1/chat/completions',
                data=json.dumps(value).encode()),timeout=3) as r:
                return json.loads(r.read())
        try:
            values=[dict(id=i,stream=False,seed=42) for i in range(24)]
            with ThreadPoolExecutor(max_workers=16) as executor:
                self.assertEqual(list(executor.map(post,values)),values)
            self.assertEqual(len(received),24)
            self.assertEqual(len({port for port,_ in received}),2)
            with self.assertRaises(urllib.error.HTTPError) as ctx: post(dict(fail=True))
            self.assertEqual(ctx.exception.code,503)
            ctx.exception.close()
            self.assertEqual(len(received),25)
            with pool.condition:
                self.assertTrue(pool.condition.wait_for(lambda:sum(pool.failed)==1,timeout=1))
            self.assertEqual(sum(pool.snapshot()['failed']),1)
        finally:
            router.shutdown();router.server_close()
            for server in backends:server.shutdown();server.server_close()


class TransitionTests(unittest.TestCase):
    def test_close_releases_router_port_for_next_experiment(self):
        obj=object.__new__(Experiment)
        obj.pool=BackendPool(['http://127.0.0.1:1'])
        obj.router=start_router(0,obj.pool)
        port=obj.router.server_port
        obj.stop_monitor=threading.Event()
        obj.executor=ThreadPoolExecutor(max_workers=1)
        obj.processes={}
        obj.close()
        replacement=start_router(port,BackendPool(['http://127.0.0.1:1']))
        try:
            self.assertEqual(replacement.server_port,port)
        finally:
            replacement.shutdown();replacement.server_close()

    def test_sleep_transition_keeps_first_solver_alive(self):
        obj=object.__new__(Experiment)
        calls=[]
        obj.pool=BackendPool(['a','b'])
        obj.sleep=lambda n:calls.append(('sleep',n))
        obj.wake=lambda n:calls.append(('wake',n))
        obj.check_model=lambda n:calls.append(('check',n))
        obj.event=lambda *a,**kw:None
        obj.to_scoring('sleep')
        self.assertEqual(obj.pool.snapshot()['enabled'],[0,1])
        obj.to_proposing('sleep')
        self.assertEqual(obj.pool.snapshot()['enabled'],[0])
        self.assertEqual(calls,[('sleep','proposer'),('wake','solver1'),('check','solver1'),
                                ('sleep','solver1'),('wake','proposer'),('check','proposer')])


if __name__=='__main__':unittest.main()
