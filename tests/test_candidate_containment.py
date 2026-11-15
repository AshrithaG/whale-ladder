"""Timeout, result-integrity, selection, and interrupted-owner regressions."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures'
sys.path[:0] = [str(ROOT / 'tools'), str(FIXTURES)]
from candidate_containment import CandidateRejected, run_worker, timeout_setting, validate_candidate
from two_gpu_search import Experiment
from switch_router import BackendPool, start_router
import candidate_test_worker

HEALTHY = 'def propose_action(board): return ""\ndef is_legal_action(board, action): return True\n'
CONFIG = dict(task_profiles=dict(p=dict(dataset_path='unchanged', limit=2)),
              models=[dict(model='mock')], eval=dict(assistant_token_budget=8129))


class ContainmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.logs = self.root / 'logs'
        self.obj = object.__new__(Experiment)
        self.obj.event = Mock()

    def harness(self, name, source=HEALTHY):
        path = self.root / 'harnesses' / name / 'harness.py'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        return path

    def score(self, name='h1', mode='healthy', **extra):
        config = dict(CONFIG, mode=mode, **extra)
        return self.obj.contained_score(candidate_test_worker.sweep, config, name,
                                        self.root / 'harnesses' / name / 'harness.py', self.logs)

    def test_timeout_settings_reject_nonpositive_nonfinite_values(self):
        for value in ('0', '-1', 'nan', 'inf', '-inf', 'bad'):
            with self.subTest(value=value), patch.dict(os.environ, TIMEOUT=value):
                with self.assertRaises(ValueError): timeout_setting('TIMEOUT', 30)
        self.assertEqual(timeout_setting('ABSENT_TIMEOUT', 30), 30)

    def test_healthy_complete_scores_are_published_unchanged(self):
        self.harness('h1')
        outcome = self.score()
        self.assertEqual(outcome['status'], 'evaluated')
        value = json.loads((self.logs / 'p/h1/mock/val.json').read_text())
        self.assertEqual(value['success_rate'], .5)
        self.assertEqual(value['mean_turn_count'], 3.25)
        self.assertEqual(value['num_examples'], 2)

    def test_hung_module_validation_is_bounded(self):
        path = self.harness('h1', HEALTHY + '\nwhile True: pass\n')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with patch.dict(os.environ, LADDER_CANDIDATE_VALIDATION_TIMEOUT='1.5'):
            started = time.monotonic()
            with self.assertRaises(CandidateRejected) as ctx:
                validate_candidate(self.root, 'h1', module='candidate_test_worker')
            self.assertLess(time.monotonic() - started, 8)
            self.assertEqual(ctx.exception.status, 'rejected_timeout')
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_exact_h1_parser_times_out_while_parent_stays_responsive(self):
        path = self.harness('h1', (FIXTURES / 'c4-h1.py').read_text())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        ticks = []
        stopped = threading.Event()
        def heartbeat():
            while not stopped.wait(.02): ticks.append(time.monotonic())
        thread = threading.Thread(target=heartbeat)
        thread.start()
        try:
            with patch.dict(os.environ, LADDER_CANDIDATE_SCORE_TIMEOUT='1.5'):
                started = time.monotonic()
                outcome = self.score()
            self.assertLess(time.monotonic() - started, 8)
            self.assertEqual(outcome['status'], 'rejected_timeout')
            self.assertGreater(len(ticks), 20)
            self.assertTrue((Path(outcome['evidence']) / 'worker.log').is_file())
            self.assertFalse(list(self.logs.rglob('val.json')))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        finally:
            stopped.set(); thread.join()

    def test_crash_malformed_failed_partial_and_invalid_scores_are_rejected(self):
        self.harness('h1')
        for mode in ('crash', 'no_result', 'malformed', 'failed', 'partial', 'invalid_metric', 'invalid_document'):
            with self.subTest(mode=mode):
                stale = self.logs / 'p/h1/mock/val.json'
                stale.parent.mkdir(parents=True, exist_ok=True)
                stale.write_text('{"success_rate":1}')
                outcome = self.score(mode=mode)
                self.assertEqual(outcome['status'], 'rejected_evaluation')
                self.assertFalse(list(self.logs.rglob('val.json')))
                self.assertTrue(list(Path(outcome['evidence']).rglob('val.json')))

    def test_worker_timeout_kills_descendants(self):
        self.harness('h1')
        pidfile = self.root / 'descendant.pid'
        with patch.dict(os.environ, LADDER_CANDIDATE_SCORE_TIMEOUT='1.5'):
            outcome = self.score(mode='descendants', pid_file=str(pidfile))
        self.assertEqual(outcome['status'], 'rejected_timeout')
        self.assert_dead(int(pidfile.read_text()))

    def assert_dead(self, pid):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True)
            if result.returncode or result.stdout.strip().startswith('Z'):
                return
            time.sleep(.05)
        self.fail(f'Owned descendant {pid} remains alive')

    def test_interrupted_parent_reaps_private_worker_group(self):
        path = self.harness('h1')
        pidfile = self.root / 'descendant.pid'
        request = dict(action='score', module='candidate_test_worker', function='sweep',
                       config=dict(CONFIG, mode='descendants', pid_file=str(pidfile)), name='h1',
                       path=str(path), logs=str(self.root / 'staged'))
        script = ('import sys,json; sys.path[:0]=' + repr([str(ROOT / 'tools'), str(FIXTURES)]) + ';'
                  'from candidate_containment import run_worker;'
                  'run_worker(json.loads(sys.argv[1]),sys.argv[2],60)')
        parent = subprocess.Popen([sys.executable, '-c', script, json.dumps(request), str(self.root / 'worker')])
        try:
            deadline = time.monotonic() + 8
            while not pidfile.exists() and parent.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue(pidfile.exists())
            parent.terminate()
            self.assertEqual(parent.wait(timeout=8), 143)
            outcome = json.loads((self.root / 'worker/outcome.json').read_text())
            self.assertEqual(outcome['status'], 'interrupted')
            self.assertEqual(outcome['exit_code'], 143)
            self.assertIn('resource_policy', outcome)
            self.assertIn('usage', outcome)
            self.assert_dead(int(pidfile.read_text()))
        finally:
            if parent.poll() is None: parent.kill()
            parent.wait(timeout=5)

    def test_mixed_round_retains_servers_and_scores_survivor(self):
        self.harness('h1', (FIXTURES / 'c4-h1.py').read_text())
        self.harness('h2')
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
        backends = [ThreadingHTTPServer(('127.0.0.1', 0), Handler) for _ in range(2)]
        for server in backends: threading.Thread(target=server.serve_forever, daemon=True).start()
        obj = self.obj
        obj.pool = BackendPool([f'http://127.0.0.1:{s.server_port}' for s in backends])
        router = start_router(0, obj.pool)
        obj.executor = ThreadPoolExecutor(max_workers=1)
        obj.phase = 'search'; obj.replicas = 2; obj.processes = {'server': SimpleNamespace(pid=123, poll=lambda:None)}
        obj.status = Mock(); obj.to_scoring = lambda mode: obj.pool.enable([0, 1])
        obj.to_proposing = Mock(); obj.check_model = Mock()
        try:
            with patch.dict(os.environ, LADDER_CANDIDATE_SCORE_TIMEOUT='1.5'):
                rows = obj.score(candidate_test_worker.sweep, dict(CONFIG, router=f'http://127.0.0.1:{router.server_port}'),
                                 [(n, self.root / 'harnesses' / n / 'harness.py') for n in ('h1', 'h2')], self.logs)
            self.assertEqual([r['status'] for r in rows], ['rejected_timeout', 'evaluated'])
            self.assertFalse(list(self.logs.glob('*/h1/*/val.json')))
            self.assertTrue(list(self.logs.glob('*/h2/*/val.json')))
            self.assertEqual(obj.processes['server'].pid, 123)
            self.assertTrue(all(obj.pool.snapshot()['completed']))
            obj.to_proposing.assert_called_once()
        finally:
            obj.executor.shutdown(); router.shutdown(); router.server_close()
            for server in backends: server.shutdown(); server.server_close()

    def score_boundary(self, handler, *, capacity=128):
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        obj = self.obj
        obj.pool = BackendPool([f'http://127.0.0.1:{server.server_port}'], capacity=capacity)
        router = start_router(0, obj.pool)
        obj.phase = 'search'; obj.replicas = 1; obj.processes = {}
        obj.status = Mock(); obj.to_scoring = Mock(); obj.to_proposing = Mock(); obj.check_model = Mock()
        self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
        self.addCleanup(router.server_close); self.addCleanup(router.shutdown)
        return f'http://127.0.0.1:{router.server_port}'

    def test_timed_out_requests_drain_before_next_candidate_deadline(self):
        completed = threading.Event()
        calls = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                calls.append(time.monotonic())
                if len(calls) == 1:
                    time.sleep(2.5)
                    completed.set()
                self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
        router = self.score_boundary(Handler, capacity=1)
        self.harness('h1'); self.harness('h2')
        begins = []
        def event(name, **fields):
            if name == 'candidate_score_begin': begins.append((fields['harness'], completed.is_set(), len(calls)))
        self.obj.event = event
        with patch.dict(os.environ, LADDER_CANDIDATE_SCORE_TIMEOUT='1.5'):
            rows = self.obj.score(candidate_test_worker.sweep,
                dict(CONFIG, router=router, candidate_modes={'h1': 'outstanding_request'}, outstanding_requests=2),
                [(n, self.root / 'harnesses' / n / 'harness.py') for n in ('h1', 'h2')], self.logs)
        self.assertEqual([r['status'] for r in rows], ['rejected_timeout', 'evaluated'])
        self.assertEqual(begins, [('h1', False, 0), ('h2', True, 2)])
        self.assertEqual(self.obj.pool.snapshot()['failed'], [0])

    def test_upstream_http_failure_aborts_score_boundary(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(503); self.end_headers(); self.wfile.write(b'{}')
        router = self.score_boundary(Handler)
        with self.assertRaisesRegex(RuntimeError, 'Solver HTTP request failure'):
            self.obj.score(candidate_test_worker.sweep, dict(CONFIG, router=router),
                           [('h1', self.harness('h1'))], self.logs)
        self.assertEqual(self.obj.pool.snapshot()['failed'], [1])
        self.obj.to_proposing.assert_not_called()

    def test_extreme_or_nested_scores_and_nested_worker_result_are_rejected(self):
        self.harness('h1')
        for mode in ('huge_metric', 'deep_score', 'deep_result'):
            with self.subTest(mode=mode):
                outcome = self.score(mode=mode)
                self.assertEqual(outcome['status'], 'rejected_evaluation')
                self.assertFalse(list(self.logs.rglob('val.json')))

    def test_nonfinite_metrics_are_rejected(self):
        self.harness('h1')
        for metric in ('success_rate', 'mean_turn_count', 'avg_reward'):
            for value in ('nan', 'inf', '-inf'):
                with self.subTest(metric=metric, value=value):
                    outcome = self.score(mode='metric_override', metric=metric, metric_value=value)
                    self.assertEqual(outcome['status'], 'rejected_evaluation')
                    self.assertFalse(list(self.logs.rglob('val.json')))

    def test_original_token_budget_is_enforced_even_if_config_and_score_agree(self):
        self.harness('h1')
        config = dict(CONFIG, mode='budget_override', eval=dict(assistant_token_budget=123))
        outcome = self.obj.contained_score(candidate_test_worker.sweep, config, 'h1',
                                          self.root / 'harnesses/h1/harness.py', self.logs)
        self.assertEqual(outcome['status'], 'rejected_evaluation')
        self.assertIn('8129', outcome['reason'])
        self.assertFalse(list(self.logs.rglob('val.json')))

    def test_validation_preserves_invalid_and_prompt_only_errors(self):
        for source, prompt_only, status, reason in (
                ('def broken(:', False, 'rejected_invalid', 'SyntaxError'),
                ('import os\n' + HEALTHY, False, 'rejected_invalid', 'Forbidden import: os'),
                (HEALTHY, True, 'rejected_prompt_only', 'prompt-only constraint')):
            with self.subTest(status=status, reason=reason):
                self.harness('h1', source)
                with self.assertRaises(CandidateRejected) as ctx:
                    validate_candidate(self.root, 'h1', module='candidate_test_worker', prompt_only=prompt_only)
                self.assertEqual(ctx.exception.status, status)
                self.assertIn(reason, str(ctx.exception))
                self.assertTrue(Path(ctx.exception.outcome['evidence']).is_dir())

    def test_real_search_installs_contained_validation_and_restores_hooks(self):
        mh = candidate_test_worker
        self.harness('h1', HEALTHY + '\nwhile True: pass\n')
        self.harness('h2')
        obj = self.obj
        obj.out = self.root / 'out'; obj.out.mkdir()
        obj.processes = {}; obj.proposals = 2; obj.limit = 2; obj.status = Mock()
        obj.score = lambda sweep, config, harnesses, logs, **kw: [
            obj.contained_score(sweep, config, n, p, logs) for n, p in harnesses]
        continued = []
        def evolve(args):
            rejects = []
            try:
                mh.validate_candidate(self.root, 'h1')
            except CandidateRejected as exc:
                rejects.append(dict(iteration=1, harness='h1', status=exc.status, reason=str(exc)))
            mh.validate_candidate(self.root, 'h2')
            continued.append('h2')
            results = mh.run_sweep(CONFIG, [('h2', self.root / 'harnesses/h2/harness.py')], self.logs)
            self.assertEqual(results[0]['status'], 'evaluated')
            self.logs.mkdir(exist_ok=True)
            (self.logs / 'accepted_harness.txt').write_text('h2')
            (self.logs / 'evolution_summary.jsonl').write_text('\n'.join(map(json.dumps,
                rejects + [dict(iteration=1, harness='h2', status='evaluated')])))
        original_popen = subprocess.Popen
        def spawn(command, **kwargs):
            if command[0] == 'litellm': return SimpleNamespace(poll=lambda: None)
            return original_popen(command, **kwargs)
        attrs = dict(run_sweep=candidate_test_worker.sweep, render_task_prompt=lambda *a: 'prompt',
            PROPOSER_SYSTEM_PROMPT='system', PROMPT_ONLY=True, run_evolve=evolve,
            parse_args=lambda: SimpleNamespace(), get_run_dir=lambda _: self.root)
        argv = sys.argv[:]
        with patch.multiple(mh, create=True, **attrs), \
             patch.dict(os.environ, LADDER_HOME=str(ROOT), LITELLM_PORT='1', SWITCH_CONFIG='fixture',
                        LADDER_CYCLE_SEARCH_NAME='test-contained-validation', LADDER_CANDIDATE_VALIDATION_TIMEOUT='1.5'), \
             patch('two_gpu_search.subprocess.check_output', return_value=b'{}'), \
             patch('two_gpu_search.subprocess.Popen', side_effect=spawn), \
             patch('two_gpu_search.urllib.request.urlopen'):
            original_validator = mh.validate_candidate
            outcome = obj.real_search(mh, iterations=1, baseline_two_gpu=True)
            self.assertIs(mh.validate_candidate, original_validator)
            self.assertIs(mh.run_sweep, attrs['run_sweep'])
            self.assertIs(mh.render_task_prompt, attrs['render_task_prompt'])
            self.assertEqual(mh.PROPOSER_SYSTEM_PROMPT, 'system')
            self.assertTrue(mh.PROMPT_ONLY)
        self.assertEqual(sys.argv, argv)
        self.assertEqual(continued, ['h2'])
        self.assertEqual(outcome['candidates_evaluated'], 1)
        self.assertEqual(outcome['candidates_rejected'], 1)

    def test_matched_benchmark_rejections_cannot_create_pass_receipts(self):
        mh = SimpleNamespace(run_sweep=candidate_test_worker.sweep)
        obj = self.obj
        obj.reference = self.root; obj.out = self.root / 'out'; obj.out.mkdir()
        for name in ('h1', 'h2', 'h3'): self.harness(name)
        obj.checkpoint = self.root; obj.config = CONFIG; obj.processes = {}
        obj.start = lambda name: obj.processes.update({name: SimpleNamespace(pid=123)})
        obj.status = Mock(); obj.check_model = Mock(); obj.monitor = Mock(); obj.sleep = Mock(); obj.wake = Mock()
        obj.real_search = Mock()
        healthy = [dict(harness=n, status='evaluated') for n in ('h1', 'h2', 'h3')]
        rejected = [dict(harness=n, status='rejected_timeout' if n == 'h1' else 'evaluated')
                    for n in ('h1', 'h2', 'h3')]
        with patch.dict(sys.modules, {'meta_harness': SimpleNamespace(meta_harness_chess_puzzle=mh)}):
            for rejected_round in range(3):
                with self.subTest(round=rejected_round):
                    obj.score = Mock(side_effect=[healthy] * rejected_round + [rejected])
                    with self.assertRaisesRegex(RuntimeError, 'every candidate evaluated'):
                        obj.run()
                    self.assertFalse((obj.out / 'matched-passed.json').exists())
                    self.assertFalse((obj.out / 'passed.json').exists())
        obj.real_search.assert_not_called()

    def test_real_qwen_model_name_normalizes_directory_and_outcome(self):
        path = self.harness('h1')
        config = dict(CONFIG, models=[dict(model='Qwen/Qwen3.5-4B')])
        outcome = self.obj.contained_score(candidate_test_worker.sweep, config, 'h1', path, self.logs)
        self.assertEqual(outcome['status'], 'evaluated')
        self.assertEqual(outcome['rows'], [['p/h1/qwen3.5-4b_seed42', True]])
        value = json.loads((self.logs / 'p/h1/qwen3.5-4b/val.json').read_text())
        self.assertEqual(value['num_examples'], 2)
        self.assertEqual(value['metadata']['assistant_token_budget'], 8129)
        self.assertEqual(value['success_rate'], .5)

    def test_baseline_failure_fails_phase_clearly(self):
        obj = self.obj
        obj.phase = 'search'; obj.status = Mock(); obj.pool = BackendPool(['unused'])
        obj.to_scoring = Mock(); obj.to_proposing = Mock()
        for mode, source, status in (
                ('healthy', (FIXTURES / 'c4-h1.py').read_text(), 'rejected_timeout'),
                ('partial', HEALTHY, 'rejected_evaluation')):
            with self.subTest(status=status), patch.dict(os.environ, LADDER_CANDIDATE_SCORE_TIMEOUT='1.5'):
                path = self.harness('h0', source)
                with self.assertRaisesRegex(RuntimeError, 'Baseline h0 failed: ' + status):
                    obj.score(candidate_test_worker.sweep, dict(CONFIG, mode=mode), [('h0', path)], self.logs)
                self.assertFalse(list(self.logs.rglob('val.json')))
        obj.to_proposing.assert_not_called()


if __name__ == '__main__': unittest.main()
