"""Resource containment. Linux-only memory cases must also run in the real image."""
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures'
sys.path[:0] = [str(ROOT / 'tools'), str(FIXTURES)]
from candidate_containment import (CandidateRejected, JSON_LIMIT_BYTES, ResourceExceeded,
    THREAD_ENV, WorkerSetupError, read_json_bounded, resource_policy, validate_candidate)
from two_gpu_search import Experiment
from switch_router import BackendPool, start_router
import candidate_test_worker

HEALTHY = 'def propose_action(board): return ""\ndef is_legal_action(board, action): return True\n'
CONFIG = dict(task_profiles=dict(p=dict(dataset_path='unchanged', limit=2)),
              models=[dict(model='mock')], eval=dict(assistant_token_budget=8129))
LINUX = sys.platform.startswith('linux')


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.logs = self.root / 'logs'
        self.obj = object.__new__(Experiment)
        self.obj.event = Mock()
        self.harness('h1')

    def harness(self, name, source=HEALTHY):
        p = self.root / 'harnesses' / name / 'harness.py'
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text(source)
        return p

    def score(self, mode='healthy', **extra):
        return self.obj.contained_score(candidate_test_worker.sweep, dict(CONFIG, mode=mode, **extra),
                                        'h1', self.root / 'harnesses/h1/harness.py', self.logs)

    def assert_resource(self, outcome, kind):
        self.assertEqual(outcome['status'], 'rejected_resource_limit', outcome)
        self.assertEqual(outcome['resource']['kind'], kind)
        self.assertIn('usage', outcome)
        self.assertIn('resource_policy', outcome)
        self.assertEqual(outcome['applied_policy']['core_rlimit'], [0, 0])
        self.assertEqual(outcome['applied_policy']['cpu_rlimit'],
                         [outcome['resource_policy']['cpu_seconds'], outcome['resource_policy']['cpu_hard_seconds']])
        self.assertFalse(list(self.logs.rglob('val.json')))
        self.assertTrue((Path(outcome['evidence']) / 'worker.log').is_file())

    def test_policy_defaults_and_positive_integer_validation(self):
        clean = {k: v for k, v in os.environ.items() if not k.startswith('LADDER_CANDIDATE_')}
        with patch.dict(os.environ, clean, clear=True):
            validation = resource_policy('validate'); scoring = resource_policy('score')
            self.assertEqual(validation['cpu_seconds'], 10)
            self.assertEqual(scoring['cpu_seconds'], 120)
            self.assertEqual(scoring['address_space_bytes'], 32 * 1024**3)
            self.assertEqual(scoring['rss_bytes'], 8 * 1024**3)
            self.assertEqual(scoring['output_bytes'], 16 * 1024**2)
            self.assertEqual(scoring['json_bytes'], 1024**2)
            self.assertEqual(scoring['enforcement']['address_space'], 'RLIMIT_AS' if LINUX else 'unavailable')
        settings = ('VALIDATION_CPU_SECONDS', 'SCORE_CPU_SECONDS', 'AS_BYTES', 'RSS_BYTES', 'OUTPUT_BYTES')
        for setting in settings:
            for value in ('0', '-1', 'nan', 'inf', '1.5', str(10**400)):
                with self.subTest(setting=setting, value=value), patch.dict(
                        os.environ, {'LADDER_CANDIDATE_' + setting: value}):
                    with self.assertRaises(ValueError):
                        resource_policy('validate' if setting.startswith('VALIDATION') else 'score')

    def test_limit_setup_failure_is_fatal_with_explicit_receipt(self):
        policy = resource_policy('score')
        policy.update(cpu_seconds=2, cpu_hard_seconds=1)
        with patch('candidate_containment.resource_policy', return_value=policy):
            with self.assertRaisesRegex(WorkerSetupError, 'Resource policy setup failed'):
                self.score()
        receipts = list(self.root.glob('logs-candidate-evidence/*/outcome.json'))
        self.assertEqual(len(receipts), 1)
        receipt = json.loads(receipts[0].read_text())
        self.assertEqual(receipt['status'], 'failed_resource_setup')
        self.assertEqual(receipt['resource_policy']['cpu_hard_seconds'], 1)
        self.assertFalse(list(self.logs.rglob('val.json')))

    def test_launch_and_selector_failures_are_fatal_setup_failures(self):
        import subprocess
        for target in ('candidate_containment.subprocess.Popen', 'candidate_containment.selectors.DefaultSelector'):
            with self.subTest(target=target), patch(target, side_effect=OSError('launch blocked')):
                with self.assertRaisesRegex(WorkerSetupError, 'launch/monitor setup failed'):
                    self.score()
        receipts = list(self.root.glob('logs-candidate-evidence/*/outcome.json'))
        self.assertEqual(len(receipts), 2)
        self.assertTrue(all(json.loads(p.read_text())['status'] == 'failed_resource_setup' for p in receipts))

    def test_selector_registration_failure_reaps_spawned_worker(self):
        import subprocess
        real_popen = subprocess.Popen
        children = []
        def spawn(*args, **kwargs):
            child = real_popen(*args, **kwargs); children.append(child); return child
        with patch('candidate_containment.subprocess.Popen', side_effect=spawn), \
             patch('candidate_containment.selectors.DefaultSelector') as selector:
            selector.return_value.__enter__.return_value.register.side_effect = OSError('selector registration blocked')
            with self.assertRaisesRegex(WorkerSetupError, 'selector registration blocked'):
                self.score()
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())
        self.assertFalse(list(self.logs.rglob('val.json')))

    def test_startup_crash_without_handshake_is_fatal(self):
        import subprocess
        real_popen = subprocess.Popen
        def crash(command, **kwargs):
            return real_popen([sys.executable, '-c', 'import os; os._exit(23)'], **kwargs)
        with patch('candidate_containment.subprocess.Popen', side_effect=crash):
            with self.assertRaisesRegex(WorkerSetupError, 'startup failed.*exit code 23'):
                self.score()
        receipt = json.loads(next(self.root.glob('logs-candidate-evidence/*/outcome.json')).read_text())
        self.assertEqual(receipt['status'], 'failed_resource_setup')

    @unittest.skipUnless(LINUX, 'Tiny Linux RLIMIT_AS setup failure must run in the real image')
    def test_tiny_address_space_is_fatal_setup_failure(self):
        with patch.dict(os.environ, LADDER_CANDIDATE_AS_BYTES='1'):
            with self.assertRaises(WorkerSetupError):
                self.score()
        receipt = json.loads(next(self.root.glob('logs-candidate-evidence/*/outcome.json')).read_text())
        self.assertEqual(receipt['status'], 'failed_resource_setup')
        self.assertFalse(list(self.logs.rglob('val.json')))

    def test_candidate_memory_error_after_handshake_remains_resource_rejection(self):
        outcome = self.score('memory_error')
        self.assert_resource(outcome, 'allocation')
        self.assertEqual(outcome['resource']['trigger'], 'MemoryError')
        marker = Path(outcome['evidence']) / 'execution-started.json'
        self.assertEqual(json.loads(marker.read_text()), {'started': True})

    def test_cpu_spin_is_confirmed_by_kernel_signal_before_wall_deadline(self):
        with patch.dict(os.environ, LADDER_CANDIDATE_SCORE_CPU_SECONDS='1',
                        LADDER_CANDIDATE_SCORE_TIMEOUT='12'):
            started = time.monotonic(); outcome = self.score('cpu_spin')
        self.assertLess(time.monotonic() - started, 10)
        self.assert_resource(outcome, 'cpu')
        self.assertEqual(outcome['resource']['signal'], 'SIGXCPU')
        self.assertEqual(outcome['resource_policy']['cpu_seconds'], 1)
        if LINUX: self.assertGreater(outcome['usage']['cpu_seconds'], .5)

    def test_validation_resource_rejection_preserves_policy_usage_and_source(self):
        source = HEALTHY + '\nwhile True: pass\n'
        path = self.harness('h1', source)
        with patch.dict(os.environ, LADDER_CANDIDATE_VALIDATION_CPU_SECONDS='1',
                        LADDER_CANDIDATE_VALIDATION_TIMEOUT='12'):
            with self.assertRaises(CandidateRejected) as ctx:
                validate_candidate(self.root, 'h1', module='candidate_test_worker')
        self.assert_resource(ctx.exception.outcome, 'cpu')
        self.assertEqual(path.read_text(), source)

    @unittest.skipUnless(LINUX, 'Linux RLIMIT_AS enforcement must run in the real image')
    def test_hard_address_space_limit_rejects_failed_allocation(self):
        with patch.dict(os.environ, LADDER_CANDIDATE_AS_BYTES=str(128 * 1024**2),
                        LADDER_CANDIDATE_RSS_BYTES=str(96 * 1024**2)):
            outcome = self.score('allocation_excess')
        self.assert_resource(outcome, 'allocation')
        self.assertEqual(outcome['resource']['trigger'], 'MemoryError')
        self.assertTrue(outcome['resource']['address_space_enforced'])

    @unittest.skipUnless(LINUX, 'Linux sampled /proc RSS enforcement must run in the real image')
    def test_sampled_rss_limit_kills_worker_with_measured_trigger(self):
        limit = 64 * 1024**2
        with patch.dict(os.environ, LADDER_CANDIDATE_AS_BYTES=str(1024**3),
                        LADDER_CANDIDATE_RSS_BYTES=str(limit)):
            outcome = self.score('rss_excess')
        self.assert_resource(outcome, 'rss')
        self.assertGreater(outcome['resource']['observed_bytes'], limit)
        self.assertEqual(outcome['resource']['enforcement'], 'sampled')

    def test_combined_stdout_stderr_is_bounded_and_rejected(self):
        limit = 32 * 1024
        with patch.dict(os.environ, LADDER_CANDIDATE_OUTPUT_BYTES=str(limit)):
            outcome = self.score('output_flood')
        self.assert_resource(outcome, 'output')
        self.assertGreater(outcome['usage']['output_bytes'], limit)
        self.assertEqual(outcome['usage']['captured_output_bytes'], limit)
        self.assertEqual((Path(outcome['evidence']) / 'worker.log').stat().st_size, limit)

    def test_oversized_result_and_score_artifacts_are_rejected_without_publication(self):
        for mode in ('oversized_result', 'oversized_score'):
            with self.subTest(mode=mode):
                outcome = self.score(mode)
                self.assert_resource(outcome, 'json')
                self.assertEqual(outcome['resource']['limit_bytes'], JSON_LIMIT_BYTES)
                self.assertGreater(outcome['resource']['observed_bytes'], JSON_LIMIT_BYTES)
                self.assertTrue(Path(outcome['resource']['artifact']).is_file())

    def test_malformed_applied_policy_stays_within_rejection_boundary(self):
        for mode, status in (('bad_applied', 'rejected_evaluation'), ('deep_applied', 'rejected_evaluation'),
                             ('oversized_applied', 'rejected_resource_limit')):
            with self.subTest(mode=mode):
                outcome = self.score(mode)
                self.assertEqual(outcome['status'], status)
                self.assertIsNone(outcome['applied_policy'])
                self.assertFalse(list(self.logs.rglob('val.json')))

    def test_nonregular_json_artifact_is_rejected_without_blocking(self):
        p = self.root / 'result.fifo'
        os.mkfifo(p)
        with self.assertRaisesRegex(ValueError, 'regular file'):
            read_json_bounded(p)

    def test_sparse_gigabyte_artifact_is_rejected_before_full_read(self):
        p = self.root / 'gigabyte.json'
        with p.open('wb') as f: f.truncate(1024**3)
        with patch('candidate_containment.json.loads') as decode:
            with self.assertRaises(ResourceExceeded) as ctx: read_json_bounded(p)
        decode.assert_not_called()
        self.assertEqual(ctx.exception.details['observed_bytes'], 1024**3)

    def test_unexplained_sigkill_is_not_labeled_cpu_failure(self):
        outcome = self.score('sigkill')
        self.assertEqual(outcome['status'], 'rejected_evaluation')
        self.assertNotIn('resource', outcome)
        self.assertIn(str(-signal.SIGKILL), outcome['reason'])

    def test_native_thread_limits_are_child_only_and_success_keeps_real_score(self):
        inherited = {key: os.environ.get(key) for key in THREAD_ENV}
        outcome = self.score('thread_settings')
        self.assertEqual(outcome['status'], 'evaluated')
        self.assertEqual({key: os.environ.get(key) for key in THREAD_ENV}, inherited)
        self.assertEqual(json.loads(next(self.logs.rglob('val.json')).read_text())['success_rate'], .5)
        self.assertGreater(outcome['usage']['cpu_seconds'], 0)
        self.assertGreater(outcome['usage']['peak_rss_bytes'], 0)
        if LINUX:
            self.assertGreater(outcome['usage']['peak_virtual_bytes'], 0)
            self.assertGreater(outcome['usage']['rss_bytes'], 0)

    def backend(self, delay=0):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                time.sleep(delay)
                self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.obj.pool = BackendPool([f'http://127.0.0.1:{server.server_port}'])
        router = start_router(0, self.obj.pool)
        self.obj.phase = 'search'; self.obj.replicas = 1; self.obj.processes = {}
        self.obj.status = Mock(); self.obj.to_scoring = Mock(); self.obj.to_proposing = Mock(); self.obj.check_model = Mock()
        self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
        self.addCleanup(router.server_close); self.addCleanup(router.shutdown)
        return f'http://127.0.0.1:{router.server_port}'

    def test_all_candidates_reject_before_http_and_return_to_proposing(self):
        router = self.backend()
        backend = SimpleNamespace(pid=321, poll=lambda: None)
        self.obj.processes = {'shared': backend}
        path = self.harness('h2')
        rows = self.obj.score(candidate_test_worker.sweep, dict(CONFIG, mode='malformed', router=router),
                              [('h1', self.root / 'harnesses/h1/harness.py'), ('h2', path)], self.logs)
        self.assertEqual([r['status'] for r in rows], ['rejected_evaluation', 'rejected_evaluation'])
        self.assertEqual(self.obj.pool.snapshot()['completed'], [0])
        self.assertEqual(self.obj.pool.snapshot()['failed'], [0])
        self.assertEqual(self.obj.pool.snapshot()['running'], [0])
        self.assertEqual(self.obj.pool.snapshot()['waiting'], 0)
        self.assertIs(self.obj.processes['shared'], backend)
        self.assertIsNone(backend.poll())
        self.obj.to_proposing.assert_called_once()
        self.assertFalse(list(self.logs.rglob('val.json')))

    def test_http_wait_does_not_deplete_cpu_budget(self):
        router = self.backend(delay=2)
        with patch.dict(os.environ, LADDER_CANDIDATE_SCORE_CPU_SECONDS='1',
                        LADDER_CANDIDATE_SCORE_TIMEOUT='10'):
            outcome = self.score(router=router, requests=1)
        self.assertEqual(outcome['status'], 'evaluated', outcome)
        self.assertGreater(outcome['seconds'], 2)
        self.assertLess(outcome['usage']['cpu_seconds'], 1)

    def test_resource_rejection_then_healthy_survivor_retains_shared_backend(self):
        router = self.backend()
        path = self.harness('h2')
        with patch.dict(os.environ, LADDER_CANDIDATE_OUTPUT_BYTES=str(32 * 1024)):
            rows = self.obj.score(candidate_test_worker.sweep,
                dict(CONFIG, router=router, candidate_modes={'h1': 'output_flood'}),
                [('h1', self.root / 'harnesses/h1/harness.py'), ('h2', path)], self.logs)
        self.assertEqual([r['status'] for r in rows], ['rejected_resource_limit', 'evaluated'])
        self.assertEqual(self.obj.pool.snapshot()['failed'], [0])
        self.assertFalse(list(self.logs.glob('*/h1/*/val.json')))
        self.assertTrue(list(self.logs.glob('*/h2/*/val.json')))


if __name__ == '__main__': unittest.main()
