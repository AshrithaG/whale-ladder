"""Failure-oriented checks for sleeping servers and checkpoint replacement."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from persistent_search import PersistentSearch, checkpoint_signature, memory_guard, validate_sleep_receipt
from persistent_cycles import validate_mode
from switch_router import BackendPool


def checkpoint(path, config='{}'):
    path.mkdir()
    for name, text in {'.merge_complete': '', 'config.json': config,
                       'tokenizer_config.json': '{}', 'model.safetensors': 'weights'}.items():
        (path / name).write_text(text)
    return path


class AdmissionTests(unittest.TestCase):
    def test_checkpoint_requires_complete_weights_and_matching_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = checkpoint(Path(tmp) / 'a')
            q = checkpoint(Path(tmp) / 'b', '{"hidden_size":128}')
            self.assertNotEqual(checkpoint_signature(p), checkpoint_signature(q))
            (p / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x': 'missing.safetensors'}}))
            with self.assertRaises(ValueError): checkpoint_signature(p)
            (q / '.merge_complete').unlink()
            with self.assertRaises(ValueError): checkpoint_signature(q)

    def test_awake_server_memory_refuses_training(self):
        with patch('persistent_search.subprocess.check_output', return_value='6000, 81920\n7000, 81920\n'):
            self.assertEqual(len(memory_guard(['0', '1'])), 2)
        with patch('persistent_search.subprocess.check_output', return_value='6000, 81920\n61000, 81920\n'):
            with self.assertRaises(RuntimeError): memory_guard(['0', '1'])

    @patch.dict(os.environ, {'SLURM_JOB_ID': '42', 'LADDER_JOB_GPUS': '0,1'})
    @patch('persistent_search.memory_guard', return_value=[])
    def test_sleep_receipt_requires_same_job_freshness_and_all_servers_sleeping(self, guard):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'receipt.json'
            receipt = dict(time=time.time(), job_id='42', gpus=['0', '1'],
                           ports={'solver0': 30001, 'solver1': 30002, 'proposer': 30003})
            path.write_text(json.dumps(receipt))
            with patch('persistent_search.urllib.request.urlopen') as urlopen:
                response = urlopen.return_value.__enter__.return_value
                response.read.return_value = b'{"is_sleeping":true}'
                validate_sleep_receipt(path)
                self.assertEqual(urlopen.call_count, 3)
                response.read.return_value = b'{"is_sleeping":false}'
                with self.assertRaises(RuntimeError): validate_sleep_receipt(path)
                receipt.update(job_id='another-job')
                path.write_text(json.dumps(receipt))
                with self.assertRaises(RuntimeError): validate_sleep_receipt(path)
                receipt.update(job_id='42', time=time.time() - 601)
                path.write_text(json.dumps(receipt))
                with self.assertRaises(RuntimeError): validate_sleep_receipt(path)

    @patch.dict(os.environ, {'PROPOSER_BACKEND': 'local', 'LADDER_JOB_GPUS': '0,1', 'LADDER_GPU_CHECK': 'skip'})
    def test_supported_arms_and_incompatible_backend_or_topology(self):
        for arm in ('whale_qwen4b', 'whale_qwen27b', 'whale_self'):
            validate_mode(arm)
        with self.assertRaises(ValueError): validate_mode('whale_deepseek')
        with patch.dict(os.environ, {'PROPOSER_BACKEND': 'api'}):
            validate_mode('whale_deepseek')
            with self.assertRaises(ValueError): validate_mode('whale_self')
        with patch.dict(os.environ, {'LADDER_SHARE_ONE_GPU': '1'}):
            with self.assertRaises(ValueError): validate_mode('whale_qwen4b')

    @patch.dict(os.environ, {'SLURM_JOB_ID': '42', 'LADDER_JOB_GPUS': '0,1', 'PROPOSER_BACKEND': 'api'})
    @patch('persistent_search.memory_guard', return_value=[])
    def test_api_sleep_receipt_requires_only_the_two_solvers(self, guard):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'receipt.json'
            receipt = dict(time=time.time(), job_id='42', gpus=['0', '1'], backend='api',
                           ports={'solver0': 30001, 'solver1': 30002})
            path.write_text(json.dumps(receipt))
            with patch('persistent_search.urllib.request.urlopen') as urlopen:
                urlopen.return_value.__enter__.return_value.read.return_value = b'{"is_sleeping":true}'
                validate_sleep_receipt(path)
                self.assertEqual(urlopen.call_count, 2)
                receipt['backend'] = 'local'
                path.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(RuntimeError, 'backend differs'):
                    validate_sleep_receipt(path)


class ReloadTests(unittest.TestCase):
    def trial(self, path):
        obj = object.__new__(PersistentSearch)
        obj.local_proposer, obj.proposer_tracks_solver = True, False
        obj.signature = checkpoint_signature(path)
        obj.parked, obj.digest, obj.proposer_digest = True, 'old', 'fixed'
        obj.pool = BackendPool(['a', 'b'])
        obj.pool.enable([])
        obj.gpus = ['0', '1']
        obj.pids = {'solver0': 1, 'solver1': 2, 'proposer': 3}
        obj.processes = {n: SimpleNamespace(pid=p, poll=lambda: None) for n, p in obj.pids.items()}
        obj.model_checks = {'solver0': 'old-output', 'solver1': 'old-output', 'proposer': 'fixed'}
        obj.event = Mock()
        obj.calls = []
        def request(name, url, data=None, **kw):
            obj.calls.append((name, url, json.loads(data) if data else None))
            return {'success': True}
        obj.request = request
        obj.weight_digest = Mock(side_effect=lambda name: 'fixed' if name == 'proposer' else 'new')
        obj.wake = lambda n: obj.calls.append((n, 'wake', None))
        obj.sleep = lambda n: obj.calls.append((n, 'sleep', None))
        obj.check_model = lambda n: obj.calls.append((n, 'check', None))
        return obj

    @patch('persistent_search.memory_guard', return_value=[])
    def test_self_proposer_reloads_and_matches_the_new_solver(self, guard):
        with tempfile.TemporaryDirectory() as tmp:
            p = checkpoint(Path(tmp) / 'a')
            obj = self.trial(p)
            obj.proposer_tracks_solver = True
            obj.weight_digest.side_effect = lambda name: 'new'
            obj.refresh(p)
            self.assertEqual(obj.proposer_digest, obj.digest)
            reloads = [name for name, url, _ in obj.calls if url == '/collective_rpc']
            self.assertEqual(reloads, ['solver0', 'solver1', 'proposer'])
            self.assertNotIn('proposer', obj.model_checks)
            self.assertEqual(obj.pool.snapshot()['enabled'], [0])

    @patch('persistent_search.memory_guard', return_value=[])
    def test_api_arm_reloads_only_solvers_and_enables_both(self, guard):
        with tempfile.TemporaryDirectory() as tmp:
            p = checkpoint(Path(tmp) / 'a')
            obj = self.trial(p)
            obj.local_proposer = False
            obj.pids.pop('proposer')
            obj.processes.pop('proposer')
            obj.refresh(p)
            self.assertFalse(any(name == 'proposer' for name, _, _ in obj.calls))
            self.assertEqual(obj.pool.snapshot()['enabled'], [0, 1])

    @patch('persistent_search.memory_guard', return_value=[])
    def test_stale_self_proposer_blocks_routing_after_solver_reload(self, guard):
        with tempfile.TemporaryDirectory() as tmp:
            p = checkpoint(Path(tmp) / 'a')
            obj = self.trial(p)
            obj.proposer_tracks_solver = True
            obj.weight_digest.side_effect = ['new', 'new', 'old-proposer']
            with self.assertRaisesRegex(RuntimeError, 'self proposer differs'):
                obj.refresh(p)
            self.assertEqual(obj.pool.snapshot()['enabled'], [])

    @patch('persistent_search.memory_guard', return_value=[])
    def test_reload_both_and_clear_old_cache_before_enabling_requests(self, guard):
        with tempfile.TemporaryDirectory() as tmp:
            p = checkpoint(Path(tmp) / 'a')
            obj = self.trial(p)
            obj.refresh(p)
            self.assertEqual(obj.digest, 'new')
            self.assertFalse(obj.parked)
            self.assertEqual(obj.pool.snapshot()['enabled'], [0])
            for name in ('solver0', 'solver1'):
                calls = [x[1] for x in obj.calls if x[0] == name]
                self.assertLess(calls.index('/collective_rpc'), calls.index('/reset_prefix_cache'))
                self.assertLess(calls.index('/reset_prefix_cache'), calls.index('wake'))
            reloads = [x[2]['kwargs']['weights_path'] for x in obj.calls if x[1] == '/collective_rpc']
            self.assertEqual(reloads, [str(p.resolve())] * 2)
            self.assertEqual(obj.model_checks, {'proposer': 'fixed'})

    @patch('persistent_search.memory_guard', return_value=[])
    def test_stale_or_disagreeing_weights_and_timeout_keep_routing_disabled(self, guard):
        with tempfile.TemporaryDirectory() as tmp:
            p = checkpoint(Path(tmp) / 'a')
            for values in [('new', 'different'), ('old', 'old'), ('new', 'new', 'changed-proposer')]:
                obj = self.trial(p)
                obj.weight_digest.side_effect = values
                with self.assertRaises(RuntimeError): obj.refresh(p)
                self.assertEqual(obj.pool.snapshot()['enabled'], [])
            obj = self.trial(p)
            obj.request = Mock(side_effect=TimeoutError('reload outcome unknown'))
            with self.assertRaises(TimeoutError): obj.refresh(p)
            self.assertEqual(obj.pool.snapshot()['enabled'], [])

    def test_incompatible_metadata_or_dead_process_rejected_before_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = checkpoint(Path(tmp) / 'a')
            q = checkpoint(Path(tmp) / 'b', '{"hidden_size":128}')
            obj = self.trial(p)
            with self.assertRaises(ValueError): obj.refresh(q)
            self.assertEqual(obj.calls, [])
            obj.processes['solver0'].poll = lambda: 1
            with self.assertRaises(RuntimeError): obj.refresh(p)
            self.assertEqual(obj.calls, [])

    @patch('persistent_search.memory_guard', return_value=[])
    def test_failed_cache_reset_never_reenables_solver(self, guard):
        with tempfile.TemporaryDirectory() as tmp:
            p = checkpoint(Path(tmp) / 'a')
            obj = self.trial(p)
            obj.request = Mock(return_value={'success': False})
            with self.assertRaises(RuntimeError): obj.refresh(p)
            self.assertEqual(obj.pool.snapshot()['enabled'], [])
            obj.weight_digest.assert_not_called()

    def test_search_preserves_pr4_outcomes_and_restores_module_state(self):
        obj = object.__new__(PersistentSearch)
        mh = SimpleNamespace(render_task_prompt='prompt', run_sweep='sweep',
                             PROPOSER_SYSTEM_PROMPT='system', PROMPT_ONLY=True)
        original, argv = vars(mh).copy(), sys.argv[:]
        outcome = dict(candidates_evaluated=0, candidates_rejected=3, accepted_harness='h0')
        def search(*args, **kw):
            mh.run_sweep = 'closure'
            mh.PROPOSER_SYSTEM_PROMPT = 'changed'
            sys.argv = ['changed']
            return outcome
        obj.real_search = search
        obj.stop = Mock()
        self.assertEqual(obj.search(mh, 1), outcome)
        self.assertEqual(vars(mh), original)
        self.assertEqual(sys.argv, argv)
        obj.real_search = Mock(side_effect=RuntimeError('server failed'))
        with self.assertRaises(RuntimeError): obj.search(mh, 1)
        self.assertEqual(vars(mh), original)
        self.assertEqual(obj.stop.call_count, 2)


if __name__ == '__main__':
    unittest.main()
