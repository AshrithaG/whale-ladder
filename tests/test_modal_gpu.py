"""CPU-only wrapper checks. No model, GPU or provider invocation is exercised."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import modal_gpu
import modal_gpu_probe


class ModalWrapperTests(unittest.TestCase):
    def trial(self, failed_phase=None, timeout_phase=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root, code = Path(tmp.name) / 'work', Path(tmp.name) / 'code'
        (root / 'runs').mkdir(parents=True)
        (root / 'WHALE/domains/chess_puzzles/meta_harness').mkdir(parents=True)
        code.mkdir()
        (root / 'cpu-ready.json').write_text(json.dumps(dict(cpu_installation_passed=True,
            revision=modal_gpu.RUNTIME_REVISION, data_fingerprint='91c2fe70efad772c')))
        sources = {'modal_gpu.py': '# test launcher\n', 'modal_gpu_probe.py': '# test probe\n'}
        plan = dict(runtime_revision=modal_gpu.RUNTIME_REVISION, runtime_source_sha256={},
            launcher_sha256={n: hashlib.sha256(s.encode()).hexdigest() for n, s in sources.items()},
            soft_deadline_seconds=60, arm='whale_qwen4b', profile='modal-smoke', results_volume='test-only')
        phases, envs = [], []
        processes = []
        def start(argv, **kwargs):
            phase = Path(kwargs['stdout'].name).stem
            phases.append(phase)
            envs.append(kwargs['env'].copy())
            if phase == 'host':
                Path(argv[-1]).write_text(json.dumps(dict(ladder_job_gpus='GPU-assigned7,GPU-assigned3')))
            if phase == 'logging-ready':
                (root / 'runs/modal-control/live.json').write_text('{}')
            proc = Mock(pid=123456)
            proc.wait.side_effect = ([subprocess.TimeoutExpired(argv, 1), 0]
                                    if phase == timeout_phase else None)
            proc.wait.return_value = 7 if phase == failed_phase else 0
            processes.append(proc)
            return proc
        volume = Mock()
        return root, code, plan, sources, phases, envs, start, volume, processes

    def execute(self, fixture):
        root, code, plan, sources, phases, envs, start, volume, processes = fixture
        with patch.object(modal_gpu, 'WORK_ROOT', str(root)), patch.object(modal_gpu, 'CODE_ROOT', str(code)), \
             patch('subprocess.check_output', return_value=modal_gpu.RUNTIME_REVISION), \
             patch('subprocess.Popen', side_effect=start), \
             patch.dict(os.environ, {'PYTHONPATH': '/pkg:/root', 'PYTHONHOME': '/provider'}):
            return modal_gpu.run_gpu(plan, sources, volume)

    def test_failed_preflight_stops_before_training_and_preserves_failure(self):
        f = self.trial(failed_phase='preflight')
        with self.assertRaises(subprocess.CalledProcessError):
            self.execute(f)
        root, _, _, _, phases, envs, _, volume, _ = f
        self.assertEqual(phases, ['host', 'logging-ready', 'observer', 'nccl', 'kernels', 'preflight'])
        failure = json.loads((root / 'runs/modal-control/failure.json').read_text())
        self.assertEqual(failure['phase'], 'preflight')
        self.assertFalse((root / 'runs/modal-control/execution-completed.json').exists())
        self.assertTrue((root / 'WHALE/domains/chess_puzzles/meta_harness/runs').is_symlink())
        self.assertTrue((root / 'preflight').is_symlink())
        self.assertNotIn('PYTHONPATH', envs[0])
        self.assertNotIn('PYTHONHOME', envs[0])
        self.assertEqual(envs[-1]['CUDA_VISIBLE_DEVICES'], 'GPU-assigned7,GPU-assigned3')
        self.assertGreaterEqual(volume.commit.call_count, 2)

    def test_replay_does_not_resume_or_generate_new_candidates(self):
        f = self.trial()
        self.assertTrue(self.execute(f)['raw_audit_required'])
        first_phases = f[4][:]
        with self.assertRaises(FileExistsError):
            self.execute(f)
        self.assertEqual(f[4], first_phases)

    def test_deadline_signals_owned_group_and_retains_failure(self):
        f = self.trial(timeout_phase='kernels')
        with patch('os.killpg') as kill:
            with self.assertRaises(subprocess.TimeoutExpired):
                self.execute(f)
        kill.assert_called_once()
        self.assertEqual(kill.call_args.args[0], 123456)
        failure = json.loads((f[0] / 'runs/modal-control/failure.json').read_text())
        self.assertEqual(failure['type'], 'TimeoutExpired')
        self.assertNotIn('preflight', f[4])

    def test_image_source_mismatch_stops_before_any_gpu_command(self):
        f = self.trial()
        (f[1] / 'changed.py').write_text('changed')
        f[2]['runtime_source_sha256']['changed.py'] = 'wrong-digest'
        with self.assertRaisesRegex(RuntimeError, 'Image source mismatch'):
            self.execute(f)
        self.assertEqual(f[4], [])


class DeviceMappingTests(unittest.TestCase):
    def inspect(self, second_name='NVIDIA H100 80GB HBM3'):
        torch = SimpleNamespace(__version__='test', version=SimpleNamespace(cuda='13.0'),
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                get_device_properties=lambda i: SimpleNamespace(uuid=('uuid3', 'uuid7')[i]),
                get_device_capability=lambda i: (9, 0), nccl=SimpleNamespace(version=lambda: (2, 28, 9))))
        raw = f'7, GPU-uuid7, {second_name}, 81559, 580.1\n3, GPU-uuid3, NVIDIA H100 80GB HBM3, 81559, 580.1\n'
        with patch.dict(sys.modules, {'torch': torch}), \
             patch.object(sys, 'prefix', '/opt/whale-work/venv'), \
             patch('subprocess.check_output', return_value=raw), \
             patch('os.sched_getaffinity', return_value=set(range(48)), create=True), \
             patch('pathlib.Path.exists', return_value=False), \
             patch('pathlib.Path.read_text', return_value='test meminfo'), \
             patch('shutil.disk_usage', return_value=SimpleNamespace(free=300 * 1024**3)):
            return modal_gpu_probe.inspect_host()

    def test_cuda_order_maps_to_noncontiguous_physical_devices(self):
        host = self.inspect()
        self.assertEqual(host['ladder_job_gpus'], 'GPU-uuid3,GPU-uuid7')
        self.assertEqual([d['index'] for d in host['devices']], ['3', '7'])

    def test_h200_substitution_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'Expected H100'):
            self.inspect('NVIDIA H200')


if __name__ == '__main__':
    unittest.main()
