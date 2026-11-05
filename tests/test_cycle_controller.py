"""Exercise shared cycle handoffs and resume guards without model dependencies."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import cycle_controller as controller


class CycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.old_cwd = Path.cwd()
        self.addCleanup(os.chdir, self.old_cwd)
        (self.root / 'domain').mkdir()
        base = self.root / 'h0.py'
        base.write_text('base harness\n')
        self.env = dict(LADDER_HOME=str(ROOT), OUT_ROOT=str(self.root / 'out'),
                        DOMAIN_DIR=str(self.root / 'domain'), BASE_HARNESS=str(base),
                        CYCLES='2', STEPS_PER_CYCLE='1', SEARCH_ITERS='1', SEED='7',
                        SOLVER_MODEL='solver', SOLVER_MODEL_REVISION='solver-revision',
                        LADDER_PERSISTENT_SEARCH='0', LADDER_GPU_CHECK='skip')
        self.calls = []
        self.state = self.root / 'out/cycles/rsft_only-unit-s7'

    def fake_command(self, argv, env=None):
        if argv[0] == 'bash':
            self.calls.append(dict(env))
            out = Path(argv[-1])
            out.mkdir(parents=True, exist_ok=True)
            if env['HYBRID_MODE'] == 'train':
                previous = env.get('HYBRID_RESUME_CHECKPOINT')
                step = int(previous.rsplit('_', 1)[1]) + 1 if previous else 1
                ckpt = out / 'checkpoints' / env['HYBRID_RUN_NAME'] / f'global_step_{step}'
                (ckpt / 'actor').mkdir(parents=True)
                (ckpt / 'data.pt').write_text('dataloader')
            (out / 'passed.json').write_text('{"passed":true}')
        else:
            (self.state / 'results.json').write_text('{"problems":[]}')

    def run_controller(self):
        with patch.dict(os.environ, self.env, clear=True), patch.object(controller, 'command', side_effect=self.fake_command):
            controller.run('rsft_only', 'unit')

    def test_training_only_uses_common_handoffs_final_evaluation_and_resume(self):
        self.run_controller()
        self.assertEqual([e['HYBRID_MODE'] for e in self.calls], ['train', 'train', 'eval'])
        self.assertNotIn('HYBRID_RESUME_CHECKPOINT', self.calls[0])
        self.assertTrue(self.calls[1]['HYBRID_RESUME_CHECKPOINT'].endswith('global_step_1'))
        self.assertTrue(self.calls[2]['HYBRID_SOURCE_CHECKPOINT'].endswith('global_step_2'))
        self.assertEqual({e['HARNESS_PATH'] for e in self.calls[:2]},
                         {str(self.state / f'c{c}-trained-with.py') for c in (1, 2)})
        self.assertFalse(list(self.state.glob('*-search')))
        self.assertTrue(json.loads((self.state / 'passed.json').read_text())['passed'])
        self.calls.clear()
        self.run_controller()
        self.assertEqual(self.calls, [], 'Resume must not train or evaluate completed phases again')

    def test_changed_schedule_or_final_budget_cannot_reuse_old_phases(self):
        self.run_controller()
        self.calls.clear()
        self.env['HYBRID_EVAL_ATTEMPTS'] = '2'
        with self.assertRaisesRegex(ValueError, 'Saved cycle plan differs'):
            self.run_controller()
        self.assertEqual(self.calls, [])

    def test_changed_incoming_state_fails_resume_and_removes_stale_pass(self):
        self.run_controller()
        (self.state / 'c2-train-inputs.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'incoming state'):
            self.run_controller()
        self.assertFalse((self.state / 'passed.json').exists())

    def test_zero_exit_without_phase_receipt_is_a_failure(self):
        with patch.dict(os.environ, self.env, clear=True), patch.object(controller, 'command'):
            with self.assertRaisesRegex(RuntimeError, 'without a pass receipt'):
                controller.run('rsft_only', 'unit')
        self.assertFalse((self.state / 'passed.json').exists())


if __name__ == '__main__':
    unittest.main()
