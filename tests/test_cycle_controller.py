"""Exercise shared cycle handoffs and resume guards without model dependencies."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import cycle_controller as controller
from candidate_containment import CandidateRejected


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

    def test_stalling_selected_harness_blocks_handoff_before_next_training(self):
        self.env.update(PROPOSER_BACKEND='local', WEAK_PROPOSER_MODEL='proposer',
                        WEAK_PROPOSER_REVISION='base-revision',
                        LADDER_CANDIDATE_VALIDATION_TIMEOUT='1.5')
        self.state = self.root / 'out/cycles/whale_qwen4b-unit-s7'
        searches = []

        def command(argv, env=None):
            if argv[:2] == ['bash', str(ROOT / 'ladder/search_two_gpu.sh')]:
                search = Path(argv[2])
                searches.append(search)
                run = search / 'harness-run'
                (run / 'logs').mkdir(parents=True)
                selected = run / 'harnesses/h1/harness.py'
                selected.parent.mkdir(parents=True)
                selected.write_text('def propose_action(board): return ""\n'
                                    'def is_legal_action(board, action): return True\n'
                                    'while True: pass\n')
                (run / 'logs/accepted_harness.txt').write_text('h1')
                (search / 'harness-run-dir').write_text(str(run))
                (search / 'passed.json').write_text('{"passed":true}')
            else:
                self.fake_command(argv, env)

        # The controller's real validator launches a fresh interpreter that
        # imports the lightweight fixture, then executes the selected module.
        mh = SimpleNamespace(__name__='candidate_test_worker', PROMPT_ONLY=False)
        with patch.dict(os.environ, self.env, clear=True), \
             patch.object(sys, 'path', [str(ROOT / 'tests/fixtures')] + sys.path), \
             patch.dict(sys.modules, {'meta_harness': SimpleNamespace(meta_harness_chess_puzzle=mh)}), \
             patch.object(controller, 'command', side_effect=command):
            started = time.monotonic()
            with self.assertRaises(CandidateRejected) as caught:
                controller.run('whale_qwen4b', 'unit')
            self.assertLess(time.monotonic() - started, 8)

        self.assertEqual(caught.exception.status, 'rejected_timeout')
        evidence = Path(caught.exception.outcome['evidence'])
        self.assertEqual(json.loads((evidence / 'outcome.json').read_text())['status'],
                         'rejected_timeout')
        self.assertEqual(len(searches), 1)
        self.assertTrue((searches[0] / 'passed.json').is_file(), 'Reach the accepted-harness handoff')
        self.assertEqual([call['HYBRID_MODE'] for call in self.calls], ['train'])
        self.assertFalse((self.state / 'c1-accepted.py').exists())
        self.assertFalse((self.state / 'c2-train').exists())
        self.assertFalse((self.state / 'final-test').exists())
        self.assertFalse((self.state / 'final-harness.py').exists())
        self.assertFalse((self.state / 'passed.json').exists())

    def test_ten_cycles_resume_reuses_c4_training_and_reruns_unfinished_search(self):
        self.env.update(CYCLES='10', LADDER_PERSISTENT_SEARCH='1',
                        PROPOSER_BACKEND='local', WEAK_PROPOSER_MODEL='proposer',
                        WEAK_PROPOSER_REVISION='base-revision')
        self.state = self.root / 'out/cycles/whale_qwen4b-unit-s7'
        searches, refreshes, parks, starts = [], [], [], []

        class Engine:
            def __init__(engine):
                engine.out = Path(os.environ['SWITCH_OUT'])
                engine.parked = False
                engine.pids = [1, 2, 3]
                engine.server_log_directory = str(engine.out)
                engine.digest, engine.proposer_digest = 'updated', 'frozen'

            def initialize(engine): starts.append(True)
            def verify_pids(engine): pass
            def close(engine): pass

            def park(engine, receipt):
                parks.append(receipt)
                receipt.write_text('{}')
                engine.parked = True

            def refresh(engine, checkpoint):
                refreshes.append(checkpoint)
                engine.parked = False

            def search(engine, mh, iterations):
                searches.append(iterations)
                if len(searches) == 4:
                    (engine.out / 'partial-evidence').write_text('unfinished c4 search')
                    raise RuntimeError('interrupted c4 search')
                run = engine.out / 'harness-run'
                (run / 'logs').mkdir(parents=True)
                (run / 'harnesses/h1').mkdir(parents=True)
                (run / 'logs/accepted_harness.txt').write_text('h1')
                (run / 'harnesses/h1/harness.py').write_text(f'cycle {len(searches) - (len(searches) > 4)}\n')
                (engine.out / 'harness-run-dir').write_text(str(run))
                return {}

        def command(argv, env=None):
            if argv[:2] == ['bash', '-c']:
                (Path(argv[-1]) / 'hf').mkdir(exist_ok=True)
            elif 'make_mh_config.py' in argv[1]:
                Path(argv[-1]).write_text('{}')
            else:
                self.fake_command(argv, env)

        mh = SimpleNamespace(__name__='fixture', PROMPT_ONLY=False)
        with patch.dict(os.environ, self.env, clear=True), \
             patch.dict(sys.modules, {'meta_harness': SimpleNamespace(meta_harness_chess_puzzle=mh)}), \
             patch.object(controller, 'command', side_effect=command), \
             patch.object(controller, 'validate_mode'), patch.object(controller, 'PersistentSearch', Engine), \
             patch.object(controller, 'validate_candidate'):
            with self.assertRaisesRegex(RuntimeError, 'interrupted c4 search'):
                controller.run('whale_qwen4b', 'unit')
            self.assertEqual(len(self.calls), 4)
            self.assertTrue((self.state / 'c4-train/passed.json').is_file())
            self.assertFalse((self.state / 'whale_qwen4b-c4-search/passed.json').exists())
            controller.run('whale_qwen4b', 'unit')
        self.assertEqual(searches, [1] * 11)
        self.assertEqual(len(starts), 2)
        preserved = list(self.state.glob('whale_qwen4b-c4-search.unfinished-*'))
        self.assertEqual(len(preserved), 1)
        self.assertEqual((preserved[0] / 'partial-evidence').read_text(), 'unfinished c4 search')
        self.assertEqual(len(parks), 9)
        self.assertEqual(len(refreshes), 9)
        self.assertEqual([e['HYBRID_MODE'] for e in self.calls], ['train'] * 10 + ['eval'])
        self.assertTrue(self.calls[-1]['HYBRID_SOURCE_CHECKPOINT'].endswith('global_step_10'))
        self.assertEqual(Path(self.calls[-1]['HARNESS_PATH']).read_text(), 'cycle 10\n')
        self.assertEqual(json.loads((self.state / 'passed.json').read_text())['cycles'], 10)


if __name__ == '__main__':
    unittest.main()
