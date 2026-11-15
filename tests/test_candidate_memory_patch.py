"""Run real pinned WHALE function bodies after the MemoryError compatibility patch."""
import ast
import json
from pathlib import Path
import re
import subprocess
import tempfile
import traceback
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures'
FILES = {'domains/chess_puzzles/autoharness_chess_puzzle/runner.py': 'pinned_runner_before_memory_errors.py',
         'domains/chess_puzzles/meta_harness/chess_puzzle_benchmark.py': 'pinned_benchmark_before_memory_errors.py'}


def functions(source, names, namespace):
    tree = ast.parse(source)
    body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    body += [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), 'pinned-whale', 'exec'), namespace)
    return namespace


class NativeMemoryPatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for destination, fixture in FILES.items():
            p = self.root / destination; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((FIXTURES / fixture).read_bytes())
        patch = str(ROOT / 'patches/candidate-memory-errors.patch')
        subprocess.run(['git', 'apply', '--check', patch], cwd=self.root, check=True)
        subprocess.run(['git', 'apply', patch], cwd=self.root, check=True)
        subprocess.run(['git', 'apply', '--reverse', '--check', patch], cwd=self.root, check=True)
        self.runner = functions((self.root / next(iter(FILES))).read_text(),
            {'normalize_uci', 'format_observation_with_harness', 'policy_messages_with_harness',
             'parse_model_action', 'verify_and_step'},
            dict(re=re, UCI_RE=re.compile(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', re.IGNORECASE), build_observation=lambda _: 'original observation', effective_max_turns=lambda _: 12,
                 retry_budgets=lambda _: (0, 0), StepResult=lambda **kw: kw,
                 chess=SimpleNamespace(WHITE=True, InvalidMoveError=ValueError,
                                       Move=SimpleNamespace(from_uci=lambda move: move))))
        self.state = SimpleNamespace(accepted_solver_moves=[], last_opponent_move=None, event_log=[],
            completion_records=[], board=SimpleNamespace(turn=True, legal_moves=['e2e4']),
            illegal_retries_used=0, format_retries_used=0)
        self.harness = SimpleNamespace(format_observation=None, parse_action=None, system_prompt=None,
                                       user_prompt=None, is_legal_action=lambda *a: True)

    def test_generated_formatter_memory_error_cannot_fall_back(self):
        self.harness.format_observation = Mock(side_effect=MemoryError('formatter allocation'))
        with self.assertRaisesRegex(MemoryError, 'formatter allocation'):
            self.runner['format_observation_with_harness'](self.harness, self.state)

    def test_prompt_format_memory_error_cannot_fall_back(self):
        class Template(str):
            def format(self, **kw): raise MemoryError('prompt allocation')
        self.harness.user_prompt = Template('{observation}')
        with self.assertRaisesRegex(MemoryError, 'prompt allocation'):
            self.runner['policy_messages_with_harness'](self.harness, self.state)

    def test_generated_parser_memory_error_cannot_fall_back_to_native_parser(self):
        self.harness.parse_action = Mock(side_effect=MemoryError('parser allocation'))
        with self.assertRaisesRegex(MemoryError, 'parser allocation'):
            self.runner['parse_model_action'](self.harness, '<move>e2e4</move>')

    def test_generated_legality_memory_error_cannot_become_a_failed_puzzle(self):
        self.harness.is_legal_action = Mock(side_effect=MemoryError('legality allocation'))
        with self.assertRaisesRegex(MemoryError, 'legality allocation'):
            self.runner['verify_and_step'](harness=self.harness, state=self.state, parsed_action='e2e4')
        self.assertEqual(self.state.event_log, [])

    def test_sweep_memory_error_reaches_worker_without_success_or_failure_scores(self):
        name = 'domains/chess_puzzles/meta_harness/chess_puzzle_benchmark.py'
        ns = functions((self.root / name).read_text(),
            {'_models_from_config', 'get_model_short_name', 'run_dir', 'run_sweep'},
            dict(Path=Path, DEFAULT_SEED=42, json=json, traceback=traceback,
                 evaluate_harness=Mock(side_effect=MemoryError('evaluation allocation'))))
        config = dict(task_profiles=dict(p=dict(dataset_path='original', limit=2)),
                      models=[dict(model='Qwen/Qwen3.5-4B')])
        with self.assertRaisesRegex(MemoryError, 'evaluation allocation'):
            ns['run_sweep'](config, [('h1', self.root / 'harness.py')], self.root / 'logs')
        self.assertFalse(list((self.root / 'logs').rglob('*')))
        ns['evaluate_harness'] = Mock(side_effect=ValueError('ordinary evaluation failure'))
        self.assertEqual(ns['run_sweep'](config, [('h1', self.root / 'harness.py')], self.root / 'logs'),
                         [('p/h1/qwen3.5-4b_seed42', False)])


if __name__ == '__main__': unittest.main()
