"""Execute the patched pinned WHALE loop with controlled scorer outcomes."""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures'
sys.path.insert(0, str(ROOT / 'tools'))
from candidate_containment import CandidateRejected, WorkerSetupError


class PatchedLoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        file = root / 'domains/chess_puzzles/meta_harness/meta_harness_chess_puzzle.py'
        file.parent.mkdir(parents=True)
        file.write_bytes((FIXTURES / 'pinned_meta_harness_before_validation.py').read_bytes())
        subprocess.run(['git', 'apply', '--check', str(ROOT / 'patches/candidate-validation.patch')], cwd=root, check=True)
        subprocess.run(['git', 'apply', str(ROOT / 'patches/candidate-validation.patch')], cwd=root, check=True)
        # Use actual patched function bodies; dependency-heavy inference imports
        # are supplied by the controlled CPU test boundary.
        tree = ast.parse(file.read_text())
        functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        self.ns = dict(argparse=argparse, json=json, os=os, shutil=shutil, sys=sys, time=time,
                       Path=Path, Any=Any, PROMPT_ONLY=False)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future] + functions, type_ignores=[])), str(file), 'exec'), self.ns)
        self.run = root / 'run'
        (self.run / 'logs').mkdir(parents=True)
        for name in ('h0', 'h1', 'h2'):
            p = self.run / 'harnesses' / name / 'harness.py'
            p.parent.mkdir(parents=True); p.write_text('fixture')
        self.ns.update(get_run_dir=lambda _: self.run, load_config=lambda _: {},
                       ensure_run_layout=lambda _: None, baseline_harness=lambda: self.run / 'harnesses/h0/harness.py',
                       init_h0=lambda *a: None, SKILL_DIR='fixture', CONFIG_PATH='unused',
                       next_harness_names=lambda *a: ['h1', 'h2'], render_task_prompt=lambda *a: 'prompt',
                       validate_candidate=lambda *a: None)
        self.ns['set_accepted_harness'](self.run, 'h0')
        self.args = SimpleNamespace(config='fixture', proposer_model='', proposer_effort='', propose_timeout=None,
            proposals_per_iter=2, early_stop_success_rate=1, run_name='fixture', fresh=False,
            iterations=1, force=False, eval_only=True)

    def val(self, name, score):
        p = self.run / 'logs/p' / name / 'mock/val.json'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(success_rate=score, mean_turn_count=2, avg_reward=score)))

    def run_round(self, outcomes):
        def sweep(config, harnesses, logs, **kwargs):
            if harnesses[0][0] == 'h0':
                self.val('h0', .4)
                return [dict(harness='h0', status='evaluated')]
            # Rejected h1 has an old apparently perfect score. It must be
            # removed before frontier/summary/early-stop logic can inspect it.
            self.val('h1', 1)
            for row in outcomes:
                if row.get('status') == 'evaluated': self.val(row['harness'], .6)
            return outcomes
        self.ns['run_sweep'] = sweep
        frontier_calls = []
        def frontier(logs, config):
            names = [p.parts[-3] for p in logs.glob('*/*/*/val.json')]
            frontier_calls.append(names)
            best = max(names, key=lambda name: json.loads(next(logs.glob(f'*/{name}/*/val.json')).read_text())['success_rate'])
            return dict(_best=dict(harness=best), _pareto=[])
        self.ns['print_frontier'] = frontier
        self.ns['run_evolve'](self.args)
        rows = [json.loads(line) for line in (self.run / 'logs/evolution_summary.jsonl').read_text().splitlines()]
        comparison = json.loads((self.run / 'logs/iteration_001/comparison.json').read_text())
        return rows, comparison, frontier_calls

    def test_timeout_then_healthy_counts_slots_and_excludes_stale_early_stop(self):
        rows, comparison, calls = self.run_round([
            dict(harness='h1', status='rejected_timeout', reason='deadline', evidence='worker.log'),
            dict(harness='h2', status='evaluated')])
        self.assertEqual([(r['harness'], r['status']) for r in rows], [('h1', 'rejected_timeout'), ('h2', 'evaluated')])
        self.assertEqual(comparison['accepted_harness'], 'h2')
        self.assertIsNone(comparison['early_stop'])
        self.assertEqual(len(rows), 2)
        self.assertNotIn('h1', calls[-1])
        self.assertFalse(list((self.run / 'logs').glob('*/h1/*/val.json')))
        self.assertTrue(list((self.run / 'rejected-score-evidence').rglob('val.json')))

    def test_all_rejected_keeps_incumbent_without_scores_or_early_stop(self):
        rows, comparison, calls = self.run_round([
            dict(harness='h1', status='rejected_timeout', reason='deadline'),
            dict(harness='h2', status='rejected_evaluation', reason='partial')])
        self.assertEqual(self.ns['get_accepted_harness'](self.run), 'h0')
        self.assertEqual(comparison['summary'], [])
        self.assertEqual(len(comparison['rejected']), 2)
        self.assertEqual(len(calls), 1, 'No post-rejection frontier selection')
        self.assertEqual({r['status'] for r in rows}, {'rejected_timeout', 'rejected_evaluation'})

    def test_missing_duplicate_or_unknown_outcomes_reject_without_fabricated_scores(self):
        for results in (None, [], [dict(harness='h1', status='ok')],
                        [dict(harness='h1', status='evaluated')] * 2):
            with self.subTest(results=results):
                good, rejects = self.ns['scoring_outcomes'](self.run, [dict(name='h1')], results, 1)
                self.assertEqual(good, [])
                self.assertEqual(rejects[0]['status'], 'rejected_evaluation')

    def test_worker_setup_failure_bypasses_candidate_validation_rejection(self):
        self.ns['validate_candidate'] = Mock(side_effect=WorkerSetupError('policy setup failed'))
        with self.assertRaisesRegex(WorkerSetupError, 'policy setup failed'):
            self.ns['validate_proposals'](self.run, [dict(name='h1')], ['h1'], 1)
        self.assertFalse((self.run / 'logs/evolution_summary.jsonl').exists())

    def test_resource_rejection_details_survive_scoring_and_validation_rows(self):
        details = dict(status='rejected_resource_limit', reason='CPU cap', evidence='worker',
                       resource=dict(kind='cpu', signal='SIGXCPU'),
                       resource_policy=dict(cpu_seconds=1), applied_policy=dict(cpu_rlimit=[1, 2]),
                       usage=dict(cpu_seconds=.99))
        rows, comparison, _ = self.run_round([dict(details, harness='h1'), dict(harness='h2', status='evaluated')])
        rejected = next(r for r in rows if r['harness'] == 'h1')
        for key in ('status', 'resource', 'resource_policy', 'applied_policy', 'usage', 'evidence'):
            self.assertEqual(rejected[key], details[key])
        self.assertEqual(comparison['accepted_harness'], 'h2')
        def reject(*args): raise CandidateRejected(details)
        self.ns['validate_candidate'] = reject
        _, _, rejects, _ = self.ns['validate_proposals'](self.run, [dict(name='h1')], ['h1'], 2)
        for key in ('status', 'resource', 'resource_policy', 'applied_policy', 'usage', 'evidence'):
            self.assertEqual(rejects[0][key], details[key])

    def test_unhashable_status_rejects_and_quarantines(self):
        self.val('h1', 1)
        names, rejected = self.ns['scoring_outcomes'](
            self.run, [dict(name='h1')], [dict(harness='h1', status=[])], 1)
        self.assertEqual(names, [])
        self.assertEqual(rejected[0]['status'], 'rejected_evaluation')
        self.assertFalse(list((self.run / 'logs').glob('*/h1/*/val.json')))
        self.assertTrue(list((self.run / 'rejected-score-evidence').rglob('val.json')))

    def test_validation_timeout_is_named_and_duplicate_manifest_consumes_one_slot(self):
        def reject(*args):
            raise CandidateRejected(dict(status='rejected_timeout', reason='validation deadline', evidence='validation/worker.log'))
        self.ns['validate_candidate'] = reject
        good, candidates, rejected, prompt = self.ns['validate_proposals'](
            self.run, [dict(name='h1'), dict(name='h1')], ['h1', 'h2'], 1)
        self.assertEqual(good, [])
        self.assertEqual([(r['harness'], r['status']) for r in rejected],
                         [('h1', 'rejected_timeout'), ('h2', 'rejected_missing')])
        self.assertTrue(all(r['requested_slot'] for r in rejected))
        self.assertEqual(rejected[0]['evidence'], 'validation/worker.log')


if __name__ == '__main__': unittest.main()
