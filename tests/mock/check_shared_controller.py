"""Audit simulated all-arm persistence, checkpoint policy, resume and final test."""
import json
from pathlib import Path
import sys

root, logs = map(Path, sys.argv[1:])


def read(path):
    return json.loads(path.read_text())


for arm in ('rsft_only', 'whale_deepseek', 'whale_qwen27b', 'whale_qwen4b', 'whale_self'):
    state = root / f'{arm}-mock-persistent-s7'
    plan = read(state / 'cycle-plan.json')
    assert plan['controller_version'] == 1
    assert read(state / 'passed.json')['passed']
    results = read(state / 'results.json')
    assert len(results['cycles']) == 2 and not results['problems']
    assert results['final_test']['rollouts'] == 32
    resume = (logs / ('resume.log' if arm == 'whale_qwen4b' else f'{arm}-resume.log')).read_text()
    assert resume.count('training already passed') == 2
    if arm == 'rsft_only':
        assert not plan['persistent_search'] and not list(state.glob('*-search'))
        continue
    assert resume.count('search already passed') == 2
    directories = [state / f'{arm}-c{c}-search' for c in (1, 2)]
    receipts = [read(p / 'passed.json') for p in directories]
    assert receipts[0]['server_pids'] == receipts[1]['server_pids']
    assert receipts[0]['solver_weight_digest'] != receipts[1]['solver_weight_digest']
    events = [json.loads(line) for p in directories
              for line in (p / 'timing.jsonl').read_text().splitlines()]
    starts = [e for e in events if e['event'] == 'server_start' and e['name'] != 'proxy']
    assert len(starts) == (2 if arm == 'whale_deepseek' else 3)
    assert sum(e['event'] == 'checkpoint_refreshed' for e in events) == 1
    if arm == 'whale_deepseek':
        assert all('proposer' not in r['server_pids'] for r in receipts)
        assert all(r['proposer_weight_digest'] is None for r in receipts)
    elif arm == 'whale_self':
        assert all(r['proposer_weight_digest'] == r['solver_weight_digest'] for r in receipts)
        assert receipts[0]['proposer_weight_digest'] != receipts[1]['proposer_weight_digest']
    else:
        assert receipts[0]['proposer_weight_digest'] == receipts[1]['proposer_weight_digest']
    for r in receipts:
        assert r['candidates_evaluated'] + r['candidates_rejected'] == 6
    receipt = read(state / 'c2-sleep.json')
    assert set(receipt['ports']) == set(receipts[0]['server_pids'])
print('PASS: one controller; all five arms; simulated persistence, weight policies, handoffs, resume and final evaluation.')
