"""Check real orchestration with simulated model RPCs; no GPU claims."""
import json
from pathlib import Path
import subprocess
import sys

root, mock = map(Path, sys.argv[1:])
def read(path): return json.loads(path.read_text())
def rows(path): return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

assert read(root / 'passed.json')['passed']
results = read(root / 'results.json')
assert len(results['cycles']) == 2 and not results['problems'] and results['final_test']
searches = [root / f'whale_qwen4b-c{c}-search' for c in (1, 2)]
receipts = [read(p / 'passed.json') for p in searches]
assert receipts[0]['server_pids'] == receipts[1]['server_pids'], 'Search servers restarted'
assert receipts[0]['solver_weight_digest'] != receipts[1]['solver_weight_digest'], 'Solver weights did not change'
events = [event for p in searches for event in rows(p / 'timing.jsonl')]
assert sum(e['event'] == 'server_start' and e.get('name') != 'proxy' for e in events) == 3
assert sum(e['event'] == 'checkpoint_refreshed' for e in events) == 1
assert sum(e['event'] == 'parked_for_training' for e in events) == 1
digests = {e['digest'] for e in events if e['event'] == 'weight_digest' and e.get('name') == 'proposer'}
assert len(digests) == 1, 'Frozen proposer changed'
for p, receipt in zip(searches, receipts):
    assert receipt['candidates_requested'] == 6
    assert receipt['candidates_evaluated'] + receipt['candidates_rejected'] == 6
    run = Path((p / 'harness-run-dir').read_text().strip())
    assert not list(p.glob('*repair*'))
    assert not list(run.glob('*repair*'))
assert receipts[0]['candidates_rejected'] == 2
assert receipts[1]['candidates_missing'] == 3
assert receipts[1]['candidates_evaluated'] == 3
lineage = next((root / 'c2-train/checkpoints').glob('*/global_step_2/mock_lineage.json'))
assert 'c1-train/checkpoints' in read(lineage)['resumed_from']
resume = (mock / 'resume.log').read_text()
assert 'training already passed' in resume and 'search already passed' in resume
assert int((mock / 'claude-calls-qwen4b').read_text()) == 4, 'Resume reproposed candidates'
left = subprocess.run(['pgrep', '-f', str(Path(__file__).parent / 'bin')], capture_output=True, text=True).stdout.split()
assert not left, f'Mock processes remain: {left}'
print('PASS: two persistent cycles, unchanged server PIDs, refreshed solver weights, frozen proposer,')
print('      invalid/missing candidates accounted for, no repairs, correct handoffs, resume, and cleanup.')
