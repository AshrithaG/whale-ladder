"""CPU worker fixtures exercise the production interpreter boundary."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

from pinned_harness_loader import load_harness

PROMPT_ONLY = False


def validate_candidate(run_dir, name):
    load_harness(run_dir / 'harnesses' / name / 'harness.py')
    if PROMPT_ONLY:
        raise ValueError('violates the prompt-only constraint')


def sweep(config, harnesses, logs, *, force=False):
    name, path = harnesses[0]
    module = load_harness(path)
    mode = config.get('candidate_modes', {}).get(name, config.get('mode', 'healthy'))
    if mode == 'memory_error':
        raise MemoryError('candidate allocation fixture')
    if mode == 'cpu_spin':
        while True: pass
    if mode == 'allocation_excess':
        data = bytearray(512 * 1024**2)
        time.sleep(30)
    if mode == 'rss_excess':
        data = bytearray(128 * 1024**2)
        time.sleep(30)
    if mode == 'output_flood':
        while True:
            os.write(1, b'o' * (24 * 1024))
            os.write(2, b'e' * (24 * 1024))
    if mode == 'sigkill':
        os.kill(os.getpid(), signal.SIGKILL)
    if mode == 'oversized_result':
        Path(sys.argv[2]).write_text(json.dumps(dict(status='ok', padding='x' * (2 * 1024**2))))
        os._exit(0)
    if mode == 'thread_settings':
        expected = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')
        assert all(os.environ[key] == '1' for key in expected)
    if mode == 'crash':
        os._exit(17)
    if mode == 'no_result':
        os._exit(0)
    if mode == 'malformed':
        return None
    if mode == 'deep_result':
        # Exit successfully after writing a result that fails parent decoding.
        Path(sys.argv[2]).write_text('[' * 2000 + '0' + ']' * 2000)
        os._exit(0)
    if mode == 'outstanding_request':
        # The parent's router retains this request after this worker dies.
        import threading
        def request():
            with urllib.request.urlopen(urllib.request.Request(config['router'] + '/v1/chat/completions',
                    data=b'{"stream":false}'), timeout=10) as response:
                response.read()
        for _ in range(config.get('outstanding_requests', 1)):
            threading.Thread(target=request, daemon=True).start()
        while True: time.sleep(1)
    if mode == 'descendants':
        child = subprocess.Popen([sys.executable, '-c',
            'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(90)'])
        Path(config['pid_file']).write_text(str(child.pid))
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(90)
    if module.parse_action:
        module.parse_action('<x>' + 'z' * 11353)
    if config.get('router'):
        for i in range(config.get('requests', 4)):
            with urllib.request.urlopen(urllib.request.Request(config['router'] + '/v1/chat/completions',
                    data=b'{"stream":false}'), timeout=5) as response:
                response.read()
    value = dict(num_examples=2, success_rate=.5, mean_turn_count=3.25, avg_reward=.5,
                 metadata=dict(assistant_token_budget=8129))
    if mode == 'partial':
        value['num_examples'] = 1
    if mode == 'invalid_metric':
        value.pop('success_rate')
    if mode == 'invalid_document':
        value = []
    if mode == 'huge_metric':
        value['success_rate'] = 10 ** 400
    if mode == 'metric_override':
        value[config['metric']] = float(config['metric_value'])
    if mode == 'budget_override':
        value['metadata']['assistant_token_budget'] = config['eval']['assistant_token_budget']
    if mode in ('bad_applied', 'deep_applied', 'oversized_applied'):
        applied = Path(sys.argv[2]).with_name('applied-policy.json')
        applied.write_text('[]' if mode == 'bad_applied' else
            '[' * 2000 + '0' + ']' * 2000 if mode == 'deep_applied' else 'x' * (2 * 1024**2))
    if mode == 'oversized_score':
        value['padding'] = 'x' * (2 * 1024**2)
    model = str(config.get('models', [{'model': 'mock'}])[0].get('model', 'mock')).split('/')[-1].lower()
    out = logs / 'p' / name / model
    out.mkdir(parents=True)
    (out / 'val.json').write_text('[' * 2000 + '0' + ']' * 2000 if mode == 'deep_score' else json.dumps(value))
    return [(f'p/{name}/{model}_seed42', mode != 'failed')]
