"""Search servers retained between training phases on two GPUs.

Ported from the PSC development implementation. Candidate generation and
validation remain in PR4's search loop; there is no repair/retry path here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request

from two_gpu_search import Experiment


def checkpoint_signature(path):
    path = Path(path).resolve()
    if not (path / '.merge_complete').is_file():
        raise ValueError(f'Checkpoint merge incomplete: {path}')
    shards = list(path.glob('*.safetensors'))
    if not shards or any(p.stat().st_size == 0 for p in shards):
        raise ValueError('Missing or empty checkpoint weights')
    index = path / 'model.safetensors.index.json'
    if index.exists():
        for name in set(json.loads(index.read_text())['weight_map'].values()):
            if Path(name).name != name or not (path / name).is_file():
                raise ValueError(f'Missing checkpoint shard: {name}')
    result = {}
    for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json',
                 'generation_config.json', 'special_tokens_map.json', 'chat_template.jinja'):
        p = path / name
        if name in ('config.json', 'tokenizer_config.json') and not p.is_file():
            raise ValueError(f'Missing {name}')
        result[name] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    return result


def memory_guard(gpus, limit_mib=8192):
    raw = subprocess.check_output(['nvidia-smi', '-i', ','.join(gpus),
        '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], text=True)
    rows = [tuple(int(v.strip()) for v in line.split(',')) for line in raw.splitlines()]
    if len(rows) != 2 or any(used > min(limit_mib, total * .10) for used, total in rows):
        raise RuntimeError(f'Sleeping servers leave insufficient training memory: {rows}')
    return rows


def validate_sleep_receipt(path):
    receipt = json.loads(Path(path).read_text())
    run_id = os.environ.get('SLURM_JOB_ID') or os.environ['LADDER_PERSISTENT_RUN_ID']
    gpus = os.environ['LADDER_JOB_GPUS'].split(',')
    if receipt['job_id'] != run_id or receipt['gpus'] != gpus or len(gpus) != 2:
        raise RuntimeError('Sleep receipt belongs to another run or GPU assignment')
    if not 0 <= time.time() - receipt['time'] < 600:
        raise RuntimeError('Stale sleep receipt')
    backend = receipt.get('backend', 'local')
    if backend not in ('local', 'api') or backend != os.environ.get('PROPOSER_BACKEND', 'local'):
        raise RuntimeError('Sleep receipt proposer backend differs from this run')
    expected_ports = {'solver0', 'solver1'} | ({'proposer'} if backend == 'local' else set())
    if set(receipt['ports']) != expected_ports:
        raise RuntimeError('Incomplete sleep receipt')
    for port in receipt['ports'].values():
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError('Invalid server port')
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/is_sleeping', timeout=10) as r:
            if json.load(r).get('is_sleeping') is not True:
                raise RuntimeError('Search server is awake; training refused')
    return memory_guard(gpus)


class PersistentSearch(Experiment):
    def __init__(self):
        super().__init__()
        self.proposer_tracks_solver = os.environ.get('PROPOSER_MODEL_PATH') == '@checkpoint'
        if not self.local_proposer:
            self.ports.pop('proposer')

    def initialize(self):
        if self.replicas != 2:
            raise ValueError('Persistence requires two solver replicas')
        self.signature = checkpoint_signature(self.checkpoint)
        self.pool.enable([])
        self.start('solver0')
        self.check_model('solver0')
        self.digest = self.weight_digest('solver0')
        self.proposer_digest = None
        if self.local_proposer:
            self.start('proposer')
            self.check_model('proposer')
            self.proposer_digest = self.weight_digest('proposer')
            if self.proposer_tracks_solver and self.proposer_digest != self.digest:
                raise RuntimeError('Self proposer does not match the incoming solver checkpoint')
            self.sleep('proposer')
        self.start('solver1')
        self.check_model('solver1')
        if self.weight_digest('solver1') != self.digest:
            raise RuntimeError('Initial solver replicas have different weights')
        if self.local_proposer:
            self.sleep('solver1')
            self.wake('proposer')
        self.pids = {name: self.processes[name].pid for name in self.ports}
        self.server_log_directory = str(self.out)
        self.parked = False
        self.pool.enable([0] if self.local_proposer else [0, 1])
        threading.Thread(target=self.monitor, daemon=True).start()

    def weight_digest(self, name):
        result = self.request(name, '/collective_rpc', json.dumps({
            'method': 'compute_weight_checksums', 'timeout': 180}).encode(), timeout=200)
        values = result.get('results') if isinstance(result, dict) else None
        if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict) or not values[0]:
            raise RuntimeError('Expected full weight checksums from one TP=1 worker')
        digest = hashlib.sha256(json.dumps(values[0], sort_keys=True).encode()).hexdigest()
        with (self.out / 'weight-checksums.jsonl').open('a') as log:
            log.write(json.dumps(dict(time=time.time(), name=name,
                pid=self.processes[name].pid, digest=digest, checksums=values[0])) + '\n')
        self.event('weight_digest', name=name, digest=digest, parameters=len(values[0]))
        return digest

    def verify_pids(self):
        for name, pid in self.pids.items():
            process = self.processes[name]
            if process.pid != pid or process.poll() is not None:
                raise RuntimeError(f'Persistent server changed or exited: {name}')

    def park(self, receipt):
        self.pool.enable([])
        for i in (0, 1):
            self.pool.drain(i)
        self.verify_pids()
        for name in self.ports:
            if not self.request(name, '/is_sleeping')['is_sleeping']:
                self.sleep(name)
        usage = memory_guard(self.gpus)
        self.parked = True
        Path(receipt).write_text(json.dumps(dict(time=time.time(),
            job_id=os.environ.get('SLURM_JOB_ID') or os.environ['LADDER_PERSISTENT_RUN_ID'],
            gpus=self.gpus, ports=self.ports, pids=self.pids, memory=usage,
            backend='local' if self.local_proposer else 'api'), indent=2) + '\n')
        self.event('parked_for_training', memory=usage)

    def refresh(self, checkpoint):
        self.pool.enable([])
        checkpoint = Path(checkpoint).resolve()
        if not self.parked:
            raise RuntimeError('Park servers before replacing weights')
        if checkpoint_signature(checkpoint) != self.signature:
            raise ValueError('Checkpoint architecture/tokenizer/defaults changed; restart required')
        self.verify_pids()
        memory_guard(self.gpus)
        started = time.monotonic()
        def reload(name):
            self.request(name, '/wake_up?tags=weights', data=b'')
            self.request(name, '/collective_rpc', json.dumps({'method': 'reload_weights',
                'timeout': 300, 'kwargs': {'weights_path': str(checkpoint)}}).encode(), timeout=330)
            if self.request(name, '/reset_prefix_cache', data=b'').get('success') is not True:
                raise RuntimeError('Could not invalidate old prefix cache')
            digest = self.weight_digest(name)
            self.wake(name)
            self.model_checks.pop(name, None)
            self.check_model(name)
            return digest

        digests = []
        for name in ('solver0', 'solver1'):
            digests.append(reload(name))
            if name == 'solver1' and self.local_proposer:
                self.sleep(name)
        if digests[0] != digests[1] or digests[0] == self.digest:
            raise RuntimeError('Reloaded replicas disagree or still contain old weights')
        if self.local_proposer:
            if self.proposer_tracks_solver:
                proposer_digest = reload('proposer')
                if proposer_digest != digests[0]:
                    raise RuntimeError('Reloaded self proposer differs from solver weights')
                self.proposer_digest = proposer_digest
            else:
                self.wake('proposer')
                self.check_model('proposer')
                if self.weight_digest('proposer') != self.proposer_digest:
                    raise RuntimeError('Fixed proposer weights changed')
        self.checkpoint, self.digest, self.parked = checkpoint, digests[0], False
        self.event('checkpoint_refreshed', checkpoint=str(checkpoint), pids=self.pids,
                   digest=self.digest, proposer_digest=self.proposer_digest,
                   proposer_tracks_solver=self.proposer_tracks_solver,
                   seconds=time.monotonic() - started)
        self.pool.enable([0] if self.local_proposer else [0, 1])

    def search(self, mh, iterations):
        attrs = ('render_task_prompt', 'run_sweep', 'PROPOSER_SYSTEM_PROMPT', 'PROMPT_ONLY')
        saved = {key: getattr(mh, key) for key in attrs}
        argv = sys.argv[:]
        try:
            return self.real_search(mh, iterations=iterations, baseline_two_gpu=True)
        finally:
            for key, value in saved.items():
                setattr(mh, key, value)
            sys.argv = argv
            self.stop('proxy')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-sleep-receipt', required=True)
    print(json.dumps(validate_sleep_receipt(parser.parse_args().check_sleep_receipt)))
