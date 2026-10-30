"""Two-H100 candidate scoring and sleep/wake search runner.

Uses a completed solver checkpoint and a fixed local 4B proposer. The optional
benchmark replays saved candidates before two real search iterations.
Owned subprocesses are cleaned up; the allocation is never signalled.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from switch_router import BackendPool, start_router


class Experiment:
    def __init__(self):
        self.out = Path(os.environ['SWITCH_OUT'])
        self.out.mkdir(parents=True, exist_ok=True)
        self.processes = {}
        self.launch_counts = {}
        self.lock = threading.Lock()
        self.gpus = os.environ['LADDER_JOB_GPUS'].split(',')
        assert len(self.gpus) == 2, self.gpus
        base = int(os.environ['SOLVER_PORT'])
        self.ports = {'solver0': base + 3, 'solver1': base + 4,
                      'proposer': int(os.environ['PROPOSER_PORT'])}
        self.pool = BackendPool([self.url('solver0'), self.url('solver1')])
        self.checkpoint = Path(os.environ['SWITCH_CHECKPOINT']) / 'hf'
        assert (self.checkpoint / '.merge_complete').exists()
        self.reference = Path(os.environ['SWITCH_REFERENCE_RUN']) if os.environ.get('SWITCH_REFERENCE_RUN') else None
        self.config = json.loads(Path(os.environ['SWITCH_CONFIG']).read_text())
        self.config['models'][0]['batch_concurrency'] = 256
        self.router = start_router(base, self.pool)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.phase = 'initializing'
        self.stop_monitor = threading.Event()
        self.model_checks = {}

    def event(self, event, **fields):
        record = dict(time=time.time(), event=event, **fields)
        with self.lock:
            with (self.out / 'timing.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)

    def status(self, message):
        self.phase = message
        (self.out / 'status').write_text(message + '\n')
        self.event('status', message=message)

    def url(self, name):
        return f'http://127.0.0.1:{self.ports[name]}'

    def request(self, name, path, data=None, timeout=180):
        req = urllib.request.Request(self.url(name) + path, data=data,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            return json.loads(raw) if raw else None

    def wait_ready(self, name, timeout=900):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.processes[name].poll() is not None:
                raise RuntimeError(f'{name} exited: inspect server logs')
            try:
                self.request(name, '/v1/models', timeout=5)
                return
            except (OSError, ValueError):
                time.sleep(2)
        raise TimeoutError(f'{name} startup exceeded {timeout}s')

    def start(self, name):
        assert name not in self.processes
        proposer = name == 'proposer'
        model = os.environ['PROPOSER_MODEL_PATH'] if proposer else str(self.checkpoint)
        served = 'proposer' if proposer else os.environ['SOLVER_SERVED_NAME']
        cmd = ['vllm', 'serve', model, '--host', '127.0.0.1', '--port', str(self.ports[name]),
               '--served-model-name', served, '--tensor-parallel-size', '1',
               '--gpu-memory-utilization', '0.75', '--enable-sleep-mode',
               '--max-model-len', os.environ.get('PROPOSER_MAX_MODEL_LEN', '131072') if proposer else '32768',
               '--enable-prefix-caching', '--trust-remote-code', '--gdn-prefill-backend', 'triton']
        if proposer:
            cmd += ['--enable-auto-tool-choice', '--tool-call-parser', os.environ.get('TOOL_CALL_PARSER', 'qwen3_coder'),
                    '--reasoning-parser', os.environ.get('REASONING_PARSER', 'qwen3')]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.gpus[0 if name == 'solver0' else 1],
                   VLLM_SERVER_DEV_MODE='1')
        self.launch_counts[name] = self.launch_counts.get(name, 0) + 1
        logfile = self.out / f'{name}-{self.launch_counts[name]}.log'
        started = time.monotonic()
        with logfile.open('w') as log:
            self.processes[name] = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                   start_new_session=True)
        self.wait_ready(name)
        self.event('server_start', name=name, seconds=time.monotonic()-started,
                   pid=self.processes[name].pid, command=cmd)

    def stop(self, name):
        process = self.processes.pop(name, None)
        if process is None:
            return
        # start_new_session creates a private group. No pkill, no Slurm signals.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        # Reap lingering engine descendants in this invocation's process group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        time.sleep(2)

    def sleep(self, name):
        started = time.monotonic()
        self.request(name, '/sleep?level=1', data=b'')
        assert self.request(name, '/is_sleeping')['is_sleeping'] is True
        self.event('sleep', name=name, seconds=time.monotonic()-started)

    def wake(self, name):
        started = time.monotonic()
        self.request(name, '/wake_up', data=b'')
        assert self.request(name, '/is_sleeping')['is_sleeping'] is False
        self.event('wake', name=name, seconds=time.monotonic()-started)

    def check_model(self, name):
        """Deterministic short request checks engine usability before/after sleep."""
        payload = dict(model='proposer' if name == 'proposer' else os.environ['SOLVER_SERVED_NAME'],
                       messages=[dict(role='user', content='Reply with the word ready.')],
                       max_tokens=16, temperature=0, seed=42, stream=False)
        result = self.request(name, '/v1/chat/completions', json.dumps(payload).encode())
        assert result.get('choices'), result
        message = result['choices'][0]['message']
        digest = hashlib.sha256(json.dumps(message, sort_keys=True).encode()).hexdigest()
        previous = self.model_checks.setdefault(name, digest)
        self.event('model_check', name=name, response_sha256=digest,
                   matches_first=previous == digest, usage=result.get('usage'))
        if previous != digest:
            raise RuntimeError(f'{name} deterministic check changed after transition')

    def to_scoring(self, mode):
        started = time.monotonic()
        if mode == 'restart':
            self.stop('proposer')
            self.start('solver1')
        else:
            self.sleep('proposer')
            self.wake('solver1')
        self.check_model('solver1')
        self.pool.enable([0, 1])
        self.event('transition_to_scoring', mode=mode, seconds=time.monotonic()-started)

    def to_proposing(self, mode):
        started = time.monotonic()
        self.pool.drain(1)
        if mode == 'restart':
            self.stop('solver1')
            self.start('proposer')
        else:
            self.sleep('solver1')
            self.wake('proposer')
        self.check_model('proposer')
        self.event('transition_to_proposing', mode=mode, seconds=time.monotonic()-started)

    def monitor(self):
        while not self.stop_monitor.wait(5):
            record = dict(time=time.time(), phase=self.phase, router=self.pool.snapshot())
            try:
                record['gpu'] = subprocess.check_output(['nvidia-smi', '-i', ','.join(self.gpus),
                    '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                    text=True, timeout=5)
                for name in ('solver0', 'solver1'):
                    if name in self.processes:
                        with urllib.request.urlopen(self.url(name)+'/metrics', timeout=2) as r:
                            lines = r.read().decode().splitlines()
                            record[name] = [x for x in lines if x.startswith('vllm:') and
                                any(k in x.split('{')[0] for k in ('num_requests_running', 'num_requests_waiting',
                                    'generation_tokens_total', 'kv_cache_usage_perc', 'num_preemptions_total'))]
            except Exception as exc:
                record['monitor_error'] = str(exc)
            with (self.out/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')

    def validate_scores(self, logs, names):
        for name in names:
            values = list(Path(logs).glob(f'*/{name}/*/val.json'))
            assert len(values) == 1, (name, values)
            value = json.loads(values[0].read_text())
            assert value['num_examples'] == 256
            assert value['metadata']['assistant_token_budget'] == 8129

    def score(self, sweep, config, harnesses, logs, overlap=False, mode='sleep'):
        self.status(f'{self.phase.split(":")[0]}: candidate scoring ({mode}, overlap={overlap})')
        started = time.monotonic()
        before = self.pool.snapshot()
        transition = self.executor.submit(self.to_scoring, mode) if overlap else None
        if not overlap:
            self.to_scoring(mode)
        rows = []
        try:
            for name, path in harnesses:
                stage = time.monotonic()
                measured = sweep(copy.deepcopy(config), [(name, path)], logs, force=True)
                self.event('candidate_score', harness=name, seconds=time.monotonic()-stage, results=measured)
                if any(not ok for _, ok in measured):
                    raise RuntimeError(f'Candidate evaluation failed: {name}')
                self.validate_scores(logs, [name])
                rows.extend(measured)
        finally:
            if transition:
                transition.result()
        after = self.pool.snapshot()
        assert after['failed'] == before['failed'], 'Solver HTTP request failure'
        assert all(after['completed'][i] > before['completed'][i] for i in (0, 1)), 'Both GPUs must serve requests'
        self.check_model('solver0')
        self.event('scoring_with_transition', seconds=time.monotonic()-started, mode=mode,
                   overlap=overlap, before=before, after=after)
        self.to_proposing(mode)
        self.event('complete_round_trip', seconds=time.monotonic()-started, mode=mode, overlap=overlap)
        return rows

    def run_search(self, iterations=1):
        from meta_harness import meta_harness_chess_puzzle as mh
        self.status('search: initialize solver replicas and fixed proposer')
        self.start('solver0')
        self.start('proposer')
        self.check_model('solver0')
        self.check_model('proposer')
        threading.Thread(target=self.monitor, daemon=True).start()
        self.sleep('proposer')
        self.start('solver1')
        self.check_model('solver1')
        self.sleep('solver1')
        self.wake('proposer')
        pids = {name: proc.pid for name, proc in self.processes.items()}
        self.real_search(mh, iterations=iterations, baseline_two_gpu=True)
        assert all(self.processes[name].pid == pid for name, pid in pids.items())
        (self.out/'passed.json').write_text(json.dumps(dict(passed=True,
            search_iterations=iterations, candidates=3*iterations,
            puzzles_per_candidate=256, server_pids=pids), indent=2)+'\n')
        self.status('search passed; allocation retained')

    def run(self):
        from meta_harness import meta_harness_chess_puzzle as mh
        if self.reference is None:
            raise ValueError('Benchmark requires SWITCH_REFERENCE_RUN')
        self.status('switch experiment: initializing independent servers')
        self.start('solver0')
        self.start('proposer')
        stable_solver_pid = self.processes['solver0'].pid
        self.check_model('solver0')
        self.check_model('proposer')
        threading.Thread(target=self.monitor, daemon=True).start()
        candidates = [(n, self.reference/'harnesses'/n/'harness.py') for n in ('h1','h2','h3')]
        for _, path in candidates:
            assert path.is_file(), path
        self.event('inputs', checkpoint=str(self.checkpoint), gpu_memory_utilization=.75,
                   candidates={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in candidates},
                   config=self.config)
        self.phase = 'matched-restart'
        self.score(mh.run_sweep, self.config, candidates, self.out/'matched-restart', mode='restart')
        self.status('switch experiment: initialize sleeping second solver once')
        self.sleep('proposer')
        self.start('solver1')
        self.check_model('solver1')
        self.sleep('solver1')
        self.wake('proposer')
        for label, overlap in [('matched-sleep', False), ('matched-sleep-overlap', True)]:
            self.phase = label
            self.score(mh.run_sweep, self.config, candidates, self.out/label, overlap=overlap)
            assert self.processes['solver0'].pid == stable_solver_pid
        (self.out/'matched-passed.json').write_text(json.dumps({'passed':True, 'cases':3,
            'candidate_rollouts_per_case':768, 'solver0_pid':stable_solver_pid}, indent=2))
        self.real_search(mh)
        assert self.processes['solver0'].pid == stable_solver_pid
        (self.out/'passed.json').write_text(json.dumps({'passed':True, 'matched_cases':3,
            'real_search_iterations':2, 'solver0_pid':stable_solver_pid}, indent=2))
        self.status('switch experiment passed; allocation retained')

    def real_search(self, mh, iterations=2, baseline_two_gpu=False):
        self.status(f'real-search: {iterations} feedback-driven iterations')
        proxy_config = self.out/'litellm.json'
        proxy_config.write_bytes(subprocess.check_output([sys.executable,
            str(Path(os.environ['LADDER_HOME'])/'tools/make_litellm_config.py')]))
        with (self.out/'litellm.log').open('w') as log:
            self.processes['proxy'] = subprocess.Popen(['litellm','--config',str(proxy_config),
                '--host','127.0.0.1','--port',os.environ['LITELLM_PORT']], stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        end = time.monotonic()+120
        while True:
            try:
                with urllib.request.urlopen('http://127.0.0.1:'+os.environ['LITELLM_PORT']+'/health/liveliness',timeout=3):
                    break
            except OSError:
                if time.monotonic()>end or self.processes['proxy'].poll() is not None:
                    raise RuntimeError('Local proposer proxy did not start')
                time.sleep(2)
        run_name = os.environ.get('LADDER_CYCLE_SEARCH_NAME') or 'mh-switch-'+os.environ['SLURM_JOB_ID']+'-'+self.out.name
        sys.argv = [sys.argv[0], '--run-name',run_name,'--config',os.environ['SWITCH_CONFIG'],
            '--iterations',str(iterations),'--proposals-per-iter','3','--proposer-model','ladder-proposer',
            '--proposer-effort','','--propose-timeout','1200']
        args = mh.parse_args()
        run_dir = mh.get_run_dir(run_name).resolve()
        (self.out/'harness-run-dir').write_text(str(run_dir)+'\n')
        original_prompt, sweep = mh.render_task_prompt, mh.run_sweep
        mh.render_task_prompt = lambda iteration, names: (
            f'Your actual working directory is {run_dir}. Use relative paths from here. '
            'Write valid Python without backslash-escaped identifiers. '
            'Harness code must not import sys or modify sys.path. Allowed imports are '
            '__future__, collections, itertools, math, random, re, statistics.\n'+original_prompt(iteration,names))
        mh.PROPOSER_SYSTEM_PROMPT += f' Your working directory is exactly {run_dir}.'
        mh.PROMPT_ONLY = False
        def wrapped(config, harnesses, logs_dir, *, force=False):
            config = copy.deepcopy(config)
            if all(name == 'h0' for name,_ in harnesses):
                if baseline_two_gpu:
                    config['models'][0]['batch_concurrency'] = 256
                    return self.score(sweep,config,harnesses,logs_dir,overlap=False)
                config['models'][0]['batch_concurrency'] = 128
                results = sweep(config,harnesses,logs_dir,force=force)
                assert all(ok for _,ok in results), results
                self.validate_scores(logs_dir,['h0'])
                return results
            config['models'][0]['batch_concurrency'] = 256
            return self.score(sweep,config,harnesses,logs_dir,overlap=False)
        mh.run_sweep = wrapped
        mh.run_evolve(args)
        rows = [json.loads(x) for x in (run_dir/'logs/evolution_summary.jsonl').read_text().splitlines()]
        names = {r['harness'] for r in rows if r.get('status')=='evaluated'}
        assert names == {f'h{i}' for i in range(1,3*iterations+1)}, names
        self.validate_scores(run_dir/'logs', names)
        # Invalid generated code fails visibly. No unrecorded repair or retries.

    def close(self):
        self.stop_monitor.set()
        self.executor.shutdown(wait=True)
        with self.pool.condition:
            self.pool.closed = True
            self.pool.condition.notify_all()
        self.router.shutdown()
        self.router.server_close()
        for name in list(self.processes):
            self.stop(name)


if __name__ == '__main__':
    experiment = Experiment()
    try:
        if os.environ.get('SWITCH_MODE', 'search') == 'benchmark':
            experiment.run()
        else:
            experiment.run_search(iterations=int(os.environ.get('SWITCH_ITERATIONS', '1')))
    except BaseException as exc:
        experiment.event('failure', type=type(exc).__name__, message=str(exc))
        raise
    finally:
        experiment.close()
