"""Shared cycle controller for all proposer arms and the training-only baseline.

Server persistence is a lifecycle option, not a separate experiment loop.
Finished phases require matching inputs; unfinished work is preserved and rerun.
No candidate repairs, automatic service restarts, or Slurm job mutations.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import uuid

from persistent_search import PersistentSearch


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def timed(state, phase):
    """Sequential controller boundaries; inner trainer/search timers can overlap."""
    started, clock = time.time(), time.monotonic()
    record = dict(phase=phase, started_unix=started, status='running')
    temporary = state / 'current-phase.tmp'
    write(temporary, record)
    temporary.replace(state / 'current-phase.json')
    try:
        yield
    except BaseException as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    else:
        record['status'] = 'completed'
    finally:
        record.update(ended_unix=time.time(), elapsed_seconds=time.monotonic() - clock)
        with (state / 'phase-timing.jsonl').open('a') as log:
            log.write(json.dumps(record) + '\n')
        write(temporary, record)
        temporary.replace(state / 'current-phase.json')


def phase_ok(path):
    receipt = path / 'passed.json'
    return receipt.is_file() and json.loads(receipt.read_text()).get('passed') is True


def set_aside(path):
    if path.exists() and not phase_ok(path):
        destination = path.with_name(path.name + f'.unfinished-{time.time_ns()}')
        path.rename(destination)
        return destination


def command(args, env=None):
    """Forward interruption to this command's private process group."""
    child = subprocess.Popen(args, env=env, start_new_session=True)
    try:
        code = child.wait()
    except BaseException:
        try:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)
        except ProcessLookupError:
            pass
        raise
    if code:
        raise subprocess.CalledProcessError(code, args)


def validate_mode(arm):
    backends = {'rsft_only': '', 'whale_deepseek': 'api', 'whale_qwen27b': 'local',
                'whale_qwen4b': 'local', 'whale_self': 'local'}
    if arm not in backends or os.environ.get('PROPOSER_BACKEND', '') != backends[arm]:
        raise ValueError('Arm and proposer backend disagree')
    if arm == 'rsft_only':
        return
    if len(os.environ['LADDER_JOB_GPUS'].split(',')) != 2:
        raise ValueError('Persistent mode requires exactly two assigned GPUs')
    if (int(os.environ.get('HYBRID_NUM_GPUS', '2')) != 2
            or int(os.environ.get('LADDER_SOLVER_REPLICAS', '2')) != 2
            or os.environ.get('LADDER_SHARE_ONE_GPU') == '1'):
        raise ValueError('Persistent mode requires two training ranks and two solver replicas')
    expected = os.environ.get('LADDER_GPU_CHECK', 'H100')
    if expected != 'skip':
        names = subprocess.check_output(['nvidia-smi', '-i', os.environ['LADDER_JOB_GPUS'],
            '--query-gpu=name', '--format=csv,noheader'], text=True).splitlines()
        if len(names) != 2 or any(expected not in name for name in names):
            raise ValueError(f'Persistent profile requires two {expected} GPUs: {names}')


def proposer_identity(arm):
    if arm == 'rsft_only':
        return dict(policy='none')
    if arm == 'whale_self':
        return dict(policy='follow_solver', model=os.environ['SOLVER_MODEL'],
                    revision=os.environ['SOLVER_MODEL_REVISION'])
    if arm == 'whale_deepseek':
        return dict(policy='api', provider=os.environ['DEEPSEEK_PROVIDER'],
                    model=os.environ['FIREWORKS_DEEPSEEK_MODEL'] if os.environ['DEEPSEEK_PROVIDER'] == 'fireworks'
                    else os.environ['DEEPSEEK_MODEL'])
    prefix = 'QWEN27B' if arm == 'whale_qwen27b' else 'WEAK_PROPOSER'
    return dict(policy='frozen', model=os.environ[prefix + '_MODEL'],
                revision=os.environ[prefix + '_REVISION'])


def verify_inputs(marker, expected):
    if not marker.is_file() or json.loads(marker.read_text()) != expected:
        raise ValueError('Completed phase has different or missing incoming state')


def run(arm, profile):
    if os.environ.get('LADDER_PERSISTENT_SEARCH', '0') not in ('0', '1'):
        raise ValueError('LADDER_PERSISTENT_SEARCH must be 0 or 1')
    persistent = os.environ.get('LADDER_PERSISTENT_SEARCH', '0') == '1' and arm != 'rsft_only'
    if persistent:
        validate_mode(arm)
    home = Path(os.environ['LADDER_HOME'])
    cycles = int(os.environ['CYCLES'])
    steps = int(os.environ['STEPS_PER_CYCLE'])
    iterations = int(os.environ['SEARCH_ITERS'])
    if cycles < 1 or not 1 <= steps <= 99 or not 1 <= iterations <= 6:
        raise ValueError('Invalid cycle/training/search schedule')
    name = f"{arm}-{profile}-s{os.environ['SEED']}"
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', name):
        raise ValueError('Invalid experiment name')
    state = Path(os.environ['OUT_ROOT']) / 'cycles' / name
    state.mkdir(parents=True, exist_ok=True)
    plan = dict(controller_version=1, arm=arm, persistent_search=persistent, cycles=cycles, steps_per_cycle=steps,
        seed=int(os.environ['SEED']), data_fingerprint=os.environ.get('CHESS_DATA_FINGERPRINT'),
        search_iterations=iterations, train_batch=int(os.environ.get('HYBRID_TRAIN_BATCH', '256')),
        rollout_n=int(os.environ.get('HYBRID_ROLLOUT_N', '8')),
        evaluation_puzzles=int(os.environ.get('HYBRID_EVAL_PUZZLES', '256')),
        evaluation_attempts=int(os.environ.get('HYBRID_EVAL_ATTEMPTS', '8')),
        proposals=int(os.environ.get('PROPOSALS_PER_ITER', '3')),
        scoring_puzzles=int(os.environ.get('LADDER_MH_LIMIT', '256')),
        solver=os.environ['SOLVER_MODEL'], solver_revision=os.environ['SOLVER_MODEL_REVISION'],
        proposer=proposer_identity(arm),
        optimizer_policy='model+extra resume; optimizer resets between phases')
    plan_path = state / 'cycle-plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Saved cycle plan differs; use a new output root for a changed experiment')
    if not plan_path.exists() and any(state.glob('c*-train')):
        raise ValueError('Existing outputs were produced by another runner; use a new output root')
    write(plan_path, plan)
    (state / 'passed.json').unlink(missing_ok=True)
    invocation = uuid.uuid4().hex[:12]
    os.environ['LADDER_PERSISTENT_RUN_ID'] = f'local{os.getpid()}'
    job_tag = os.environ.get('SLURM_JOB_ID', os.environ['LADDER_PERSISTENT_RUN_ID'])
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=home, text=True, capture_output=True)
    constraint = os.environ.get('INSTALL_CONSTRAINTS')
    provenance = dict(invocation=invocation, job_id=job_tag, started_utc=time.strftime('%FT%TZ', time.gmtime()),
        revision=revision.stdout.strip() if revision.returncode == 0 else None,
        constraints_sha256=sha(constraint) if constraint else None,
        source_sha256={str(p.relative_to(home)): sha(p) for folder in ('tools', 'ladder', 'patches', 'setup')
            for p in sorted((home / folder).rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.sh', '.env', '.patch', '.txt') and p.name != 'secrets.env'},
        execution_mode='mock' if os.environ.get('LADDER_GPU_CHECK') == 'skip' else 'real')
    write(state / f'provenance-{invocation}.json', provenance)
    harness = Path(os.environ['BASE_HARNESS'])
    previous = None
    engine = None
    # WHALE's benchmark config contains paths relative to the chess domain,
    # just as in search_two_gpu.sh. Keep that contract for the in-process loop.
    os.chdir(os.environ['DOMAIN_DIR'])
    mh = None
    if arm != 'rsft_only':
        from meta_harness import meta_harness_chess_puzzle as mh

    def status(text):
        (state / 'status').write_text(text + '\n')
        with (state / 'cycles.log').open('a') as log:
            log.write(f"[cycles {time.strftime('%FT%TZ', time.gmtime())}] {text}\n")
        print(text, flush=True)

    def train(directory, mode, checkpoint=None, receipt=None):
        env = dict(os.environ, HYBRID_RUN_NAME=name if mode == 'train' else name + '-test',
                   HYBRID_MODE=mode, HYBRID_TRAINING_STEPS=str(steps), HARNESS_PATH=str(harness))
        for key in ('HYBRID_RESUME_CHECKPOINT', 'HYBRID_SOURCE_CHECKPOINT', 'HYBRID_PERSISTENT_SLEEP_RECEIPT'):
            env.pop(key, None)
        if checkpoint:
            env['HYBRID_RESUME_CHECKPOINT' if mode == 'train' else 'HYBRID_SOURCE_CHECKPOINT'] = str(checkpoint)
        if receipt:
            env['HYBRID_PERSISTENT_SLEEP_RECEIPT'] = str(receipt)
        with timed(state, directory.name):
            command(['bash', str(home / 'ladder/hybrid_test.sh'), str(directory)], env)
            if not phase_ok(directory):
                raise RuntimeError(f'Launcher returned without a pass receipt: {directory}')

    def preserve_unfinished(directory):
        destination = set_aside(directory)
        if destination:
            status(f'moved unfinished {directory.name} aside to {destination.name}')

    def persistent_search(search, checkpoint, harness, cycle):
        nonlocal engine
        (search / 'state/claude-config').mkdir(parents=True)
        with timed(state, f'c{cycle}-merge-checkpoint'):
            command(['bash', '-c', 'source "$LADDER_HOME/ladder/lib.sh"; merge_ckpt "$1"', 'merge', str(checkpoint)])
        os.environ.update(SWITCH_OUT=str(search), RUN_OUT=str(search), SWITCH_CHECKPOINT=str(checkpoint),
            SWITCH_CONFIG=str(search / 'state/mh_config.json'),
            CLAUDE_CONFIG_DIR=str(search / 'state/claude-config'),
            LADDER_COST_LEDGER=str(search / 'proposer_cost.jsonl'),
            LADDER_CYCLE_SEARCH_NAME=f'mh-persistent-j{job_tag}-{name}-c{cycle}-{invocation}',
            BASELINE_HARNESS_OVERRIDE=str(harness), CHESS_PUZZLE_PROPOSER_RETRIES='1')
        with timed(state, f'c{cycle}-prepare-search-config'):
            command(['python', str(home / 'tools/make_mh_config.py'),
                     str(Path(os.environ['DOMAIN_DIR']) / 'meta_harness/config-chess-puzzle.json'),
                     os.environ['SWITCH_CONFIG']])
        status(f'cycle {cycle}/{cycles}: search')
        if engine is None:
            engine = PersistentSearch()
            with timed(state, f'c{cycle}-search-server-startup'):
                engine.initialize()
        else:
            engine.out = search
            engine.config = json.loads(Path(os.environ['SWITCH_CONFIG']).read_text())
            if not engine.parked:
                with timed(state, f'c{cycle}-park-search-servers'):
                    engine.park(state / f'c{cycle}-sleep.json')
            with timed(state, f'c{cycle}-refresh-search-checkpoint'):
                engine.refresh(checkpoint / 'hf')
        with timed(state, f'c{cycle}-search'):
            outcome = engine.search(mh, iterations)
        engine.verify_pids()
        write(search / 'passed.json', dict(passed=True, search_iterations=iterations,
            candidates_requested=plan['proposals'] * iterations, puzzles_per_candidate=plan['scoring_puzzles'],
            server_pids=engine.pids, server_log_directory=engine.server_log_directory,
            solver_weight_digest=engine.digest, proposer_weight_digest=engine.proposer_digest,
            proposer_policy=plan['proposer']['policy'], persistent_search=True, **outcome))

    try:
        for cycle in range(1, cycles + 1):
            train_dir = state / f'c{cycle}-train'
            used = state / f'c{cycle}-trained-with.py'
            inputs = dict(harness_sha256=sha(harness), resume_checkpoint=str(previous) if previous else None)
            input_marker = state / f'c{cycle}-train-inputs.json'
            if phase_ok(train_dir):
                verify_inputs(input_marker, inputs)
                if sha(used) != inputs['harness_sha256']:
                    raise ValueError('Completed training phase has different incoming harness')
                status(f'cycle {cycle}: training already passed')
            else:
                preserve_unfinished(train_dir)
                train_dir.mkdir(parents=True)
                shutil.copyfile(harness, used)
                write(input_marker, inputs)
                receipt = state / f'c{cycle}-sleep.json' if engine else None
                if engine:
                    with timed(state, f'c{cycle}-park-search-servers'):
                        engine.park(receipt)
                status(f'cycle {cycle}/{cycles}: training')
                original = harness
                harness = used
                try:
                    train(train_dir, 'train', previous, receipt)
                finally:
                    harness = original
            checkpoint = train_dir / 'checkpoints' / name / f'global_step_{cycle * steps}'
            if not (checkpoint / 'actor').is_dir() or not (checkpoint / 'data.pt').is_file():
                raise RuntimeError(f'Missing resume checkpoint: {checkpoint}')
            if arm == 'rsft_only':
                previous = checkpoint
                status(f'cycle {cycle}: training completed; no harness search')
                continue
            search = state / f'{arm}-c{cycle}-search'
            search_inputs = dict(checkpoint=str(checkpoint), harness_sha256=sha(harness))
            search_marker = state / f'c{cycle}-search-inputs.json'
            if phase_ok(search):
                verify_inputs(search_marker, search_inputs)
                status(f'cycle {cycle}: search already passed')
            else:
                preserve_unfinished(search)
                write(search_marker, search_inputs)
                if not persistent:
                    env = dict(os.environ, ARM=arm, BASELINE_HARNESS_OVERRIDE=str(harness))
                    with timed(state, f'c{cycle}-search'):
                        command(['bash', str(home / 'ladder/search_two_gpu.sh'), str(search),
                                 str(checkpoint), str(iterations)], env)
                        if not phase_ok(search):
                            raise RuntimeError(f'Search returned without a pass receipt: {search}')
                else:
                    persistent_search(search, checkpoint, harness, cycle)
            run_dir = Path((search / 'harness-run-dir').read_text().strip())
            accepted = (run_dir / 'logs/accepted_harness.txt').read_text().strip()
            if not re.fullmatch(r'h[0-9]+', accepted):
                raise ValueError('Invalid accepted harness name')
            mh.validate_candidate(run_dir, accepted)
            selected = state / f'c{cycle}-accepted.py'
            shutil.copyfile(run_dir / 'harnesses' / accepted / 'harness.py', selected)
            harness, previous = selected, checkpoint
            status(f'cycle {cycle}: completed; accepted {accepted}')
        # Evaluation uses the shared training launcher. Release resident search
        # processes first so its ordinary idle-GPU check remains applicable.
        if engine:
            with timed(state, 'search-server-shutdown'):
                engine.close()
            engine = None
        final = state / 'final-test'
        shutil.copyfile(harness, state / 'final-harness.py')
        final_inputs = dict(checkpoint=str(previous), harness_sha256=sha(harness))
        final_marker = state / 'final-inputs.json'
        if phase_ok(final):
            verify_inputs(final_marker, final_inputs)
        else:
            preserve_unfinished(final)
            write(final_marker, final_inputs)
            status('final held-out test')
            train(final, 'eval', previous)
        command(['python', str(home / 'tools/summarize_cycles.py'), str(state), '--arm', arm, '--profile', profile])
        result = json.loads((state / 'results.json').read_text())
        if result['problems']:
            raise RuntimeError(f"Run summary reports problems: {result['problems']}")
        status('cycles and final test passed')
        write(state / 'passed.json', dict(passed=True, controller_version=1, persistent_search=persistent, cycles=cycles, invocation=invocation))
    except BaseException as exc:
        write(state / f'failure-{invocation}.json', dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        if engine:
            engine.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True)
    parser.add_argument('--profile', required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    run(args.arm, args.profile)


if __name__ == '__main__':
    main()
