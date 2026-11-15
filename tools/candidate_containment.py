"""Killable fresh-interpreter execution; evidence lives outside scored logs."""
import json
import math
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid

JSON_LIMIT_BYTES = 1024 * 1024
SAMPLE_SECONDS = .05
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')


class CandidateRejected(ValueError):
    def __init__(self, outcome):
        self.outcome = outcome
        self.status = outcome['status']
        super().__init__(outcome['reason'])


class WorkerSetupError(RuntimeError):
    """Resource policy/monitoring failure is fatal to the phase, not a candidate."""
    fatal_worker_setup = True


class ResourceExceeded(ValueError):
    def __init__(self, reason, **details):
        self.details = details
        super().__init__(reason)


def timeout_setting(name, default):
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return value


def _positive_integer(name, default):
    try:
        value = int(os.environ.get(name, str(default)))
    except (ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a positive integer') from exc
    if not 0 < value < sys.maxsize:
        raise ValueError(f'{name} must be a positive integer below {sys.maxsize}')
    return value


def resource_policy(action):
    if action not in ('validate', 'score'):
        raise ValueError('Unknown candidate action')
    cpu = _positive_integer('LADDER_CANDIDATE_' + ('VALIDATION' if action == 'validate' else 'SCORE')
                            + '_CPU_SECONDS', 10 if action == 'validate' else 120)
    linux = sys.platform.startswith('linux')
    return dict(version=1, cpu_seconds=cpu, cpu_hard_seconds=cpu + 1,
        address_space_bytes=_positive_integer('LADDER_CANDIDATE_AS_BYTES', 32 * 1024**3),
        rss_bytes=_positive_integer('LADDER_CANDIDATE_RSS_BYTES', 8 * 1024**3),
        output_bytes=_positive_integer('LADDER_CANDIDATE_OUTPUT_BYTES', 16 * 1024**2),
        json_bytes=JSON_LIMIT_BYTES, rss_sample_seconds=SAMPLE_SECONDS, native_threads=1,
        enforcement=dict(cpu='RLIMIT_CPU', address_space='RLIMIT_AS' if linux else 'unavailable',
                         rss='sampled Linux /proc worker RSS' if linux else 'unavailable',
                         output='parent bounds combined stdout/stderr pipe'))


def read_json_bounded(path):
    """Never read more than one MiB plus the oversize sentinel into the parent."""
    path = Path(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK), 'rb') as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('JSON artifact must be a regular file')
        size = info.st_size
        if size > JSON_LIMIT_BYTES:
            raise ResourceExceeded('JSON artifact exceeds 1 MiB', kind='json',
                limit_bytes=JSON_LIMIT_BYTES, observed_bytes=size, artifact=str(path))
        raw = source.read(JSON_LIMIT_BYTES + 1)
    if len(raw) > JSON_LIMIT_BYTES:
        raise ResourceExceeded('JSON artifact exceeds 1 MiB', kind='json',
            limit_bytes=JSON_LIMIT_BYTES, observed_bytes=len(raw), artifact=str(path))
    return json.loads(raw)


def evidence_dir(logs, name, phase):
    logs = Path(logs).resolve()
    path = logs.parent / (logs.name + '-candidate-evidence') / f'{name}-{phase}-{uuid.uuid4().hex}'
    path.mkdir(parents=True)
    return path


def quarantine(logs, name, destination):
    """Move old/partial scores out of every frontier scanner's search tree."""
    for source in Path(logs).glob(f'*/{name}'):
        if source.is_dir():
            target = Path(destination) / 'quarantine' / source.parent.name / name
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)


def _signal_group(pid, sig):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def _sample_worker(pid, usage):
    """Linux RSS is sampled, not a hard resident-memory ceiling."""
    if not sys.platform.startswith('linux'):
        return
    try:
        status = Path(f'/proc/{pid}/status').read_text()
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise WorkerSetupError(f'Linux resource sampling failed: {exc}') from exc
    rss = next((int(line.split()[1]) * 1024 for line in status.splitlines()
                if line.startswith('VmRSS:')), 0)
    usage['rss_bytes'] = rss
    usage['peak_rss_bytes'] = max(usage['peak_rss_bytes'], rss)
    virtual = next((int(line.split()[1]) * 1024 for line in status.splitlines()
                    if line.startswith('VmPeak:')), 0)
    usage['peak_virtual_bytes'] = max(usage['peak_virtual_bytes'], virtual)
    # stat starts at field 3 after the command. utime/stime are fields 14/15
    # and already include the process's threads, excluding HTTP wait time.
    usage['cpu_seconds'] = max(usage['cpu_seconds'],
                               (int(stat[11]) + int(stat[12])) / os.sysconf('SC_CLK_TCK'))


def run_worker(request, directory, timeout):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Worker timeout must be positive and finite')
    policy = resource_policy(request['action'])
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    request = dict(request, resource_policy=policy,
                   python_path=[str(Path(p or '.').resolve()) for p in sys.path])
    request_path = directory / 'request.json'
    result_path = directory / 'result.json'
    started = time.monotonic()
    previous = None
    child = None
    usage = dict(cpu_seconds=0., rss_bytes=0, peak_rss_bytes=0, peak_virtual_bytes=0,
                 output_bytes=0, captured_output_bytes=0)
    outcome = None
    applied_policy = None
    phase = 'launch'

    def rejected(status, reason, **details):
        return dict(status=status, reason=reason, **details)

    def write_failure(status, reason, **details):
        failure = dict(status=status, reason=reason, resource_policy=policy,
                       applied_policy=applied_policy, usage=usage, evidence=str(directory),
                       seconds=time.monotonic() - started, **details)
        (directory / 'outcome.json').write_text(json.dumps(failure, indent=2) + '\n')

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    try:
        if sys.platform.startswith('linux') and not Path('/proc/self/status').is_file():
            raise WorkerSetupError('Linux RSS watchdog requires readable /proc')
        request_path.write_text(json.dumps(request) + '\n')
        if threading.current_thread() is threading.main_thread():
            previous = signal.signal(signal.SIGTERM, interrupted)
        env = dict(os.environ, **{key: '1' for key in THREAD_ENV})
        with (directory / 'worker.log').open('wb') as log, selectors.DefaultSelector() as selector:
            child = subprocess.Popen([sys.executable, str(Path(__file__).with_name('candidate_worker.py')),
                str(request_path), str(result_path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                start_new_session=True, env=env)
            os.set_blocking(child.stdout.fileno(), False)
            selector.register(child.stdout, selectors.EVENT_READ)
            phase = 'monitor'
            next_sample = 0.
            deadline = started + timeout
            while True:
                now = time.monotonic()
                if now >= next_sample:
                    _sample_worker(child.pid, usage)
                    next_sample = now + SAMPLE_SECONDS
                    if usage['peak_rss_bytes'] > policy['rss_bytes']:
                        outcome = rejected('rejected_resource_limit', 'Worker RSS exceeded sampled limit',
                            resource=dict(kind='rss', limit_bytes=policy['rss_bytes'],
                                          observed_bytes=usage['peak_rss_bytes'], enforcement='sampled'))
                        break
                if now >= deadline:
                    outcome = rejected('rejected_timeout', f"{request['action']} exceeded {timeout:g}s")
                    break
                for key, _ in selector.select(min(SAMPLE_SECONDS, max(0, deadline - now))):
                    raw = os.read(key.fd, 65536)
                    if not raw:
                        selector.unregister(key.fileobj)
                        continue
                    usage['output_bytes'] += len(raw)
                    captured = raw[:max(0, policy['output_bytes'] - usage['captured_output_bytes'])]
                    log.write(captured)
                    usage['captured_output_bytes'] += len(captured)
                    if usage['output_bytes'] > policy['output_bytes']:
                        outcome = rejected('rejected_resource_limit', 'Combined stdout/stderr exceeded limit',
                            resource=dict(kind='output', limit_bytes=policy['output_bytes'],
                                          observed_bytes=usage['output_bytes']))
                        break
                if outcome:
                    break
                if child.poll() is not None:
                    # A descendant must not keep this pipe open after the
                    # interpreter exits; it belongs to this private group.
                    _signal_group(child.pid, signal.SIGKILL)
                    if not selector.get_map():
                        break
        phase = 'read_results'
        marker = directory / 'execution-started.json'
        execution_started = marker.is_file() and read_json_bounded(marker) == {'started': True}
        if not execution_started:
            reason = 'Worker startup failed before candidate execution/applied-policy handshake'
            if result_path.is_file():
                setup_result = read_json_bounded(result_path)
                if isinstance(setup_result, dict) and setup_result.get('status') == 'failed_resource_setup':
                    reason = str(setup_result.get('reason', reason))
            if child.returncode:
                reason += f' (exit code {child.returncode})'
            elif outcome:
                reason += ': ' + outcome['reason']
            raise WorkerSetupError(reason)
        applied = directory / 'applied-policy.json'
        if applied.is_file():
            receipt = read_json_bounded(applied)
            if (not isinstance(receipt, dict)
                    or receipt.get('cpu_rlimit') != [policy['cpu_seconds'], policy['cpu_hard_seconds']]
                    or receipt.get('core_rlimit') != [0, 0]
                    or (sys.platform.startswith('linux') and receipt.get('address_space_rlimit')
                        != [policy['address_space_bytes'], policy['address_space_bytes']])):
                raise ValueError('Malformed or changed applied resource policy receipt')
            applied_policy = receipt
        if outcome is None:
            if child.returncode == -signal.SIGXCPU:
                outcome = rejected('rejected_resource_limit', 'Kernel CPU limit terminated worker (SIGXCPU)',
                    resource=dict(kind='cpu', limit_seconds=policy['cpu_seconds'],
                                  observed_seconds=usage['cpu_seconds'], signal='SIGXCPU'))
            elif child.returncode:
                # SIGKILL alone cannot establish that CPU was the cause.
                outcome = rejected('rejected_evaluation', f'Worker exited with code {child.returncode}')
            else:
                result = read_json_bounded(result_path)
                if not isinstance(result, dict):
                    raise ValueError('Worker result must be an object')
                status = result.get('status')
                if status == 'failed_resource_setup':
                    raise WorkerSetupError(str(result.get('reason', 'Worker resource setup failed')))
                if status != 'ok' and not (
                        request['action'] == 'validate'
                        and status in ('rejected_invalid', 'rejected_prompt_only')
                        and isinstance(result.get('reason'), str)) and not (
                        status == 'rejected_resource_limit' and isinstance(result.get('resource'), dict)
                        and isinstance(result.get('reason'), str)):
                    raise ValueError('Invalid worker result status or reason')
                measured = result.pop('worker_usage', {})
                if isinstance(measured, dict):
                    for key in ('cpu_seconds', 'rss_bytes', 'peak_rss_bytes', 'peak_virtual_bytes'):
                        value = measured.get(key, 0)
                        if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                            usage[key] = value if key == 'rss_bytes' else max(usage[key], value)
                outcome = result
                if sys.platform.startswith('linux') and usage['peak_rss_bytes'] > policy['rss_bytes']:
                    outcome = rejected('rejected_resource_limit', 'Worker peak RSS exceeded sampled limit',
                        resource=dict(kind='rss', limit_bytes=policy['rss_bytes'],
                                      observed_bytes=usage['peak_rss_bytes'], enforcement='reported peak'))
        if applied_policy is None and outcome.get('status') == 'ok':
            raise ValueError('Missing applied resource policy receipt')
    except WorkerSetupError as exc:
        write_failure('failed_resource_setup', str(exc))
        raise
    except ResourceExceeded as exc:
        outcome = rejected('rejected_resource_limit', str(exc), resource=exc.details)
    except OSError as exc:
        if phase in ('launch', 'monitor'):
            write_failure('failed_resource_setup', f'Worker launch/monitor setup failed: {exc}')
            raise WorkerSetupError(f'Worker launch/monitor setup failed: {exc}') from exc
        outcome = rejected('rejected_evaluation', str(exc)[:1000])
    except (ValueError, OverflowError, RecursionError) as exc:
        outcome = rejected('rejected_evaluation', str(exc)[:1000])
    except (KeyboardInterrupt, SystemExit) as exc:
        write_failure('interrupted', f'Worker owner interrupted: {type(exc).__name__}',
                      exit_code=exc.code if isinstance(exc, SystemExit) else 130)
        raise
    finally:
        if child is not None:
            _signal_group(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                pass
            _signal_group(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
            if child.stdout:
                child.stdout.close()
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
    return dict(outcome, seconds=time.monotonic() - started, evidence=str(directory),
                resource_policy=policy, applied_policy=applied_policy, usage=usage)


def validate_candidate(run_dir, name, *, module='meta_harness.meta_harness_chess_puzzle', prompt_only=False):
    logs = Path(run_dir) / 'logs'
    directory = evidence_dir(logs, name, 'validation')
    outcome = run_worker(dict(action='validate', module=module, run_dir=str(Path(run_dir).resolve()),
        name=name, prompt_only=prompt_only), directory,
        timeout_setting('LADDER_CANDIDATE_VALIDATION_TIMEOUT', 30))
    (directory / 'outcome.json').write_text(json.dumps(outcome, indent=2) + '\n')
    if outcome['status'] != 'ok':
        quarantine(logs, name, directory)
        raise CandidateRejected(outcome)
