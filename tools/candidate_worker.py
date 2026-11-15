"""One generated candidate per interpreter; no server ownership in this process."""
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import traceback


def apply_limits(policy):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (policy['cpu_seconds'], policy['cpu_hard_seconds']))
    if sys.platform.startswith('linux'):
        resource.setrlimit(resource.RLIMIT_AS, (policy['address_space_bytes'], policy['address_space_bytes']))
    # Reading procfs is part of setup, so an unavailable Linux watchdog is fatal.
    if sys.platform.startswith('linux'):
        Path('/proc/self/status').read_text()


def worker_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    measured = dict(cpu_seconds=usage.ru_utime + usage.ru_stime,
                    peak_rss_bytes=usage.ru_maxrss * (1024 if sys.platform.startswith('linux') else 1))
    if sys.platform.startswith('linux'):
        status = Path('/proc/self/status').read_text()
        for field, key in (('VmRSS:', 'rss_bytes'), ('VmPeak:', 'peak_virtual_bytes')):
            measured[key] = next((int(line.split()[1]) * 1024 for line in status.splitlines()
                                  if line.startswith(field)), 0)
    return measured


def main():
    request_path, result_path = map(Path, sys.argv[1:])
    request = json.loads(request_path.read_text())
    sys.path[:] = request['python_path']
    policy = request['resource_policy']
    reserve = None
    executing_candidate = False
    def publish(result):
        result['worker_usage'] = worker_usage()
        temporary = result_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, allow_nan=False) + '\n')
        temporary.replace(result_path)
    try:
        reserve = bytearray(256 * 1024)  # Reserve before an AS ceiling can prevent setup reporting.
        try:
            apply_limits(policy)
            applied = dict(platform=sys.platform, cpu_rlimit=list(resource.getrlimit(resource.RLIMIT_CPU)),
                core_rlimit=list(resource.getrlimit(resource.RLIMIT_CORE)),
                address_space_rlimit=list(resource.getrlimit(resource.RLIMIT_AS))
                    if sys.platform.startswith('linux') else None,
                native_thread_env={key: os.environ.get(key) for key in ('OMP_NUM_THREADS',
                    'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')})
            applied_path = result_path.with_name('applied-policy.json')
            temporary = applied_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(applied) + '\n')
            temporary.replace(applied_path)
        except (OSError, ValueError, OverflowError) as exc:
            publish(dict(status='failed_resource_setup', reason=f'Resource policy setup failed: {exc}'))
            return
        module = importlib.import_module(request['module'])
        if request['action'] == 'validate':
            module.PROMPT_ONLY = request['prompt_only']
            callback = module.validate_candidate
        elif request['action'] == 'score':
            callback = getattr(module, request['function'])
        else:
            raise ValueError('Unknown candidate action')
        started = result_path.with_name('execution-started.json')
        temporary = started.with_suffix('.tmp')
        temporary.write_text('{"started": true}\n')
        temporary.replace(started)
        executing_candidate = True
        if request['action'] == 'validate':
            try:
                callback(Path(request['run_dir']), request['name'])
            except (SyntaxError, ValueError) as exc:
                result = dict(status='rejected_prompt_only' if request['prompt_only'] else 'rejected_invalid',
                              reason=f'{type(exc).__name__}: {exc}')
            else:
                result = dict(status='ok')
        elif request['action'] == 'score':
            rows = callback(request['config'],
                [(request['name'], Path(request['path']))], Path(request['logs']), force=True)
            result = dict(status='ok', rows=rows)
        else:
            raise ValueError('Unknown candidate action')
        publish(result)
    except MemoryError:
        reserve = None  # Leave room for a small structured rejection.
        result = dict(status='rejected_resource_limit' if executing_candidate else 'failed_resource_setup',
                      reason='Worker memory allocation failed (MemoryError)' if executing_candidate
                          else 'Worker resource setup or import failed (MemoryError)',
                      resource=dict(kind='allocation', trigger='MemoryError',
                                    address_space_limit_bytes=policy['address_space_bytes'],
                                    address_space_enforced=sys.platform.startswith('linux')))
        publish(result)
    except BaseException:
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
