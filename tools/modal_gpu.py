"""One bounded Modal GPU validation using the saved CPU image.

Default: print the plan locally. --launch spends credits; obtain approval first.
The prepared experiment source is never replaced by the local checkout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

IMAGE_ID = 'im-1uRsQ0NES8BmdMyIrsoQSR'
RUNTIME_REVISION = '877a4f4466533c0f0879a65ba565a7803afe00f5'
WORK_ROOT = '/opt/whale-work'
CODE_ROOT = '/opt/whale-ladder'
RESOURCES = dict(gpu='H100!:2', cpu=(24, 24), memory=(262144, 262144),
                 ephemeral_disk=524288, timeout=13800, startup_timeout=600,
                 max_containers=1, min_containers=0, buffer_containers=0,
                 retries=0, single_use_containers=True)


def prepare_plan(trial, cpu_receipt=None):
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,49}', trial):
        raise ValueError('Trial must be 1–50 lowercase letters, digits or hyphens')
    home = Path(__file__).resolve().parents[1]
    image_id, runtime_revision = IMAGE_ID, RUNTIME_REVISION
    if cpu_receipt:
        prepared = json.loads(cpu_receipt.read_text())
        if prepared.get('cpu_installation_passed') is not True:
            raise ValueError('CPU receipt does not report a successful installation')
        if prepared.get('data_fingerprint') != '91c2fe70efad772c':
            raise ValueError('CPU receipt uses a different dataset')
        image_id = prepared['modal_image_id']
        runtime_revision = prepared.get('source_revision') or prepared['revision']
    if not re.fullmatch(r'[0-9a-f]{40}', runtime_revision):
        raise ValueError('CPU runtime must use a full commit SHA')
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=home)
    # Check the actual files the image will execute, not just the branch name.
    paths = git('ls-tree', '-r', '--name-only', runtime_revision,
                'ladder', 'setup', 'tools', 'patches').decode().splitlines()
    hashes = {}
    for name in paths:
        content = git('show', f'{runtime_revision}:{name}')
        if (home / name).read_bytes() != content:
            raise ValueError(f'{name} differs from the prepared image; reconcile runtime before launch')
        hashes[name] = hashlib.sha256(content).hexdigest()
    sources = {name: (home / 'tools' / name).read_text()
               for name in ('modal_gpu.py', 'modal_gpu_probe.py')}
    return dict(trial=trial, image_id=image_id, runtime_revision=runtime_revision,
        launcher_checkout_revision=git('rev-parse', 'HEAD').decode().strip(),
        launcher_checkout_status=git('status', '--porcelain').decode(),
        launcher_sha256={n: hashlib.sha256(s.encode()).hexdigest() for n, s in sources.items()},
        runtime_source_sha256=hashes, resources=RESOURCES,
        results_volume=f'whale-ladder-gpu-{trial}', models_volume='whale-ladder-candidate-models',
        mount='/opt/whale-work/runs', arm='whale_qwen4b', profile='modal-smoke',
        schedule=dict(cycles=2, training_puzzles_per_cycle=256, training_attempts=8,
            search_iterations_per_cycle=1, candidate_slots_per_iteration=3,
            scoring_puzzles=256, scoring_attempts=1, final_test_puzzles=256, final_test_attempts=8),
        logging=dict(heartbeat_seconds=10, rollout_completion_counters=True,
            resource_and_inference_metrics=True, full_parameter_checksums=True),
        soft_deadline_seconds=13200, hourly_estimate_usd=11.075184,
        four_hour_compute_estimate_usd=44.300736,
        cost_note='Estimate, not a provider-enforced dollar cap; storage/egress are additional.'), sources


def run_gpu(plan, sources, results):
    # Serialized from this launcher. Keep provider imports/environment in the parent.
    import hashlib
    import json
    import os
    from pathlib import Path
    import shutil
    import signal
    import subprocess
    import time
    import traceback

    root, code = Path(WORK_ROOT), Path(CODE_ROOT)
    out = root / 'runs'
    control = out / 'modal-control'
    # A committed marker refuses duplicate invocation/preemption replay, rather
    # than silently adding training or proposal opportunities.
    control.mkdir(exist_ok=False)
    def write(name, value):
        (control / name).write_text(json.dumps(value, indent=2) + '\n')
    write('plan.json', plan)
    for name, source in sources.items():
        if hashlib.sha256(source.encode()).hexdigest() != plan['launcher_sha256'][name]:
            raise RuntimeError(f'Launcher hash mismatch: {name}')
        (control / name).write_text(source)
    write('started.json', dict(started_unix=time.time(), task_id=os.environ.get('MODAL_TASK_ID')))
    results.commit()
    began = time.monotonic()
    deadline = began + plan['soft_deadline_seconds']
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    env.update(LADDER_ROOT=str(root), LADDER_HOME=str(code), LADDER_MODEL_CACHE='/models/hf',
        LADDER_GCC_MODULE='', LADDER_ALLOW_NO_SLURM='1', PROFILE='modal-smoke',
        CC='/usr/bin/gcc', CXX='/usr/bin/g++', CUDAHOSTCXX='/usr/bin/g++',
        PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
        CHESS_DATA_FINGERPRINT='91c2fe70efad772c',
        PATH=str(root / 'venv/bin') + ':' + env['PATH'],
        LD_LIBRARY_PATH=str(root / 'conda/lib') + ':' + env.get('LD_LIBRARY_PATH', ''))
    python = str(root / 'venv/bin/python')
    phase = 'identity'
    observer = None

    def command(name, argv):
        nonlocal phase
        phase = name
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Trial soft deadline reached')
        write('status.json', dict(phase=name, started_unix=time.time(), argv=argv))
        print(f'Modal validation phase: {name}', flush=True)
        started = time.monotonic()
        status, failure = None, None
        # The log goes straight to persistent storage, including subprocess
        # output produced immediately before a crash or disconnect.
        try:
            with (control / f'{name}.log').open('x') as log:
                process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, cwd=code)
                try:
                    status = process.wait(timeout=remaining)
                except BaseException:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                    except ProcessLookupError:
                        pass
                    raise
        except BaseException as exc:
            failure = dict(type=type(exc).__name__, message=str(exc))
            raise
        finally:
            with (control / 'phases.jsonl').open('a') as log:
                log.write(json.dumps(dict(phase=name, returncode=status, failure=failure,
                    started_unix=time.time() - (time.monotonic() - started), ended_unix=time.time(),
                    elapsed_seconds=time.monotonic() - started)) + '\n')
            results.commit()
        print(f'Modal validation phase ended: {name}; exit={status}', flush=True)
        if status:
            raise subprocess.CalledProcessError(status, argv)

    try:
        actual = subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != plan['runtime_revision']:
            raise RuntimeError(f'Image runtime mismatch: {actual}')
        for name, expected in plan['runtime_source_sha256'].items():
            if hashlib.sha256((code / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Image source mismatch: {name}')
        shutil.copy2(root / 'cpu-ready.json', control / 'cpu-ready.json')
        cpu = json.loads((root / 'cpu-ready.json').read_text())
        if (cpu.get('cpu_installation_passed') is not True
                or cpu.get('revision') != actual
                or cpu.get('data_fingerprint') != env['CHESS_DATA_FINGERPRINT']):
            raise RuntimeError('Baked CPU audit does not match the requested runtime/data')
        # These paths otherwise live on ephemeral image storage. Preserve all
        # generated candidates and preflight configs, including failure evidence.
        for original, target in [
            (root / 'WHALE/domains/chess_puzzles/meta_harness/runs', out / 'harness-runs'),
            (root / 'preflight', out / 'preflight')]:
            if original.exists():
                shutil.move(str(original), str(target))
            else:
                target.mkdir()
            original.symlink_to(target, target_is_directory=True)
        probe = str(control / 'modal_gpu_probe.py')
        command('host', [python, probe, '--output', str(control / 'host.json')])
        host = json.loads((control / 'host.json').read_text())
        env.update(CUDA_VISIBLE_DEVICES=host['ladder_job_gpus'], LADDER_JOB_GPUS=host['ladder_job_gpus'])
        command('logging-ready', [python, str(code / 'tools/observe_run.py'), str(out), '--once'])
        live = json.loads((control / 'live.json').read_text())
        if live.get('observation_error') or live.get('gpu_observation_error'):
            raise RuntimeError(f'Live logging did not initialize: {live}')
        with (control / 'observer.log').open('x') as log:
            observer = subprocess.Popen([python, str(code / 'tools/observe_run.py'), str(out)],
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=code)
        command('nccl', [python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', probe, '--nccl'])
        command('kernels', ['bash', str(code / 'setup/install_gpu_kernels.sh')])
        command('preflight', ['bash', str(code / 'setup/preflight.sh'), plan['arm'], plan['profile']])
        command('cycles', ['bash', str(code / 'ladder/run_cycles.sh'), plan['arm'], plan['profile']])
        write('execution-completed.json', dict(execution_completed=True, raw_audit_required=True,
                                             elapsed_seconds=time.monotonic() - began))
        write('status.json', dict(phase='completed', state='completed', ended_unix=time.time()))
    except BaseException as exc:
        write('status.json', dict(phase=phase, state='failed', ended_unix=time.time()))
        write('failure.json', dict(phase=phase, type=type(exc).__name__, message=str(exc),
                                  traceback=traceback.format_exc(), elapsed_seconds=time.monotonic() - began))
        raise
    finally:
        if observer is not None and observer.poll() is None:
            observer.terminate()
            try:
                observer.wait(timeout=15)
            except subprocess.TimeoutExpired:
                observer.kill()
                observer.wait(timeout=5)
        # Normal failure/deadline cleanup is provided by the existing controller.
        # Single-use container termination also releases any surviving children.
        for name in ('pip-freeze.txt', 'install.log'):
            if (root / name).exists():
                shutil.copy2(root / name, control / name)
        results.commit()
    return dict(results_volume=plan['results_volume'], execution_completed=True, raw_audit_required=True)


def launch(plan, sources, receipt):
    import modal

    receipt.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the local identity before any provider mutation; ambiguous client
    # failure must be investigated via the App, never blindly resubmitted.
    with receipt.open('x') as f:
        json.dump(dict(state='submission-prepared', plan=plan), f, indent=2)
    results = modal.Volume.from_name(plan['results_volume'], create_if_missing=True, version=2)
    if results.listdir('/'):
        raise ValueError('Results Volume is not empty; inspect the previous invocation before any restart')
    models = modal.Volume.from_name(plan['models_volume'])
    app = modal.App('whale-ladder-gpu-' + plan['trial'])
    function = app.function(image=modal.Image.from_id(plan['image_id']), serialized=True,
        include_source=False, volumes={plan['mount']: results, '/models': models},
        **RESOURCES)(run_gpu)
    with modal.enable_output(), app.run(detach=True):
        call = function.spawn(plan, sources, results)
        record = dict(state='submitted', app_id=app.app_id, function_call_id=call.object_id, plan=plan)
        receipt.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', required=True)
    parser.add_argument('--launch', action='store_true', help='Spend credits on this single GPU invocation')
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--cpu-receipt', type=Path, help='Use your own successful Modal CPU preparation receipt')
    args = parser.parse_args()
    plan, sources = prepare_plan(args.trial, args.cpu_receipt)
    if args.launch:
        if args.receipt is None:
            parser.error('--launch requires --receipt')
        launch(plan, sources, args.receipt)
    else:
        print(json.dumps(plan, indent=2))
