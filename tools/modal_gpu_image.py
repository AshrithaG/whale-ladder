"""Derive a CPU-verified runtime image from the saved installation, without reinstalling it.

The input is a clean local commit. Its Git bundle supplies the exact source;
no GitHub push or GPU allocation is needed to prepare this image.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import modal

BASE_IMAGE = 'im-1uRsQ0NES8BmdMyIrsoQSR'


def audit_runtime(revision):
    import os
    from pathlib import Path
    import shutil
    import subprocess

    root, code = Path('/opt/whale-work'), Path('/opt/whale-ladder')
    actual = subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != revision:
        raise RuntimeError(f'Unexpected runtime: {actual}')
    shutil.copy2(root / 'cpu-ready.json', root / 'parent-cpu-ready.json')
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    # Populate official matching kernel wheels on CPUs. Actual CUDA compilation,
    # device execution and the actor probe still run in GPU preflight.
    subprocess.run(['bash', str(code / 'setup/install_gpu_kernels.sh')], env=env, check=True)
    subprocess.run(['bash', '-euc', '''
set -a
source /opt/whale-ladder/ladder/config.env
source /opt/whale-ladder/ladder/profiles/modal-smoke.env
source "$VENV/bin/activate"
python "$LADDER_HOME/setup/install_rollout_progress.py" "$WHALE_DIR/domains/chess_puzzles" "$LADDER_ROOT/logging-source-backups"
python "$LADDER_HOME/setup/check_cpu_install.py" --fingerprint "$CHESS_DATA_FINGERPRINT" --output "$LADDER_ROOT/cpu-ready.json"
python "$LADDER_HOME/tools/observe_run.py" /tmp/logging-cpu-check --once
'''], env=env, check=True)


def read_receipt():
    import json
    from pathlib import Path
    root = Path('/opt/whale-work')
    return dict(audit=json.loads((root / 'cpu-ready.json').read_text()),
        parent_audit=json.loads((root / 'parent-cpu-ready.json').read_text()),
        instrumentation=json.loads((root / 'logging-source-backups/rollout-progress-install.json').read_text()))


def prepare(output, base_image=BASE_IMAGE):
    home = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=home, text=True).strip():
        raise ValueError('Commit the runtime changes before preparing an image')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=home, text=True).strip()
    models = modal.Volume.from_name('whale-ladder-candidate-models')
    with tempfile.TemporaryDirectory() as directory:
        bundle = Path(directory) / 'runtime.bundle'
        subprocess.run(['git', 'bundle', 'create', str(bundle), 'HEAD'], cwd=home, check=True)
        image = (modal.Image.from_id(base_image)
            .add_local_file(bundle, '/tmp/whale-runtime.bundle', copy=True)
            .run_commands('git -C /opt/whale-ladder fetch /tmp/whale-runtime.bundle HEAD',
                          f'git -C /opt/whale-ladder checkout --detach {revision}')
            .run_function(audit_runtime, args=(revision,), cpu=4, memory=8192, timeout=900,
                          volumes={'/models': models}, include_source=False))
        app = modal.App('whale-ladder-logging-cpu')
        receipt = app.function(image=image, serialized=True, include_source=False,
                               cpu=1, memory=1024, timeout=60)(read_receipt)
        with modal.enable_output(), app.run():
            result = receipt.remote()
            record = dict(result['audit'], modal_image_id=image.object_id,
                source_revision=revision, parent_image_id=base_image,
                parent_runtime_revision=result['parent_audit']['revision'],
                logging_instrumentation=result['instrumentation'],
                gpu_requested=False, gpu_validated=False,
                note='Existing CPU installation reused; source updated and official kernel wheels added on CPU. GPU preflight remains required.')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2) + '\n')
        print(f'CPU-verified logging image {record["modal_image_id"]}; receipt: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-image', default=BASE_IMAGE, help='Your existing CPU-prepared image ID')
    args=parser.parse_args()
    prepare(args.output,args.base_image)
