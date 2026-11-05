"""CPU-only Modal preparation. No function or image build requests a GPU.

Run with the Modal SDK installed:
  python tools/modal_candidate.py --data /path/to/shared/parquets --fingerprint FINGERPRINT

The checkout must be clean and its HEAD published. Inputs are verified before
upload. The existing installer builds a cached image; this does not run an experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import modal

# NVIDIA CUDA 13.0.0 devel / Ubuntu 24.04, pinned linux/amd64 manifest.
BASE_IMAGE = 'nvidia/cuda@sha256:435220c0fef35cbf712e11999f8670a83835ef3cdd18564e5e8122f83078c88c'
CODE = '/opt/whale-ladder'
ROOT = '/opt/whale-work'
inputs = modal.Volume.from_name('whale-ladder-candidate-inputs', create_if_missing=True, version=2)
models = modal.Volume.from_name('whale-ladder-candidate-models', create_if_missing=True, version=2)


def install_on_cpu():
    import os
    import subprocess
    from pathlib import Path

    code, root = Path('/opt/whale-ladder'), Path('/opt/whale-work')
    root.mkdir(parents=True, exist_ok=True)
    # Modal injects /pkg into PYTHONPATH, which also exposes its own dependency
    # metadata. Keep that runtime in the parent, outside experiment processes.
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    with (root / 'install.log').open('w') as log:
        process = subprocess.Popen(['bash', str(code / 'setup/install.sh')],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
        if process.wait():
            raise RuntimeError('CPU installation failed; inspect the image-build log')
    subprocess.run(['bash', '-euc', '''
set -a
source "$LADDER_HOME/ladder/config.env"
source "$LADDER_HOME/ladder/profiles/modal-smoke.env"
source "$VENV/bin/activate"
python "$LADDER_HOME/setup/check_cpu_install.py" --fingerprint "$CHESS_DATA_FINGERPRINT" --output "$LADDER_ROOT/cpu-ready.json"
'''], check=True, env=env)
    models.commit()


def verify_inputs(data, expected):
    from data_fingerprint import FILES, sha256
    lines = [f'{sha256(data / name)}  {name}' for name in FILES]
    actual = hashlib.sha256('\n'.join(lines).encode()).hexdigest()[:16]
    if actual != expected:
        raise ValueError(f'Dataset fingerprint {actual} differs from {expected}')
    return FILES


def read_receipt():
    import json
    from pathlib import Path
    return json.loads(Path('/opt/whale-work/cpu-ready.json').read_text())


def prepare(data, fingerprint, output):
    home = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=home, text=True).strip():
        raise ValueError('Commit and publish the candidate before building its image')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=home, text=True).strip()
    files = verify_inputs(data, fingerprint)
    with inputs.batch_upload(force=True) as batch:
        for name in files:
            batch.put_file(data / name, f'/datasets/{fingerprint}/{name}')
    image = (modal.Image.from_registry(BASE_IMAGE, add_python='3.14')
        .apt_install('git', 'curl', 'ca-certificates', 'build-essential', 'libgl1', 'libglib2.0-0')
        .run_commands(f'git clone https://github.com/AshrithaG/whale-ladder.git {CODE}',
                      f'git -C {CODE} checkout --detach {revision}')
        .env(dict(LADDER_HOME=CODE, LADDER_ROOT=ROOT, LADDER_MODEL_CACHE='/models/hf',
            CHESS_DATA_FROM=f'/inputs/datasets/{fingerprint}', CHESS_DATA_FINGERPRINT=fingerprint,
            INSTALL_PROFILE='modal-smoke', SKIP_27B='1', INSTALL_SKIP_GPU_KERNELS='1',
            LADDER_GCC_MODULE='', CC='/usr/bin/gcc', CXX='/usr/bin/g++', CUDAHOSTCXX='/usr/bin/g++',
            PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', PIP_NO_CACHE_DIR='1'))
        .run_function(install_on_cpu, cpu=4, memory=8192, timeout=3600,
                      volumes={'/inputs': inputs, '/models': models}))
    app = modal.App('whale-ladder-candidate-cpu')

    receipt = app.function(image=image, cpu=1, memory=1024, timeout=60)(read_receipt)

    with modal.enable_output(), app.run():
        result = receipt.remote()
        result.update(modal_image_id=image.object_id, base_image=BASE_IMAGE,
                      model_volume='whale-ladder-candidate-models', source_revision=revision)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'CPU preparation passed: {output}; image {result["modal_image_id"]}. GPU validation remains pending.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--fingerprint', required=True)
    parser.add_argument('--output', type=Path, default=Path('/tmp/whale-ladder-modal-cpu.json'))
    args = parser.parse_args()
    prepare(args.data, args.fingerprint, args.output)
