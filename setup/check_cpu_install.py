"""Check a prepared CPU installation; GPU compatibility remains unverified."""
import argparse
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import pyarrow.parquet as pq
import torch

home = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(home / 'tools'))
from data_fingerprint import FILES, sha256


def check(fingerprint):
    root = Path(os.environ['LADDER_ROOT'])
    assert Path(sys.prefix).resolve() == (root / 'venv').resolve(), sys.prefix
    constraint = home / 'setup/constraints/psc-py311-cu130.txt'
    installed = {canonicalize_name(d.metadata['Name']): d.version for d in distributions()}
    mismatches = []
    for line in constraint.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        req = Requirement(line)
        version = installed.get(canonicalize_name(req.name))
        if version is not None and not req.specifier.contains(version):
            mismatches.append(f'{req}: installed {version}')
    if mismatches:
        raise RuntimeError('Constraint mismatches: ' + '; '.join(mismatches))
    required = ['vllm', 'transformers', 'ray', 'cupy-cuda13x', 'flash-linear-attention',
                'torch-c-dlpack-ext', 'litellm', 'openai']
    assert all(canonicalize_name(name) in installed for name in required), 'Missing core dependency'
    assert platform.python_version() == os.environ['INSTALL_PYTHON_VERSION'], platform.python_version()
    assert torch.version.cuda.startswith('13.'), torch.version.cuda
    assert not torch.cuda.is_available(), 'CPU preparation unexpectedly has an accessible GPU'
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    subprocess.run([os.environ.get('CXX', 'g++'), '-std=c++20', '-x', 'c++', '-fsyntax-only', '-'],
                   input='#include <type_traits>\nstatic_assert(std::is_same_v<int,int>);\n', text=True, check=True)
    data = root / 'WHALE/domains/chess_puzzles/data/chess_puzzle'
    lines = [f'{sha256(data / name)}  {name}' for name in FILES]
    actual = hashlib.sha256('\n'.join(lines).encode()).hexdigest()[:16]
    assert actual == fingerprint, (actual, fingerprint)
    rows = {name: pq.read_metadata(data / name).num_rows for name in FILES}
    assert list(rows.values()) == [16384, 256, 256], rows
    whale_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root / 'WHALE', text=True).strip()
    assert whale_revision == os.environ['WHALE_COMMIT'], whale_revision
    from huggingface_hub import snapshot_download
    model_revision = os.environ['SOLVER_MODEL_REVISION']
    model = Path(snapshot_download(os.environ['SOLVER_MODEL'], revision=model_revision, local_files_only=True))
    shards = list(model.glob('*.safetensors'))
    assert shards and all(p.stat().st_size > 0 for p in shards), 'Missing model weights'
    return dict(cpu_installation_passed=True, gpu_validated=False,
        revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=home, text=True).strip(),
        python=platform.python_version(), python_prefix=sys.prefix, python_search_path=sys.path,
        torch_cuda=torch.version.cuda,
        whale_revision=whale_revision,
        constraint_sha256=sha256(constraint), installed_packages=installed, constraint_mismatches=mismatches,
        data_fingerprint=actual, dataset_rows=rows, dataset_sha256=dict(zip(FILES, [sha256(data / n) for n in FILES])),
        model=os.environ['SOLVER_MODEL'], model_revision=model_revision,
        model_weight_bytes=sum(p.stat().st_size for p in shards),
        compiler=subprocess.check_output([os.environ.get('CXX', 'g++'), '--version'], text=True).splitlines()[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fingerprint', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = check(args.fingerprint)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'installed_packages'}, indent=2))
