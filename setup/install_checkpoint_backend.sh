#!/usr/bin/env bash
# Install the CUDA-matched dependency required by WHALE's NCCL weight transfer.
# Called by setup; it can also repair this dependency in an older environment.
set -euo pipefail
LADDER_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$LADDER_HOME/ladder/config.env"
source "$VENV/bin/activate"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$LADDER_ROOT/cache/pip}"
cuda_major="$(python -c 'import torch; print((torch.version.cuda or "").split(".")[0])')"
case "$cuda_major" in
  12|13) wheel="cupy-cuda${cuda_major}x" ;;
  *) echo "Unsupported CUDA runtime for NCCL checkpoint transfer: $cuda_major" >&2; exit 1 ;;
esac
# Refuse conflicting CuPy distributions instead of installing over them.
python - "$wheel" <<'PY'
import sys
from importlib.metadata import distributions
names = {d.metadata['Name'].lower().replace('_', '-') for d in distributions()}
conflicts = {n for n in names if (n == 'cupy' or n.startswith('cupy-cuda')) and n != sys.argv[1]}
if conflicts:
    raise SystemExit('Conflicting CuPy distributions; resolve explicitly: ' + ', '.join(sorted(conflicts)))
PY
if ! python - "$wheel" <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version
try:
    print("NCCL dependency:", sys.argv[1], version(sys.argv[1]), "fastrlock", version("fastrlock"))
except PackageNotFoundError:
    sys.exit(1)
PY
then
  # Constrain every installed package so this repair cannot upgrade torch,
  # vLLM, NumPy, or another existing dependency as a side effect.
  constraints="$(mktemp "$LADDER_ROOT/checkpoint-constraints.XXXXXX")"
  trap 'rm -f "$constraints"' EXIT
  python - <<'PY' > "$constraints"
from importlib.metadata import distributions
for d in sorted(distributions(), key=lambda d: d.metadata['Name'].lower()):
    print(d.metadata['Name'] + '==' + d.version)
PY
  python -m pip install --only-binary=:all: -c "$constraints" "$wheel==14.2.0" fastrlock
  python -m pip freeze > "$LADDER_ROOT/pip-freeze.txt"
fi
python -m pip check
