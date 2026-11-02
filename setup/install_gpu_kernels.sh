#!/usr/bin/env bash
# Run on a compute node after installing vLLM. Keep current packages fixed
# while its official helper adds the matching precompiled FlashInfer kernels.
set -euo pipefail
LADDER_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$LADDER_HOME/ladder/config.env"
source "$VENV/bin/activate"
export PIP_CACHE_DIR="$LADDER_ROOT/cache/pip"
constraints="$(mktemp "$LADDER_ROOT/kernel-constraints.XXXXXX")"
trap 'rm -f "$constraints"' EXIT
python - <<'PY' > "$constraints"
from importlib.metadata import distributions
for d in sorted(distributions(), key=lambda d: d.metadata['Name'].lower()):
    print(d.metadata['Name'] + '==' + d.version)
PY
PIP_CONSTRAINT="$constraints" vllm download-kernels
python -m pip check
python -m pip freeze > "$LADDER_ROOT/pip-freeze.txt"
