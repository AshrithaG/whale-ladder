#!/usr/bin/env bash
# CPU-only install for the layer-1 mock run (ladder/mock_run.sh): WHALE with all
# ladder patches plus a light Python venv. No GPU, CUDA, vLLM, torch or Node.
#
#   export LADDER_ROOT=$HOME/whale-ladder-mock
#   bash setup/install_mock.sh
#
# Needs python3 >= 3.10 and internet. Refuses to run on macOS.
set -euo pipefail
[[ "$(uname -s)" != Darwin ]] || { echo "install_mock.sh does not run on macOS; use cerlab23 or PSC" >&2; exit 2; }
LADDER_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${LADDER_ROOT:?set LADDER_ROOT to a scratch folder for the mock, e.g. \$HOME/whale-ladder-mock}"
export LADDER_ROOT LADDER_HOME
set -a
source "$LADDER_HOME/ladder/config.env"
set +a
mkdir -p "$LADDER_ROOT" "$OUT_ROOT"

bash "$LADDER_HOME/setup/prepare_whale.sh"

if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
    || { echo "python3 >= 3.10 required" >&2; exit 1; }
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install -q -U pip
"$VENV/bin/pip" install -q pandas pyarrow python-chess "openai>=1.40" huggingface_hub packaging
"$VENV/bin/pip" install -q -e "$LADDER_HOME/shim"
echo "[install_mock] done. Next: bash ladder/mock_run.sh"
