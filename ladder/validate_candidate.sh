#!/usr/bin/env bash
# Failures return to the allocation-retaining outer Slurm controller.
set -euo pipefail
repo="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
exports="$(python3 "$repo/setup/load_candidate_config.py" "${1:?job configuration required}")"
eval "$exports"
cd "$LADDER_CODE_DIR"
git diff --quiet && git diff --cached --quiet
[[ "$(git rev-parse HEAD)" == "$EXPECTED_REVISION" ]] || { echo 'Unexpected candidate code revision' >&2; exit 2; }
[[ "$(cat "$LADDER_ROOT/installed-revision")" == "$EXPECTED_REVISION" ]] || {
  echo 'Installation differs from this revision; rerun candidate setup' >&2; exit 2;
}
export LADDER_GCC_MODULE="${LADDER_GCC_MODULE:-gcc/13.3.1-p20240614}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
bash setup/install_gpu_kernels.sh
bash setup/preflight.sh whale_qwen4b persistent-smoke
bash ladder/run_cycles.sh whale_qwen4b persistent-smoke
