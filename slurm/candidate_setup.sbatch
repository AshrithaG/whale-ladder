#!/bin/bash -l
# CPU preparation for the isolated PSC installation. Submit from the checkout:
#   python3 tools/submit_candidate.py --account YOUR_ACCOUNT \
#     --root /absolute/new/candidate-work --setup-only
# The submitter passes an explicit JSON config path as the script argument.
#SBATCH --job-name=ladder-setup
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=7600M
#SBATCH --time=01:00:00
#SBATCH --export=ALL
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
exports="$(python3 setup/load_candidate_config.py "${1:?job configuration required}")"
eval "$exports"
cd "$LADDER_CODE_DIR"
git diff --quiet && git diff --cached --quiet
[[ "$(git rev-parse HEAD)" == "$EXPECTED_REVISION" ]] || { echo 'Unexpected candidate revision' >&2; exit 2; }
mkdir -p "$LADDER_ROOT"
marker="$LADDER_ROOT/.candidate-install"
if [[ ! -f "$marker" && ( -e "$LADDER_ROOT/venv" || -e "$LADDER_ROOT/WHALE" ) ]]; then
  echo 'Existing non-candidate environment; choose a new LADDER_ROOT' >&2
  exit 2
fi
touch "$marker"
export INSTALL_PROFILE=persistent-smoke SKIP_27B=1 INSTALL_SKIP_GPU_KERNELS=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"
export OMP_NUM_THREADS=1
bash setup/install.sh 2>&1 | tee "$LADDER_ROOT/install.log"
"$LADDER_ROOT/venv/bin/python" -m pip check
git rev-parse HEAD > "$LADDER_ROOT/installed-revision"
printf 'CPU installation passed; GPU kernel/preflight checks remain\n' > "$LADDER_ROOT/setup-status"
