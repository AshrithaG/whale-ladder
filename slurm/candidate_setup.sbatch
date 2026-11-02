#!/bin/bash -l
# CPU preparation for the isolated PSC installation. Submit from the checkout:
#   export LADDER_ROOT=/absolute/new/candidate-work
#   export LADDER_MODEL_CACHE=/absolute/existing/hf       # optional
#   export CHESS_DATA_FROM=/absolute/shared/parquet-dir  # recommended
#   export CHESS_DATA_FINGERPRINT=<shared fingerprint>
#   sbatch -A YOUR_ACCOUNT slurm/candidate_setup.sbatch
#SBATCH --job-name=ladder-setup
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=7600M
#SBATCH --time=01:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${LADDER_ROOT:?Use a separate candidate installation root}"
git diff --quiet && git diff --cached --quiet
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
