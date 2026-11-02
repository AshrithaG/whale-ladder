#!/bin/bash -l
# Submit after candidate_setup.sbatch succeeds, from the same clean checkout
# and with the same LADDER_ROOT/model-cache/data settings.
#SBATCH --job-name=ladder-persist
#SBATCH --partition=GPU-shared
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --gres=gpu:h100-80:2
#SBATCH --time=04:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${LADDER_ROOT:?Use the isolated candidate installation root}"
git diff --quiet && git diff --cached --quiet
[[ "$(cat "$LADDER_ROOT/installed-revision")" == "$(git rev-parse HEAD)" ]] || {
  echo 'Code differs from the prepared installation; rerun candidate_setup.sbatch' >&2; exit 2;
}
export LADDER_GCC_MODULE="${LADDER_GCC_MODULE:-gcc/13.3.1-p20240614}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export INSTALL_PROFILE=persistent-smoke
control="$LADDER_ROOT/jobs/candidate-$SLURM_JOB_ID"
mkdir -p "$control"
git rev-parse HEAD > "$control/revision"
validate() {
  printf 'GPU kernels and preflight\n' > "$control/status"
  bash setup/install_gpu_kernels.sh || return $?
  bash setup/preflight.sh whale_qwen4b persistent-smoke || return $?
  printf 'Two persistent training/search cycles and final test\n' > "$control/status"
  bash ladder/run_cycles.sh whale_qwen4b persistent-smoke || return $?
}
while true; do
  if validate > "$control/attempt-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1; then
    printf 'PASSED: two cycles and final test\n' > "$control/status"
    break
  else
    result=$?
    printf 'FAILED exit=%s; allocation retained for inspection. Create %s/retry to retry.\n' "$result" "$control" > "$control/status"
    # Batch jobs survive laptop disconnections. Waiting here preserves the
    # allocation for diagnosis, bounded by its normal four-hour wall time.
    while [[ ! -f "$control/retry" ]]; do sleep 15; done
    mv "$control/retry" "$control/retry-$(date -u +%Y%m%dT%H%M%SZ)"
  fi
done
