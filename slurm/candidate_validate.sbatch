#!/bin/bash -l
# Allocation owner: setup and experiment failures remain inside an srun child.
# Keep the allocation after success, worker failure, or bootstrap errors.
#SBATCH --job-name=ladder-persist
#SBATCH --partition=GPU-shared
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --gres=gpu:h100-80:2
#SBATCH --time=04:00:00
#SBATCH --export=ALL
set +e
set -uo pipefail
repo="${SLURM_SUBMIT_DIR:-$PWD}"
config="${1:-}"
parent="${config:+$(dirname "$config")}"
parent="${parent:-$repo/.candidate-control}"
control="$parent/jobs/candidate-${SLURM_JOB_ID:-local}"
mkdir -p "$control" || control="$(mktemp -d /tmp/ladder-control-XXXXXX)"
hold() {
  local result="$1"
  printf 'exit=%s; allocation retained; control=%s; create retry after diagnosis\n' "$result" "$control" | tee "$control/status"
  while [[ ! -f "$control/retry" ]]; do sleep 15; done
  mv "$control/retry" "$control/retry-$(date -u +%Y%m%dT%H%M%SZ)"
}
bootstrap_failure() {
  local result=$?
  trap - EXIT
  hold "$result"
  exec bash "$repo/slurm/candidate_validate.sbatch" "$config"
}
trap bootstrap_failure EXIT
trap 'trap - EXIT; exit 143' TERM
trap 'trap - EXIT; exit 130' INT
[[ -n "$config" && -r "$config" ]] || { echo 'Missing explicit job configuration' >&2; exit 2; }
cd "$repo" || exit 2
while true; do
  attempt="$control/attempt-$(date -u +%Y%m%dT%H%M%SZ).log"
  printf 'RUNNING: %s\n' "$attempt" > "$control/status"
  if srun --unbuffered --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-24}" \
      --gpus=h100-80:2 --export=ALL bash "$repo/ladder/validate_candidate.sh" "$config" > "$attempt" 2>&1; then
    printf 'PASSED: worker completed\n' >> "$attempt"
    result=0
  else
    result=$?
    printf 'FAILED: worker exit=%s\n' "$result" >> "$attempt"
  fi
  hold "$result"
done
