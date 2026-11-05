#!/bin/bash -l
# Release the allocation on success or failure. Optional failure diagnosis is
# bounded across manual retries; it never extends the Slurm allocation.
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
hold_seconds="${LADDER_FAILURE_HOLD_SECONDS:-0}"
if [[ ! "$hold_seconds" =~ ^[0-9]{1,4}$ ]] || (( 10#$hold_seconds > 1200 )); then
  echo 'LADDER_FAILURE_HOLD_SECONDS must be an integer from 0 to 1200' >&2
  exit 2
fi
hold_seconds=$(( 10#$hold_seconds ))
hold_deadline=0
repo="${SLURM_SUBMIT_DIR:-$PWD}"
config="${1:-}"
parent="${config:+$(dirname "$config")}"
parent="${parent:-$repo/.candidate-control}"
control="$parent/jobs/candidate-${SLURM_JOB_ID:-local}"
mkdir -p "$control" || control="$(mktemp -d /tmp/ladder-control-XXXXXX)"
wait_for_retry() {
  local result="$1"
  if (( hold_seconds == 0 )); then
    printf 'FAILED: exit=%s; allocation released\n' "$result" | tee "$control/status"
    return 1
  fi
  # One deadline for this invocation: failed retries cannot reset the hold.
  if (( hold_deadline == 0 )); then hold_deadline=$(( SECONDS + hold_seconds )); fi
  printf 'FAILED: exit=%s; allocation retained for diagnosis until elapsed=%ss; create %s/retry to retry\n' \
    "$result" "$hold_deadline" "$control" | tee "$control/status"
  while (( SECONDS < hold_deadline )); do
    if [[ -f "$control/retry" ]]; then
      mv "$control/retry" "$control/retry-$(date -u +%Y%m%dT%H%M%SZ)-$SECONDS" || return 1
      return 0
    fi
    sleep 1
  done
  printf 'FAILED: exit=%s; diagnosis hold expired; allocation released\n' "$result" | tee "$control/status"
  return 1
}
trap 'exit 143' TERM
trap 'exit 130' INT
while true; do
  # Handle bootstrap failures in the same bounded loop, without re-exec.
  if [[ -z "$config" || ! -r "$config" ]]; then
    echo 'Missing explicit job configuration' >&2
    wait_for_retry 2 && continue
    exit 2
  fi
  if ! cd "$repo"; then
    wait_for_retry 2 && continue
    exit 2
  fi
  attempt="$control/attempt-$(date -u +%Y%m%dT%H%M%SZ).log"
  printf 'RUNNING: %s\n' "$attempt" > "$control/status"
  if srun --unbuffered --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-24}" \
      --gpus=h100-80:2 --export=ALL bash "$repo/ladder/validate_candidate.sh" "$config" > "$attempt" 2>&1; then
    printf 'PASSED: worker completed\n' >> "$attempt"
    printf 'PASSED: worker completed; allocation released\n' | tee "$control/status"
    exit 0
  else
    result=$?
    printf 'FAILED: worker exit=%s\n' "$result" >> "$attempt"
  fi
  wait_for_retry "$result" || exit "$result"
done
