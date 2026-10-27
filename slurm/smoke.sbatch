#!/bin/bash
# One-cycle smoke test of one arm on 4 H100s.
#
#   sbatch -A <psc-account> slurm/smoke.sbatch whale_deepseek
#
# Submit from the repository root. Check the GPU spec for this cluster with
#   sinfo -p GPU-shared -o "%P %G"
# and change --gpus below if it is not h100-80.
#SBATCH --job-name=ladder-smoke
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:4
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --signal=B:USR1@300

set -uo pipefail
ARM="${1:-whale_deepseek}"
PROFILE=smoke
cd "$SLURM_SUBMIT_DIR"
export LADDER_NUM_GPUS=4

echo "== preflight =="
bash setup/preflight.sh "$ARM" || { echo "preflight failed; nothing was run"; exit 1; }

# The run is started in its own process group so a timeout or scancel can stop
# every server and Ray worker it started, not just the top-level script.
child=""
cleanup() {
  echo "== stopping (signal) =="
  [[ -n "$child" ]] && kill -TERM -- "-$child" 2>/dev/null
  # Only this job's servers and Ray processes; other jobs on the node are untouched.
  ( export ARM PROFILE; source ladder/arm_env.sh; bash ladder/hooks/post_harness.sh 0 0; stop_job_ray ) || true
}
trap cleanup USR1 TERM INT

echo "== run $ARM / $PROFILE =="
setsid bash ladder/run_arm.sh "$ARM" "$PROFILE" &
child=$!
wait "$child"; status=$?

echo "== report =="
( export ARM PROFILE; source ladder/arm_env.sh
  python tools/report.py --run-name "$RUN_NAME" --run-out "$RUN_OUT" \
    --ckpt-dir "$CKPT_ROOT/$RUN_NAME" --mh-runs "$DOMAIN_DIR/meta_harness/runs" \
    --gpus "$LADDER_NUM_GPUS" --smoke-batch "$TRAIN_BATCH_SIZE" --smoke-steps "$ROUND_STEPS" \
    --smoke-proposals "$PROPOSALS_PER_ITER" ) || echo "report failed"
exit "$status"
