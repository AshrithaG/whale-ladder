#!/usr/bin/env bash
# Stage 1 of the pipeline: tiny end-to-end test in an interactive GPU session.
#
#   # on a login node, inside tmux so a dropped connection does not kill it:
#   tmux new -s ladder
#   srun -A <account> -p GPU-shared --gpus=h100-80:4 -t 01:00:00 --pty bash
#   cd $PROJECT/whale-ladder && bash ladder/interactive_test.sh whale_self
#
# ARM defaults to whale_self (no internet or API key needed, and it exercises
# the most code). Each call starts from scratch: earlier tiny outputs of the
# same arm are deleted, because a resumed run would skip the stages under test.
# Expect about 30-45 minutes. Ends with a PASS/FAIL line per stage.
set -uo pipefail

ARM="${1:-whale_self}"
PROFILE=tiny
LADDER_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LADDER_HOME"
export LADDER_NUM_GPUS=4
t0=$SECONDS

echo "== preflight =="
bash setup/preflight.sh "$ARM" || { echo "preflight failed; fix the FAIL lines first"; exit 1; }

echo "== fresh start for $ARM / $PROFILE =="
RUN_OUT="$(
  export ARM PROFILE; source ladder/arm_env.sh >/dev/null 2>&1 || { echo "arm setup failed" >&2; exit 1; }
  # Delete only paths that are set, end in this tiny run's name, and sit under
  # the ladder's own output folders.
  [[ -n "${RUN_NAME:-}" && "$RUN_NAME" == *-tiny-* ]] || { echo "refusing to clean: run name '${RUN_NAME:-}' is not a tiny run" >&2; exit 1; }
  [[ -n "${OUT_ROOT:-}" && -n "${CKPT_ROOT:-}" && -n "${DOMAIN_DIR:-}" \
     && "$RUN_OUT" == "$OUT_ROOT/$RUN_NAME" && "$CKPT_ROOT" == "$OUT_ROOT/ckpt" ]] \
    || { echo "refusing to clean: unexpected paths" >&2; exit 1; }
  rm -rf -- "$OUT_ROOT/$RUN_NAME" "$OUT_ROOT/ckpt/$RUN_NAME" "$DOMAIN_DIR/meta_harness/runs/mh-$RUN_NAME-round"*
  mkdir -p "$RUN_OUT/state"
  # A 32-puzzle slice of the test set, so the validation pass is quick.
  python - "$DOMAIN_DIR/data/chess_puzzle/lichess_puzzles_test_256.parquet" "$RUN_OUT/state/test_32.parquet" <<'PY'
import sys
import pandas as pd
pd.read_parquet(sys.argv[1]).head(32).to_parquet(sys.argv[2], index=False)
PY
  echo "$RUN_OUT"
)" || exit 1
export VAL_FILES="$RUN_OUT/state/test_32.parquet"

cleanup() {
  echo "== interrupted: stopping this job's servers =="
  ( export ARM PROFILE; source ladder/arm_env.sh >/dev/null 2>&1; bash ladder/hooks/post_harness.sh 0 0; stop_job_ray ) || true
  exit 130
}
trap cleanup INT TERM

echo "== run (logs: $RUN_OUT/logs/driver.log) =="
bash ladder/run_arm.sh "$ARM" "$PROFILE"
run_status=$?
trap - INT TERM

echo "== checks =="
(
  export ARM PROFILE; source ladder/arm_env.sh >/dev/null 2>&1
  ck="$CKPT_ROOT/$RUN_NAME/global_step_$ROUND_STEPS"
  mh="$DOMAIN_DIR/meta_harness/runs/mh-$RUN_NAME-round1"
  fails=0
  say() { if eval "$2"; then echo "PASS  $1"; else echo "FAIL  $1"; fails=$(( fails + 1 )); fi; }
  say "run exited cleanly (status $run_status)"           "(( run_status == 0 ))"
  say "weight phase saved a checkpoint"                   "[[ -d '$ck/actor' ]]"
  say "validation ran after the step"                     "grep -q -E 'val-(core|aux)/|val/' '$CKPT_ROOT/$RUN_NAME'/logs/*.log 2>/dev/null"
  if [[ "$ARM" != rsft_only ]]; then
    say "checkpoint merged for serving"                   "[[ -f '$ck/hf/config.json' ]]"
    say "solver server came up"                           "grep -q 'solver ready' '$RUN_OUT/logs/driver.log'"
    say "proposer session ran and exited 0"               "grep -q '\"exit_code\": 0' '$RUN_OUT/proposer_cost.jsonl' 2>/dev/null"
    say "proposer wrote a candidate that was scored"      "grep -q '\"status\": \"evaluated\"' '$mh/logs/evolution_summary.jsonl' 2>/dev/null"
    say "a harness was accepted"                          "[[ -s '$mh/logs/accepted_harness.txt' ]]"
    say "servers stopped afterwards"                      "[[ ! -s '$RUN_OUT/state/pids' ]]"
  fi
  python tools/report.py --run-name "$RUN_NAME" --run-out "$RUN_OUT" \
    --ckpt-dir "$CKPT_ROOT/$RUN_NAME" --mh-runs "$DOMAIN_DIR/meta_harness/runs" \
    --gpus 4 --smoke-batch "$TRAIN_BATCH_SIZE" --smoke-steps "$ROUND_STEPS" \
    --smoke-proposals "$PROPOSALS_PER_ITER" > /dev/null 2>&1 \
    && echo "PASS  report written: $RUN_OUT/report.md" || { echo "FAIL  report"; fails=$(( fails + 1 )); }
  echo "elapsed: $(( (SECONDS - t0) / 60 )) min"
  if (( fails == 0 )); then
    echo "ALL STAGES PASSED for $ARM. Next: sbatch the smoke test (slurm/smoke.sbatch)."
  else
    echo "$fails check(s) failed. Send $RUN_OUT/logs/ and the output above."
    exit 1
  fi
)
