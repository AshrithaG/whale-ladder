#!/usr/bin/env bash
# The whole experiment for one arm:
#
#   CYCLES x ( train STEPS_PER_CYCLE steps with the current harness
#              -> search SEARCH_ITERS iterations, accept the best harness )
#   then score the final model + harness on the held-out test puzzles.
#
#   bash ladder/run_cycles.sh ARM PROFILE
#
# ARM      rsft_only | whale_deepseek | whale_qwen27b | whale_qwen4b | whale_self
#          (rsft_only trains the same steps and never searches)
# PROFILE  ladder/profiles/<PROFILE>.env: h100 (real runs), mini4090, mock
#
# Training is ladder/hybrid_test.sh and search is ladder/search_two_gpu.sh;
# this script only chains them and hands the checkpoint and the accepted
# harness from one phase to the next. Every phase leaves passed.json when it
# succeeds, so relaunching resumes at the first unfinished phase; an
# interrupted phase is moved aside (never deleted) and redone.
# Results: $OUT_ROOT/cycles/<ARM>-<PROFILE>-s<SEED>/results.json
set -euo pipefail

ARM="${1:?usage: run_cycles.sh ARM PROFILE}"
PROFILE="${2:?usage: run_cycles.sh ARM PROFILE}"
LADDER_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$ARM" in
  rsft_only|whale_deepseek|whale_qwen27b|whale_qwen4b|whale_self) ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac

set -a
source "$LADDER_HOME/ladder/config.env"
source "$LADDER_HOME/ladder/profiles/$PROFILE.env"
set +a
: "${CYCLES:?profile must set CYCLES}" "${STEPS_PER_CYCLE:?profile must set STEPS_PER_CYCLE}"
: "${SEARCH_ITERS:?profile must set SEARCH_ITERS}" "${SEED:?profile must set SEED}"
if [[ " ${LADDER_SKIP_ARMS:-} " == *" $ARM "* ]]; then
  echo "$ARM cannot run with profile $PROFILE (LADDER_SKIP_ARMS)" >&2
  exit 3
fi
export PROFILE

# The persistent mode keeps the PR4 output layout and candidate-validation
# rules. It is opt-in while GPU validation is pending.
if [[ "${LADDER_PERSISTENT_SEARCH:-0}" == 1 ]]; then
  export ARM
  source "$LADDER_HOME/ladder/arm_env.sh"
  exec python "$LADDER_HOME/tools/persistent_cycles.py" --arm "$ARM" --profile "$PROFILE"
fi

run_name="${ARM}-${PROFILE}-s${SEED}"
state="$OUT_ROOT/cycles/$run_name"
mkdir -p "$state"
log() { echo "[cycles $(date '+%F %T')] $*" | tee -a "$state/cycles.log" >&2; }
phase_ok() { [[ -f "$1/passed.json" ]]; }
# An unfinished phase directory is kept for inspection under a new name.
set_aside() {
  if [[ -e "$1" ]] && ! phase_ok "$1"; then
    local dest; dest="$1.unfinished-$(date +%Y%m%d-%H%M%S)"
    mv "$1" "$dest"
    log "moved unfinished $(basename "$1") aside to $(basename "$dest")"
  fi
}

# The released WHALE base harness h0 is where every arm starts.
harness="$WHALE_DIR/domains/chess_puzzles/environments/chess_puzzle/base_harness.py"
[[ -f "$harness" ]] || { log "ERROR: base harness missing: $harness"; exit 1; }
prev_ckpt=""
log "=== $ARM / $PROFILE: $CYCLES cycle(s) x $STEPS_PER_CYCLE step(s), $SEARCH_ITERS search iteration(s) ==="

for (( c = 1; c <= CYCLES; c++ )); do
  train_dir="$state/c${c}-train"
  target=$(( c * STEPS_PER_CYCLE ))
  cp "$harness" "$state/c${c}-trained-with.py"
  if phase_ok "$train_dir"; then
    log "cycle $c: training already passed"
  else
    set_aside "$train_dir"
    log "cycle $c: training steps $(( target - STEPS_PER_CYCLE + 1 ))-$target"
    (
      export HARNESS_PATH="$state/c${c}-trained-with.py" HYBRID_MODE=train
      export HYBRID_TRAINING_STEPS="$STEPS_PER_CYCLE" HYBRID_RUN_NAME="$run_name"
      if [[ -n "$prev_ckpt" ]]; then export HYBRID_RESUME_CHECKPOINT="$prev_ckpt"; else unset HYBRID_RESUME_CHECKPOINT; fi
      bash "$LADDER_HOME/ladder/hybrid_test.sh" "$train_dir"
    ) || { log "ERROR: cycle $c training failed; logs in $train_dir"; exit 1; }
  fi
  ckpt="$train_dir/checkpoints/$run_name/global_step_$target"
  [[ -d "$ckpt/actor" ]] || { log "ERROR: expected checkpoint $ckpt"; exit 1; }

  if [[ "$ARM" != rsft_only ]]; then
    # The arm name keeps WHALE's search run name unique when several arms run
    # in one job (the mock runs all five).
    search_dir="$state/${ARM}-c${c}-search"
    if phase_ok "$search_dir"; then
      log "cycle $c: search already passed"
    else
      set_aside "$search_dir"
      log "cycle $c: search, $SEARCH_ITERS iteration(s) from $(basename "$harness")"
      (
        export ARM BASELINE_HARNESS_OVERRIDE="$harness"
        bash "$LADDER_HOME/ladder/search_two_gpu.sh" "$search_dir" "$ckpt" "$SEARCH_ITERS"
      ) || { log "ERROR: cycle $c search failed; logs in $search_dir"; exit 1; }
    fi
    run_dir="$(cat "$search_dir/harness-run-dir")"
    accepted="$(tr -d '[:space:]' < "$run_dir/logs/accepted_harness.txt")"
    cp "$run_dir/harnesses/$accepted/harness.py" "$state/c${c}-accepted.py"
    log "cycle $c: accepted $accepted"
    harness="$state/c${c}-accepted.py"
  fi
  prev_ckpt="$ckpt"
done

test_dir="$state/final-test"
if phase_ok "$test_dir"; then
  log "final test already passed"
else
  set_aside "$test_dir"
  log "final test: last checkpoint + last accepted harness on the held-out test puzzles"
  cp "$harness" "$state/final-harness.py"
  (
    export HARNESS_PATH="$state/final-harness.py" HYBRID_MODE=eval
    export HYBRID_SOURCE_CHECKPOINT="$prev_ckpt" HYBRID_RUN_NAME="${run_name}-test"
    bash "$LADDER_HOME/ladder/hybrid_test.sh" "$test_dir"
  ) || { log "ERROR: final test failed; logs in $test_dir"; exit 1; }
fi

source "$VENV/bin/activate"
python "$LADDER_HOME/tools/summarize_cycles.py" "$state" --arm "$ARM" --profile "$PROFILE"
log "=== $ARM done: $state/results.json ==="
