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

# One cycle controller for every arm. Persistence controls server lifetime;
# it does not select another training/search/evaluation loop.
export ARM
source "$LADDER_HOME/ladder/arm_env.sh"
exec python "$LADDER_HOME/tools/cycle_controller.py" --arm "$ARM" --profile "$PROFILE"
