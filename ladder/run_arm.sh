#!/usr/bin/env bash
# Run one arm of the proposer ladder on WHALE's chess domain.
#
#   bash ladder/run_arm.sh ARM PROFILE
#
# ARM is one of rsft_only, whale_deepseek, whale_qwen27b, whale_qwen4b,
# whale_self (see ladder/arm_env.sh). PROFILE is a file in ladder/profiles/
# without .env, e.g. smoke. Relaunching the same ARM/PROFILE/SEED resumes:
# finished weight phases and harness phases are skipped.
set -euo pipefail

export ARM="${1:?usage: run_arm.sh ARM PROFILE}"
export PROFILE="${2:?usage: run_arm.sh ARM PROFILE}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/arm_env.sh"

ladder_log "arm=$ARM profile=$PROFILE run=$RUN_NAME rung=$LADDER_RUNG out=$RUN_OUT"
timing_event run_start "\"arm\": \"$ARM\", \"profile\": \"$PROFILE\", \"gpus\": ${LADDER_NUM_GPUS:-4}"
trap 'timing_event run_end "\"status\": $?"' EXIT

if [[ "$ARM" == rsft_only ]]; then
  # Same weight-update budget as one whale arm, no harness search.
  bash "$DOMAIN_DIR/run/rsft_only.sh" 2>&1 | tee -a "$RUN_OUT/logs/driver.log"
  exit "${PIPESTATUS[0]}"
fi

python "$LADDER_HOME/tools/make_mh_config.py" \
  "$DOMAIN_DIR/meta_harness/config-chess-puzzle.json" "$RUN_OUT/state/mh_config.json"
export MH_CONFIG="$RUN_OUT/state/mh_config.json"
export PRE_HARNESS_HOOK="$LADDER_HOME/ladder/hooks/pre_harness.sh"
export POST_HARNESS_HOOK="$LADDER_HOME/ladder/hooks/post_harness.sh"

bash "$WHALE_DIR/run/lib/alternate.sh" 2>&1 | tee -a "$RUN_OUT/logs/driver.log"
exit "${PIPESTATUS[0]}"
