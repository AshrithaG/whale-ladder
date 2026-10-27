#!/usr/bin/env bash
# Runs before every harness-search phase (called by WHALE's patched alternate.sh).
#
#   $1  checkpoint dir just produced by the weight phase (.../global_step_N)
#   $2  round number
#
# Serves that checkpoint as the solver on $SOLVER_PORT (what the candidate
# harnesses are scored against), serves the proposer model if the rung hosts
# one, and starts the LiteLLM proxy that Claude Code talks to.
set -euo pipefail
source "$LADDER_HOME/ladder/lib.sh"

ckpt="$1"; round="$2"
timing_event pre_harness_start "\"round\": $round"

wait_gpus_free 300
merge_ckpt "$ckpt"
timing_event merge_done "\"round\": $round"

# Solver: plain chat, no reasoning or tool parser, so the scored completions
# look exactly like the verl rollouts the model was trained on.
start_vllm solver "$ckpt/hf" "$SOLVER_SERVED_NAME" "$SOLVER_GPUS" "$SOLVER_PORT" \
  "${SOLVER_MAX_MODEL_LEN:-32768}" 1 "$SOLVER_DP"

case "$PROPOSER_BACKEND" in
  api)
    ;;
  local)
    # "self" proposes with the checkpoint just trained; "qwen4b" and "qwen27b"
    # with fixed Hugging Face weights.
    model="$PROPOSER_MODEL_PATH"
    [[ "$model" == "@checkpoint" ]] && model="$ckpt/hf"
    start_vllm proposer "$model" proposer "$PROPOSER_GPUS" "$PROPOSER_PORT" \
      "${PROPOSER_MAX_MODEL_LEN:-131072}" "$PROPOSER_TP" 1 \
      --enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER:-qwen3_coder}" \
      --reasoning-parser "${REASONING_PARSER:-qwen3}"
    ;;
  *)
    ladder_log "ERROR: unknown PROPOSER_BACKEND=$PROPOSER_BACKEND"; exit 2 ;;
esac

python "$LADDER_HOME/tools/make_litellm_config.py" > "$RUN_OUT/state/litellm.yaml"
start_bg litellm "$RUN_OUT/logs/litellm.log" \
  litellm --config "$RUN_OUT/state/litellm.yaml" --host 127.0.0.1 --port "$LITELLM_PORT"

wait_http "http://127.0.0.1:$SOLVER_PORT/v1/models" 1200 solver
if [[ "$PROPOSER_BACKEND" == local ]]; then
  wait_http "http://127.0.0.1:$PROPOSER_PORT/v1/models" 1800 proposer
fi
wait_http "http://127.0.0.1:$LITELLM_PORT/health/liveliness" 300 litellm

timing_event pre_harness_end "\"round\": $round"
