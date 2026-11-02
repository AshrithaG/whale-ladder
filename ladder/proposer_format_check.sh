#!/usr/bin/env bash
# Layer 2: does this rung's proposer write valid candidates? Minutes, no training.
#
#   bash ladder/proposer_format_check.sh ARM PROFILE SAVED_RUN_DIR [SESSIONS] [CHECKPOINT_HF]
#
# ARM            whale_deepseek | whale_qwen27b | whale_qwen4b | whale_self
# SAVED_RUN_DIR  a finished WHALE search run directory (e.g. B1's search records)
# SESSIONS       proposer sessions to sample (default 3)
# CHECKPOINT_HF  for whale_self: the merged checkpoint to propose with
#                (default: the base solver weights)
# Serves the proposer on the job's first GPU(s) (none for the API rung), starts
# LiteLLM, runs tools/proposer_format_check.py, and stops everything it started.
# Output: $OUT_ROOT/<ARM>-<PROFILE>-s<SEED>/format_check/format_check.json
set -euo pipefail

export ARM="${1:?usage: proposer_format_check.sh ARM PROFILE SAVED_RUN_DIR [SESSIONS] [CHECKPOINT_HF]}"
export PROFILE="${2:?profile required}"
saved="$(cd "${3:?saved run directory required}" && pwd)"
sessions="${4:-3}"
checkpoint_hf="${5:-}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/arm_env.sh"
[[ -n "$PROPOSER_BACKEND" ]] || { ladder_log "ERROR: $ARM has no proposer"; exit 2; }
trap 'stop_servers' EXIT
out="$RUN_OUT/format_check"
mkdir -p "$out"
export LADDER_COST_LEDGER="$out/proposer_cost.jsonl"

if [[ "$PROPOSER_BACKEND" == local ]]; then
  model="$PROPOSER_MODEL_PATH"
  if [[ "$model" == "@checkpoint" ]]; then model="${checkpoint_hf:-$SOLVER_MODEL_PATH}"; fi
  tp="$PROPOSER_TP"
  start_vllm proposer "$model" proposer "$(seq -s, 0 $(( tp - 1 )))" "$PROPOSER_PORT" \
    "${PROPOSER_MAX_MODEL_LEN:-131072}" "$tp" 1 \
    --enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER:-qwen3_coder}" \
    --reasoning-parser "${REASONING_PARSER:-qwen3}"
  wait_http "http://127.0.0.1:$PROPOSER_PORT/v1/models" 1800 proposer
fi
python "$LADDER_HOME/tools/make_litellm_config.py" > "$RUN_OUT/state/litellm.yaml"
start_bg litellm "$RUN_OUT/logs/litellm.log" \
  litellm --config "$RUN_OUT/state/litellm.yaml" --host 127.0.0.1 --port "$LITELLM_PORT"
wait_http "http://127.0.0.1:$LITELLM_PORT/health/liveliness" 300 litellm

cd "$DOMAIN_DIR"
python "$LADDER_HOME/tools/proposer_format_check.py" "$saved" "$out" "$sessions"
