#!/usr/bin/env bash
# Check one proposer rung end to end in a few minutes, before spending GPU hours:
# serve the model (if local), start LiteLLM, and run one Claude Code session
# through WHALE's own claude_wrapper that must use the Write tool.
#
#   bash ladder/proposer_check.sh ARM      (whale_deepseek | whale_qwen27b | whale_qwen4b | whale_self)
#
# whale_self has no trained checkpoint yet, so it is checked with the base
# weights, i.e. exactly as whale_qwen4b.
set -euo pipefail

export ARM="${1:?usage: proposer_check.sh ARM}"
export PROFILE=check
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/arm_env.sh"
[[ -n "$PROPOSER_BACKEND" ]] || { ladder_log "ERROR: $ARM has no proposer"; exit 2; }
trap 'stop_servers' EXIT

if [[ "$PROPOSER_BACKEND" == local ]]; then
  model="$PROPOSER_MODEL_PATH"
  [[ "$model" == "@checkpoint" ]] && model="$SOLVER_MODEL_PATH"
  # In the real run the proposer gets only its own GPUs; here it may use the
  # same number of GPUs, starting from GPU 0.
  gpus="$(seq -s, 0 $(( PROPOSER_TP - 1 )))"
  start_vllm proposer "$model" proposer "$gpus" "$PROPOSER_PORT" \
    "${PROPOSER_MAX_MODEL_LEN:-131072}" "$PROPOSER_TP" 1 \
    --enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER:-qwen3_coder}" \
    --reasoning-parser "${REASONING_PARSER:-qwen3}"
  wait_http "http://127.0.0.1:$PROPOSER_PORT/v1/models" 1800 proposer
fi

python "$LADDER_HOME/tools/make_litellm_config.py" > "$RUN_OUT/state/litellm.yaml"
start_bg litellm "$RUN_OUT/logs/litellm.log" \
  litellm --config "$RUN_OUT/state/litellm.yaml" --host 127.0.0.1 --port "$LITELLM_PORT"
wait_http "http://127.0.0.1:$LITELLM_PORT/health/liveliness" 300 litellm

work="$RUN_OUT/check_workdir"
rm -rf "$work"; mkdir -p "$work"
cd "$DOMAIN_DIR"
status=0
python - "$work" <<'PY' || status=$?
import json, sys, time
from pathlib import Path
from meta_harness import claude_wrapper

work = Path(sys.argv[1])
prompt = (
    "Use the Write tool to create a file named check.txt in the current directory "
    "whose entire content is the single word ladder. Then reply with the single word DONE."
)
t0 = time.time()
result = claude_wrapper.run(
    prompt=prompt,
    model="ladder-proposer",
    allowed_tools=["Read", "Write", "Edit", "Glob", "Grep"],
    cwd=str(work),
    log_dir=str(work / "claude_sessions"),
    name="proposer-check",
    timeout_seconds=900,
)
target = work / "check.txt"
ok_file = target.exists() and target.read_text().strip() == "ladder"
summary = {
    "exit_code": result.exit_code,
    "wrote_file_correctly": ok_file,
    "final_text": result.text.strip()[:200],
    "tool_calls": [tc.name for tc in result.tool_calls],
    "tool_errors": sum(1 for tc in result.tool_calls if tc.is_error),
    "token_usage": result.token_usage,
    "seconds": round(time.time() - t0, 1),
    "stderr_tail": result.stderr[-800:],
}
print(json.dumps(summary, indent=2))
(work / "check_result.json").write_text(json.dumps(summary, indent=2))
sys.exit(0 if (result.exit_code == 0 and ok_file) else 1)
PY
if (( status == 0 )); then
  ladder_log "PASS: $ARM proposer path works (result in $work/check_result.json)"
else
  ladder_log "FAIL: $ARM proposer path; see $work/check_result.json and $RUN_OUT/logs/"
fi
exit "$status"
