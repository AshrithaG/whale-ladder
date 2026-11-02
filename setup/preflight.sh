#!/usr/bin/env bash
# Check that a run can start, without starting one. Run on the GPU node (the
# Slurm scripts call it first) so the GPU and network checks are real.
#
#   bash setup/preflight.sh ARM [PROFILE]
#
# Prints PASS/FAIL per check and exits 1 if anything failed.
set -uo pipefail

ARM="${1:-whale_deepseek}"
PROFILE="${2:-${PROFILE:-smoke}}"
LADDER_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a
source "$LADDER_HOME/ladder/config.env" || exit 1
source "$LADDER_HOME/ladder/profiles/$PROFILE.env" || exit 1
[[ -f "$LADDER_HOME/ladder/secrets.env" ]] && source "$LADDER_HOME/ladder/secrets.env"
set +a
fails=0
check() {  # check NAME CMD...
  local name="$1"; shift
  if out="$("$@" 2>&1)"; then echo "PASS  $name ${out:+($out)}"
  else echo "FAIL  $name: ${out:-failed}"; fails=$(( fails + 1 )); fi
}

source "$VENV/bin/activate" 2>/dev/null || { echo "FAIL  venv missing at $VENV; run setup/install.sh"; exit 1; }
DOMAIN_DIR="$WHALE_DIR/domains/chess_puzzles"
export PYTHONPATH="$DOMAIN_DIR"

# Count the job's own GPUs (CUDA_VISIBLE_DEVICES under Slurm), not every GPU
# nvidia-smi can see on a shared node.
check "job has 4 GPUs" bash -c 'n=$(if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then tr "," "\n" <<< "$CUDA_VISIBLE_DEVICES" | wc -l; else nvidia-smi -L | wc -l; fi); [[ $n -ge 4 ]] && echo "$n GPUs (${CUDA_VISIBLE_DEVICES:-all}), $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"'
check "WHALE patch applied" grep -q PRE_HARNESS_HOOK "$WHALE_DIR/run/lib/alternate.sh"
check "torch sees CUDA" python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.device_count())'
check "declared package dependencies" python -m pip check
check "host compiler supports CUDA kernel C++ requirements" bash -c 'printf "#include <type_traits>\nstatic_assert(std::is_same_v<int, int>);\n" | "${CXX:-g++}" -std=c++20 -x c++ -fsyntax-only -'
check "vllm >= 0.17" python -c 'import vllm; from packaging.version import Version as V; assert V(vllm.__version__) >= V("0.17"), vllm.__version__; print(vllm.__version__)'
check "vLLM CLI startup" vllm --version
check "transformers >= 5" python -c 'import transformers; assert int(transformers.__version__.split(".")[0]) >= 5, transformers.__version__; print(transformers.__version__)'
check "verl trainer imports" python -c 'import verl.trainer.main_textarena_disagg_rsft'
check "chess agent-loop imports" python -c 'import verl.experimental.agent_loop.chess_puzzle_agent_loop'
check "disaggregated FSDP worker imports" python -c 'import verl.workers.disagg_fsdp_workers'
check "training padding forward/backward" python "$LADDER_HOME/setup/check_padding_backend.py"
check "vLLM rollout integration imports" python -c 'import verl.workers.rollout.vllm_rollout.vllm_async_server'
check "streamed tied-weight validation" python "$LADDER_HOME/setup/check_streamed_weights.py"
check "NCCL checkpoint backend available" python "$LADDER_HOME/setup/check_checkpoint_backend.py"
check "Hydra trainer config composes" python -c 'from hydra import compose, initialize_config_module; ctx = initialize_config_module(config_module="verl.trainer.config", version_base=None); ctx.__enter__(); config = compose(config_name="ppo_trainer"); assert "data" in config; ctx.__exit__(None, None, None)'
check "chess runner imports (shim)" python -c 'import autoharness_chess_puzzle.runner'
check "litellm installed" python -c 'import litellm'
check "claude CLI" "$VENV/bin/claude" --version
for f in lichess_puzzles_train_16384 lichess_puzzles_mh_val_256 lichess_puzzles_test_256; do
  check "data $f" test -f "$DOMAIN_DIR/data/chess_puzzle/$f.parquet"
done
# Everyone running an arm must have identical puzzles; compare this fingerprint.
data_fp() {
  local fp; fp="$(python "$LADDER_HOME/tools/data_fingerprint.py" "$DOMAIN_DIR/data/chess_puzzle" | tail -1)" || return 1
  if [[ -n "$CHESS_DATA_FINGERPRINT" && "$fp" != "$CHESS_DATA_FINGERPRINT" ]]; then
    echo "$fp, expected $CHESS_DATA_FINGERPRINT"; return 1
  fi
  echo "$fp"
}
check "chess data fingerprint" data_fp
check "solver weights @ ${SOLVER_MODEL_REVISION:0:7}" python -c "from huggingface_hub import snapshot_download as s; print(s('$SOLVER_MODEL', revision='$SOLVER_MODEL_REVISION', local_files_only=True))"
check "weight-transfer buffer fits model tensors" python "$LADDER_HOME/setup/check_transfer_buffer.py" \
  "$HF_HOME/hub/models--${SOLVER_MODEL//\//--}/snapshots/$SOLVER_MODEL_REVISION" "${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB:-3072}"
# Use the real launcher's overrides and entrypoint, without creating workers.
# Keep its generated logs separate from training logs.
training_config() (
  export ARM PROFILE
  source "$LADDER_HOME/ladder/arm_env.sh" || exit 1
  export RUN_NAME="$RUN_NAME-preflight" OUTPUT_ROOT="$LADDER_ROOT/preflight"
  log="$LADDER_ROOT/preflight/$RUN_NAME.log"
  mkdir -p "$(dirname "$log")"
  if timeout 180 bash "$TRAIN_SCRIPT" --cfg job > "$log" 2>&1; then
    echo "composed actual launcher configuration: $log"
  else
    tail -60 "$log"; return 1
  fi
)
check "actual training launcher configuration" training_config
if [[ "$ARM" == whale_qwen27b ]]; then
  check "27B weights @ ${QWEN27B_REVISION:0:7}" python -c "from huggingface_hub import snapshot_download as s; print(s('$QWEN27B_MODEL', revision='$QWEN27B_REVISION', local_files_only=True))"
fi
if [[ "$ARM" == whale_deepseek ]]; then
  check "price set for the dollar cap" bash -c "[[ '$DEEPSEEK_PRICE_IN_PER_M' != 0 && '$DEEPSEEK_PRICE_OUT_PER_M' != 0 ]]"
  if [[ "$DEEPSEEK_PROVIDER" == fireworks ]]; then
    check "FIREWORKS_API_KEY set" test -n "${FIREWORKS_API_KEY:-}"
    check "FIREWORKS_DEEPSEEK_MODEL set" test -n "$FIREWORKS_DEEPSEEK_MODEL"
    check "outbound internet to Fireworks" curl -sS -o /dev/null -m 15 -w '%{http_code}' https://api.fireworks.ai/inference/v1/models
  else
    check "DEEPSEEK_API_KEY set" test -n "${DEEPSEEK_API_KEY:-}"
    check "outbound internet to DeepSeek" curl -sS -o /dev/null -m 15 -w '%{http_code}' https://api.deepseek.com
  fi
fi
check "free space under LADDER_ROOT >= 200 GB" bash -c "df -BG --output=avail '$LADDER_ROOT' | tail -1 | tr -dc 0-9 | { read g; [[ \$g -ge 200 ]] && echo \${g}G; }"

if (( fails > 0 )); then echo "$fails check(s) failed"; exit 1; fi
echo "all checks passed"
