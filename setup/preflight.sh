#!/usr/bin/env bash
# Check that a run can start, without starting one. Run on the GPU node (the
# Slurm scripts call it first) so the GPU and network checks are real.
#
#   bash setup/preflight.sh ARM
#
# Prints PASS/FAIL per check and exits 1 if anything failed.
set -uo pipefail

ARM="${1:-whale_deepseek}"
LADDER_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a
source "$LADDER_HOME/ladder/config.env"
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

check "4 GPUs visible" bash -c '[[ $(nvidia-smi -L | wc -l) -ge 4 ]] && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1'
check "WHALE patch applied" grep -q PRE_HARNESS_HOOK "$WHALE_DIR/run/lib/alternate.sh"
check "torch sees CUDA" python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.device_count())'
check "vllm >= 0.17" python -c 'import vllm; from packaging.version import Version as V; assert V(vllm.__version__) >= V("0.17"), vllm.__version__; print(vllm.__version__)'
check "transformers >= 5" python -c 'import transformers; assert int(transformers.__version__.split(".")[0]) >= 5, transformers.__version__; print(transformers.__version__)'
check "verl trainer imports" python -c 'import verl.trainer.main_textarena_disagg_rsft'
check "chess runner imports (shim)" python -c 'import autoharness_chess_puzzle.runner'
check "litellm installed" python -c 'import litellm'
check "claude CLI" "$VENV/bin/claude" --version
for f in lichess_puzzles_train_16384 lichess_puzzles_mh_val_256 lichess_puzzles_test_256; do
  check "data $f" test -f "$DOMAIN_DIR/data/chess_puzzle/$f.parquet"
done
check "solver weights cached" python -c "from huggingface_hub import snapshot_download as s; print(s('$SOLVER_MODEL', local_files_only=True))"
if [[ "$ARM" == whale_qwen27b ]]; then
  check "27B weights cached" python -c "from huggingface_hub import snapshot_download as s; print(s('$QWEN27B_MODEL', local_files_only=True))"
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
