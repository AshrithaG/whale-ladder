#!/usr/bin/env bash
# One standalone search phase in an existing allocation, using a saved solver.
set -euo pipefail
out="${1:?fresh output directory required}"
checkpoint="${2:?completed global_step checkpoint directory required}"
iterations="${3:-1}"
[[ "$iterations" =~ ^[1-6]$ ]] || { echo 'Expected 1-6 search iterations' >&2; exit 2; }
[[ ! -e "$out" ]] || { echo 'Use a fresh output directory; previous results are preserved' >&2; exit 2; }
checkpoint="$(cd "$checkpoint" && pwd)"
test -d "$checkpoint/actor"
test -f "$checkpoint/data.pt"
: "${SLURM_JOB_ID:?Run inside an existing two-H100 allocation}"
export ARM=whale_qwen4b PROFILE=smoke LADDER_NUM_GPUS=2
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/arm_env.sh"
[[ "$(job_gpus 0,1)" == "$LADDER_JOB_GPUS" ]] || { echo 'Exactly two assigned GPUs are required' >&2; exit 2; }
export SOLVER_GPUS=0 SOLVER_DP=1 PROPOSER_GPUS=1 PROPOSER_TP=1
case "${SWITCH_MODE:-search}" in
  search) ;;
  benchmark) : "${SWITCH_REFERENCE_RUN:?benchmark mode needs saved h1/h2/h3 candidates}" ;;
  *) echo 'SWITCH_MODE must be search or benchmark' >&2; exit 2 ;;
esac
mkdir -p "$out"
out="$(cd "$out" && pwd)"
export RUN_OUT="$out" SWITCH_OUT="$out" SWITCH_CHECKPOINT="$checkpoint"
export SWITCH_ITERATIONS="$iterations" SWITCH_CONFIG="$out/state/mh_config.json"
export CLAUDE_CONFIG_DIR="$out/state/claude-config" LADDER_COST_LEDGER="$out/proposer_cost.jsonl"
export LADDER_CYCLE_SEARCH_NAME="mh-search-j$SLURM_JOB_ID-$(basename "$out")"
[[ "$LADDER_CYCLE_SEARCH_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Output name must use letters, digits, underscores or hyphens' >&2; exit 2; }
export BASELINE_HARNESS_OVERRIDE="${BASELINE_HARNESS_OVERRIDE:-$BASE_HARNESS}"
export CHESS_PUZZLE_PROPOSER_RETRIES=1 PROPOSALS_PER_ITER=3 LADDER_MH_LIMIT=256
mkdir -p "$CLAUDE_CONFIG_DIR" "$out/logs"
python - <<'PY'
import torch
assert torch.cuda.device_count() == 2
assert all('H100' in torch.cuda.get_device_name(i) for i in range(2))
PY
merge_ckpt "$checkpoint"
python "$LADDER_HOME/tools/make_mh_config.py" "$DOMAIN_DIR/meta_harness/config-chess-puzzle.json" "$SWITCH_CONFIG"
cd "$DOMAIN_DIR"
python "$LADDER_HOME/tools/two_gpu_search.py"
