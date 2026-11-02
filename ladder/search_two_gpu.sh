#!/usr/bin/env bash
# One standalone search phase in an existing allocation, using a saved solver.
#
# Defaults are the H100 run: ARM=whale_qwen4b, 2 GPUs, 256 scoring puzzles.
# ladder/run_cycles.sh sets ARM for the other rungs; profiles override
# LADDER_MH_LIMIT, PROPOSALS_PER_ITER, LADDER_GPU_CHECK (H100 | 4090 | skip),
# LADDER_SHARE_ONE_GPU=1 (solver and proposer take turns on one GPU),
# LADDER_ALLOW_NO_SLURM=1 (cerlab23 and the mock have no Slurm).
set -euo pipefail
out="${1:?fresh output directory required}"
checkpoint="${2:?completed global_step checkpoint directory required}"
iterations="${3:-1}"
[[ "$iterations" =~ ^[1-6]$ ]] || { echo 'Expected 1-6 search iterations' >&2; exit 2; }
[[ ! -e "$out" ]] || { echo 'Use a fresh output directory; previous results are preserved' >&2; exit 2; }
checkpoint="$(cd "$checkpoint" && pwd)"
test -d "$checkpoint/actor"
test -f "$checkpoint/data.pt"
[[ -n "${SLURM_JOB_ID:-}" || "${LADDER_ALLOW_NO_SLURM:-0}" == 1 ]] || { echo 'Run inside an existing two-H100 allocation' >&2; exit 2; }
job_tag="${SLURM_JOB_ID:-local$$}"
export ARM="${ARM:-whale_qwen4b}" PROFILE="${PROFILE:-smoke}" LADDER_NUM_GPUS=2
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/arm_env.sh"
if [[ "${LADDER_SHARE_ONE_GPU:-0}" == 1 ]]; then
  [[ "$(job_gpus 0)" == "$LADDER_JOB_GPUS" ]] || { echo 'LADDER_SHARE_ONE_GPU=1 needs exactly one assigned GPU' >&2; exit 2; }
else
  [[ "$(job_gpus 0,1)" == "$LADDER_JOB_GPUS" ]] || { echo 'Exactly two assigned GPUs are required' >&2; exit 2; }
fi
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
export LADDER_CYCLE_SEARCH_NAME="mh-search-j$job_tag-$(basename "$out")"
[[ "$LADDER_CYCLE_SEARCH_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Output name must use letters, digits, underscores or hyphens' >&2; exit 2; }
export BASELINE_HARNESS_OVERRIDE="${BASELINE_HARNESS_OVERRIDE:-$BASE_HARNESS}"
export CHESS_PUZZLE_PROPOSER_RETRIES=1 PROPOSALS_PER_ITER="${PROPOSALS_PER_ITER:-3}" LADDER_MH_LIMIT="${LADDER_MH_LIMIT:-256}"
mkdir -p "$CLAUDE_CONFIG_DIR" "$out/logs"
if [[ "${LADDER_GPU_CHECK:-H100}" != skip ]]; then
  python - "${LADDER_GPU_CHECK:-H100}" "$([[ "${LADDER_SHARE_ONE_GPU:-0}" == 1 ]] && echo 1 || echo 2)" <<'PY'
import sys
import torch
name, count = sys.argv[1], int(sys.argv[2])
assert torch.cuda.device_count() == count, (torch.cuda.device_count(), count)
assert all(name in torch.cuda.get_device_name(i) for i in range(count)), name
PY
fi
merge_ckpt "$checkpoint"
python "$LADDER_HOME/tools/make_mh_config.py" "$DOMAIN_DIR/meta_harness/config-chess-puzzle.json" "$SWITCH_CONFIG"
cd "$DOMAIN_DIR"
python "$LADDER_HOME/tools/two_gpu_search.py"
