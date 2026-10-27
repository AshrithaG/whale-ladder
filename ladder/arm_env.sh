# Environment for one arm of the proposer ladder. Sourced by run_arm.sh and
# proposer_check.sh after they set ARM and PROFILE; do not execute it.
#
# ARM      rsft_only       weight updates only, no proposer (the baseline)
#          whale_deepseek  WHALE, proposer = DeepSeek Flash over the API (strong)
#          whale_qwen27b   WHALE, proposer = Qwen ~27B served on 2 of the GPUs (medium)
#          whale_qwen4b    WHALE, proposer = frozen base Qwen3.5-4B (weak)
#          whale_self      WHALE, proposer = the checkpoint just trained (self-improvement)
#
# Every whale_* arm uses the same Claude Code proposer scaffold and WHALE skill
# files; only the model behind it changes.

: "${ARM:?ARM must be set}" "${PROFILE:?PROFILE must be set}"

LADDER_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LADDER_HOME

set -a
source "$LADDER_HOME/ladder/config.env"
[[ -f "$LADDER_HOME/ladder/secrets.env" ]] && source "$LADDER_HOME/ladder/secrets.env"
source "$LADDER_HOME/ladder/profiles/$PROFILE.env"
set +a

source "$VENV/bin/activate"
source "$LADDER_HOME/ladder/lib.sh"

# ---- run identity and output layout -------------------------------------
export RUN_NAME="${ARM}-${PROFILE}-s${SEED}"
export RUN_OUT="$OUT_ROOT/$RUN_NAME"
export CKPT_ROOT="$OUT_ROOT/ckpt"          # alternate.sh: $CKPT_ROOT/$RUN_NAME
export OUTPUT_ROOT="$CKPT_ROOT"            # training script: same directory
export DATA_SEED="$SEED"
mkdir -p "$RUN_OUT/logs" "$RUN_OUT/state"

# Ports derived from the Slurm job id, so jobs sharing a GPU-shared node do not
# collide. Outside Slurm the shell pid is used.
base=$(( 20000 + ( ${SLURM_JOB_ID:-$$} % 2000 ) * 10 ))
export SOLVER_PORT=$(( base + 1 )) PROPOSER_PORT=$(( base + 2 )) LITELLM_PORT=$(( base + 3 ))
export LADDER_SOLVER_KEY="EMPTY"

# The GPUs Slurm gave this job, captured once before anything narrows
# CUDA_VISIBLE_DEVICES. lib.sh maps the rung's job-relative indices onto these.
export LADDER_JOB_GPUS="${LADDER_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
# A private Ray temp dir per job: two jobs on one node then never share a Ray
# cluster, and this job's Ray processes can be found and stopped without
# touching the other's. Kept short and node-local (Ray's socket paths have a
# ~107 character limit, which $PROJECT paths can exceed).
export RAY_TMPDIR="/tmp/ray-${USER:-u}-${SLURM_JOB_ID:-$$}"
mkdir -p "$RAY_TMPDIR"

# ---- WHALE chess domain, as in domains/chess_puzzles/run/whale.sh ---------
export DOMAIN_DIR="$WHALE_DIR/domains/chess_puzzles"
export PYTHONPATH="$DOMAIN_DIR:${PYTHONPATH:-}"
export TRAIN_SCRIPT="$DOMAIN_DIR/scripts/train_chess_puzzle_multinode_disagg.sh"
export MH_MODULE="meta_harness.meta_harness_chess_puzzle"
export BASE_HARNESS="$DOMAIN_DIR/environments/chess_puzzle/base_harness.py"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # weights are pre-downloaded by setup

# model_path REPO REVISION : local directory of that exact downloaded commit.
# Training and serving load from this path, so the pinned revision is what
# runs even if the Hub repo has moved on. Falls back to the repo id (with a
# warning) only where the snapshot is missing, e.g. during cleanup on a
# machine without the weights; preflight fails in that case before any run.
model_path() {
  python - "$1" "$2" 2>/dev/null <<'PY' || { ladder_log "WARN: no local snapshot of $1@${2:0:7}; using the repo id"; echo "$1"; }
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2], local_files_only=True))
PY
}
export SOLVER_MODEL_PATH="$(model_path "$SOLVER_MODEL" "$SOLVER_MODEL_REVISION")"
export BASE_MODEL="$SOLVER_MODEL_PATH"

# ---- proposer rung ----------------------------------------------------------
export LADDER_RUNG="${ARM#whale_}"
export PROPOSER_BACKEND=""
case "$ARM" in
  rsft_only)
    LADDER_RUNG="none" ;;
  whale_deepseek)
    PROPOSER_BACKEND=api
    export SOLVER_GPUS=0,1,2,3 SOLVER_DP=4
    export LADDER_PRICE_IN_PER_M="$DEEPSEEK_PRICE_IN_PER_M"
    export LADDER_PRICE_CACHED_IN_PER_M="$DEEPSEEK_PRICE_CACHED_IN_PER_M"
    export LADDER_PRICE_OUT_PER_M="$DEEPSEEK_PRICE_OUT_PER_M"
    if [[ "$DEEPSEEK_PRICE_IN_PER_M" == 0 || "$DEEPSEEK_PRICE_OUT_PER_M" == 0 ]]; then
      ladder_log "ERROR: set DEEPSEEK_PRICE_*_PER_M in ladder/config.env; the dollar cap cannot work at price 0"
      return 2 2>/dev/null || exit 2
    fi ;;
  whale_qwen27b)
    PROPOSER_BACKEND=local
    export PROPOSER_MODEL_PATH="$(model_path "$QWEN27B_MODEL" "$QWEN27B_REVISION")" PROPOSER_TP=2
    export SOLVER_GPUS=0,1 SOLVER_DP=2 PROPOSER_GPUS=2,3 ;;
  whale_qwen4b)
    PROPOSER_BACKEND=local
    export PROPOSER_MODEL_PATH="$SOLVER_MODEL_PATH" PROPOSER_TP=1
    export SOLVER_GPUS=0,1,2 SOLVER_DP=3 PROPOSER_GPUS=3 ;;
  whale_self)
    PROPOSER_BACKEND=local
    export PROPOSER_MODEL_PATH="@checkpoint" PROPOSER_TP=1
    export SOLVER_GPUS=0,1,2 SOLVER_DP=3 PROPOSER_GPUS=3 ;;
  *)
    ladder_log "ERROR: unknown arm $ARM"
    return 2 2>/dev/null || exit 2 ;;
esac
if [[ "$PROPOSER_BACKEND" == local ]]; then
  export LADDER_PRICE_IN_PER_M=0 LADDER_PRICE_CACHED_IN_PER_M=0 LADDER_PRICE_OUT_PER_M=0
fi

# ---- proposer: Claude Code pointed at this job's LiteLLM proxy -------------
if [[ -n "$PROPOSER_BACKEND" ]]; then
  export LITELLM_MASTER_KEY="sk-ladder-$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  unset ANTHROPIC_API_KEY
  export ANTHROPIC_BASE_URL="http://127.0.0.1:$LITELLM_PORT"
  export ANTHROPIC_AUTH_TOKEN="$LITELLM_MASTER_KEY"
  export ANTHROPIC_MODEL=ladder-proposer
  export ANTHROPIC_DEFAULT_OPUS_MODEL=ladder-proposer ANTHROPIC_DEFAULT_SONNET_MODEL=ladder-proposer
  export ANTHROPIC_DEFAULT_HAIKU_MODEL=ladder-proposer ANTHROPIC_SMALL_FAST_MODEL=ladder-proposer
  export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_TELEMETRY=1 DISABLE_ERROR_REPORTING=1 DISABLE_AUTOUPDATER=1
  export CLAUDE_CONFIG_DIR="$RUN_OUT/state/claude-config"
  export PROPOSER_MODEL=ladder-proposer
  export PROPOSER_EFFORT=""                # patched alternate.sh: empty = no --effort flag
  export LADDER_COST_LEDGER="$RUN_OUT/proposer_cost.jsonl"
  mkdir -p "$CLAUDE_CONFIG_DIR"
fi
