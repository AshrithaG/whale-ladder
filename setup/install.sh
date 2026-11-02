#!/usr/bin/env bash
# One-time setup on PSC Bridges-2 (or any Linux box with conda and internet).
#
#   bash setup/install.sh            everything
#   SKIP_27B=1 bash setup/install.sh skip the ~55 GB Qwen 27B download
#   INSTALL_PROFILE=mini4090 SKIP_27B=1 bash setup/install.sh   cerlab23's 4090
#
# Run installation on a compute allocation with internet access.
# Rerunning can resolve newer dependencies; do not rerun during an experiment.
set -euo pipefail

LADDER_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a
source "$LADDER_HOME/ladder/config.env"
[[ -f "$LADDER_HOME/ladder/secrets.env" ]] && source "$LADDER_HOME/ladder/secrets.env"
# INSTALL_PROFILE=mini4090 downloads that profile's solver as well.
[[ -n "${INSTALL_PROFILE:-}" ]] && source "$LADDER_HOME/ladder/profiles/$INSTALL_PROFILE.env"
set +a
if [[ -n "${LADDER_GCC_MODULE:-}" ]]; then
  export HOSTNAME="$(hostname)"
  module load "$LADDER_GCC_MODULE"
  export CC="$(command -v gcc)" CXX="$(command -v g++)" CUDAHOSTCXX="$(command -v g++)"
fi
say() { echo "[install $(date '+%T')] $*"; }
# The PSC profiles select a captured constraints file. All pip invocations,
# including the helper scripts, inherit the same version restrictions.
if [[ -n "${INSTALL_CONSTRAINTS:-}" ]]; then
  [[ -f "$INSTALL_CONSTRAINTS" ]] || { say "ERROR: missing constraints: $INSTALL_CONSTRAINTS"; exit 2; }
  export PIP_CONSTRAINT="$(cd "$(dirname "$INSTALL_CONSTRAINTS")" && pwd)/$(basename "$INSTALL_CONSTRAINTS")"
fi
mkdir -p "$LADDER_ROOT" "$HF_HOME" "$OUT_ROOT"

# ---- 1. WHALE at the pinned commit, with every ladder patch ---------------
LADDER_HOME="$LADDER_HOME" bash "$LADDER_HOME/setup/prepare_whale.sh"

# ---- 2. Python env (conda for Python 3.11 + Node.js, venv on top) ----------
CONDA_ENV="$LADDER_ROOT/conda"
if [[ ! -x "$CONDA_ENV/bin/python" ]]; then
  command -v conda >/dev/null || module load anaconda3 2>/dev/null || true
  if ! command -v conda >/dev/null; then
    # No conda module (e.g. cerlab23): install conda-forge's Miniforge into LADDER_ROOT.
    if [[ ! -x "$LADDER_ROOT/miniforge/bin/conda" ]]; then
      say "conda not found; installing Miniforge into $LADDER_ROOT/miniforge"
      curl -fsSL -o "$LADDER_ROOT/miniforge.sh" \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh"
      bash "$LADDER_ROOT/miniforge.sh" -b -p "$LADDER_ROOT/miniforge"
    fi
    export PATH="$LADDER_ROOT/miniforge/bin:$PATH"
  fi
  say "creating conda env with python 3.11 and nodejs 20"
  conda create -y -q -p "$CONDA_ENV" --override-channels -c conda-forge "python=${INSTALL_PYTHON_VERSION:-3.11}" "nodejs=20"
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  "$CONDA_ENV/bin/python" -m venv "$VENV"
  # Node for Claude Code, without putting conda's python on PATH.
  ln -sf "$CONDA_ENV/bin/node" "$VENV/bin/node"
fi
source "$VENV/bin/activate"
pip install -q -U pip wheel

if ! python -c "import vllm" 2>/dev/null; then
  say "installing vllm (pulls a matching torch); Qwen3.5 needs vllm >= 0.17"
  pip install -q "vllm>=0.17"
fi
say "installing transformers v5, WHALE chess requirements, litellm, shim"
pip install -q "transformers>=5"
# WHALE pins numpy<2.0, which current vllm/torch wheels do not accept; keep the
# rest of the list as released.
grep -v -E '^\s*numpy' "$WHALE_DIR/domains/chess_puzzles/requirements.txt" > "$LADDER_ROOT/requirements-chess.txt"
pip install -q -r "$LADDER_ROOT/requirements-chess.txt"
pip install -q "litellm[proxy]" "huggingface_hub[cli]" openai
bash "$LADDER_HOME/setup/install_checkpoint_backend.sh"
pip install -q -e "$LADDER_HOME/shim"
# Faster Gated DeltaNet kernels for Qwen3.5 training; transformers falls back
# to a slower path without them.
pip install -q flash-linear-attention || say "WARN: flash-linear-attention failed to install; training will be slower"
if [[ "${INSTALL_SKIP_GPU_KERNELS:-0}" != 1 ]]; then
  bash "$LADDER_HOME/setup/install_gpu_kernels.sh"
fi
pip freeze > "$LADDER_ROOT/pip-freeze.txt"

# ---- 3. Claude Code CLI (the WHALE proposer scaffold) -----------------------
if [[ ! -x "$VENV/bin/claude" ]]; then
  say "installing Claude Code CLI into the venv"
  PATH="$CONDA_ENV/bin:$PATH" npm install -g -s --prefix "$VENV" "@anthropic-ai/claude-code${CLAUDE_CODE_VERSION:+@$CLAUDE_CODE_VERSION}"
fi
say "claude $("$VENV/bin/claude" --version 2>/dev/null || echo 'version unknown')"

# ---- 4. Model weights -------------------------------------------------------
# hf_get REPO REVISION : download one exact commit of a model.
hf_get() {
  if command -v hf >/dev/null; then hf download "$1" --revision "$2"
  else huggingface-cli download "$1" --revision "$2"; fi > /dev/null
}
say "downloading $SOLVER_MODEL @ ${SOLVER_MODEL_REVISION:0:7}"
hf_get "$SOLVER_MODEL" "$SOLVER_MODEL_REVISION"
if [[ "$WEAK_PROPOSER_MODEL@$WEAK_PROPOSER_REVISION" != "$SOLVER_MODEL@$SOLVER_MODEL_REVISION" ]]; then
  say "downloading weak proposer $WEAK_PROPOSER_MODEL @ ${WEAK_PROPOSER_REVISION:0:7}"
  hf_get "$WEAK_PROPOSER_MODEL" "$WEAK_PROPOSER_REVISION"
fi
if [[ "${SKIP_27B:-0}" != 1 ]]; then
  say "downloading $QWEN27B_MODEL @ ${QWEN27B_REVISION:0:7} (large)"
  hf_get "$QWEN27B_MODEL" "$QWEN27B_REVISION"
fi

# ---- 5. Chess puzzle data (D_weight 16,384 / D_harness 256 / D_test 256) ----
data="$WHALE_DIR/domains/chess_puzzles/data/chess_puzzle"
if [[ -f "$data/lichess_puzzles_test_256.parquet" ]]; then
  say "chess data already in place"
elif [[ -n "${CHESS_DATA_FROM:-}" ]]; then
  say "copying the shared chess data from $CHESS_DATA_FROM"
  mkdir -p "$data"
  cp "$CHESS_DATA_FROM"/lichess_puzzles_{train_16384,mh_val_256,test_256}.parquet "$data"/
else
  say "preparing chess data from Lichess/chess-puzzles (the Hub dataset is not pinned; date recorded)"
  ( cd "$WHALE_DIR/domains/chess_puzzles" && python scripts/prepare_chess_puzzle_data.py )
  date -u +%FT%TZ > "$data/PREPARED_AT"
fi
python "$LADDER_HOME/tools/data_fingerprint.py" "$data" | tee "$data/SHA256SUMS"
fp="$(tail -1 "$data/SHA256SUMS")"
if [[ -z "${CHESS_DATA_FINGERPRINT:-}" ]]; then
  say "chess data fingerprint: $fp"
  say "If other people run arms, set CHESS_DATA_FINGERPRINT=\"$fp\" in ladder/config.env, push it, and send them the three .parquet files in $data"
elif [[ "$fp" != "$CHESS_DATA_FINGERPRINT" ]]; then
  say "ERROR: chess data fingerprint $fp does not match CHESS_DATA_FINGERPRINT=$CHESS_DATA_FINGERPRINT"
  exit 1
else
  say "chess data matches the shared fingerprint $fp"
fi

say "done. Next: bash setup/preflight.sh (on a GPU node) or sbatch slurm/proposer_check.sbatch"
