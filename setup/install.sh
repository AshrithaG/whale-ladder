#!/usr/bin/env bash
# One-time setup on PSC Bridges-2 (or any Linux box with conda and internet).
#
#   bash setup/install.sh            everything
#   SKIP_27B=1 bash setup/install.sh skip the ~55 GB Qwen 27B download
#
# Safe to rerun: every step skips work that is already done. Needs internet,
# so run it on a login node (or an interactive RM-shared session if a login
# node kills it). It uses no GPU.
set -euo pipefail

LADDER_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a
source "$LADDER_HOME/ladder/config.env"
[[ -f "$LADDER_HOME/ladder/secrets.env" ]] && source "$LADDER_HOME/ladder/secrets.env"
set +a
say() { echo "[install $(date '+%T')] $*"; }
mkdir -p "$LADDER_ROOT" "$HF_HOME" "$OUT_ROOT"

# ---- 1. WHALE at the pinned commit, with the ladder patch -------------------
if [[ ! -d "$WHALE_DIR/.git" ]]; then
  say "cloning WHALE"
  git clone -q "$WHALE_REPO" "$WHALE_DIR"
fi
git -C "$WHALE_DIR" fetch -q origin
git -C "$WHALE_DIR" -c advice.detachedHead=false checkout -q "$WHALE_COMMIT"
patch="$LADDER_HOME/patches/whale-ladder.patch"
if git -C "$WHALE_DIR" apply --check -R "$patch" 2>/dev/null; then
  say "WHALE patch already applied"
else
  git -C "$WHALE_DIR" apply "$patch"
  say "applied $patch"
fi

# ---- 2. Python env (conda for Python 3.11 + Node.js, venv on top) ----------
CONDA_ENV="$LADDER_ROOT/conda"
if [[ ! -x "$CONDA_ENV/bin/python" ]]; then
  command -v conda >/dev/null || module load anaconda3 2>/dev/null || true
  command -v conda >/dev/null || { echo "conda not found; on Bridges-2 run: module load anaconda3"; exit 1; }
  say "creating conda env with python 3.11 and nodejs 20"
  conda create -y -q -p "$CONDA_ENV" -c conda-forge python=3.11 "nodejs=20"
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
pip install -q -e "$LADDER_HOME/shim"
# Faster Gated DeltaNet kernels for Qwen3.5 training; transformers falls back
# to a slower path without them.
pip install -q flash-linear-attention || say "WARN: flash-linear-attention failed to install; training will be slower"
pip freeze > "$LADDER_ROOT/pip-freeze.txt"

# ---- 3. Claude Code CLI (the WHALE proposer scaffold) -----------------------
if [[ ! -x "$VENV/bin/claude" ]]; then
  say "installing Claude Code CLI into the venv"
  PATH="$CONDA_ENV/bin:$PATH" npm install -g -s --prefix "$VENV" @anthropic-ai/claude-code
fi
say "claude $("$VENV/bin/claude" --version 2>/dev/null || echo 'version unknown')"

# ---- 4. Model weights -------------------------------------------------------
hf_get() { if command -v hf >/dev/null; then hf download "$1"; else huggingface-cli download "$1"; fi > /dev/null; }
say "downloading $SOLVER_MODEL"
hf_get "$SOLVER_MODEL"
if [[ "${SKIP_27B:-0}" != 1 ]]; then
  say "downloading $QWEN27B_MODEL (large)"
  hf_get "$QWEN27B_MODEL"
fi

# ---- 5. Chess puzzle data (D_weight 16,384 / D_harness 256 / D_test 256) ----
data="$WHALE_DIR/domains/chess_puzzles/data/chess_puzzle"
if [[ -f "$data/lichess_puzzles_test_256.parquet" ]]; then
  say "chess data already prepared"
else
  say "preparing chess data from Lichess/chess-puzzles (the Hub dataset is not pinned; date recorded)"
  ( cd "$WHALE_DIR/domains/chess_puzzles" && python scripts/prepare_chess_puzzle_data.py )
  date -u +%FT%TZ > "$data/PREPARED_AT"
fi

say "done. Next: bash setup/preflight.sh (on a GPU node) or sbatch slurm/proposer_check.sbatch"
