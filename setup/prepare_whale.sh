#!/usr/bin/env bash
# WHALE at the pinned commit with every ladder patch applied. Called by
# setup/install.sh (full GPU install) and setup/install_mock.sh (CPU-only test
# install); safe to rerun, each patch is skipped when already applied.
set -euo pipefail
: "${LADDER_HOME:?}" "${WHALE_DIR:?}" "${WHALE_REPO:?}" "${WHALE_COMMIT:?}"
say() { echo "[prepare-whale $(date '+%T')] $*"; }

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

# Restore missing files from this same pinned release, then apply runtime fixes.
python3 "$LADDER_HOME/setup/repair_chess_release.py" "$WHALE_DIR"
for compat_patch in vllm-routed-experts vllm-streamed-tied-weights torch-padding-fallback vllm-pause-generation shared-fsdp2 candidate-validation; do
  patch="$LADDER_HOME/patches/$compat_patch.patch"
  if ! git -C "$WHALE_DIR" apply --check -R "$patch" 2>/dev/null; then
    git -C "$WHALE_DIR" apply "$patch"
  fi
done
install -m 644 "$LADDER_HOME/setup/streamed_weight_loading.py" "$WHALE_DIR/domains/chess_puzzles/verl/utils/vllm/streamed_weight_loading.py"
