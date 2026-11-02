#!/usr/bin/env bash
# Layer 1: run the whole experiment for all five arms on CPU, with fakes.
#
#   export LADDER_ROOT=$HOME/whale-ladder-mock     # a scratch folder just for the mock
#   bash setup/install_mock.sh                       # once
#   bash ladder/mock_run.sh
#
# Real: ladder/run_cycles.sh, hybrid_test.sh, search_two_gpu.sh,
# tools/two_gpu_search.py, the router, WHALE's search loop and chess scoring.
# Fake (tests/mock/): training, checkpoint merging, model servers, the
# proposer, LiteLLM and nvidia-smi. The fake proposer deliberately writes bad
# candidates (broken Python, missing functions, wrong names, no output, a crash)
# and one training phase is made to fail once, to prove the pipeline records
# and survives them. Ends with PASS/FAIL per check (tests/mock/check_mock.py).
# Runs on cerlab23 or PSC; refuses to run on macOS.
set -uo pipefail

[[ "$(uname -s)" != Darwin ]] || { echo "mock_run.sh does not run on macOS; use cerlab23 or PSC" >&2; exit 2; }
LADDER_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${LADDER_ROOT:?set LADDER_ROOT to a scratch folder for the mock, e.g. \$HOME/whale-ladder-mock}"
export LADDER_ROOT
set -a
source "$LADDER_HOME/ladder/config.env"
set +a
[[ -x "$VENV/bin/python" && -d "$WHALE_DIR/.git" ]] || { echo "run setup/install_mock.sh first" >&2; exit 2; }
cd "$LADDER_HOME"

MOCK="$LADDER_HOME/tests/mock"
export PATH="$MOCK/bin:$VENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1 MOCK_NUM_GPUS=2
unset SLURM_JOB_ID LADDER_JOB_GPUS
export LADDER_TRAIN_LAUNCHER="$MOCK/fake_train.sh"
export LADDER_MERGE_CMD="$VENV/bin/python $MOCK/fake_merge.py"
export MOCK_STATE_DIR="$OUT_ROOT/mock-state"
chmod +x "$MOCK"/bin/* "$MOCK/fake_train.sh"

# Fresh start: only this mock's own outputs, all under LADDER_ROOT.
[[ "$OUT_ROOT" == "$LADDER_ROOT"/* && "$WHALE_DIR" == "$LADDER_ROOT"/* ]] || { echo "unexpected paths" >&2; exit 2; }
rm -rf -- "$OUT_ROOT/cycles/"*-mock-s7 "$OUT_ROOT/"whale_*-mock-s7 "$MOCK_STATE_DIR" \
  "$WHALE_DIR/domains/chess_puzzles/meta_harness/runs/mh-search-jlocal"*
mkdir -p "$MOCK_STATE_DIR"

echo "== mock data =="
python "$MOCK/make_data.py" "$WHALE_DIR/domains/chess_puzzles" || exit 1

# What each rung's fake proposer does on successive calls (6 calls per arm).
export MOCK_PROPOSER_PLAN_DEEPSEEK="good"
export MOCK_PROPOSER_PLAN_QWEN27B="synonym,good"
export MOCK_PROPOSER_PLAN_QWEN4B="broken_python,garbage_manifest,missing_function,wrong_slot,good,no_output"
export MOCK_PROPOSER_PLAN_SELF="no_output,crash,good"
# Cycle 2's training (target step 4) in the qwen4b arm fails once.
export MOCK_FAIL_ONCE_TRAIN="whale_qwen4b-mock-s7:4"

started=$SECONDS
for arm in rsft_only whale_deepseek whale_qwen27b whale_qwen4b whale_self; do
  echo "== $arm (log: $MOCK_STATE_DIR/$arm.log) =="
  bash ladder/run_cycles.sh "$arm" mock >> "$MOCK_STATE_DIR/$arm.log" 2>&1
  status=$?
  if [[ "$arm" == whale_qwen4b ]]; then
    if (( status == 0 )); then
      echo "   the injected training failure did not stop the run"
    else
      echo "   injected failure stopped the run as expected; relaunching to test resume"
      bash ladder/run_cycles.sh "$arm" mock >> "$MOCK_STATE_DIR/$arm.log" 2>&1
      status=$?
    fi
  fi
  echo "   exit $status"
done
echo "== layer-2 tool: proposer format check on a saved search run =="
saved="$(cat "$OUT_ROOT/cycles/whale_deepseek-mock-s7/whale_deepseek-c1-search/harness-run-dir" 2>/dev/null)"
for arm in whale_deepseek whale_qwen4b; do
  bash ladder/proposer_format_check.sh "$arm" mock "$saved" 2 >> "$MOCK_STATE_DIR/format-$arm.log" 2>&1
  echo "   $arm exit $?"
done
echo "== checks ($(( SECONDS - started ))s) =="
python "$MOCK/check_mock.py" "$OUT_ROOT/cycles" "$MOCK_STATE_DIR"
