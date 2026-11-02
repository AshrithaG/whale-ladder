#!/usr/bin/env bash
# CPU-only persistence integration check, after setup/install_mock.sh.
set -euo pipefail
[[ "$(uname -s)" != Darwin ]] || { echo 'Run the mock on Linux (or in the documented Docker container)' >&2; exit 2; }
LADDER_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${LADDER_ROOT:?Use an isolated mock root}"
set -a
source "$LADDER_HOME/ladder/config.env"
set +a
MOCK="$LADDER_HOME/tests/mock"
export PATH="$MOCK/bin:$VENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1 MOCK_NUM_GPUS=2
unset SLURM_JOB_ID LADDER_JOB_GPUS MOCK_FAIL_ONCE_TRAIN
export LADDER_TRAIN_LAUNCHER="$MOCK/fake_train.sh"
export LADDER_MERGE_CMD="$VENV/bin/python $MOCK/fake_merge.py"
export MOCK_STATE_DIR="$OUT_ROOT/persistent-mock-state"
[[ ! -e "$MOCK_STATE_DIR" ]] || { echo 'Use a fresh mock root or preserve/rename the previous persistent mock results' >&2; exit 2; }
mkdir -p "$MOCK_STATE_DIR"
chmod +x "$MOCK"/bin/* "$MOCK/fake_train.sh"
python "$MOCK/make_data.py" "$WHALE_DIR/domains/chess_puzzles"
export MOCK_PROPOSER_PLAN_QWEN4B="broken_python,missing_function,no_output,good"
bash "$LADDER_HOME/ladder/run_cycles.sh" whale_qwen4b mock-persistent > "$MOCK_STATE_DIR/run.log" 2>&1
# Finished phases must be reusable without generating or training again.
bash "$LADDER_HOME/ladder/run_cycles.sh" whale_qwen4b mock-persistent > "$MOCK_STATE_DIR/resume.log" 2>&1
python "$MOCK/check_persistent.py" "$OUT_ROOT/cycles/whale_qwen4b-mock-persistent-s7" "$MOCK_STATE_DIR"
