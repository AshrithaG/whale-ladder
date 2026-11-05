#!/usr/bin/env bash
# Failures return to the outer Slurm controller, which releases by default.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exports="$(python3 "$repo/setup/load_candidate_config.py" "${1:?job configuration required}")"
eval "$exports"
cd "$LADDER_CODE_DIR"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo 'Candidate checkout is dirty; commit or stash tracked changes before validation' >&2
  exit 2
fi
[[ "$(git rev-parse HEAD)" == "$EXPECTED_REVISION" ]] || { echo 'Unexpected candidate code revision' >&2; exit 2; }
installed="$(cat "$LADDER_ROOT/installed-revision")"
if [[ "$installed" != "$EXPECTED_REVISION" ]]; then
  [[ "${LADDER_REUSE_INSTALL_BASE:-}" == "$installed" ]] || {
    echo 'Installation differs; explicit verified reuse or fresh setup is required' >&2; exit 2;
  }
  "$LADDER_ROOT/venv/bin/python" setup/check_install_reuse.py --code "$LADDER_CODE_DIR" --prepared "$installed" \
    --runtime "$EXPECTED_REVISION" --output "$LADDER_ROOT/jobs/compatibility-$EXPECTED_REVISION.json"
fi
export LADDER_GCC_MODULE="${LADDER_GCC_MODULE:-gcc/13.3.1-p20240614}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
source ladder/config.env
profile="${LADDER_VALIDATION_PROFILE:-persistent-smoke}"
source "$LADDER_ROOT/venv/bin/activate"
export LADDER_JOB_GPUS="${LADDER_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:?missing Slurm GPU assignment}}"
control="$LADDER_ROOT/runs/modal-control"
mkdir -p "$control"
python tools/observe_run.py "$LADDER_ROOT/runs" > "$control/observer.log" 2>&1 &
observer=$!
phase=starting
completed=0
finish() {
  result=$?
  # A signal can reach EXIT while the last shell command still has status 0.
  # Only the normal end of all phases can mark the invocation completed.
  if [[ "$completed" != 1 && "$result" == 0 ]]; then result=1; fi
  python - "$control/status.json" "$phase" "$result" <<'PY'
import json,sys,time
from pathlib import Path
path=Path(sys.argv[1])
temporary=path.with_suffix('.tmp')
temporary.write_text(json.dumps(dict(provider='psc',phase=sys.argv[2],
    state='completed' if sys.argv[3]=='0' else 'failed',returncode=int(sys.argv[3]),ended_unix=time.time())))
temporary.replace(path)
PY
  kill -TERM "$observer" 2>/dev/null || true
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
run_phase() {
  phase="$1"; shift
  start="$(date +%s)"
  python - "$control/status.json" "$phase" <<'PY'
import json,sys,time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(dict(provider='psc',phase=sys.argv[2],started_unix=time.time())))
PY
  set +e
  "$@" 2>&1 | tee "$control/$phase.log"
  result=${PIPESTATUS[0]}
  set -e
  python - "$control/phases.jsonl" "$phase" "$start" "$result" <<'PY'
import json,sys,time
with open(sys.argv[1],'a') as f:f.write(json.dumps(dict(phase=sys.argv[2],started_unix=int(sys.argv[3]),
    ended_unix=time.time(),returncode=int(sys.argv[4]),elapsed_seconds=time.time()-int(sys.argv[3])))+'\n')
PY
  return "$result"
}
run_phase kernels bash setup/install_gpu_kernels.sh
run_phase preflight bash setup/preflight.sh whale_qwen4b "$profile"
run_phase cycles bash ladder/run_cycles.sh whale_qwen4b "$profile"
completed=1
