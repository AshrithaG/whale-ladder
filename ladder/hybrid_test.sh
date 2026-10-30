#!/usr/bin/env bash
# Two GPUs share generation and FSDP2 training. Isolated validation only.
set -euo pipefail
test_dir="${1:?output directory required}"
mkdir -p "$test_dir"
test_dir="$(cd "$test_dir" && pwd)"
dry_run="${2:-}"
mode="${HYBRID_MODE:-train}"
training_steps="${HYBRID_TRAINING_STEPS:-2}"
[[ "$training_steps" =~ ^[12]$ ]] || { echo 'Hybrid smoke requires one or two training steps' >&2; exit 2; }
start_step=0
if [[ "$mode" == train && -n "${HYBRID_RESUME_CHECKPOINT:-}" ]]; then
  [[ "$HYBRID_RESUME_CHECKPOINT" =~ /global_step_([0-9]+)$ ]] || { echo 'Invalid resume checkpoint path' >&2; exit 2; }
  start_step="${BASH_REMATCH[1]}"
  test -d "$HYBRID_RESUME_CHECKPOINT/actor"
  test -f "$HYBRID_RESUME_CHECKPOINT/data.pt"
fi
target_step=$((start_step+training_steps))
case "$mode" in train|eval) ;; *) echo 'HYBRID_MODE must be train or eval' >&2; exit 2 ;; esac
export ARM=rsft_only PROFILE=smoke LADDER_NUM_GPUS=2
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/arm_env.sh"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use an existing Slurm allocation' >&2; exit 2; }
[[ "$(job_gpus 0,1)" == "$LADDER_JOB_GPUS" ]] || { echo 'Exactly two assigned GPUs are required' >&2; exit 2; }
export RUN_NAME="${HYBRID_RUN_NAME:-hybrid-check-j${SLURM_JOB_ID:?}}"
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid hybrid run name' >&2; exit 2; }
export RUN_OUT="$test_dir" OUTPUT_ROOT="$test_dir/checkpoints"
export PROJECT_DIR="$DOMAIN_DIR"
export RAY_TMPDIR="/tmp/ray-${USER}-${SLURM_JOB_ID}-hy"
export N_GPUS_PER_NODE=2 NNODES=1 ROLLOUT_GPUS_PER_NODE=2 ROLLOUT_NNODES=1
export TRAIN_BATCH_SIZE=256 PPO_MINI_BATCH_SIZE=256 ROLLOUT_N=8 VAL_ROLLOUT_N=2
validation_puzzles=16
if [[ "$mode" == eval ]]; then
  : "${HYBRID_SOURCE_CHECKPOINT:?evaluation requires a checkpoint}"
  validation_puzzles=256
  [[ "$HYBRID_SOURCE_CHECKPOINT" =~ /global_step_([0-9]+)$ ]] || { echo "Invalid evaluation checkpoint" >&2; exit 2; }
  export HYBRID_EVAL_STEP="${BASH_REMATCH[1]}"
  export VAL_ROLLOUT_N=8
fi
export CHESS_PUZZLE_ALGO=rsft ENABLE_ONLINE_RSFT=true
# update_sft_policy receives a shard and interprets this field per worker.
# Keep the global SFT minibatch at 64 across the two data-parallel workers.
export RSFT_SFT_EPOCHS=1 RSFT_SFT_MINI_BATCH_SIZE=32 RSFT_SFT_MICRO_PER_GPU=1
export TRAINER_SAVE_FREQ=1 TRAINER_TEST_FREQ=1 TRAINER_TOTAL_EPOCHS=1 TRAINER_VAL_BEFORE_TRAIN=False
export FSDP_STRATEGY=fsdp2 FSDP_NATIVE_CPU_OFFLOAD=True
export FSDP_PARAM_OFFLOAD=False FSDP_OPTIM_OFFLOAD=False USE_KL_LOSS=False
export ENFORCE_EAGER=false GPU_MEM_UTIL="${HYBRID_GPU_MEM_UTIL:-0.75}" AGENT_NUM_WORKERS="${HYBRID_AGENT_NUM_WORKERS:-8}" REWARD_NUM_WORKERS=1
export TRAINER_LOGGER='[console,file]' TRAINER_LOGGER_IF_NO_WANDB='[console,file]'
export VERL_FILE_LOGGER_PATH="$test_dir/training-metrics.jsonl"
mkdir -p "$test_dir"
if [[ -f "$test_dir/passed.json" ]]; then
  echo 'Hybrid test already passed; preserving results.'
  exit 0
fi
if [[ -f "$VERL_FILE_LOGGER_PATH" && "$dry_run" != --dry-run ]]; then
  echo 'Use a fresh output directory for another hybrid attempt.' >&2
  exit 2
fi
python "$LADDER_HOME/setup/prepare_hybrid_launcher.py" "$DOMAIN_DIR" "$test_dir/launcher"
export LADDER_HYBRID_PREAMBLE="$test_dir/launcher/preamble_hybrid.py"
export VAL_FILES="$test_dir/validation-$validation_puzzles.parquet"
python - "$DOMAIN_DIR/data/chess_puzzle/lichess_puzzles_test_256.parquet" "$VAL_FILES" "$validation_puzzles" <<'PY'
import sys
import pyarrow.parquet as pq
pq.write_table(pq.read_table(sys.argv[1]).slice(0,int(sys.argv[3])),sys.argv[2])
PY
unset VAL_FILES_HYDRA TRAIN_FILES_HYDRA VAL_DATA_DIR ROLLOUT_DATA_DIR
args=(trainer.resume_mode=disable "trainer.total_training_steps=$target_step"
      trainer.use_legacy_worker_impl=enable
      actor_rollout_ref.rollout.checkpoint_engine.backend=naive
      actor_rollout_ref.rollout.free_cache_engine=True
      actor_rollout_ref.rollout.disable_log_stats=False
      "actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=${HYBRID_GDN_PREFILL_BACKEND:-triton}"
      actor_rollout_ref.rollout.tensor_model_parallel_size=1
      actor_rollout_ref.rollout.data_parallel_size=1
      actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
      '~trainer.disaggregated_rsft')
if [[ "$mode" == train && -n "${HYBRID_RESUME_CHECKPOINT:-}" ]]; then
  args+=(trainer.resume_mode=resume_path "trainer.resume_from_path=$HYBRID_RESUME_CHECKPOINT")
fi
if [[ "$mode" == eval ]]; then
  args+=(trainer.resume_mode=resume_path "trainer.resume_from_path=$HYBRID_SOURCE_CHECKPOINT"
         trainer.val_only=True trainer.val_before_train=True)
fi
if [[ "$dry_run" == --dry-run ]]; then
  bash "$test_dir/launcher/train_hybrid.sh" "${args[@]}" --cfg job > "$test_dir/composed-config.log" 2>&1
  echo 'Hybrid config composed; GPU execution is not yet validated.'
  exit 0
fi
cleanup() {
  stop_job_ray
}
trap cleanup EXIT
wait_gpus_free 120
echo "HYBRID CHECK START: mode=$mode; 2 GPUs; memory=$GPU_MEM_UTIL; backend=${HYBRID_GDN_PREFILL_BACKEND:-triton}; 2048 rollouts per full batch"
timeout --signal=TERM --kill-after=60s "${HYBRID_TIMEOUT_S:-5400}s" bash "$test_dir/launcher/train_hybrid.sh" "${args[@]}" 2>&1 | tee "$test_dir/console.log"
export HYBRID_VALIDATION_START_STEP="$start_step"
python - "$VERL_FILE_LOGGER_PATH" "$OUTPUT_ROOT/$RUN_NAME" "$test_dir/passed.json" "$mode" "$training_steps" <<'PY'
import json,sys
from pathlib import Path
rows=[json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()]
steps={r['step']:r['data'] for r in rows}
if sys.argv[4]=='eval':
    import os
    step=int(os.environ['HYBRID_EVAL_STEP'])
    samples=[json.loads(x) for x in (Path(sys.argv[2])/'val_rollouts'/f'{step}.jsonl').read_text().splitlines()]
    assert len(samples)==2048
    assert any(k.startswith('val-core/') for m in steps.values() for k in m), 'No evaluation metrics'
    Path(sys.argv[3]).write_text(json.dumps({'passed':True,'mode':'eval','validation_rollouts':2048,
        'output_characters':sum(len(x.get('output') or '') for x in samples),
        'assistant_tokens':sum(v for x in samples if isinstance((v:=x.get('assistant_generated_tokens_tokenized')),(int,float))),
        'metrics':steps},indent=2)+'\n')
    sys.exit(0)
training_steps=int(sys.argv[5])
import os
start_step=int(os.environ.get('HYBRID_VALIDATION_START_STEP','0'))
target_step=start_step+training_steps
for step in range(start_step+1,target_step+1):
    m=steps[step]
    assert m.get('online_rsft/total')==2048, 'Wrong rollout count'
    assert m.get('online_rsft/accepted',0)>0 and m.get('actor/sft_updates',0)>0, 'No actual weight update'
    assert m['actor/sft_updates']==(m['online_rsft/accepted']+63)//64, 'Wrong global SFT minibatch schedule'
    assert m.get('timing_s/update_weights',0)>0, 'No weight handoff'
    assert len((Path(sys.argv[2])/'rollouts'/f'{step}.jsonl').read_text().splitlines())==2048
assert len((Path(sys.argv[2])/'val_rollouts'/f'{target_step}.jsonl').read_text().splitlines())==32
assert any(k.startswith('val-core/') for k in steps[target_step]), 'No post-update validation'
Path(sys.argv[3]).write_text(json.dumps({'passed':True,'steps':training_steps,'start_step':start_step,'target_step':target_step,'rollouts_per_step':2048,'metrics':steps},indent=2)+'\n')
PY
if [[ "$mode" == eval ]]; then
  echo 'HYBRID EVALUATION PASSED: 256 puzzles x 8 attempts'
else
  echo "HYBRID CHECK PASSED: $training_steps generation/update transitions plus post-update validation"
fi
