# Shared helpers for the ladder scripts. Source it; do not execute it.
#
# Expects run_arm.sh to have exported RUN_OUT, WHALE_DIR, the port variables and
# the rung variables. Every background server started here is recorded in
# $RUN_OUT/state/pids so stop_servers can always clean up.

ladder_log() { echo "[ladder $(date '+%F %T')] $*" >&2; }

# Append one timing event to $RUN_OUT/timing.jsonl; tools/report.py reads it.
timing_event() {
  local event="$1"; shift
  mkdir -p "$RUN_OUT"
  printf '{"t": %s, "event": "%s"%s}\n' "$(date +%s.%N)" "$event" "${1:+, $1}" >> "$RUN_OUT/timing.jsonl"
}

pid_file() { mkdir -p "$RUN_OUT/state"; echo "$RUN_OUT/state/pids"; }

# start_bg NAME LOGFILE CMD... : run CMD in the background, remember its pid.
start_bg() {
  local name="$1" logfile="$2"; shift 2
  mkdir -p "$(dirname "$logfile")"
  nohup "$@" > "$logfile" 2>&1 &
  local pid=$!
  echo "$pid $name" >> "$(pid_file)"
  ladder_log "started $name (pid $pid), log $logfile"
}

# wait_http URL TIMEOUT_S NAME : poll until URL answers 200 or the process died.
wait_http() {
  local url="$1" timeout="$2" name="$3" waited=0
  while (( waited < timeout )); do
    if curl -sf -o /dev/null "$url"; then
      ladder_log "$name ready after ${waited}s"
      return 0
    fi
    if ! pid_alive "$name"; then
      ladder_log "ERROR: $name exited before becoming ready; see its log in $RUN_OUT/logs"
      return 1
    fi
    sleep 5; waited=$(( waited + 5 ))
  done
  ladder_log "ERROR: $name not ready after ${timeout}s"
  return 1
}

pid_alive() {
  local name="$1" pid
  pid="$(awk -v n="$name" '$2 == n {p=$1} END {print p}' "$(pid_file)" 2>/dev/null)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

stop_servers() {
  local f; f="$(pid_file)"
  [[ -f "$f" ]] || return 0
  while read -r pid name; do
    [[ -n "$pid" ]] || continue
    if kill -0 "$pid" 2>/dev/null; then
      ladder_log "stopping $name (pid $pid)"
      # vLLM forks engine workers; signal the whole process group too.
      kill -TERM "$pid" 2>/dev/null || true
      pkill -TERM -P "$pid" 2>/dev/null || true
    fi
  done < "$f"
  sleep 10
  while read -r pid name; do
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && { kill -KILL "$pid" 2>/dev/null || true; pkill -KILL -P "$pid" 2>/dev/null || true; }
  done < "$f"
  rm -f "$f"
}

# Block until every visible GPU uses less than 2 GiB, so the next phase does not
# start while the previous one's processes are still releasing memory.
wait_gpus_free() {
  local timeout="${1:-300}" waited=0 used
  while (( waited < timeout )); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)"
    if (( ${used:-0} < 2048 )); then return 0; fi
    sleep 5; waited=$(( waited + 5 ))
  done
  ladder_log "WARN: GPUs still hold ${used} MiB after ${timeout}s; stopping stray Ray processes"
  ray stop --force >/dev/null 2>&1 || true
  sleep 10
}

# merge_ckpt CKPT_DIR : verl saves sharded FSDP actor weights; vLLM needs a plain
# Hugging Face directory. Produces CKPT_DIR/hf once and reuses it afterwards.
merge_ckpt() {
  local ckpt="$1" hf="$1/hf"
  if [[ -f "$hf/config.json" ]] && ls "$hf"/*.safetensors >/dev/null 2>&1; then
    ladder_log "merged checkpoint already present: $hf"
    return 0
  fi
  ladder_log "merging FSDP checkpoint $ckpt/actor -> $hf"
  ( cd "$WHALE_DIR/domains/chess_puzzles" && \
    PYTHONPATH="$PWD" python -m verl.model_merger merge --backend fsdp \
      --local_dir "$ckpt/actor" --target_dir "$hf" )
  # Some verl versions write weights only; vLLM also needs the tokenizer files.
  if [[ ! -f "$hf/tokenizer_config.json" && -d "$ckpt/actor/huggingface" ]]; then
    cp -n "$ckpt/actor/huggingface"/* "$hf"/ 2>/dev/null || true
  fi
  [[ -f "$hf/config.json" ]] || { ladder_log "ERROR: merge produced no config.json in $hf"; return 1; }
}

# start_vllm NAME MODEL SERVED_NAME GPUS PORT MAX_LEN TP DP [extra vllm args...]
start_vllm() {
  local name="$1" model="$2" served="$3" gpus="$4" port="$5" max_len="$6" tp="$7" dp="$8"; shift 8
  start_bg "$name" "$RUN_OUT/logs/${name}_vllm.log" \
    env CUDA_VISIBLE_DEVICES="$gpus" vllm serve "$model" \
      --served-model-name "$served" \
      --host 127.0.0.1 --port "$port" \
      --tensor-parallel-size "$tp" --data-parallel-size "$dp" \
      --max-model-len "$max_len" \
      --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.90}" \
      --enable-prefix-caching \
      --trust-remote-code \
      "$@"
}
