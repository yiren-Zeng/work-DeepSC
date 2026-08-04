#!/bin/bash
# Wait for a genuinely idle GPU, then launch the fresh stagewise rate047 run.
set -uo pipefail

PROJECT_ROOT="/workspace/yi/work/RQ-VAE"
TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train/current/run_stagewise_residual_simvq_k8x2-2x2_rate047.sh"
QUEUE_STATE_DIR="${PROJECT_ROOT}/experiments/queue"
QUEUE_NAME="stagewise_residual_simvq_k8x2-2x2_rate047"
LOCK_FILE="${QUEUE_STATE_DIR}/${QUEUE_NAME}.lock"
WATCHER_PID_FILE="${QUEUE_STATE_DIR}/${QUEUE_NAME}.watcher.pid"
TRAIN_PID_FILE="${QUEUE_STATE_DIR}/${QUEUE_NAME}.train.pid"

GPU_QUEUE_CANDIDATES="${GPU_QUEUE_CANDIDATES:-0,1,2,3,4}"
GPU_QUEUE_POLL_SECONDS="${GPU_QUEUE_POLL_SECONDS:-30}"
GPU_QUEUE_IDLE_CONFIRMATIONS="${GPU_QUEUE_IDLE_CONFIRMATIONS:-2}"
GPU_QUEUE_MAX_MEMORY_MIB="${GPU_QUEUE_MAX_MEMORY_MIB:-1024}"
GPU_QUEUE_MAX_UTILIZATION="${GPU_QUEUE_MAX_UTILIZATION:-5}"

mkdir -p "$QUEUE_STATE_DIR"
cd "$PROJECT_ROOT"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*"
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    log "Invalid ${name}=${value}; expected a positive integer."
    exit 2
  fi
}

require_nonnegative_integer() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    log "Invalid ${name}=${value}; expected a non-negative integer."
    exit 2
  fi
}

require_positive_integer "GPU_QUEUE_POLL_SECONDS" "$GPU_QUEUE_POLL_SECONDS"
require_positive_integer \
  "GPU_QUEUE_IDLE_CONFIRMATIONS" "$GPU_QUEUE_IDLE_CONFIRMATIONS"
require_nonnegative_integer \
  "GPU_QUEUE_MAX_MEMORY_MIB" "$GPU_QUEUE_MAX_MEMORY_MIB"
require_nonnegative_integer \
  "GPU_QUEUE_MAX_UTILIZATION" "$GPU_QUEUE_MAX_UTILIZATION"

if [ ! -x "$TRAIN_SCRIPT" ]; then
  log "Training script is missing or not executable: ${TRAIN_SCRIPT}"
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  log "nvidia-smi is unavailable."
  exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
  log "flock is unavailable."
  exit 2
fi
if ! command -v setsid >/dev/null 2>&1; then
  log "setsid is unavailable."
  exit 2
fi

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  log "Another ${QUEUE_NAME} watcher or training process already holds the lock."
  exit 0
fi

train_pid=""
cleanup() {
  rm -f -- "$WATCHER_PID_FILE" "$TRAIN_PID_FILE"
}
terminate() {
  log "Termination requested."
  if [ -n "$train_pid" ] && kill -0 "$train_pid" 2>/dev/null; then
    # The training shell starts in its own session, so this reaches Python,
    # tee, and data-loader descendants instead of orphaning them.
    kill -TERM -- "-$train_pid" 2>/dev/null \
      || kill -TERM "$train_pid"
    wait "$train_pid"
  fi
  exit 143
}
trap cleanup EXIT
trap terminate INT TERM

echo "$$" >"$WATCHER_PID_FILE"
IFS=',' read -r -a candidate_gpus <<<"$GPU_QUEUE_CANDIDATES"
declare -A idle_counts

log "Queue watcher started with PID $$."
log "Candidates=${GPU_QUEUE_CANDIDATES}; idle means memory<=${GPU_QUEUE_MAX_MEMORY_MIB} MiB and utilization<=${GPU_QUEUE_MAX_UTILIZATION}% for ${GPU_QUEUE_IDLE_CONFIRMATIONS} consecutive polls."

while true; do
  selected_gpu=""
  gpu_states=()

  for raw_gpu in "${candidate_gpus[@]}"; do
    gpu="${raw_gpu//[[:space:]]/}"
    if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
      log "Ignoring invalid GPU candidate: ${raw_gpu}"
      continue
    fi

    query="$(
      nvidia-smi \
        --id="$gpu" \
        --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null
    )"
    query_status=$?
    if [ "$query_status" -ne 0 ] || [ -z "$query" ]; then
      idle_counts["$gpu"]=0
      gpu_states+=("gpu${gpu}:unavailable")
      continue
    fi

    IFS=',' read -r memory_used utilization <<<"$query"
    memory_used="${memory_used//[[:space:]]/}"
    utilization="${utilization//[[:space:]]/}"
    if ! [[ "$memory_used" =~ ^[0-9]+$ ]] \
      || ! [[ "$utilization" =~ ^[0-9]+$ ]]; then
      idle_counts["$gpu"]=0
      gpu_states+=("gpu${gpu}:invalid")
      continue
    fi

    if [ "$memory_used" -le "$GPU_QUEUE_MAX_MEMORY_MIB" ] \
      && [ "$utilization" -le "$GPU_QUEUE_MAX_UTILIZATION" ]; then
      idle_counts["$gpu"]=$(( ${idle_counts["$gpu"]:-0} + 1 ))
    else
      idle_counts["$gpu"]=0
    fi
    gpu_states+=(
      "gpu${gpu}:${memory_used}MiB/${utilization}%#${idle_counts["$gpu"]}"
    )

    if [ -z "$selected_gpu" ] \
      && [ "${idle_counts["$gpu"]}" -ge "$GPU_QUEUE_IDLE_CONFIRMATIONS" ]; then
      selected_gpu="$gpu"
    fi
  done

  log "Poll: ${gpu_states[*]}"
  if [ -n "$selected_gpu" ]; then
    log "GPU ${selected_gpu} confirmed idle; launching training."
    setsid env GPU_ID="$selected_gpu" bash "$TRAIN_SCRIPT" &
    train_pid=$!
    echo "$train_pid" >"$TRAIN_PID_FILE"
    log "Training shell started with PID ${train_pid} on physical GPU ${selected_gpu}."

    wait "$train_pid"
    train_status=$?
    log "Training shell exited with status ${train_status}."
    exit "$train_status"
  fi

  sleep "$GPU_QUEUE_POLL_SECONDS"
done
