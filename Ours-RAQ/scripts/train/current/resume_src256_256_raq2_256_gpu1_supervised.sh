#!/bin/bash
set -uo pipefail

PROJECT_ROOT="/workspace/yi/work/Ours-RAQ"
TRAIN_SCRIPT="$PROJECT_ROOT/scripts/train/current/run_src256_256_raq2_256_curriculum_ch256_512_res6.sh"
GPU_ID="${GPU_ID:-1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-45000}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SUPERVISOR_LOG="$PROJECT_ROOT/experiments/logs/supervisor_src256_256_raq2_256_gpu${GPU_ID}.log"
LOCK_FILE="$PROJECT_ROOT/experiments/logs/supervisor_src256_256_raq2_256_gpu${GPU_ID}.lock"

mkdir -p "$PROJECT_ROOT/experiments/logs"
exec >>"$SUPERVISOR_LOG" 2>&1
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[$(date '+%F %T')] Another supervisor already holds $LOCK_FILE; exiting."
    exit 1
fi

echo "[$(date '+%F %T')] Supervisor started for physical GPU $GPU_ID."
echo "[$(date '+%F %T')] Waiting for at least ${MIN_FREE_MIB} MiB free memory before each launch."

attempt=0
while (( attempt <= MAX_RESTARTS )); do
    while true; do
        free_mib="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]')"
        if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
            break
        fi
        echo "[$(date '+%F %T')] GPU $GPU_ID has ${free_mib:-unknown} MiB free; waiting."
        sleep 30
    done

    attempt=$((attempt + 1))
    run_id="src256-256_raq2-256_curriculum_ch256-512-res6_gpu${GPU_ID}-resume$(date '+%Y%m%d-%H%M%S')-a${attempt}"
    echo "[$(date '+%F %T')] Launching resume attempt $attempt with run ID $run_id."

    GPU_ID="$GPU_ID" \
    SIMVQ_RESUME=1 \
    PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    EXPERIMENT_RUN_ID="$run_id" \
        bash "$TRAIN_SCRIPT"
    status=$?

    if (( status == 0 )); then
        echo "[$(date '+%F %T')] Training completed successfully; supervisor exiting."
        exit 0
    fi

    echo "[$(date '+%F %T')] Training exited with status $status; it will resume from the latest checkpoint."
    if (( attempt >= MAX_RESTARTS )); then
        echo "[$(date '+%F %T')] Reached restart limit $MAX_RESTARTS; supervisor exiting."
        exit "$status"
    fi
    sleep 60
done
