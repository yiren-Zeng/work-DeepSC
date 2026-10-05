#!/bin/bash
# Keep the single-scale src64 training alive and resume from the latest
# epoch-level checkpoint after an unexpected exit.
set -uo pipefail

PROJECT_ROOT="/workspace/yi/work/RAQ-RVQ"
TRAIN_SCRIPT="$PROJECT_ROOT/scripts/train/current/run_independent_raq_rvq_single_scale_src64_k2_64_d2_curriculum_ch256_res6_6_mixture_per_k.sh"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/shiyan_independent_raq_rvq_single_scale_src64_trg2-64_d2_curriculum_rate094_A_patch_ch256_res6-6_mixture_per_k_unet1_ds8_k64"
GPU_ID="${GPU_ID:-4}"
MIN_FREE_MIB="${MIN_FREE_MIB:-45000}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SUPERVISOR_LOG="$PROJECT_ROOT/experiments/logs/supervisor_independent_raq_rvq_single_scale_src64_mixture_per_k_gpu${GPU_ID}.log"
LOCK_FILE="$PROJECT_ROOT/experiments/logs/supervisor_independent_raq_rvq_single_scale_src64_mixture_per_k_gpu${GPU_ID}.lock"

mkdir -p "$PROJECT_ROOT/experiments/logs"
exec >>"$SUPERVISOR_LOG" 2>&1
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[$(date '+%F %T')] Another supervisor already holds $LOCK_FILE; exiting."
    exit 1
fi

echo "[$(date '+%F %T')] Supervisor started for physical GPU $GPU_ID."
echo "[$(date '+%F %T')] Checkpoint directory: $CHECKPOINT_DIR"
echo "[$(date '+%F %T')] Waiting for at least ${MIN_FREE_MIB} MiB free memory before each launch."

attempt=0
while (( attempt < MAX_RESTARTS )); do
    while true; do
        free_mib="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]')"
        if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
            break
        fi
        echo "[$(date '+%F %T')] GPU $GPU_ID has ${free_mib:-unknown} MiB free; waiting."
        sleep 30
    done

    attempt=$((attempt + 1))
    if [[ -f "$CHECKPOINT_DIR/last_checkpoint.pth" ]]; then
        resume=1
    else
        resume=0
    fi
    run_id="independent-raq-rvq-single-scale-src64-d2-mixture-per-k_gpu${GPU_ID}-resume-$(date '+%Y%m%d-%H%M%S')-a${attempt}"
    echo "[$(date '+%F %T')] Launching attempt $attempt with resume=$resume and run ID $run_id."

    GPU_ID="$GPU_ID" \
    SIMVQ_RESUME="$resume" \
    PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    EXPERIMENT_RUN_ID="$run_id" \
        bash "$TRAIN_SCRIPT"
    status=$?

    if (( status == 0 )); then
        echo "[$(date '+%F %T')] Training completed successfully; supervisor exiting."
        exit 0
    fi

    echo "[$(date '+%F %T')] Training exited with status $status; the next attempt will resume from last_checkpoint.pth when available."
    if (( attempt >= MAX_RESTARTS )); then
        echo "[$(date '+%F %T')] Reached restart limit $MAX_RESTARTS; supervisor exiting."
        exit "$status"
    fi
    sleep 60
done
