#!/bin/bash
set -euo pipefail

eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/Simvq-dc-64-Multi-pro-2ceng-GN
mkdir -p checkpoints
mkdir -p experiments/logs

EXPERIMENT_NAME="${SIMVQ_EXPERIMENT_NAME:-gn_v1_k64}"
RUN_ID="${RUN_ID:-${EXPERIMENT_NAME}-$(date +%Y%m%d-%H%M%S)}"
GPU_ID="${GPU_ID:-3}"
LOG_FILE="experiments/logs/train_${RUN_ID}.log"

export EXPERIMENT_RUN_ID="$RUN_ID"
export PYTHONUNBUFFERED=1
echo "Run ID: ${RUN_ID}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "GPU ID: ${GPU_ID}"
echo "Log file: ${LOG_FILE}"
CUDA_VISIBLE_DEVICES="$GPU_ID" python -u train.py 2>&1 | tee "$LOG_FILE"
