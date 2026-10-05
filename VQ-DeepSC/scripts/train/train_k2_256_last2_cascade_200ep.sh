#!/bin/bash
set -euo pipefail
eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/VQ-DeepSC
export VQ_DEEPSC_EXPERIMENT_NAME="k2_256_last2_copy_200ep"
export VQ_DEEPSC_NUM_EMBEDDINGS_LIST="2,256"
export VQ_DEEPSC_NUM_EPOCHS=200
export VQ_DEEPSC_CHECKPOINT_DIR="/home/yi/vq_deepsc_runs_20260920/checkpoints/k2_256_last2_copy_200ep"
export VQ_DEEPSC_LOG_DIR="/home/yi/vq_deepsc_runs_20260920/logs/k2_256_last2_copy_200ep"
export GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export VQ_DEEPSC_DEVICE="cuda:0"
export VQ_DEEPSC_RESUME="${VQ_DEEPSC_RESUME:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
mkdir -p /home/yi/vq_deepsc_runs_20260920/console
python -B -u train_last2_cascade.py 2>&1 | tee -a "/home/yi/vq_deepsc_runs_20260920/console/train_${VQ_DEEPSC_EXPERIMENT_NAME}.log"
