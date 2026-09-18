#!/bin/bash
set -euo pipefail
eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/VQ-DeepSC
export VQ_DEEPSC_EXPERIMENT_NAME="k64_64_32_16_last2_copy"
export VQ_DEEPSC_NUM_EMBEDDINGS_LIST="64,64,32,16"
export GPU_ID="${GPU_ID:-2}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export VQ_DEEPSC_DEVICE="cuda:4"
export VQ_DEEPSC_RESUME="${VQ_DEEPSC_RESUME:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
mkdir -p experiments/logs
python -B -u train_last2_cascade.py 2>&1 | tee -a "experiments/logs/train_${VQ_DEEPSC_EXPERIMENT_NAME}.log"
