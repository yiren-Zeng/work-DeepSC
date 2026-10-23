#!/bin/bash
set -euo pipefail
eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/VQ-DeepSC
export VQ_DEEPSC_EXPERIMENT_NAME="k32_first_branch"
export VQ_DEEPSC_NUM_EMBEDDINGS_LIST="32"
export VQ_DEEPSC_TEST_DATASET_PATH="/workspace/yi/work/Kodak"
export GPU_ID="${GPU_ID:-2}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export VQ_DEEPSC_DEVICE="cuda:0"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TF_CPP_MIN_LOG_LEVEL=3
CHECKPOINT="${CHECKPOINT:-checkpoints/${VQ_DEEPSC_EXPERIMENT_NAME}/best_vq_deepsc.pth}"
SNRS="${SNRS:-10}"
NO_CHANNEL="${NO_CHANNEL:-0}"
IMAGE_OUTPUT_DIR="${IMAGE_OUTPUT_DIR:-/workspace/yi/work/VQ-DeepSC/generated_images/k32_first_branch}"
read -r -a SNR_ARGS <<< "$SNRS"
ARGS=(--checkpoint "$CHECKPOINT" 
      --snrs "${SNR_ARGS[@]}" 
      --modulation bpsk 
      --stream-packing combined 
      --ldpc_n 256 
      --ldpc_k 0.5
      --image-output-dir "$IMAGE_OUTPUT_DIR")
if [[ "$NO_CHANNEL" == "1" ]]; then ARGS+=(--no-channel); fi
python -B -u test_real_first_branch.py "${ARGS[@]}" "$@"
