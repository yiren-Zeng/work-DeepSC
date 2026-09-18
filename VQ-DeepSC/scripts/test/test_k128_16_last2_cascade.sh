#!/bin/bash
set -euo pipefail
eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/VQ-DeepSC
export VQ_DEEPSC_EXPERIMENT_NAME="k128_16_last2_copy"
export VQ_DEEPSC_NUM_EMBEDDINGS_LIST="128,16"
export GPU_ID="${GPU_ID:-3}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export VQ_DEEPSC_DEVICE="cuda:0"
export PYTHONUNBUFFERED=1
mkdir -p experiments/eval
CHECKPOINT="${CHECKPOINT:-checkpoints/k64_64_128_16_last2_copy/best_vq_deepsc.pth}"
SNRS="${SNRS:-10}"
JSON_OUTPUT="${JSON_OUTPUT:-experiments/eval/${VQ_DEEPSC_EXPERIMENT_NAME}_ldpc12_qpsk_combined.json}"
NO_CHANNEL="${NO_CHANNEL:-0}"
read -r -a SNR_ARGS <<< "$SNRS"
ARGS=(--checkpoint "$CHECKPOINT" 
      --snrs "${SNR_ARGS[@]}" 
      --modulation 16qam 
      --stream-packing combined 
      --ldpc_n 256 
      --ldpc_k 0.5 
      --json-output "$JSON_OUTPUT")
if [[ "$NO_CHANNEL" == "1" ]]; then ARGS+=(--no-channel); fi
python -B -u test_real_last2_cascade.py "${ARGS[@]}" "$@"
