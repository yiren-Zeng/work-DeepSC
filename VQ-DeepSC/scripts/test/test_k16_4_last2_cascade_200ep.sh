#!/bin/bash
set -euo pipefail
eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/VQ-DeepSC
export VQ_DEEPSC_EXPERIMENT_NAME="k16_4_last2_copy_200ep"
export VQ_DEEPSC_NUM_EMBEDDINGS_LIST="16,4"
export GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export VQ_DEEPSC_DEVICE="cuda:0"
export PYTHONUNBUFFERED=1
mkdir -p /home/yi/vq_deepsc_runs_20260919/eval
CHECKPOINT="${CHECKPOINT:-/home/yi/vq_deepsc_runs_20260919/checkpoints/k16_4_last2_copy_200ep/best_vq_deepsc.pth}"
SNRS="${SNRS:--10 -8 -6 -4 -2 0}"
MODULATION="${MODULATION:-qpsk}"
LDPC_RATE="${LDPC_RATE:-0.75}"
LDPC_N="${LDPC_N:-256}"
JSON_OUTPUT="${JSON_OUTPUT:-/home/yi/vq_deepsc_runs_20260919/eval/${VQ_DEEPSC_EXPERIMENT_NAME}_ldpc34_qpsk_combined.json}"
NO_CHANNEL="${NO_CHANNEL:-0}"
CHANNEL_TYPES="${CHANNEL_TYPES:-awgn rician}"
RICIAN_K_FACTOR="${RICIAN_K_FACTOR:-10}"
CHANNEL_SEEDS="${CHANNEL_SEEDS:-42 43 44 45 46 47 48 49 50 51}"
read -r -a SNR_ARGS <<< "$SNRS"
read -r -a CHANNEL_ARGS <<< "$CHANNEL_TYPES"
read -r -a CHANNEL_SEED_ARGS <<< "$CHANNEL_SEEDS"
ARGS=(--checkpoint "$CHECKPOINT"
      --snrs "${SNR_ARGS[@]}"
      --modulation "$MODULATION"
      --stream-packing combined
      --channels "${CHANNEL_ARGS[@]}"
      --rician-k-factor "$RICIAN_K_FACTOR"
      --channel-seeds "${CHANNEL_SEED_ARGS[@]}"
      --ldpc_n "$LDPC_N"
      --ldpc_k "$LDPC_RATE"
      --json-output "$JSON_OUTPUT")
if [[ "$NO_CHANNEL" == "1" ]]; then ARGS+=(--no-channel); fi
python -B -u test_real_last2_cascade.py "${ARGS[@]}" "$@"
