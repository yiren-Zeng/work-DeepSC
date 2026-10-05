#!/bin/bash
# Evaluate the src128 per-K-mixture checkpoint with combined stream packing.
set -euo pipefail

eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/RAQ-RVQ
mkdir -p experiments/eval

export SIMVQ_EXP_FAMILY="shiyan_independent_raq_rvq_src128-128_trg2-128_d2_curriculum_rate094_A_patch_ch256-512_res6-6_mixture_per_k"
export SIMVQ_NUM_EMBEDDINGS_LIST="128,128"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="6"
export SIMVQ_DECODER_RES_BLOCKS="6"
export SIMVQ_INDEPENDENT_RAQ_RVQ_DEPTH="2"
export SIMVQ_INDEPENDENT_RAQ_RVQ_K_LISTS="${SIMVQ_INDEPENDENT_RAQ_RVQ_K_LISTS:-8,2;128,2}"

export SIMVQ_RAQ_MIN_TRG="2"
export SIMVQ_RAQ_MAX_TRG="128"

export SIMVQ_TEST_DATASET_PATH="${SIMVQ_TEST_DATASET_PATH:-/workspace/yi/work/Kodak-256-transform-resize}"
export SIMVQ_TEST_NO_RESIZE="${SIMVQ_TEST_NO_RESIZE:-1}"
export GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

CHECKPOINT="${CHECKPOINT:-/workspace/yi/work/RAQ-RVQ/checkpoints/shiyan_independent_raq_rvq_src128-128_trg2-128_d2_curriculum_rate094_A_patch_ch256-512_res6-6_mixture_per_k_unet2_ds8x2_k128/best_vq_deepsc.pth}"
SNRS="${SNRS:-10}"
MODULATION="${MODULATION:-16qam}"
LDPC_RATE="${LDPC_RATE:-0.5}"
LDPC_N="${LDPC_N:-256}"
JSON_OUTPUT="${JSON_OUTPUT:-experiments/eval/independent_raq_rvq_src128-128_mixture_per_k_combined.json}"
NO_CHANNEL="${NO_CHANNEL:-0}"
read -r -a SNR_ARGS <<< "$SNRS"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --snrs "${SNR_ARGS[@]}"
  --modulation "$MODULATION"
  --stream-packing combined
  --ldpc_n "$LDPC_N"
  --ldpc_k "$LDPC_RATE"
  --json-output "$JSON_OUTPUT"
)
if [[ "$NO_CHANNEL" == "1" ]]; then
  ARGS+=(--no-channel)
fi

python -u test_real.py "${ARGS[@]}" "$@"
