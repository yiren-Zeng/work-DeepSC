#!/bin/bash
# Evaluate the res6-6 mixture-per-K checkpoint with adaptive arithmetic-coded
# Top-K masks over combined LDPC + modulation on AWGN and Rician channels,
# using the same channel-reporting convention as the dense combined script.
set -euo pipefail

eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/RAQ-RVQ

export SIMVQ_EXP_FAMILY="shiyan_independent_raq_rvq_src64-64_trg2-64_d2_curriculum_rate094_A_patch_ch256-512_res6-6_mixture_per_k"
export SIMVQ_NUM_EMBEDDINGS_LIST="64,64"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="6"
export SIMVQ_DECODER_RES_BLOCKS="6"
export SIMVQ_INDEPENDENT_RAQ_RVQ_DEPTH="2"
RVQ_K_LISTS="${RVQ_K_LISTS:-64,2;8,16}"
export SIMVQ_INDEPENDENT_RAQ_RVQ_K_LISTS="$RVQ_K_LISTS"
export SIMVQ_RAQ_MIN_TRG="2"
export SIMVQ_RAQ_MAX_TRG="64"

# export SIMVQ_RAQ_CURRICULUM_EARLY_LIST="32"
# export SIMVQ_RAQ_CURRICULUM_MIDDLE_LIST="8,16,32"
# export SIMVQ_RAQ_CURRICULUM_LATE_LIST="2,4,8,16,32"

export SIMVQ_TEST_DATASET_PATH="${SIMVQ_TEST_DATASET_PATH:-/workspace/yi/work/Kodak-256-transform-resize}"
export SIMVQ_TEST_NO_RESIZE="${SIMVQ_TEST_NO_RESIZE:-1}"
export GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::UserWarning}"

CHECKPOINT="${CHECKPOINT:-/workspace/yi/work/RAQ-RVQ/checkpoints/shiyan_independent_raq_rvq_src64-64_trg2-64_d2_curriculum_rate094_A_patch_ch256-512_res6-6_mixture_per_k_unet2_ds8x2_k64/best_vq_deepsc.pth}"
TARGET_ACTIVE_RATES="${TARGET_ACTIVE_RATES-}"
TARGET_ACTIVE_RATE_PAIRS="${TARGET_ACTIVE_RATE_PAIRS:-0.00,0.10}"
SNRS="${SNRS:-8}"
MODULATION="${MODULATION:-16qam}"
LDPC_N="${LDPC_N:-256}"
LDPC_RATE="${LDPC_RATE:-0.4375}"
MAX_IMAGES="${MAX_IMAGES:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CHANNEL_TYPES="${CHANNEL_TYPES:-awgn rician}"
RICIAN_K_FACTOR="${RICIAN_K_FACTOR:-10}"

read -r -a TARGET_ARGS <<< "$TARGET_ACTIVE_RATES"
read -r -a TARGET_PAIR_ARGS <<< "$TARGET_ACTIVE_RATE_PAIRS"
read -r -a SNR_ARGS <<< "$SNRS"
read -r -a CHANNEL_ARGS <<< "$CHANNEL_TYPES"

echo "Adaptive combined | RVQ K=$RVQ_K_LISTS | rates=$TARGET_ACTIVE_RATES | LDPC=${LDPC_N}x${LDPC_RATE} | ${MODULATION^^} | channels=$CHANNEL_TYPES | SNRs=$SNRS dB"

python -u test_independent_raq_rvq_adaptive_topk_arithmetic_mask.py \
  --checkpoint "$CHECKPOINT" \
  --dataset "$SIMVQ_TEST_DATASET_PATH" \
  --rvq-k-lists "$RVQ_K_LISTS" \
  --target-active-rates "${TARGET_ARGS[@]}" \
  --target-active-rate-pairs "${TARGET_PAIR_ARGS[@]}" \
  --snrs "${SNR_ARGS[@]}" \
  --modulation "$MODULATION" \
  --channels "${CHANNEL_ARGS[@]}" \
  --rician-k-factor "$RICIAN_K_FACTOR" \
  --ldpc-n "$LDPC_N" \
  --ldpc-rate "$LDPC_RATE" \
  --max-images "$MAX_IMAGES" \
  --num-workers "$NUM_WORKERS" \
  --concise \
  "$@" \
  2> >(grep -v -E \
    '^(WARNING: All log messages before absl::InitializeLog|I[0-9]+ .*oneDNN custom operations are on\.)' \
    >&2)
