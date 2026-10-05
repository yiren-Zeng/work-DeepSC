#!/bin/bash
# Evaluate the per-K-mixture checkpoint over AWGN and Rician channels.
set -euo pipefail

eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/RAQ-RVQ

export SIMVQ_EXP_FAMILY="shiyan_independent_raq_rvq_src64-64_trg2-64_d2_curriculum_rate094_A_patch_ch256-512_res6-6_mixture_per_k"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="6"
export SIMVQ_DECODER_RES_BLOCKS="6"
export SIMVQ_INDEPENDENT_RAQ_RVQ_DEPTH="2"
export SIMVQ_INDEPENDENT_RAQ_RVQ_K_LISTS="${SIMVQ_INDEPENDENT_RAQ_RVQ_K_LISTS:-2,4;8,2}"

export SIMVQ_RAQ_MIN_TRG="2"
export SIMVQ_RAQ_MAX_TRG="64"

export SIMVQ_TEST_DATASET_PATH="${SIMVQ_TEST_DATASET_PATH:-/workspace/yi/work/Kodak-256-transform-resize}"
export SIMVQ_TEST_NO_RESIZE="${SIMVQ_TEST_NO_RESIZE:-1}"
export GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::UserWarning}"

CHECKPOINT="${CHECKPOINT:-/workspace/yi/work/RAQ-RVQ/checkpoints/shiyan_independent_raq_rvq_src64-64_trg2-64_d2_curriculum_rate094_A_patch_ch256-512_res6-6_mixture_per_k_unet2_ds8x2_k64/best_vq_deepsc.pth}"
SNRS="${SNRS:--10 -8 -6 -4 -2 0}"
MODULATION="${MODULATION:-qpsk}"
LDPC_RATE="${LDPC_RATE:-0.5}"
LDPC_N="${LDPC_N:-256}"
NO_CHANNEL="${NO_CHANNEL:-0}"
CHANNEL_TYPES="${CHANNEL_TYPES:-awgn rician}"
RICIAN_K_FACTOR="${RICIAN_K_FACTOR:-10}"
CHANNEL_SEEDS="${CHANNEL_SEEDS:-42 43 44 45 46 47 48 49 50 51}"
read -r -a SNR_ARGS <<< "$SNRS"
read -r -a CHANNEL_ARGS <<< "$CHANNEL_TYPES"
read -r -a CHANNEL_SEED_ARGS <<< "$CHANNEL_SEEDS"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --snrs "${SNR_ARGS[@]}"
  --modulation "$MODULATION"
  --stream-packing combined
  --channels "${CHANNEL_ARGS[@]}"
  --rician-k-factor "$RICIAN_K_FACTOR"
  --channel-seeds "${CHANNEL_SEED_ARGS[@]}"
  --ldpc_n "$LDPC_N"
  --ldpc_k "$LDPC_RATE"
  --concise
)
if [[ "$NO_CHANNEL" == "1" ]]; then
  ARGS+=(--no-channel)
fi

echo "Output: terminal only (no JSON file)"
python -u test_real.py "${ARGS[@]}" "$@" \
  2> >(grep -v -E \
    '^(WARNING: All log messages before absl::InitializeLog|I[0-9]+ .*oneDNN custom operations are on\.)' \
    >&2)
