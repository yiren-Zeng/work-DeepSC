#!/bin/bash
# Evaluate the separately trained one-spatial-scale, two-stage RAQ-RVQ model.
set -euo pipefail

eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/RAQ-RVQ

export SIMVQ_EXP_FAMILY="shiyan_independent_raq_rvq_single_scale_src64_trg2-64_d2_curriculum_rate094_A_patch_ch256_res6-6_mixture_per_k"
export SIMVQ_NUM_EMBEDDINGS_LIST="64"
export SIMVQ_DOWNSAMPLE_STRIDES="8"
export SIMVQ_UNET_DEPTH="1"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="6"
export SIMVQ_DECODER_RES_BLOCKS="6"
export SIMVQ_INDEPENDENT_RAQ_RVQ_DEPTH="2"
# One spatial scale containing two RVQ stages. Override with, for example, 2,2.
export SIMVQ_INDEPENDENT_RAQ_RVQ_K_LISTS="${SIMVQ_INDEPENDENT_RAQ_RVQ_K_LISTS:-4}"
export SIMVQ_RAQ_MIN_TRG="2"
export SIMVQ_RAQ_MAX_TRG="64"

export SIMVQ_TEST_DATASET_PATH="${SIMVQ_TEST_DATASET_PATH:-/workspace/yi/work/Kodak-256-transform-resize}"
export SIMVQ_TEST_NO_RESIZE="${SIMVQ_TEST_NO_RESIZE:-1}"
export GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::UserWarning}"

CHECKPOINT="${CHECKPOINT:-/workspace/yi/work/RAQ-RVQ/checkpoints/shiyan_independent_raq_rvq_single_scale_src64_trg2-64_d2_curriculum_rate094_A_patch_ch256_res6-6_mixture_per_k_unet1_ds8_k64/best_vq_deepsc.pth}"
SNRS="${SNRS:-0}"
MODULATION="${MODULATION:-bpsk}"
LDPC_RATE="${LDPC_RATE:-0.5}"
LDPC_N="${LDPC_N:-256}"
NO_CHANNEL="${NO_CHANNEL:-0}"
CHANNEL_TYPES="${CHANNEL_TYPES:-awgn rician}"
RICIAN_K_FACTOR="${RICIAN_K_FACTOR:-10}"
JSON_OUTPUT="${JSON_OUTPUT:-experiments/eval/independent_raq_rvq_single_scale_src64_d2_mixture_per_k_combined.json}"
read -r -a SNR_ARGS <<< "$SNRS"
read -r -a CHANNEL_ARGS <<< "$CHANNEL_TYPES"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --snrs "${SNR_ARGS[@]}"
  --modulation "$MODULATION"
  --stream-packing combined
  --channels "${CHANNEL_ARGS[@]}"
  --rician-k-factor "$RICIAN_K_FACTOR"
  --ldpc_n "$LDPC_N"
  --ldpc_k "$LDPC_RATE"
  --json-output "$JSON_OUTPUT"
  --concise
)
if [[ "$NO_CHANNEL" == "1" ]]; then
  ARGS+=(--no-channel)
fi

echo "Single spatial scale (stride 8) | two RVQ stages | K=$SIMVQ_INDEPENDENT_RAQ_RVQ_K_LISTS"
echo "Checkpoint: $CHECKPOINT"
python -u test_real.py "${ARGS[@]}" "$@" \
  2> >(grep -v -E \
    '^(WARNING: All log messages before absl::InitializeLog|I[0-9]+ .*oneDNN custom operations are on\.)' \
    >&2)
