#!/bin/bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd "$PROJECT_ROOT"

export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="shiyan_raq_src64-64_raq2-64_curriculum_rate044_A_patch_ch256-512_res6"
export SIMVQ_NUM_EMBEDDINGS_LIST="64,64"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="6"
export SIMVQ_DECODER_RES_BLOCKS="6"
export SIMVQ_QUANTIZER_TYPE="simvq"
export SIMVQ_QUANTIZER_AXIS_LIST="patch,patch"
export SIMVQ_USE_RAQ="1"
export SIMVQ_RAQ_TARGET_LIST="${SIMVQ_RAQ_TARGET_LIST:-2,16}"
export SIMVQ_RAQ_MIN_TRG="2"
export SIMVQ_RAQ_MAX_TRG="64"
export SIMVQ_TEST_DATASET_PATH="${SIMVQ_TEST_DATASET_PATH:-/workspace/yi/work/Kodak-256-transform-resize}"
export SIMVQ_TEST_NO_RESIZE="${SIMVQ_TEST_NO_RESIZE:-1}"
export GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

CHECKPOINT="${CHECKPOINT:-/workspace/yi/work/Ours-RAQ/checkpoints/shiyan_raq_src64-64_raq2-64_curriculum_rate044_A_patch_ch256-512_res6_unet2_ds8x2_k64/best_vq_deepsc.pth}"
SNRS="${SNRS:-0}"
MODULATION="${MODULATION:-bpsk}"
# LDPC_RATE is the channel coding rate k/n, e.g. 0.5 or 0.75.
LDPC_RATE="${LDPC_RATE:-0.5}"
LDPC_N="${LDPC_N:-256}"
JSON_OUTPUT="${JSON_OUTPUT:-}"
NO_CHANNEL="${NO_CHANNEL:-0}"
read -r -a SNR_ARGS <<< "$SNRS"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --snrs "${SNR_ARGS[@]}"
  --modulation "$MODULATION"
  --ldpc_n "$LDPC_N"
  --ldpc_k "$LDPC_RATE"
  --report-rate
)
if [[ "$NO_CHANNEL" == "1" ]]; then
  ARGS+=(--no-channel)
fi
if [[ -n "$JSON_OUTPUT" ]]; then
  ARGS+=(--json-output "$JSON_OUTPUT")
fi

python -u test_real.py "${ARGS[@]}" "$@"
