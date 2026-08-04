#!/bin/bash
# Real-chain stagewise K=[[64,4],[4,4]] evaluation: LDPC rate 1/2 and BPSK.
set -euo pipefail

eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/RQ-VAE
mkdir -p experiments/eval

export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="quality_v2_B_larger_rate094"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="6"
export SIMVQ_DECODER_RES_BLOCKS="6"
export SIMVQ_QUANTIZER_TYPE="stagewise_residual_simvq"
export SIMVQ_QUANTIZER_AXIS_LIST="patch,patch"
export SIMVQ_CVQ_CODEWORD_SHAPES="patch,patch"
export SIMVQ_NUM_EMBEDDINGS_LIST="64,4"
export SIMVQ_RQ_DEPTH_LIST="2,2"
export SIMVQ_RQ_CODEBOOK_SIZES="64,4;4,4"
export SIMVQ_RQ_SHARED_CODEBOOK="0"
export SIMVQ_RQ_RESTART_UNUSED_CODES="0"
export SIMVQ_LAYER_LOSS_WEIGHTS_INIT="0.25,0.50"
export SIMVQ_LAYER_LOSS_WEIGHTS_FINAL="0.25,0.25"
export SIMVQ_LPIPS_WEIGHT="0"
export SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA="0.0"
export SIMVQ_MODEL_PARALLEL="0"
export SIMVQ_RESUME="0"
unset SIMVQ_PRETRAINED_CHECKPOINT

export SIMVQ_TEST_DATASET_PATH="${SIMVQ_TEST_DATASET_PATH:-/workspace/yi/work/Kodak-256-transform-resize}"
export SIMVQ_TEST_NO_RESIZE="${SIMVQ_TEST_NO_RESIZE:-1}"
export GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

EXPERIMENT_NAME="quality_v2_B_larger_rate094_stagewise_residual_simvq_unet2_ds8x2_k64x4-4x4_d2-2"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/${EXPERIMENT_NAME}/best_vq_deepsc.pth}"
JSON_OUTPUT="${JSON_OUTPUT:-experiments/eval/stagewise_residual_simvq_k64x4-4x4_rate094_ldpc_bpsk.json}"

PYTHON_CMD=(python -u)
if [ "${DEBUG_PDB:-0}" = "1" ]; then
  PYTHON_CMD=(python -m pdb)
elif [ "${DEBUGPY:-0}" = "1" ]; then
  DEBUGPY_PORT="${DEBUGPY_PORT:-5678}"
  PYTHON_CMD=(python -u -m debugpy --listen "0.0.0.0:${DEBUGPY_PORT}" --wait-for-client)
  echo "Waiting for debugpy attach on port ${DEBUGPY_PORT}..."
fi

"${PYTHON_CMD[@]}" test_real.py \
  --checkpoint "$CHECKPOINT_PATH" \
  --snrs 0 3 6 9 12 \
  --modulation bpsk \
  --json-output "$JSON_OUTPUT" \
  "$@"
