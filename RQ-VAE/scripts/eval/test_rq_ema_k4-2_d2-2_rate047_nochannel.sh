#!/bin/bash
# Clean source reconstruction upper bound for the two-stage EMA-RQ model.
set -euo pipefail

eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/RQ-VAE

export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="quality_v2_B_larger_rate047"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="6"
export SIMVQ_DECODER_RES_BLOCKS="6"
export SIMVQ_QUANTIZER_TYPE="rq_ema"
export SIMVQ_QUANTIZER_AXIS_LIST="patch,patch"
export SIMVQ_CVQ_CODEWORD_SHAPES="patch,patch"
export SIMVQ_NUM_EMBEDDINGS_LIST="4,2"
export SIMVQ_RQ_DEPTH_LIST="2,2"
export SIMVQ_RQ_EMA_DECAY="0.99"
export SIMVQ_RQ_RESTART_UNUSED_CODES="1"
export SIMVQ_RQ_SHARED_CODEBOOK="1"
export SIMVQ_LAYER_LOSS_WEIGHTS_INIT="1,1"
export SIMVQ_LAYER_LOSS_WEIGHTS_FINAL="1,1"
export SIMVQ_LPIPS_WEIGHT="0"
export SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA="0.0"
export SIMVQ_RESUME="0"
unset SIMVQ_PRETRAINED_CHECKPOINT

export SIMVQ_TEST_DATASET_PATH="${SIMVQ_TEST_DATASET_PATH:-/workspace/yi/work/Kodak-256-transform-resize}"
export SIMVQ_TEST_NO_RESIZE="${SIMVQ_TEST_NO_RESIZE:-1}"
export GPU_ID="${GPU_ID:-3}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

CHECKPOINT_PATH="checkpoints/quality_v2_B_larger_rate047_rq_ema_unet2_ds8x2_k4-2_d2-2/best_vq_deepsc.pth"
PYTHON_CMD=(python -u)
if [ "${DEBUG_PDB:-0}" = "1" ]; then
  PYTHON_CMD=(python -m pdb)
elif [ "${DEBUGPY:-0}" = "1" ]; then
  DEBUGPY_PORT="${DEBUGPY_PORT:-5678}"
  PYTHON_CMD=(python -u -m debugpy --listen "0.0.0.0:${DEBUGPY_PORT}" --wait-for-client)
  echo "Waiting for debugpy attach on port ${DEBUGPY_PORT}..."
fi

if [ "$#" -eq 0 ]; then
  "${PYTHON_CMD[@]}" test_real.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --no-channel
else
  "${PYTHON_CMD[@]}" test_real.py "$@"
fi
