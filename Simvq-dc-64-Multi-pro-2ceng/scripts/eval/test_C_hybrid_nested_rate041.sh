#!/bin/bash
set -euo pipefail

cd /workspace/yi/work/Simvq-dc-64-Multi-pro-2ceng

export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="quality_v2_B_larger_rate041_C_hybridcvq_nested_cb65536-8192"
export SIMVQ_NUM_EMBEDDINGS_LIST="65536,8192"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="4"
export SIMVQ_DECODER_RES_BLOCKS="4"
export SIMVQ_QUANTIZER_TYPE="simvq"
export SIMVQ_QUANTIZER_AXIS_LIST="channel,patch"
export SIMVQ_CVQ_CODEWORD_SHAPES="32x32,patch"
export SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA="0.25"

PYTHON_CMD=(python -u)
if [ "${DEBUG_PDB:-0}" = "1" ]; then
  PYTHON_CMD=(python -m pdb)
elif [ "${DEBUGPY:-0}" = "1" ]; then
  DEBUGPY_PORT="${DEBUGPY_PORT:-5678}"
  PYTHON_CMD=(python -m debugpy --listen "0.0.0.0:${DEBUGPY_PORT}" --wait-for-client)
  echo "Waiting for debugger attach on port ${DEBUGPY_PORT}..."
fi

if [ "$#" -eq 0 ]; then
  "${PYTHON_CMD[@]}" test_real.py \
    --checkpoint checkpoints/quality_v2_B_larger_rate041_C_hybridcvq_nested_cb65536-8192_unet2_ds8x2_k65536-8192/best_vq_deepsc.pth \
    --snrs 0 \
    --modulation bpsk
else
  "${PYTHON_CMD[@]}" test_real.py "$@"
fi
