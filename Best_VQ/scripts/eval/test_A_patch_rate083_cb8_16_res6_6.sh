#!/bin/bash
set -euo pipefail

cd /workspace/yi/work/Best_VQ

export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="quality_v2_B_larger_rate083_A_patch_cb8-16_res6-6"
export SIMVQ_NUM_EMBEDDINGS_LIST="8,16"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="6"
export SIMVQ_DECODER_RES_BLOCKS="6"
export SIMVQ_QUANTIZER_TYPE="simvq"
export SIMVQ_QUANTIZER_AXIS_LIST="patch,patch"
export SIMVQ_CVQ_CODEWORD_SHAPES="patch,patch"
export SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA="0.0"
export SIMVQ_TEST_DATASET_PATH="/workspace/yi/work/Kodak-256-transform-resize"
export SIMVQ_TEST_NO_RESIZE="1" # 1代表不采用resize



PYTHON_CMD=(python -u) # 不设调试 → 默认：PYTHON_CMD 保持初始值 (python -u)

# 下面是调试选项，优先级：DEBUGPY < DEBUG_PDB
if [ "${DEBUG_PDB:-0}" = "1" ]; then
  PYTHON_CMD=(python -m pdb)
elif [ "${DEBUGPY:-0}" = "1" ]; then
  DEBUGPY_PORT="${DEBUGPY_PORT:-5678}"
  PYTHON_CMD=(python -m debugpy --listen "0.0.0.0:${DEBUGPY_PORT}" --wait-for-client)
  echo "Waiting for debugger attach on port ${DEBUGPY_PORT}..."
fi


if [ "$#" -eq 0 ]; then
  "${PYTHON_CMD[@]}" test_real.py \
    --checkpoint checkpoints/quality_v2_B_larger_rate083_A_patch_cb8-16_res6-6_unet2_ds8x2_k8-16/best_vq_deepsc.pth \
    --snrs 0 \
    --modulation bpsk
else
  "${PYTHON_CMD[@]}" test_real.py "$@"
fi
