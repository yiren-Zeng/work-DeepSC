#!/bin/bash
set -euo pipefail

cd /workspace/yi/work/shiyan

export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="shiyan_raq_B_larger_rate044_A_patch_cb16-2_ch512-1024"
export SIMVQ_NUM_EMBEDDINGS_LIST="64,64"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="32"
export SIMVQ_ENCODER_RES_BLOCKS="4"
export SIMVQ_DECODER_RES_BLOCKS="4"
export SIMVQ_QUANTIZER_TYPE="simvq"
export SIMVQ_QUANTIZER_AXIS_LIST="patch,patch"
export SIMVQ_CVQ_CODEWORD_SHAPES="patch,patch"
export SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA="0.0"
export SIMVQ_USE_RAQ="1"
export SIMVQ_RAQ_TARGET_LIST="64,64"
export SIMVQ_RAQ_MIN_TRG="2"
export SIMVQ_RAQ_MAX_TRG="64"
export SIMVQ_RAQ_REPULSION_WEIGHT="${SIMVQ_RAQ_REPULSION_WEIGHT:-0.00}"
export SIMVQ_TEST_DATASET_PATH="${SIMVQ_TEST_DATASET_PATH:-/workspace/yi/work/Kodak-256-transform-resize}"
export SIMVQ_TEST_NO_RESIZE="${SIMVQ_TEST_NO_RESIZE:-1}"
export GPU_ID="${GPU_ID:-0}" # 可以在终端命令行前面手动输入改变的
export CUDA_VISIBLE_DEVICES="$GPU_ID"

CHECKPOINT="${CHECKPOINT:-/workspace/yi/work/shiyan/checkpoints/shiyan_raq_src64-64_raq2-64_rate044_A_patch_ch64-128_unet2_ds8x2_k64/best_vq_deepsc.pth}"

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
    --checkpoint "$CHECKPOINT" \
    --snrs 0 \
    --modulation bpsk
else
  "${PYTHON_CMD[@]}" test_real.py "$@"
fi
