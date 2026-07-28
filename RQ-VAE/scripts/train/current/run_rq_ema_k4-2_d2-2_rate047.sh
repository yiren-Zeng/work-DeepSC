#!/bin/bash
# Two-scale U-Net with two-stage shared-codebook EMA-RQ at 4608 source bits/image.
set -euo pipefail

eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/RQ-VAE
mkdir -p checkpoints experiments/logs

export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="quality_v2_B_larger_rate047"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="6"
export SIMVQ_DECODER_RES_BLOCKS="6"
export SIMVQ_CHANNEL_PROB_START_EPOCH="80"
export SIMVQ_CHANNEL_PROB_END_EPOCH="120"
export SIMVQ_USE_SWINIR_ENHANCE="0"
export SIMVQ_USE_SWIN_BACKBONE="0"

# Preserve the baseline 256px Resize+RandomCrop/CenterCrop preprocessing and
# prevent a caller's inherited resize override from changing this fixed run.
unset SIMVQ_TRAIN_RESIZE SIMVQ_VAL_RESIZE

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

# This is an isolated RQ-EMA run: never resume or inherit another quantizer.
export SIMVQ_RESUME="0"
unset SIMVQ_PRETRAINED_CHECKPOINT

export SIMVQ_TRAIN_DATASET_PATH="${SIMVQ_TRAIN_DATASET_PATH:-/workspace/yi/work/Cars196/train_data}"
export SIMVQ_VAL_DATASET_PATH="${SIMVQ_VAL_DATASET_PATH:-/workspace/yi/work/Cars196/val_data}"
export SIMVQ_TOTAL_BATCH_SIZE="${SIMVQ_TOTAL_BATCH_SIZE:-24}"
export SIMVQ_MICRO_BATCH_SIZE="${SIMVQ_MICRO_BATCH_SIZE:-24}"
export GPU_ID="${GPU_ID:-3}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

EXPERIMENT_NAME="quality_v2_B_larger_rate047_rq_ema_unet2_ds8x2_k4-2_d2-2"
RUN_ID="rq_ema_k4-2_d2-2_rate047-$(date +%Y%m%d-%H%M%S)"
export EXPERIMENT_RUN_ID="$RUN_ID"

echo "Experiment: $EXPERIMENT_NAME"
echo "Run ID: $RUN_ID"
echo "GPU: $GPU_ID"
echo "Quantizer: rq_ema, K=[4,2], depth=[2,2], decay=0.99, shared=1, restart=1"
echo "Source rate: 4608 bits/image = 0.0703125 bpp"
echo "Test transmission ratio (LDPC1/2+BPSK): 0.04687500"
echo "Batch: total=$SIMVQ_TOTAL_BATCH_SIZE micro=$SIMVQ_MICRO_BATCH_SIZE"

PYTHON_CMD=(python -u)
if [ "${DEBUG_PDB:-0}" = "1" ]; then
  PYTHON_CMD=(python -m pdb)
  echo "[Debug] Using Python pdb debugger."
elif [ "${DEBUGPY:-0}" = "1" ]; then
  DEBUGPY_PORT="${DEBUGPY_PORT:-5678}"
  PYTHON_CMD=(python -u -m debugpy --listen "0.0.0.0:${DEBUGPY_PORT}" --wait-for-client)
  echo "[Debug] Waiting for debugpy attach on port ${DEBUGPY_PORT}..."
fi

"${PYTHON_CMD[@]}" train.py 2>&1 | tee "experiments/logs/train_${RUN_ID}.log"
