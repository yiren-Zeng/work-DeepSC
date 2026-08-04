#!/bin/bash
# Resume the trained depth-[4,4] Residual-SimVQ experiment to 200 epochs.
set -euo pipefail

eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/RQ-VAE
mkdir -p checkpoints experiments/logs

export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="quality_v2_B_larger_rate094"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="6"
export SIMVQ_DECODER_RES_BLOCKS="6"
export SIMVQ_CHANNEL_PROB_START_EPOCH="80"
export SIMVQ_CHANNEL_PROB_END_EPOCH="120"
export SIMVQ_USE_SWINIR_ENHANCE="0"
export SIMVQ_USE_SWIN_BACKBONE="0"
export SIMVQ_MODEL_PARALLEL="0"

# Preserve the original 256px Resize+RandomCrop/CenterCrop preprocessing.
unset SIMVQ_TRAIN_RESIZE SIMVQ_VAL_RESIZE

export SIMVQ_QUANTIZER_TYPE="residual_simvq"
export SIMVQ_QUANTIZER_AXIS_LIST="patch,patch"
export SIMVQ_CVQ_CODEWORD_SHAPES="patch,patch"
export SIMVQ_NUM_EMBEDDINGS_LIST="4,2"
export SIMVQ_RQ_DEPTH_LIST="4,4"
export SIMVQ_RQ_SHARED_CODEBOOK="1"
export SIMVQ_RQ_RESTART_UNUSED_CODES="0"
export SIMVQ_LAYER_LOSS_WEIGHTS_INIT="0.25,0.50"
export SIMVQ_LAYER_LOSS_WEIGHTS_FINAL="0.25,0.25"
export SIMVQ_LPIPS_WEIGHT="0"
export SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA="0.0"

export SIMVQ_RESUME="1"
unset SIMVQ_PRETRAINED_CHECKPOINT

export SIMVQ_TRAIN_DATASET_PATH="${SIMVQ_TRAIN_DATASET_PATH:-/workspace/yi/work/Cars196/train_data}"
export SIMVQ_VAL_DATASET_PATH="${SIMVQ_VAL_DATASET_PATH:-/workspace/yi/work/Cars196/val_data}"
export SIMVQ_TOTAL_BATCH_SIZE="${SIMVQ_TOTAL_BATCH_SIZE:-24}"
export SIMVQ_MICRO_BATCH_SIZE="${SIMVQ_MICRO_BATCH_SIZE:-24}"
export SIMVQ_NUM_EPOCHS="200"
export GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

EXPERIMENT_NAME="quality_v2_B_larger_rate094_residual_simvq_unet2_ds8x2_k4-2_d4-4"
CHECKPOINT_PATH="checkpoints/${EXPERIMENT_NAME}/last_checkpoint.pth"
if [ ! -s "$CHECKPOINT_PATH" ]; then
  echo "Resume checkpoint is missing or empty: $CHECKPOINT_PATH" >&2
  exit 1
fi

RUN_ID="residual_simvq_k4-2_d4-4_rate094-resume-$(date +%Y%m%d-%H%M%S)"
export EXPERIMENT_RUN_ID="$RUN_ID"

echo "Experiment: $EXPERIMENT_NAME"
echo "Run ID: $RUN_ID"
echo "GPU: $GPU_ID"
echo "Resume checkpoint: $CHECKPOINT_PATH"
echo "Quantizer: residual_simvq, shared projected codebook, K=[4,2], depth=[4,4]"
echo "Source rate: 9216 bits/image = 0.140625 bpp"
echo "Test transmission ratio (LDPC1/2+BPSK): 0.09375000"
echo "Batch: total=$SIMVQ_TOTAL_BATCH_SIZE micro=$SIMVQ_MICRO_BATCH_SIZE"
echo "Target epochs: $SIMVQ_NUM_EPOCHS, resume enabled"

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
