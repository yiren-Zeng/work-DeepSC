#!/bin/bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd "$PROJECT_ROOT"
mkdir -p checkpoints experiments/logs

export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="shiyan_raq_src256-256_raq2-256_curriculum_rate044_A_patch_ch256-512_res6"
export SIMVQ_NUM_EMBEDDINGS_LIST="256,256"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="6"
export SIMVQ_DECODER_RES_BLOCKS="6"
export SIMVQ_QUANTIZER_TYPE="simvq"
export SIMVQ_QUANTIZER_AXIS_LIST="patch,patch"
export SIMVQ_USE_RAQ="1"
unset SIMVQ_RAQ_TARGET_LIST
export SIMVQ_RAQ_MIN_TRG="2"
export SIMVQ_RAQ_MAX_TRG="256"
export SIMVQ_RAQ_USE_CURRICULUM="1"
export SIMVQ_RAQ_CURRICULUM_EARLY_LIST="32,64,128,256"
export SIMVQ_RAQ_CURRICULUM_MIDDLE_LIST="8,16,32,64,128,256"
export SIMVQ_RAQ_CURRICULUM_LATE_LIST="2,4,8,16,32,64,128,256"
export SIMVQ_RESUME="${SIMVQ_RESUME:-0}"
unset SIMVQ_PRETRAINED_CHECKPOINT

export SIMVQ_TOTAL_BATCH_SIZE="${SIMVQ_TOTAL_BATCH_SIZE:-24}"
export SIMVQ_MICRO_BATCH_SIZE="${SIMVQ_MICRO_BATCH_SIZE:-24}"
export SIMVQ_SAVE_INTERVAL="${SIMVQ_SAVE_INTERVAL:-20}"
export GPU_ID="${GPU_ID:-2}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export NUM_EPOCHS="${NUM_EPOCHS:-200}"

RUN_ID="${EXPERIMENT_RUN_ID:-src256-256_raq2-256_curriculum_ch256-512_res6_gpu${GPU_ID}-$(date +%Y%m%d-%H%M%S)}"
export EXPERIMENT_RUN_ID="$RUN_ID"
export PYTHONUNBUFFERED=1

echo "Experiment: $SIMVQ_EXP_FAMILY"
echo "Run ID: $RUN_ID"
echo "Physical GPU: $GPU_ID"
echo "Source codebooks: $SIMVQ_NUM_EMBEDDINGS_LIST"
echo "Base channels: $SIMVQ_BASE_CHANNELS"
echo "Embedding dims: 256,512"
echo "Encoder/decoder residual blocks: $SIMVQ_ENCODER_RES_BLOCKS/$SIMVQ_DECODER_RES_BLOCKS"
echo "RAQ range: [$SIMVQ_RAQ_MIN_TRG,$SIMVQ_RAQ_MAX_TRG]"
echo "RAQ curriculum: early=[$SIMVQ_RAQ_CURRICULUM_EARLY_LIST] middle=[$SIMVQ_RAQ_CURRICULUM_MIDDLE_LIST] late=[$SIMVQ_RAQ_CURRICULUM_LATE_LIST]"
echo "Batch: total=$SIMVQ_TOTAL_BATCH_SIZE micro=$SIMVQ_MICRO_BATCH_SIZE"
echo "NUM_EPOCHS: $NUM_EPOCHS"
echo "Periodic weight save interval: $SIMVQ_SAVE_INTERVAL epochs"

python -u train.py 2>&1 | tee "experiments/logs/train_${RUN_ID}.log"
