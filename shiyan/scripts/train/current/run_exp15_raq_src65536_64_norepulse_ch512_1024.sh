#!/bin/bash
# RAQ variable-rate variant:
# - source codebook K: [65536,64]
# - target/test RAQ K: [16,2]
# - RAQ codebook repulsion loss: disabled
set -euo pipefail
eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/shiyan
mkdir -p checkpoints experiments/logs

export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="shiyan_raq_src65536-64_norepulse_rate044_A_patch_ch512-1024"
export SIMVQ_NUM_EMBEDDINGS_LIST="65536,64"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="256"
export SIMVQ_ENCODER_RES_BLOCKS="4"
export SIMVQ_DECODER_RES_BLOCKS="4"
export SIMVQ_QUANTIZER_TYPE="simvq"
export SIMVQ_QUANTIZER_AXIS_LIST="patch,patch"
export SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA="0.0"

export SIMVQ_USE_RAQ="1"
export SIMVQ_RAQ_TARGET_LIST="16,2"
export SIMVQ_RAQ_MIN_TRG="2"
export SIMVQ_RAQ_MAX_TRG="16"
export SIMVQ_RAQ_REPULSION_WEIGHT="0.0"

export SIMVQ_RESUME="${SIMVQ_RESUME:-0}"
unset SIMVQ_PRETRAINED_CHECKPOINT

export SIMVQ_TOTAL_BATCH_SIZE="${SIMVQ_TOTAL_BATCH_SIZE:-24}"
export SIMVQ_MICRO_BATCH_SIZE="${SIMVQ_MICRO_BATCH_SIZE:-2}"
export SIMVQ_NUM_WORKERS="${SIMVQ_NUM_WORKERS:-8}"
export GPU_ID="${GPU_ID:-3}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_ID="exp15_raq_src65536-64_norepulse_ch512-1024-$(date +%Y%m%d-%H%M%S)"
export EXPERIMENT_RUN_ID="$RUN_ID"
export PYTHONUNBUFFERED=1

echo "Experiment: $SIMVQ_EXP_FAMILY"
echo "Run ID: $RUN_ID"
echo "GPU: $GPU_ID"
echo "Quantizer: $SIMVQ_QUANTIZER_TYPE"
echo "Quantizer axes: $SIMVQ_QUANTIZER_AXIS_LIST"
echo "Source codebooks: $SIMVQ_NUM_EMBEDDINGS_LIST"
echo "RAQ target/test K: $SIMVQ_RAQ_TARGET_LIST"
echo "RAQ train K range: [$SIMVQ_RAQ_MIN_TRG,$SIMVQ_RAQ_MAX_TRG]"
echo "RAQ repulsion weight: $SIMVQ_RAQ_REPULSION_WEIGHT"
echo "Base channels: $SIMVQ_BASE_CHANNELS"
echo "Feature dims: [$((SIMVQ_BASE_CHANNELS*2)),$((SIMVQ_BASE_CHANNELS*4))]"
echo "Test transmission ratio with target K [16,2] (LDPC1/2+BPSK): 0.04427083"
echo "Batch: total=$SIMVQ_TOTAL_BATCH_SIZE micro=$SIMVQ_MICRO_BATCH_SIZE"
echo "Pretrained: disabled, train from scratch"

python -u train.py 2>&1 | tee "experiments/logs/train_${RUN_ID}.log"
