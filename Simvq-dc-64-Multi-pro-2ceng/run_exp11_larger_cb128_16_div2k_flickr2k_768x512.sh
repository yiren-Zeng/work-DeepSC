#!/bin/bash
# Stage B larger SimVQ [128,16] trained from DIV2K + Flickr2K archives at Kodak resize.
set -euo pipefail

eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/Simvq-dc-64-Multi-pro-2ceng
mkdir -p checkpoints experiments/logs

export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="quality_v2_B_larger_cb128-16_DIV2K-Flickr2K_768x512"
export SIMVQ_NUM_EMBEDDINGS_LIST="128,16"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="4"
export SIMVQ_DECODER_RES_BLOCKS="4"
export SIMVQ_QUANTIZER_TYPE="simvq"

export SIMVQ_TRAIN_DATASET_PATH="archive:/datasets/DIV2K.tar.gz,/datasets/Flickr2K.zip"
export SIMVQ_VAL_DATASET_PATH="archive:/datasets/DIV2K.tar.gz"
export SIMVQ_TRAIN_RESIZE="768,512"
export SIMVQ_VAL_RESIZE="768,512"
export SIMVQ_TEST_RESIZE="768,512"
export SIMVQ_NUM_WORKERS="${SIMVQ_NUM_WORKERS:-1}"
export SIMVQ_ARCHIVE_SHUFFLE_BUFFER="${SIMVQ_ARCHIVE_SHUFFLE_BUFFER:-256}"
export SIMVQ_MODEL_PARALLEL="${SIMVQ_MODEL_PARALLEL:-1}"
export SIMVQ_ENCODER_DEVICE="${SIMVQ_ENCODER_DEVICE:-cuda:0}"
export SIMVQ_DECODER_DEVICE="${SIMVQ_DECODER_DEVICE:-cuda:1}"
export SIMVQ_DECODER_TAIL_DEVICE="${SIMVQ_DECODER_TAIL_DEVICE:-}"
export SIMVQ_DECODER_TAIL_BLOCKS="${SIMVQ_DECODER_TAIL_BLOCKS:-1}"
export SIMVQ_MAX_DISTANCE_ELEMENTS="${SIMVQ_MAX_DISTANCE_ELEMENTS:-1048576}"
export SIMVQ_GRADIENT_CHECKPOINTING="${SIMVQ_GRADIENT_CHECKPOINTING:-0}"

# Reuse the 27 dB expanded backbone. The codebook tensors are intentionally
# skipped where their shapes differ from the original [64,256] configuration.
export SIMVQ_PRETRAINED_CHECKPOINT="checkpoints/quality_v2_B_larger_unet2_ds8x2_k64-256/best_vq_deepsc.pth"

export SIMVQ_TOTAL_BATCH_SIZE="${SIMVQ_TOTAL_BATCH_SIZE:-24}"
export SIMVQ_MICRO_BATCH_SIZE="${SIMVQ_MICRO_BATCH_SIZE:-8}"
export GPU_IDS="${GPU_IDS:-1,2}"

RUN_ID="exp11_larger_cb128-16_div2k-flickr2k_768x512-$(date +%Y%m%d-%H%M%S)"
export EXPERIMENT_RUN_ID="$RUN_ID"
export PYTHONUNBUFFERED=1

echo "Experiment: $SIMVQ_EXP_FAMILY"
echo "Run ID: $RUN_ID"
echo "Visible GPUs: $GPU_IDS"
echo "Model parallel: $SIMVQ_MODEL_PARALLEL encoder=$SIMVQ_ENCODER_DEVICE decoder=$SIMVQ_DECODER_DEVICE"
echo "Decoder tail: device=$SIMVQ_DECODER_TAIL_DEVICE blocks=$SIMVQ_DECODER_TAIL_BLOCKS"
echo "Gradient checkpointing: $SIMVQ_GRADIENT_CHECKPOINTING"
echo "Quantizer: $SIMVQ_QUANTIZER_TYPE"
echo "Codebooks: $SIMVQ_NUM_EMBEDDINGS_LIST"
echo "Train archives: $SIMVQ_TRAIN_DATASET_PATH"
echo "Val archive: $SIMVQ_VAL_DATASET_PATH"
echo "Resize: train=$SIMVQ_TRAIN_RESIZE val=$SIMVQ_VAL_RESIZE test=$SIMVQ_TEST_RESIZE"
echo "Source BPP: 7/64 + 4/256 = 0.125"
echo "Batch: total=$SIMVQ_TOTAL_BATCH_SIZE micro=$SIMVQ_MICRO_BATCH_SIZE"

CUDA_VISIBLE_DEVICES="$GPU_IDS" python -u train.py 2>&1 | tee "experiments/logs/train_${RUN_ID}.log"

SIMVQ_MODEL_PARALLEL=0 python -u tools/auto_evaluate_completed_experiments.py \
  --experiment "quality_v2_B_larger_cb128-16_DIV2K-Flickr2K_768x512_unet2_ds8x2_k128-16" \
  --gpu "${GPU_IDS%%,*}"
