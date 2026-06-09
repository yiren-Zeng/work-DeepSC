#!/bin/bash
# Stage B larger backbone without vector quantization: autoencoder upper bound.
set -euo pipefail

eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/Simvq-dc-64-Multi-pro-2ceng
mkdir -p checkpoints experiments/logs

export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="quality_v2_B_larger_NoQuant"
export SIMVQ_NUM_EMBEDDINGS_LIST="64,256"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="4"
export SIMVQ_DECODER_RES_BLOCKS="4"
export SIMVQ_QUANTIZER_TYPE="none"

# Reuse the 27 dB expanded model's backbone. SimVQ-only parameters are skipped.
export SIMVQ_PRETRAINED_CHECKPOINT="checkpoints/quality_v2_B_larger_unet2_ds8x2_k64-256/best_vq_deepsc.pth"

export SIMVQ_TOTAL_BATCH_SIZE="${SIMVQ_TOTAL_BATCH_SIZE:-24}"
export SIMVQ_MICRO_BATCH_SIZE="${SIMVQ_MICRO_BATCH_SIZE:-24}"
export GPU_ID="${GPU_ID:-2}"

RUN_ID="exp8_larger_noquant-$(date +%Y%m%d-%H%M%S)"
export EXPERIMENT_RUN_ID="$RUN_ID"
export PYTHONUNBUFFERED=1

echo "Experiment: $SIMVQ_EXP_FAMILY"
echo "Run ID: $RUN_ID"
echo "GPU: $GPU_ID"
echo "Quantizer: $SIMVQ_QUANTIZER_TYPE (encoder features pass directly to decoder)"
echo "Batch: total=$SIMVQ_TOTAL_BATCH_SIZE micro=$SIMVQ_MICRO_BATCH_SIZE"

CUDA_VISIBLE_DEVICES="$GPU_ID" python -u train.py 2>&1 | tee "experiments/logs/train_${RUN_ID}.log"

python -u tools/auto_evaluate_completed_experiments.py \
  --experiment "quality_v2_B_larger_NoQuant_unet2_ds8x2_k64-256" \
  --gpu "$GPU_ID"
