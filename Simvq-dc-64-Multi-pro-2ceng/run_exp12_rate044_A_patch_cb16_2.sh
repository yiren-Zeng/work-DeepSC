#!/bin/bash
# Rate-0.044 A/Baseline: SimVQ patch-wise, large-to-small codebooks [16,2].
set -euo pipefail

eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/Simvq-dc-64-Multi-pro-2ceng
mkdir -p checkpoints experiments/logs

export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="quality_v2_B_larger_rate044_A_patch_cb16-2"
export SIMVQ_NUM_EMBEDDINGS_LIST="16,2"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="4"
export SIMVQ_DECODER_RES_BLOCKS="4"
export SIMVQ_QUANTIZER_TYPE="simvq"
export SIMVQ_QUANTIZER_AXIS_LIST="patch,patch"
export SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA="0.0"

export SIMVQ_PRETRAINED_CHECKPOINT="checkpoints/quality_v2_B_larger_cb128-16_unet2_ds8x2_k128-16/best_vq_deepsc.pth"

export SIMVQ_TOTAL_BATCH_SIZE="${SIMVQ_TOTAL_BATCH_SIZE:-24}"
export SIMVQ_MICRO_BATCH_SIZE="${SIMVQ_MICRO_BATCH_SIZE:-24}"
export GPU_ID="${GPU_ID:-0}"

RUN_ID="exp12_rate044_A_patch_cb16-2-$(date +%Y%m%d-%H%M%S)"
export EXPERIMENT_RUN_ID="$RUN_ID"
export PYTHONUNBUFFERED=1

echo "Experiment: $SIMVQ_EXP_FAMILY"
echo "Run ID: $RUN_ID"
echo "GPU: $GPU_ID"
echo "Quantizer: $SIMVQ_QUANTIZER_TYPE"
echo "Quantizer axes: $SIMVQ_QUANTIZER_AXIS_LIST"
echo "Codebooks: $SIMVQ_NUM_EMBEDDINGS_LIST"
echo "Test transmission ratio (LDPC1/2+BPSK): 0.04427083"
echo "Batch: total=$SIMVQ_TOTAL_BATCH_SIZE micro=$SIMVQ_MICRO_BATCH_SIZE"

CUDA_VISIBLE_DEVICES="$GPU_ID" python -u train.py 2>&1 | tee "experiments/logs/train_${RUN_ID}.log"
