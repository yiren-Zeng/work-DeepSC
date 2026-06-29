#!/bin/bash
set -euo pipefail
eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/shiyan
mkdir -p checkpoints experiments/logs

STAGE2_CKPT="checkpoints/shiyan_raq_src64-64_raq2-64_staged_curriculum_stage2_raq_warmup_rate044_A_patch_ch64-128_unet2_ds8x2_k64/best_vq_deepsc.pth"

export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="shiyan_raq_src64-64_raq2-64_staged_curriculum_stage3_raq_finetune_rate044_A_patch_ch64-128"
export SIMVQ_TRAIN_BRANCH="raq_finetune"
export SIMVQ_NUM_EMBEDDINGS_LIST="64,64"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="32"
export SIMVQ_ENCODER_RES_BLOCKS="4"
export SIMVQ_DECODER_RES_BLOCKS="4"
export SIMVQ_QUANTIZER_TYPE="simvq"
export SIMVQ_QUANTIZER_AXIS_LIST="patch,patch"
export SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA="0.0"
export SIMVQ_USE_RAQ="1"
unset SIMVQ_RAQ_TARGET_LIST
export SIMVQ_RAQ_MIN_TRG="2"
export SIMVQ_RAQ_MAX_TRG="64"
export SIMVQ_RAQ_REPULSION_WEIGHT="0.00"
export SIMVQ_RAQ_LATENT_DISTILL_WEIGHT="${SIMVQ_RAQ_LATENT_DISTILL_WEIGHT:-0.25}"
export SIMVQ_SRC_CODEBOOK_REPULSION_WEIGHT="0.00"
export SIMVQ_RAQ_USE_CURRICULUM="1"
export SIMVQ_RAQ_CURRICULUM_EARLY_LIST="32,64"
export SIMVQ_RAQ_CURRICULUM_MIDDLE_LIST="8,16,32,64"
export SIMVQ_RAQ_CURRICULUM_LATE_LIST="2,4,8,16,32,64"
export SIMVQ_CHANNEL_PROB_START_EPOCH="${SIMVQ_CHANNEL_PROB_START_EPOCH:-1000000}"
export SIMVQ_CHANNEL_PROB_END_EPOCH="${SIMVQ_CHANNEL_PROB_END_EPOCH:-1000001}"
export SIMVQ_LEARNING_RATE_G="${SIMVQ_LEARNING_RATE_G:-1e-5}"
export SIMVQ_CODEBOOK_PROJ_LR="${SIMVQ_CODEBOOK_PROJ_LR:-5e-5}"
export SIMVQ_PRETRAINED_CHECKPOINT="${SIMVQ_PRETRAINED_CHECKPOINT:-$STAGE2_CKPT}"
export SIMVQ_ALLOW_PRETRAINED="1"
export SIMVQ_RESUME="${SIMVQ_RESUME:-0}"

export SIMVQ_TOTAL_BATCH_SIZE="${SIMVQ_TOTAL_BATCH_SIZE:-24}"
export SIMVQ_MICRO_BATCH_SIZE="${SIMVQ_MICRO_BATCH_SIZE:-24}"
export GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export NUM_EPOCHS="${NUM_EPOCHS:-100}"

RUN_ID="staged_src64-64_stage3_raq_finetune_ch64-128_gpu${GPU_ID}-$(date +%Y%m%d-%H%M%S)"
export EXPERIMENT_RUN_ID="$RUN_ID"
export PYTHONUNBUFFERED=1

echo "Experiment: $SIMVQ_EXP_FAMILY"
echo "Run ID: $RUN_ID"
echo "Physical GPU: $GPU_ID"
echo "Train branch: $SIMVQ_TRAIN_BRANCH"
echo "Pretrained RAQ warmup: $SIMVQ_PRETRAINED_CHECKPOINT"
echo "RAQ curriculum: early=[$SIMVQ_RAQ_CURRICULUM_EARLY_LIST] middle=[$SIMVQ_RAQ_CURRICULUM_MIDDLE_LIST] late=[$SIMVQ_RAQ_CURRICULUM_LATE_LIST]"
echo "RAQ latent distill weight: $SIMVQ_RAQ_LATENT_DISTILL_WEIGHT"
echo "Learning rates: base=$SIMVQ_LEARNING_RATE_G codebook_proj=$SIMVQ_CODEBOOK_PROJ_LR"
echo "Channel curriculum disabled in Stage 3 via [$SIMVQ_CHANNEL_PROB_START_EPOCH,$SIMVQ_CHANNEL_PROB_END_EPOCH]"
echo "Batch: total=$SIMVQ_TOTAL_BATCH_SIZE micro=$SIMVQ_MICRO_BATCH_SIZE"
echo "NUM_EPOCHS: ${NUM_EPOCHS}"

python -u train.py 2>&1 | tee "experiments/logs/train_${RUN_ID}.log"
