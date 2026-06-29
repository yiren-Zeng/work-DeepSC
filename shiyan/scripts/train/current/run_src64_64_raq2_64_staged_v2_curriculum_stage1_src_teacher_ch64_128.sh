#!/bin/bash
set -euo pipefail
eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/shiyan
mkdir -p checkpoints experiments/logs

export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="shiyan_raq_src64-64_raq2-64_staged_v2_curriculum_stage1_src_teacher_rate044_A_patch_ch64-128"
export SIMVQ_TRAIN_BRANCH="src"
export SIMVQ_NUM_EMBEDDINGS_LIST="64,64"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="32"
export SIMVQ_ENCODER_RES_BLOCKS="4"
export SIMVQ_DECODER_RES_BLOCKS="4"
export SIMVQ_QUANTIZER_TYPE="simvq"
export SIMVQ_QUANTIZER_AXIS_LIST="patch,patch"
export SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA="0.0"
export SIMVQ_USE_RAQ="0"
export SIMVQ_RAQ_LATENT_DISTILL_WEIGHT="0.00"
export SIMVQ_SRC_CODEBOOK_REPULSION_WEIGHT="0.00"
export SIMVQ_RAQ_RECON_GRAD_MODE="ste"
export SIMVQ_RAQ_TRAIN_ENCODER="0"
export SIMVQ_CHANNEL_PROB_START_EPOCH="${SIMVQ_CHANNEL_PROB_START_EPOCH:-1000000}"
export SIMVQ_CHANNEL_PROB_END_EPOCH="${SIMVQ_CHANNEL_PROB_END_EPOCH:-1000001}"
export SIMVQ_LEARNING_RATE_G="${SIMVQ_LEARNING_RATE_G:-5e-5}"
export SIMVQ_CODEBOOK_PROJ_LR="${SIMVQ_CODEBOOK_PROJ_LR:-2e-4}"
export SIMVQ_RESUME="${SIMVQ_RESUME:-0}"
unset SIMVQ_PRETRAINED_CHECKPOINT
unset SIMVQ_ALLOW_PRETRAINED

export SIMVQ_TOTAL_BATCH_SIZE="${SIMVQ_TOTAL_BATCH_SIZE:-24}"
export SIMVQ_MICRO_BATCH_SIZE="${SIMVQ_MICRO_BATCH_SIZE:-24}"
export GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export NUM_EPOCHS="${NUM_EPOCHS:-200}"

RUN_ID="staged_v2_src64-64_stage1_src_teacher_ch64-128_gpu${GPU_ID}-$(date +%Y%m%d-%H%M%S)"
export EXPERIMENT_RUN_ID="$RUN_ID"
export PYTHONUNBUFFERED=1

echo "Experiment: $SIMVQ_EXP_FAMILY"
echo "Run ID: $RUN_ID"
echo "Physical GPU: $GPU_ID"
echo "Train branch: $SIMVQ_TRAIN_BRANCH"
echo "Source codebooks: $SIMVQ_NUM_EMBEDDINGS_LIST"
echo "Channel curriculum disabled in Stage 1 via [$SIMVQ_CHANNEL_PROB_START_EPOCH,$SIMVQ_CHANNEL_PROB_END_EPOCH]"
echo "Batch: total=$SIMVQ_TOTAL_BATCH_SIZE micro=$SIMVQ_MICRO_BATCH_SIZE"
echo "NUM_EPOCHS: ${NUM_EPOCHS}"

python -u train.py 2>&1 | tee "experiments/logs/train_${RUN_ID}.log"
