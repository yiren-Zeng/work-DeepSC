#!/bin/bash
# Train src128 independent RAQ-RVQ with an independent 50/50 choice for each K.
set -euo pipefail

eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/RAQ-RVQ
mkdir -p checkpoints experiments/logs

export SIMVQ_EXP_FAMILY="shiyan_independent_raq_rvq_src128-128_trg2-128_d2_curriculum_rate094_A_patch_ch256-512_res6-6_mixture_per_k"
export SIMVQ_NUM_EMBEDDINGS_LIST="128,128"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="6"
export SIMVQ_DECODER_RES_BLOCKS="6"
export SIMVQ_INDEPENDENT_RAQ_RVQ_DEPTH="2"
# Model initialization and codebook-monitoring reference; validation overrides K per batch.
export SIMVQ_INDEPENDENT_RAQ_RVQ_K_LISTS="16,16;4,4"
export SIMVQ_RAQ_MIN_TRG="2"
export SIMVQ_RAQ_MAX_TRG="128"
export SIMVQ_MIXTURE_MODE="per_k"
export SIMVQ_RAQ_USE_CURRICULUM="1"
export SIMVQ_RAQ_CURRICULUM_EARLY_LIST="64,128"
export SIMVQ_RAQ_CURRICULUM_MIDDLE_LIST="16,32,64,128"
export SIMVQ_RAQ_CURRICULUM_LATE_LIST="2,4,8,16,32,64,128"
export SIMVQ_LEARNING_RATE_G="5e-5"
export SIMVQ_CODEBOOK_PROJ_LR="2e-4"
export SIMVQ_CHANNEL_PROB_START_EPOCH="80"
export SIMVQ_CHANNEL_PROB_END_EPOCH="120"
export SIMVQ_RESUME="${SIMVQ_RESUME:-0}"

export SIMVQ_TRAIN_DATASET_PATH="${SIMVQ_TRAIN_DATASET_PATH:-/workspace/yi/work/Cars196/train_data}"
export SIMVQ_VAL_DATASET_PATH="${SIMVQ_VAL_DATASET_PATH:-/workspace/yi/work/Cars196/val_data}"
export SIMVQ_TOTAL_BATCH_SIZE="${SIMVQ_TOTAL_BATCH_SIZE:-24}"
export SIMVQ_MICRO_BATCH_SIZE="${SIMVQ_MICRO_BATCH_SIZE:-24}"
export NUM_EPOCHS="${NUM_EPOCHS:-200}"
export GPU_ID="${GPU_ID:-3}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

RUN_ID="${EXPERIMENT_RUN_ID:-independent-raq-rvq-src128-128-mixture-per-k_gpu${GPU_ID}-$(date +%Y%m%d-%H%M%S)}"
export EXPERIMENT_RUN_ID="$RUN_ID"

echo "Experiment: $SIMVQ_EXP_FAMILY"
echo "Run ID: $RUN_ID"
echo "Physical GPU: $GPU_ID"
echo "Source codebooks: $SIMVQ_NUM_EMBEDDINGS_LIST"
echo "Independent RAQ-RVQ depth: $SIMVQ_INDEPENDENT_RAQ_RVQ_DEPTH"
echo "Training policy: four independent 50/50 candidate-set choices per accumulation"
echo "Validation: fresh four-K samples from {2,4,8,16,32,64,128} per batch and epoch"
echo "Periodic model weights: every 20 epochs"
echo "Monitoring reference four K: $SIMVQ_INDEPENDENT_RAQ_RVQ_K_LISTS"
echo "Gradient mode: one final STE"
echo "RAQ curriculum: early=[$SIMVQ_RAQ_CURRICULUM_EARLY_LIST] middle=[$SIMVQ_RAQ_CURRICULUM_MIDDLE_LIST] late=[$SIMVQ_RAQ_CURRICULUM_LATE_LIST]"
echo "Learning rates: main/RAQ=$SIMVQ_LEARNING_RATE_G SimVQ-proj=$SIMVQ_CODEBOOK_PROJ_LR"
echo "Batch: total=$SIMVQ_TOTAL_BATCH_SIZE micro=$SIMVQ_MICRO_BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"

python -u train_mixture_src128_per_k.py 2>&1 | tee "experiments/logs/train_${RUN_ID}.log"
