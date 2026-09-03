#!/bin/bash
# Four trained RAQ codebooks: two scales x two independent residual stages.
set -euo pipefail

eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/shiyan
mkdir -p checkpoints experiments/logs

export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="shiyan_independent_raq_rvq_src256-256_trg2-256_d2_curriculum_rate094_A_patch_ch256-512"
export SIMVQ_NUM_EMBEDDINGS_LIST="256,256"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="4"
export SIMVQ_DECODER_RES_BLOCKS="4"
export SIMVQ_QUANTIZER_TYPE="simvq"
export SIMVQ_QUANTIZER_AXIS_LIST="patch,patch"
export SIMVQ_CVQ_CODEWORD_SHAPES="patch,patch"
export SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA="0.0"
export SIMVQ_LPIPS_WEIGHT="0"
export SIMVQ_MODEL_PARALLEL="0"
export SIMVQ_USE_SWINIR_ENHANCE="0"
export SIMVQ_USE_SWIN_BACKBONE="0"

export SIMVQ_USE_RAQ="1"
export SIMVQ_USE_SHARED_RAQ_RVQ="0"
export SIMVQ_USE_DYNAMIC_RAQ_RVQ="0"
export SIMVQ_USE_INDEPENDENT_RAQ_RVQ="1"
export SIMVQ_INDEPENDENT_RAQ_RVQ_DEPTH="2"
# Deterministic validation layout. Training samples all four K independently.
export SIMVQ_INDEPENDENT_RAQ_RVQ_K_LISTS="16,16;4,4"
export SIMVQ_DYNAMIC_RAQ_RVQ_ZERO_CODEWORD="0"
export SIMVQ_TEST_USE_RAQ_RVQ="0"
unset SIMVQ_TEST_RAQ_RVQ_K_LISTS

export SIMVQ_TRAIN_BRANCH="joint"
export SIMVQ_RAQ_RECON_GRAD_MODE="ste"
export SIMVQ_RAQ_GENERATOR_TYPE="encoder_decoder"
export SIMVQ_RAQ_ROUTED_SRC_ENABLED="0"
export SIMVQ_RAQ_TRAIN_ENCODER="0"
unset SIMVQ_RAQ_ROUTED_SRC_SMALL_LIST SIMVQ_RAQ_ROUTED_SRC_LARGE_LIST

# The flat pair is retained as nominal per-scale metadata. The nested four-K
# value above is authoritative for independent validation and evaluation.
export SIMVQ_RAQ_TARGET_LIST="16,4"
export SIMVQ_RAQ_MIN_TRG="2"
export SIMVQ_RAQ_MAX_TRG="256"
unset SIMVQ_RAQ_MIN_TRG_LIST SIMVQ_RAQ_MAX_TRG_LIST
export SIMVQ_RAQ_REPULSION_WEIGHT="0.00"
export SIMVQ_RAQ_LATENT_DISTILL_WEIGHT="0.00"
export SIMVQ_SRC_CODEBOOK_REPULSION_WEIGHT="0.00"
unset SIMVQ_RAQ_LATENT_DISTILL_FINAL_WEIGHT
unset SIMVQ_RAQ_LATENT_DISTILL_DECAY_END_EPOCH
export SIMVQ_RAQ_USE_CURRICULUM="1"
export SIMVQ_RAQ_CURRICULUM_EARLY_LIST="32,64,128,256"
export SIMVQ_RAQ_CURRICULUM_MIDDLE_LIST="8,16,32,64,128,256"
export SIMVQ_RAQ_CURRICULUM_LATE_LIST="2,4,8,16,32,64,128,256"
unset SIMVQ_RAQ_CURRICULUM_EARLY_LISTS
unset SIMVQ_RAQ_CURRICULUM_MIDDLE_LISTS
unset SIMVQ_RAQ_CURRICULUM_LATE_LISTS

export SIMVQ_LEARNING_RATE_G="5e-5"
export SIMVQ_CODEBOOK_PROJ_LR="2e-4"
export SIMVQ_CHANNEL_PROB_START_EPOCH="80"
export SIMVQ_CHANNEL_PROB_END_EPOCH="120"
export SIMVQ_RESUME="0"
unset SIMVQ_PRETRAINED_CHECKPOINT
unset SIMVQ_ALLOW_PRETRAINED

export SIMVQ_TRAIN_DATASET_PATH="${SIMVQ_TRAIN_DATASET_PATH:-/workspace/yi/work/Cars196/train_data}"
export SIMVQ_VAL_DATASET_PATH="${SIMVQ_VAL_DATASET_PATH:-/workspace/yi/work/Cars196/val_data}"
export SIMVQ_TOTAL_BATCH_SIZE="${SIMVQ_TOTAL_BATCH_SIZE:-24}"
export SIMVQ_MICRO_BATCH_SIZE="${SIMVQ_MICRO_BATCH_SIZE:-24}"
export NUM_EPOCHS="${NUM_EPOCHS:-200}"
export GPU_ID="${GPU_ID:-4}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

RUN_ID="independent-raq-rvq-src256-256-k2-256-d2-ch256-512_gpu${GPU_ID}-$(date +%Y%m%d-%H%M%S)"
export EXPERIMENT_RUN_ID="$RUN_ID"

echo "Experiment: $SIMVQ_EXP_FAMILY"
echo "Run ID: $RUN_ID"
echo "Physical GPU: $GPU_ID"
echo "Source codebooks: $SIMVQ_NUM_EMBEDDINGS_LIST"
echo "Independent RAQ-RVQ: enabled=$SIMVQ_USE_INDEPENDENT_RAQ_RVQ depth=$SIMVQ_INDEPENDENT_RAQ_RVQ_DEPTH"
echo "Training policy: four independent curriculum K samples per accumulation"
echo "Validation four K: $SIMVQ_INDEPENDENT_RAQ_RVQ_K_LISTS"
echo "Validation payload: 9216 bits/image = 0.140625 bpp"
echo "Validation transmission ratio (LDPC1/2+BPSK): 0.09375000"
echo "Gradient mode: $SIMVQ_RAQ_RECON_GRAD_MODE (one final STE)"
echo "Train branch: $SIMVQ_TRAIN_BRANCH"
echo "RAQ curriculum: early=[$SIMVQ_RAQ_CURRICULUM_EARLY_LIST] middle=[$SIMVQ_RAQ_CURRICULUM_MIDDLE_LIST] late=[$SIMVQ_RAQ_CURRICULUM_LATE_LIST]"
echo "Learning rates: main/RAQ=$SIMVQ_LEARNING_RATE_G SimVQ-proj=$SIMVQ_CODEBOOK_PROJ_LR"
echo "Batch: total=$SIMVQ_TOTAL_BATCH_SIZE micro=$SIMVQ_MICRO_BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"

python -u train.py 2>&1 | tee "experiments/logs/train_${RUN_ID}.log"
