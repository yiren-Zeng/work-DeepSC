#!/bin/bash
# Evaluate [[4,64],[8,2]] as one payload under LDPC 3/4 + QPSK.
set -euo pipefail

eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/shiyan
mkdir -p experiments/eval

export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="shiyan_independent_raq_rvq_src64-64_trg2-64_d2_curriculum_rate094_A_patch_ch256-512_res6-6_ep400"
export SIMVQ_NUM_EMBEDDINGS_LIST="64,64"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="6"
export SIMVQ_DECODER_RES_BLOCKS="6"
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
export SIMVQ_INDEPENDENT_RAQ_RVQ_K_LISTS="${SIMVQ_INDEPENDENT_RAQ_RVQ_K_LISTS:-4,8;8,2}"
export SIMVQ_DYNAMIC_RAQ_RVQ_ZERO_CODEWORD="0"
export SIMVQ_TEST_USE_RAQ_RVQ="0"
unset SIMVQ_TEST_RAQ_RVQ_K_LISTS

export SIMVQ_TRAIN_BRANCH="joint"
export SIMVQ_RAQ_RECON_GRAD_MODE="ste"
export SIMVQ_RAQ_GENERATOR_TYPE="encoder_decoder"
export SIMVQ_RAQ_ROUTED_SRC_ENABLED="0"
export SIMVQ_RAQ_TRAIN_ENCODER="0"
unset SIMVQ_RAQ_ROUTED_SRC_SMALL_LIST SIMVQ_RAQ_ROUTED_SRC_LARGE_LIST

export SIMVQ_RAQ_TARGET_LIST="16,4"
export SIMVQ_RAQ_MIN_TRG="2"
export SIMVQ_RAQ_MAX_TRG="64"
unset SIMVQ_RAQ_MIN_TRG_LIST SIMVQ_RAQ_MAX_TRG_LIST
export SIMVQ_RAQ_REPULSION_WEIGHT="0.00"
export SIMVQ_RAQ_LATENT_DISTILL_WEIGHT="0.00"
export SIMVQ_SRC_CODEBOOK_REPULSION_WEIGHT="0.00"
export SIMVQ_RAQ_USE_CURRICULUM="1"
export SIMVQ_RAQ_CURRICULUM_EARLY_LIST="32,64"
export SIMVQ_RAQ_CURRICULUM_MIDDLE_LIST="8,16,32,64"
export SIMVQ_RAQ_CURRICULUM_LATE_LIST="2,4,8,16,32,64"
unset SIMVQ_RAQ_CURRICULUM_EARLY_LISTS
unset SIMVQ_RAQ_CURRICULUM_MIDDLE_LISTS
unset SIMVQ_RAQ_CURRICULUM_LATE_LISTS

export SIMVQ_TEST_DATASET_PATH="${SIMVQ_TEST_DATASET_PATH:-/workspace/yi/work/Kodak-256-transform-resize}"
export SIMVQ_TEST_NO_RESIZE="${SIMVQ_TEST_NO_RESIZE:-1}"
export GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

CHECKPOINT="${CHECKPOINT:-checkpoints/shiyan_independent_raq_rvq_src64-64_trg2-64_d2_curriculum_rate094_A_patch_ch256-512_res6-6_ep400_unet2_ds8x2_k64/best_vq_deepsc.pth}"
SNRS="${SNRS:-6}"
JSON_OUTPUT="${JSON_OUTPUT:-experiments/eval/independent_raq_rvq_src64-64_res6-6_ep400_k2x2-8x2_d2_ldpc12_bpsk_combined.json}"
NO_CHANNEL="${NO_CHANNEL:-0}"
read -r -a SNR_ARGS <<< "$SNRS"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --snrs "${SNR_ARGS[@]}"
  --modulation qpsk
  --stream-packing combined
  --ldpc_n 256
  --ldpc_k 0.75
  --json-output "$JSON_OUTPUT"
)
if [[ "$NO_CHANNEL" == "1" ]]; then
  ARGS+=(--no-channel)
fi

python -u test_real.py "${ARGS[@]}" "$@"
