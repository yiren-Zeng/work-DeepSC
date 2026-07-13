#!/bin/bash
set -euo pipefail
eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/shiyan

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
export SIMVQ_CVQ_CODEWORD_SHAPES="patch,patch"
export SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA="0.0"
export SIMVQ_USE_RAQ="0"
export SIMVQ_RAQ_LATENT_DISTILL_WEIGHT="0.00"
export SIMVQ_SRC_CODEBOOK_REPULSION_WEIGHT="0.00"
export SIMVQ_RAQ_USE_CURRICULUM="0"
export SIMVQ_TEST_DATASET_PATH="${SIMVQ_TEST_DATASET_PATH:-/workspace/yi/work/Kodak-256-transform-resize}"
export SIMVQ_TEST_NO_RESIZE="${SIMVQ_TEST_NO_RESIZE:-1}"
export GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
unset SIMVQ_PRETRAINED_CHECKPOINT
unset SIMVQ_ALLOW_PRETRAINED

CHECKPOINT="${CHECKPOINT:-checkpoints/shiyan_raq_src64-64_raq2-64_staged_v2_curriculum_stage1_src_teacher_rate044_A_patch_ch64-128_unet2_ds8x2_k64/best_vq_deepsc.pth}"
SNRS="${SNRS:-0 3 6 9 12}"
MODULATION="${MODULATION:-bpsk}"
JSON_OUTPUT="${JSON_OUTPUT:-}"
NO_CHANNEL="${NO_CHANNEL:-0}"
read -r -a SNR_ARGS <<< "$SNRS"

ARGS=(--checkpoint "$CHECKPOINT" --snrs "${SNR_ARGS[@]}" --modulation "$MODULATION")
if [[ "$NO_CHANNEL" == "1" ]]; then
  ARGS+=(--no-channel)
fi
if [[ -n "$JSON_OUTPUT" ]]; then
  ARGS+=(--json-output "$JSON_OUTPUT")
fi

if [[ "$#" -eq 0 ]]; then
  python -u test_real.py "${ARGS[@]}"
else
  python -u test_real.py "$@"
fi
