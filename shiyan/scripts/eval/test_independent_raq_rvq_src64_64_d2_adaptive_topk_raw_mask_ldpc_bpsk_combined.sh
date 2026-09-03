#!/bin/bash
# Explicit one-bit Top-K masks; combined LDPC 1/2 + BPSK + AWGN at 0 dB.
set -euo pipefail

eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/shiyan

export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="shiyan_independent_raq_rvq_src64-64_trg2-64_d2_curriculum_rate094_A_patch_ch256-512"
export SIMVQ_NUM_EMBEDDINGS_LIST="64,64"
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
RVQ_K_LISTS="${RVQ_K_LISTS:-2,2;8,2}"
export SIMVQ_INDEPENDENT_RAQ_RVQ_K_LISTS="$RVQ_K_LISTS"
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
export GPU_ID="${GPU_ID:-3}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

CHECKPOINT="${CHECKPOINT:-checkpoints/shiyan_independent_raq_rvq_src64-64_trg2-64_d2_curriculum_rate094_A_patch_ch256-512_unet2_ds8x2_k64/best_vq_deepsc.pth}"
TARGET_ACTIVE_RATES="${TARGET_ACTIVE_RATES:-0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"
TARGET_ACTIVE_RATE_PAIRS="${TARGET_ACTIVE_RATE_PAIRS:-}"
SNRS="${SNRS:-0}"
MODULATION="${MODULATION:-bpsk}"
LDPC_N="${LDPC_N:-256}"
LDPC_RATE="${LDPC_RATE:-0.5}"
MAX_IMAGES="${MAX_IMAGES:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/eval/independent_raq_rvq_src64_64_d2_adaptive_topk_raw_mask_combined}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
JSON_OUTPUT="${JSON_OUTPUT:-$OUTPUT_DIR/results_${RUN_ID}.json}"
CSV_OUTPUT="${CSV_OUTPUT:-$OUTPUT_DIR/results_summary_${RUN_ID}.csv}"
PER_SCALE_CSV_OUTPUT="${PER_SCALE_CSV_OUTPUT:-$OUTPUT_DIR/per_scale_${RUN_ID}.csv}"
PER_IMAGE_CSV_OUTPUT="${PER_IMAGE_CSV_OUTPUT:-$OUTPUT_DIR/per_image_${RUN_ID}.csv}"
THRESHOLDS_CSV_OUTPUT="${THRESHOLDS_CSV_OUTPUT:-$OUTPUT_DIR/thresholds_${RUN_ID}.csv}"
LOG_PATH="${LOG_PATH:-$OUTPUT_DIR/eval_${RUN_ID}.log}"
mkdir -p "$OUTPUT_DIR"

read -r -a TARGET_ARGS <<< "$TARGET_ACTIVE_RATES"
read -r -a TARGET_PAIR_ARGS <<< "$TARGET_ACTIVE_RATE_PAIRS"
read -r -a SNR_ARGS <<< "$SNRS"

echo "Physical GPU: $GPU_ID"
echo "Independent stage K lists: $RVQ_K_LISTS"
echo "Selection: per-image/per-scale exact Top-K on stage-one residual"
echo "Mask: explicit one bit per token (1024 + 256 bit/image)"
echo "Packing: one combined stream per image"
echo "Channel: LDPC n=$LDPC_N rate=$LDPC_RATE + ${MODULATION^^} + AWGN"
echo "Activation targets: $TARGET_ACTIVE_RATES"
if [[ -n "$TARGET_ACTIVE_RATE_PAIRS" ]]; then
  echo "Independent scale targets: $TARGET_ACTIVE_RATE_PAIRS"
fi
echo "SNRs: $SNRS dB"
echo "Log: $LOG_PATH"

python -u test_independent_raq_rvq_adaptive_topk_raw_mask.py \
  --checkpoint "$CHECKPOINT" \
  --dataset "$SIMVQ_TEST_DATASET_PATH" \
  --rvq-k-lists "$RVQ_K_LISTS" \
  --target-active-rates "${TARGET_ARGS[@]}" \
  --target-active-rate-pairs "${TARGET_PAIR_ARGS[@]}" \
  --snrs "${SNR_ARGS[@]}" \
  --modulation "$MODULATION" \
  --ldpc-n "$LDPC_N" \
  --ldpc-rate "$LDPC_RATE" \
  --max-images "$MAX_IMAGES" \
  --num-workers "$NUM_WORKERS" \
  --json-output "$JSON_OUTPUT" \
  --csv-output "$CSV_OUTPUT" \
  --per-scale-csv-output "$PER_SCALE_CSV_OUTPUT" \
  --per-image-csv-output "$PER_IMAGE_CSV_OUTPUT" \
  --thresholds-csv-output "$THRESHOLDS_CSV_OUTPUT" \
  "$@" 2>&1 | tee "$LOG_PATH"
