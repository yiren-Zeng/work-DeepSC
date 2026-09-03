#!/bin/bash
# Epsilon-greedy search of the two ordered RAQ-RVQ K values for the single-scale
# src64 U-Net1 checkpoint at exact 1/32 channel-symbol ratio. All stage indices
# are concatenated into one LDPC 3/4 + QPSK payload before channel transmission.
set -euo pipefail

eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/shiyan

# Match test_independent_raq_rvq_src64_d2_unet1_combined.sh and its checkpoint.
export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="shiyan_independent_raq_rvq_src64_trg2-64_d2_curriculum_rate063_A_patch_ch256"
export SIMVQ_NUM_EMBEDDINGS_LIST="64"
export SIMVQ_DOWNSAMPLE_STRIDES="8"
export SIMVQ_UNET_DEPTH="1"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="4"
export SIMVQ_DECODER_RES_BLOCKS="4"
export SIMVQ_QUANTIZER_TYPE="simvq"
export SIMVQ_QUANTIZER_AXIS_LIST="patch"
export SIMVQ_CVQ_CODEWORD_SHAPES="patch"
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
# Checkpoint construction and post-search restore layout. Each Bandit pull
# temporarily replaces this pair with one exact-rate candidate action.
export SIMVQ_INDEPENDENT_RAQ_RVQ_K_LISTS="${SIMVQ_INDEPENDENT_RAQ_RVQ_K_LISTS:-64,8}"
export SIMVQ_DYNAMIC_RAQ_RVQ_ZERO_CODEWORD="0"
export SIMVQ_TEST_USE_RAQ_RVQ="0"
unset SIMVQ_TEST_RAQ_RVQ_K_LISTS

export SIMVQ_TRAIN_BRANCH="joint"
export SIMVQ_RAQ_RECON_GRAD_MODE="ste"
export SIMVQ_RAQ_GENERATOR_TYPE="encoder_decoder"
export SIMVQ_RAQ_ROUTED_SRC_ENABLED="0"
export SIMVQ_RAQ_TRAIN_ENCODER="0"
unset SIMVQ_RAQ_ROUTED_SRC_SMALL_LIST SIMVQ_RAQ_ROUTED_SRC_LARGE_LIST

export SIMVQ_RAQ_TARGET_LIST="16"
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
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

CHECKPOINT="${CHECKPOINT:-checkpoints/shiyan_independent_raq_rvq_src64_trg2-64_d2_curriculum_rate063_A_patch_ch256_unet1_ds8_k64/best_vq_deepsc.pth}"
SNRS="${SNRS:-10}"
CHANNEL_PROFILE="${CHANNEL_PROFILE:-ldpc12_16qam}"
STREAM_PACKING="${STREAM_PACKING:-combined}"
# Project convention: channel symbols / (C * H * W), including RGB channels.
TARGET_RATIO="${TARGET_RATIO:-1/48}"
LDPC_N="${LDPC_N:-256}"
MIN_K="${MIN_K:-2}"
MAX_K="${MAX_K:-64}"

# At the default exact rate there are four ordered arms. One common-seed pull
# per arm is forced before epsilon-greedy exploration/exploitation starts.
BANDIT_EPISODES="${BANDIT_EPISODES:-20}"
WARMUP_PULLS="${WARMUP_PULLS:-1}"
EPS_START="${EPS_START:-0.4}"
EPS_END="${EPS_END:-0.05}"
EPS_DECAY="${EPS_DECAY:-30}"
AGENT_SEED="${AGENT_SEED:-42}"
CONFIRM_TOP_K="${CONFIRM_TOP_K:-2}"

if ! [[ "$BANDIT_EPISODES" =~ ^[0-9]+$ ]] || (( BANDIT_EPISODES < 1 )); then
  echo "BANDIT_EPISODES must be a positive integer, got: $BANDIT_EPISODES" >&2
  exit 2
fi

# Partitioned seeds are the reporting default. Set FIXED_CHANNEL_SEED to a
# non-negative integer only for deterministic smoke tests or reproduction.
FIXED_CHANNEL_SEED="${FIXED_CHANNEL_SEED:-multi}"
SEARCH_SEED_BASE="${SEARCH_SEED_BASE:-42000}"
CONFIRM_SEEDS="${CONFIRM_SEEDS:-52000 52001 52002}"
REPORT_SEEDS="${REPORT_SEEDS:-62000 62001 62002 62003 62004}"

EXPECTED_IMAGES="${EXPECTED_IMAGES:-24}"
MAX_IMAGES="${MAX_IMAGES:-0}"

read -r -a SNR_ARGS <<< "$SNRS"
read -r -a CONFIRM_SEED_ARGS <<< "$CONFIRM_SEEDS"
read -r -a REPORT_SEED_ARGS <<< "$REPORT_SEEDS"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --snrs "${SNR_ARGS[@]}"
  --channel-profile "$CHANNEL_PROFILE"
  --stream-packing "$STREAM_PACKING"
  --target-ratio "$TARGET_RATIO"
  --ldpc-n "$LDPC_N"
  --min-k "$MIN_K"
  --max-k "$MAX_K"
  --episodes "$BANDIT_EPISODES"
  --warmup-pulls "$WARMUP_PULLS"
  --eps-start "$EPS_START"
  --eps-end "$EPS_END"
  --eps-decay "$EPS_DECAY"
  --agent-seed "$AGENT_SEED"
  --search-seed-base "$SEARCH_SEED_BASE"
  --confirm-seeds "${CONFIRM_SEED_ARGS[@]}"
  --report-seeds "${REPORT_SEED_ARGS[@]}"
  --confirm-top-k "$CONFIRM_TOP_K"
  --expected-images "$EXPECTED_IMAGES"
  --max-images "$MAX_IMAGES"
)

if [[ "$FIXED_CHANNEL_SEED" == "multi" ]]; then
  echo "Channel seed mode: partitioned multi-seed Monte-Carlo"
elif [[ "$FIXED_CHANNEL_SEED" =~ ^[0-9]+$ ]]; then
  echo "Channel seed mode: fixed seed $FIXED_CHANNEL_SEED"
  ARGS+=(--fixed-channel-seed "$FIXED_CHANNEL_SEED")
else
  echo "FIXED_CHANNEL_SEED must be a non-negative integer or 'multi', got: $FIXED_CHANNEL_SEED" >&2
  exit 2
fi

SAVE_RESULTS="${SAVE_RESULTS:-1}"
if [[ "$SAVE_RESULTS" == "1" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-experiments/eval/independent_raq_rvq_src64_unet1_rate1over32_bandit_psnr_${CHANNEL_PROFILE}_${STREAM_PACKING}}"
  RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
  JSON_OUTPUT="${JSON_OUTPUT:-$OUTPUT_DIR/${RUN_TAG}.json}"
  CSV_OUTPUT="${CSV_OUTPUT:-$OUTPUT_DIR/${RUN_TAG}.csv}"
  ARGS+=(--json-output "$JSON_OUTPUT" --csv-output "$CSV_OUTPUT")
elif [[ "$SAVE_RESULTS" != "0" ]]; then
  echo "SAVE_RESULTS must be 0 or 1, got: $SAVE_RESULTS" >&2
  exit 2
fi

# Extra arguments are appended for one-off diagnostics and smoke tests.
python -u bandit_independent_raq_rvq_psnr_search.py "${ARGS[@]}" "$@"
