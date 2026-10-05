#!/bin/bash
# Search four independent RAQ-RVQ K values for the randomval checkpoint.
# TARGET_RATIO is channel symbols / (C * H * W), including LDPC padding.
set -euo pipefail

eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/RAQ-RVQ

export SIMVQ_EXP_FAMILY="shiyan_independent_raq_rvq_src256-256_trg2-256_d2_curriculum_rate094_A_patch_ch256-512_res6-6_randomval"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="6"
export SIMVQ_DECODER_RES_BLOCKS="6"
export SIMVQ_INDEPENDENT_RAQ_RVQ_DEPTH="2"
# Used to construct the model; each search action replaces all four K values.
export SIMVQ_INDEPENDENT_RAQ_RVQ_K_LISTS="${SIMVQ_INDEPENDENT_RAQ_RVQ_K_LISTS:-16,2;8,2}"
export SIMVQ_RAQ_MIN_TRG="2"
export SIMVQ_RAQ_MAX_TRG="256"
# export SIMVQ_RAQ_CURRICULUM_EARLY_LIST="32"
# export SIMVQ_RAQ_CURRICULUM_MIDDLE_LIST="8,16,32"
# export SIMVQ_RAQ_CURRICULUM_LATE_LIST="2,4,8,16,32"

export SIMVQ_TEST_DATASET_PATH="${SIMVQ_TEST_DATASET_PATH:-/workspace/yi/work/Kodak-256-transform-resize}"
export SIMVQ_TEST_NO_RESIZE="${SIMVQ_TEST_NO_RESIZE:-1}"
export GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

CHECKPOINT="${CHECKPOINT:-/workspace/yi/work/RAQ-RVQ/checkpoints/shiyan_independent_raq_rvq_src256-256_trg2-256_d2_curriculum_rate094_A_patch_ch256-512_res6-6_mixture_per_k_unet2_ds8x2_k256/best_vq_deepsc.pth}"
SNRS="${SNRS:-7.5}"
# Use CHANNEL_PROFILE=ldpc12_qpsk (or another legacy preset) only when needed.
# The default custom mode lets LDPC_RATE and MODULATION be selected independently.
CHANNEL_PROFILE="${CHANNEL_PROFILE:-custom}"
LDPC_RATE="${LDPC_RATE:-3/4}"
MODULATION="${MODULATION:-qpsk}"
STREAM_PACKING="${STREAM_PACKING:-combined}"
TARGET_RATIO="${TARGET_RATIO:-1/48}"
LDPC_N="${LDPC_N:-256}"

MIN_K="${MIN_K:-2}"
MAX_K="${MAX_K:-256}"

BANDIT_EPISODES="${BANDIT_EPISODES:-100}"
WARMUP_PULLS="${WARMUP_PULLS:-1}"
EPS_START="${EPS_START:-0.4}"
EPS_END="${EPS_END:-0.05}"
EPS_DECAY="${EPS_DECAY:-30}"
AGENT_SEED="${AGENT_SEED:-42}"
CONFIRM_TOP_K="${CONFIRM_TOP_K:-2}"

# Search, confirmation, and final reporting use separate channel seeds.
FIXED_CHANNEL_SEED="${FIXED_CHANNEL_SEED:-multi}"
SEARCH_SEED_BASE="${SEARCH_SEED_BASE:-42000}"
CONFIRM_SEEDS="${CONFIRM_SEEDS:-52000 52001 52002}"
REPORT_SEEDS="${REPORT_SEEDS:-62000 62001 62002 62003 62004}"

# MAX_IMAGES is for a quick functional check; use zero for full results.
EXPECTED_IMAGES="${EXPECTED_IMAGES:-24}"
MAX_IMAGES="${MAX_IMAGES:-0}"

read -r -a SNR_ARGS <<< "$SNRS"
read -r -a CONFIRM_SEED_ARGS <<< "$CONFIRM_SEEDS"
read -r -a REPORT_SEED_ARGS <<< "$REPORT_SEEDS"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --snrs "${SNR_ARGS[@]}"
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

if [[ "$CHANNEL_PROFILE" == "custom" ]]; then
  echo "Channel: LDPC_RATE=$LDPC_RATE | LDPC_N=$LDPC_N | MODULATION=$MODULATION | TARGET_RATIO=$TARGET_RATIO"
  ARGS+=(--ldpc-rate "$LDPC_RATE" --modulation "$MODULATION")
  LDPC_RATE_TAG="${LDPC_RATE//\//over}"
  CHANNEL_TAG="ldpc${LDPC_RATE_TAG}_${MODULATION}"
else
  echo "Channel profile: $CHANNEL_PROFILE | LDPC_N=$LDPC_N | TARGET_RATIO=$TARGET_RATIO"
  ARGS+=(--channel-profile "$CHANNEL_PROFILE")
  CHANNEL_TAG="$CHANNEL_PROFILE"
fi

if [[ "$FIXED_CHANNEL_SEED" == "multi" ]]; then
  echo "Channel seed mode: partitioned multi-seed Monte Carlo"
elif [[ "$FIXED_CHANNEL_SEED" =~ ^[0-9]+$ ]]; then
  echo "Channel seed mode: fixed seed $FIXED_CHANNEL_SEED"
  ARGS+=(--fixed-channel-seed "$FIXED_CHANNEL_SEED")
else
  echo "FIXED_CHANNEL_SEED must be a non-negative integer or 'multi', got: $FIXED_CHANNEL_SEED" >&2
  exit 2
fi

SAVE_RESULTS="${SAVE_RESULTS:-1}"
if [[ "$SAVE_RESULTS" == "1" ]]; then
  RATE_TAG="${TARGET_RATIO//\//over}"
  OUTPUT_DIR="${OUTPUT_DIR:-experiments/eval/independent_raq_rvq_src64-64_d2_res6-6_randomval_rate${RATE_TAG}_bandit_psnr_${CHANNEL_TAG}_${STREAM_PACKING}}"
  RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
  JSON_OUTPUT="${JSON_OUTPUT:-$OUTPUT_DIR/${RUN_TAG}.json}"
  CSV_OUTPUT="${CSV_OUTPUT:-$OUTPUT_DIR/${RUN_TAG}.csv}"
  ARGS+=(--json-output "$JSON_OUTPUT" --csv-output "$CSV_OUTPUT")
elif [[ "$SAVE_RESULTS" != "0" ]]; then
  echo "SAVE_RESULTS must be 0 or 1, got: $SAVE_RESULTS" >&2
  exit 2
fi

# Extra CLI arguments can override defaults for quick checks.
python -u bandit_independent_raq_rvq_psnr_search.py "${ARGS[@]}" "$@"
