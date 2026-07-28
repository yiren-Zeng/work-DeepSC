#!/bin/bash
set -euo pipefail

eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/shiyan

# Keep the original experiment/model layout unchanged.
export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="shiyan_raq_src2048-2048_raq2-2048_curriculum_rate044_A_patch_ch256-512"
export SIMVQ_NUM_EMBEDDINGS_LIST="2048,2048"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="4"
export SIMVQ_DECODER_RES_BLOCKS="4"
export SIMVQ_QUANTIZER_TYPE="simvq"
export SIMVQ_QUANTIZER_AXIS_LIST="patch,patch"
export SIMVQ_CVQ_CODEWORD_SHAPES="patch,patch"
export SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA="0.0"
export SIMVQ_USE_RAQ="1"
# This construction-time target is replaced per Bandit action at evaluation.
# export SIMVQ_RAQ_TARGET_LIST="${SIMVQ_RAQ_TARGET_LIST:-2048,2048}"
export SIMVQ_RAQ_MIN_TRG="2"
export SIMVQ_RAQ_MAX_TRG="2048"
export SIMVQ_RAQ_REPULSION_WEIGHT="0.00"
export SIMVQ_RAQ_LATENT_DISTILL_WEIGHT="0.00"
export SIMVQ_SRC_CODEBOOK_REPULSION_WEIGHT="0.00"
export SIMVQ_RAQ_USE_CURRICULUM="1"
export SIMVQ_RAQ_CURRICULUM_EARLY_LIST="512,1024,2048"
export SIMVQ_RAQ_CURRICULUM_MIDDLE_LIST="64,128,256,512,1024,2048"
export SIMVQ_RAQ_CURRICULUM_LATE_LIST="2,4,8,16,32,64,128,256,512,1024,2048"
export SIMVQ_TEST_USE_RAQ_RVQ="0"
export SIMVQ_USE_DYNAMIC_RAQ_RVQ="0"
export SIMVQ_TEST_DATASET_PATH="${SIMVQ_TEST_DATASET_PATH:-/workspace/yi/work/Kodak-256-transform-resize}"
export SIMVQ_TEST_NO_RESIZE="${SIMVQ_TEST_NO_RESIZE:-1}"

export GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

# Channel switch:
#   ldpc12_bpsk   (default): LDPC 1/2 + BPSK
#   ldpc12_qpsk            : LDPC 1/2 + QPSK
#   ldpc34_qpsk            : LDPC 3/4 + QPSK
#   ldpc12_16qam           : LDPC 1/2 + 16QAM
#   all                     : run all four profiles sequentially
CHANNEL_PROFILE="${CHANNEL_PROFILE:-ldpc12_bpsk}"
case "$CHANNEL_PROFILE" in
  ldpc12_bpsk|ldpc12_qpsk|ldpc34_qpsk|ldpc12_16qam|all) ;;
  *)
    echo "Unsupported CHANNEL_PROFILE=$CHANNEL_PROFILE" >&2
    echo "Use: ldpc12_bpsk, ldpc12_qpsk, ldpc34_qpsk, ldpc12_16qam, or all" >&2
    exit 2
    ;;
esac

CHECKPOINT="${CHECKPOINT:-checkpoints/shiyan_raq_src2048-2048_raq2-2048_curriculum_rate044_A_patch_ch256-512_unet2_ds8x2_k2048/best_vq_deepsc.pth}"
SNRS="${SNRS:-0 3 6 9 12}"
TARGET_RATIO="${TARGET_RATIO:-1/64}"
LDPC_N="${LDPC_N:-256}"

# Two exact-rate arms make a short search sufficient; every value is overridable.
BANDIT_EPISODES="${BANDIT_EPISODES:-8}"
WARMUP_PULLS="${WARMUP_PULLS:-1}"
EPS_START="${EPS_START:-0.4}"
EPS_END="${EPS_END:-0.05}"
EPS_DECAY="${EPS_DECAY:-30}"
# This is the epsilon-greedy policy RNG; it does not generate channel noise.
AGENT_SEED="${AGENT_SEED:-42}"

# Actual AWGN/LDPC evaluation seed. The default reproduces the historical
# test_real.py protocol exactly. Set FIXED_CHANNEL_SEED=multi to restore the
# partitioned search/confirmation/report Monte-Carlo seeds below.
FIXED_CHANNEL_SEED="${FIXED_CHANNEL_SEED:-42}"
SEARCH_SEED_BASE="${SEARCH_SEED_BASE:-42000}"
CONFIRM_SEEDS="${CONFIRM_SEEDS:-52000 52001 52002}"
REPORT_SEEDS="${REPORT_SEEDS:-62000 62001 62002 62003 62004}"

# MAX_IMAGES is only for a development smoke test. Keep zero for paper results.
EXPECTED_IMAGES="${EXPECTED_IMAGES:-24}"
MAX_IMAGES="${MAX_IMAGES:-0}"

read -r -a SNR_ARGS <<< "$SNRS"
read -r -a CONFIRM_SEED_ARGS <<< "$CONFIRM_SEEDS"
read -r -a REPORT_SEED_ARGS <<< "$REPORT_SEEDS"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --snrs "${SNR_ARGS[@]}"
  --channel-profile "$CHANNEL_PROFILE"
  --target-ratio "$TARGET_RATIO"
  --ldpc-n "$LDPC_N"
  --episodes "$BANDIT_EPISODES"
  --warmup-pulls "$WARMUP_PULLS"
  --eps-start "$EPS_START"
  --eps-end "$EPS_END"
  --eps-decay "$EPS_DECAY"
  --agent-seed "$AGENT_SEED"
  --search-seed-base "$SEARCH_SEED_BASE"
  --confirm-seeds "${CONFIRM_SEED_ARGS[@]}"
  --report-seeds "${REPORT_SEED_ARGS[@]}"
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
  OUTPUT_DIR="${OUTPUT_DIR:-experiments/eval/bandit_psnr}"
  RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
  JSON_OUTPUT="${JSON_OUTPUT:-$OUTPUT_DIR/${RUN_TAG}_${CHANNEL_PROFILE}.json}"
  CSV_OUTPUT="${CSV_OUTPUT:-$OUTPUT_DIR/${RUN_TAG}_${CHANNEL_PROFILE}.csv}"
  ARGS+=(--json-output "$JSON_OUTPUT" --csv-output "$CSV_OUTPUT")
elif [[ "$SAVE_RESULTS" != "0" ]]; then
  echo "SAVE_RESULTS must be 0 or 1, got $SAVE_RESULTS" >&2
  exit 2
fi

# Extra CLI arguments are appended intentionally, so scalar options can be
# overridden for one-off runs without editing this script.
python -u bandit_psnr_search.py "${ARGS[@]}" "$@"
