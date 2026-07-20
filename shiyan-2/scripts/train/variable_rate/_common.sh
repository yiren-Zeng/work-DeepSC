#!/usr/bin/env bash
# Shared, source-only helpers for the isolated five-stage variable-rate pipeline.

set -euo pipefail

VR_PROJECT_ROOT="/workspace/yi/work/shiyan-2"

vr_die() {
  echo "[single-teacher-variable-rate] ERROR: $*" >&2
  exit 1
}

vr_require_file() {
  local path="$1"
  local label="${2:-file}"
  [[ -f "$path" ]] || vr_die "missing ${label}: ${path}"
}

vr_require_dir() {
  local path="$1"
  local label="${2:-directory}"
  [[ -d "$path" ]] || vr_die "missing ${label}: ${path}"
}

vr_require_local_checkpoint() {
  local path="$1"
  local label="${2:-checkpoint}"
  vr_require_file "$path" "$label"
  local resolved
  resolved="$(realpath -e "$path")"
  case "$resolved" in
    "${VR_PROJECT_ROOT}"/checkpoints/*) ;;
    *) vr_die "${label} must be stored under ${VR_PROJECT_ROOT}/checkpoints: ${resolved}" ;;
  esac
}

vr_init() {
  local default_experiment_name="$1"

  vr_require_dir "$VR_PROJECT_ROOT" "isolated project root"
  cd "$VR_PROJECT_ROOT"
  vr_require_file "$VR_PROJECT_ROOT/train_variable_rate.py" "training entrypoint"

  export GPU_ID="${GPU_ID:-2}"
  [[ "$GPU_ID" =~ ^[0-9]+$ ]] || vr_die "GPU_ID must be one physical GPU index, got: ${GPU_ID}"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  # Once one physical GPU is made visible, it is always logical cuda:0 in Python.
  export SIMVQ_DEVICE="cuda:0"

  export PYTHON_BIN="${PYTHON_BIN:-/home/yi/.conda/envs/work/bin/python}"
  [[ -x "$PYTHON_BIN" ]] || vr_die "Python executable is not available: ${PYTHON_BIN}"

  export SIMVQ_TRAIN_DATASET_PATH="${SIMVQ_TRAIN_DATASET_PATH:-/workspace/yi/work/Cars196/train_data}"
  export SIMVQ_VAL_DATASET_PATH="${SIMVQ_VAL_DATASET_PATH:-/workspace/yi/work/Cars196/val_data}"
  export SIMVQ_TEST_DATASET_PATH="${SIMVQ_TEST_DATASET_PATH:-/workspace/yi/work/Kodak-256-transform-resize}"
  vr_require_dir "$SIMVQ_TRAIN_DATASET_PATH" "training dataset"
  vr_require_dir "$SIMVQ_VAL_DATASET_PATH" "validation dataset"

  export SIMVQ_EXPERIMENT_NAME="${SIMVQ_EXPERIMENT_NAME:-$default_experiment_name}"
  if [[ "$SIMVQ_EXPERIMENT_NAME" == *"/"* || "$SIMVQ_EXPERIMENT_NAME" == *".."* ]]; then
    vr_die "SIMVQ_EXPERIMENT_NAME must be a plain directory name: ${SIMVQ_EXPERIMENT_NAME}"
  fi

  export SIMVQ_EXP_FAMILY="${SIMVQ_EXP_FAMILY:-single_teacher_variable_rate_raq}"
  export SIMVQ_CHECKPOINT_DIR="${VR_PROJECT_ROOT}/checkpoints/${SIMVQ_EXPERIMENT_NAME}"
  export SIMVQ_LOG_DIR="${VR_PROJECT_ROOT}/experiments/tensorboard/${SIMVQ_EXPERIMENT_NAME}"
  export SIMVQ_METRICS_PATH="${VR_PROJECT_ROOT}/experiments/${SIMVQ_EXPERIMENT_NAME}_epoch_metrics.csv"
  export SIMVQ_RAQ_PROFILE_METRICS_PATH="${VR_PROJECT_ROOT}/experiments/${SIMVQ_EXPERIMENT_NAME}_profile_metrics.csv"
  export SIMVQ_CODEBOOK_METRICS_PATH="${VR_PROJECT_ROOT}/experiments/${SIMVQ_EXPERIMENT_NAME}_codebook_metrics.csv"
  export SIMVQ_SNAPSHOT_DIR="${VR_PROJECT_ROOT}/experiments/snapshots/${SIMVQ_EXPERIMENT_NAME}"

  # First formal experiment: the existing two-scale backbone, with one 2048x2048
  # source teacher and lightweight rate-conditioned RAQ modules.
  export SIMVQ_UNET_DEPTH="${SIMVQ_UNET_DEPTH:-2}"
  export SIMVQ_BASE_CHANNELS="${SIMVQ_BASE_CHANNELS:-32}"
  export SIMVQ_EMBEDDING_DIM_LIST="${SIMVQ_EMBEDDING_DIM_LIST:-64,128}"
  export SIMVQ_DOWNSAMPLE_STRIDES="${SIMVQ_DOWNSAMPLE_STRIDES:-8,2}"
  export SIMVQ_ENCODER_RES_BLOCKS="${SIMVQ_ENCODER_RES_BLOCKS:-4}"
  export SIMVQ_DECODER_RES_BLOCKS="${SIMVQ_DECODER_RES_BLOCKS:-4}"
  export SIMVQ_QUANTIZER_TYPE="simvq"
  export SIMVQ_TRAIN_RESIZE="${SIMVQ_TRAIN_RESIZE:-256x256}"
  export SIMVQ_VAL_RESIZE="${SIMVQ_VAL_RESIZE:-256x256}"
  export SIMVQ_MICRO_BATCH_SIZE="${SIMVQ_MICRO_BATCH_SIZE:-4}"
  export SIMVQ_TOTAL_BATCH_SIZE="${SIMVQ_TOTAL_BATCH_SIZE:-16}"
  export SIMVQ_NUM_WORKERS="${SIMVQ_NUM_WORKERS:-8}"
  export SIMVQ_RAQ_PROFILE_SAMPLER_SEED="${SIMVQ_RAQ_PROFILE_SAMPLER_SEED:-3407}"
  export SIMVQ_RAQ_SANDWICH_NUM_RANDOM="${SIMVQ_RAQ_SANDWICH_NUM_RANDOM:-1}"
  export SIMVQ_RAQ_VAL_PROFILES="${SIMVQ_RAQ_VAL_PROFILES:-2048x2048;2048x16;16x2;1024x256;512x64;64x16}"
  export SIMVQ_RAQ_VAL_REQUIRE_MAX_PROTECTION="${SIMVQ_RAQ_VAL_REQUIRE_MAX_PROTECTION:-1}"
  export SIMVQ_RAQ_VAL_MAX_PSNR_DROP_DB="${SIMVQ_RAQ_VAL_MAX_PSNR_DROP_DB:-0.30}"

  mkdir -p \
    "$SIMVQ_CHECKPOINT_DIR" \
    "$SIMVQ_LOG_DIR" \
    "$SIMVQ_SNAPSHOT_DIR"
}

vr_run_train() {
  echo "[single-teacher-variable-rate] stage=${SIMVQ_RAQ_STAGE}"
  echo "[single-teacher-variable-rate] physical GPU=${GPU_ID}, Python device=${SIMVQ_DEVICE}"
  echo "[single-teacher-variable-rate] experiment=${SIMVQ_EXPERIMENT_NAME}"
  echo "[single-teacher-variable-rate] checkpoints=${SIMVQ_CHECKPOINT_DIR}"
  "$PYTHON_BIN" -u train_variable_rate.py "$@"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  vr_die "_common.sh is a helper; run one of run_stage*.sh instead"
fi
