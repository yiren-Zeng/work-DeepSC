#!/usr/bin/env bash
set -euo pipefail

readonly VR_EVAL_PROJECT_ROOT="/workspace/yi/work/shiyan-2"

vr_eval_die() {
  echo "[variable-rate-stage-eval] ERROR: $*" >&2
  exit 1
}

vr_eval_require_bool() {
  local value="$1"
  local name="$2"
  [[ "$value" == "0" || "$value" == "1" ]] \
    || vr_eval_die "${name} must be 0 or 1"
}

vr_eval_init() {
  cd "$VR_EVAL_PROJECT_ROOT"

  export GPU_ID="${GPU_ID:-2}"
  [[ "$GPU_ID" =~ ^[0-9]+$ ]] \
    || vr_eval_die "GPU_ID must be one physical GPU index"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"

  VR_EVAL_DEVICE="${DEVICE:-cuda:0}"
  export SIMVQ_DEVICE="$VR_EVAL_DEVICE"
  VR_EVAL_PYTHON="${PYTHON_BIN:-/home/yi/.conda/envs/work/bin/python}"
  VR_EVAL_FAMILY="${SIMVQ_EXP_FAMILY:-single_teacher_variable_rate_raq}"
  VR_EVAL_DATASET="${DATASET:-/workspace/yi/work/Kodak-256-transform-resize}"
  VR_EVAL_ALL_PROFILES="${ALL_PROFILES:-0}"
  VR_EVAL_DRY_RUN="${DRY_RUN:-0}"
  VR_EVAL_RUN_NAME="${EVAL_RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"

  [[ "$VR_EVAL_FAMILY" =~ ^[A-Za-z0-9._-]+$ ]] \
    || vr_eval_die "SIMVQ_EXP_FAMILY contains an unsafe path character"
  [[ "$VR_EVAL_RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] \
    || vr_eval_die "EVAL_RUN_NAME must be a plain directory name"
  vr_eval_require_bool "$VR_EVAL_ALL_PROFILES" "ALL_PROFILES"
  vr_eval_require_bool "$VR_EVAL_DRY_RUN" "DRY_RUN"

  if [[ "$VR_EVAL_DRY_RUN" == "0" ]]; then
    [[ -x "$VR_EVAL_PYTHON" ]] \
      || vr_eval_die "Python executable is not available: ${VR_EVAL_PYTHON}"
    [[ -f "$VR_EVAL_PROJECT_ROOT/evaluate_variable_rate.py" ]] \
      || vr_eval_die "evaluate_variable_rate.py is missing"
    [[ -d "$VR_EVAL_DATASET" ]] \
      || vr_eval_die "test dataset is missing: ${VR_EVAL_DATASET}"
  fi
}

vr_eval_require_local_checkpoint() {
  local checkpoint="$1"
  local label="$2"
  if [[ "$VR_EVAL_DRY_RUN" == "1" ]]; then
    return
  fi
  [[ -f "$checkpoint" ]] || vr_eval_die "${label} is missing: ${checkpoint}"
  local resolved
  resolved="$(realpath -e "$checkpoint")"
  case "$resolved" in
    "${VR_EVAL_PROJECT_ROOT}"/checkpoints/*) ;;
    *) vr_eval_die "${label} must be under ${VR_EVAL_PROJECT_ROOT}/checkpoints: ${resolved}" ;;
  esac
}

vr_eval_prepare() {
  local stage_label="$1"
  local checkpoint="$2"
  local expected_stage="$3"
  local profiles="$4"
  local output_suffix="${5:-}"

  if [[ "$expected_stage" == "src_teacher" ]]; then
    [[ "$VR_EVAL_ALL_PROFILES" == "0" ]] \
      || vr_eval_die "Stage 1 has only profile 2048x2048; ALL_PROFILES=1 is invalid"
    [[ "$profiles" == "2048x2048" ]] \
      || vr_eval_die "Stage 1 only accepts PROFILES=2048x2048"
  fi
  vr_eval_require_local_checkpoint "$checkpoint" "${stage_label} checkpoint"

  VR_EVAL_OUTPUT_DIR="${VR_EVAL_PROJECT_ROOT}/experiments/eval/${VR_EVAL_FAMILY}_${stage_label}_${VR_EVAL_RUN_NAME}${output_suffix}"
  if [[ "$VR_EVAL_DRY_RUN" == "0" ]]; then
    mkdir -p "$VR_EVAL_OUTPUT_DIR/per_profile"
  fi

  VR_EVAL_ARGS=(
    --checkpoint "$checkpoint"
    --expected-stage "$expected_stage"
    --dataset "$VR_EVAL_DATASET"
    --device "$VR_EVAL_DEVICE"
    --batch-size "${BATCH_SIZE:-1}"
    --num-workers "${NUM_WORKERS:-2}"
    --seed "${SEED:-42}"
    --profiles "$profiles"
    --csv "$VR_EVAL_OUTPUT_DIR/profiles.csv"
    --per-profile-csv-dir "$VR_EVAL_OUTPUT_DIR/per_profile"
    --json "$VR_EVAL_OUTPUT_DIR/results.json"
    --worst-profile-weight "${WORST_PROFILE_WEIGHT:-0.20}"
    --max-teacher-drop-db "${MAX_TEACHER_DROP_DB:-0.30}"
    --collapse-threshold "${COLLAPSE_THRESHOLD:-0.10}"
  )
  if [[ "$VR_EVAL_ALL_PROFILES" == "1" ]]; then
    VR_EVAL_ARGS+=(--all-profiles)
  fi
  if [[ -n "${MAX_BATCHES:-}" ]]; then
    VR_EVAL_ARGS+=(--max-batches "$MAX_BATCHES")
  fi
  if [[ -n "${TEST_RESIZE:-256x256}" ]]; then
    VR_EVAL_ARGS+=(--test-resize "${TEST_RESIZE:-256x256}")
  fi
  if [[ -n "${SRC_REFERENCE_PSNR:-}" ]]; then
    VR_EVAL_ARGS+=(--src-reference-psnr "$SRC_REFERENCE_PSNR")
  fi
  if [[ -n "${PROFILE_WEIGHTS:-}" ]]; then
    VR_EVAL_ARGS+=(--profile-weights "$PROFILE_WEIGHTS")
  fi
}

vr_eval_attach_teacher() {
  local teacher_checkpoint="$1"
  vr_eval_require_local_checkpoint "$teacher_checkpoint" "Stage-1 SRC teacher checkpoint"
  VR_EVAL_ARGS+=(--teacher-checkpoint "$teacher_checkpoint")
}

vr_eval_execute() {
  printf '[variable-rate-stage-eval] command:'
  printf ' %q' "$VR_EVAL_PYTHON" -u evaluate_variable_rate.py "${VR_EVAL_ARGS[@]}"
  printf '\n'
  if [[ "$VR_EVAL_DRY_RUN" == "1" ]]; then
    return
  fi
  "$VR_EVAL_PYTHON" -u evaluate_variable_rate.py "${VR_EVAL_ARGS[@]}"
  echo "[variable-rate-stage-eval] outputs: ${VR_EVAL_OUTPUT_DIR}"
}
