#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/workspace/yi/work/shiyan-2"
cd "$PROJECT_ROOT"

die() {
  echo "[variable-rate-eval] ERROR: $*" >&2
  exit 1
}

export GPU_ID="${GPU_ID:-2}"
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || die "GPU_ID must be one physical GPU index"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export SIMVQ_DEVICE="cuda:0"
export SIMVQ_TEST_NO_RESIZE=0

PYTHON_BIN="${PYTHON_BIN:-/home/yi/.conda/envs/work/bin/python}"
[[ -x "$PYTHON_BIN" ]] || die "Python executable is not available: ${PYTHON_BIN}"
[[ -f "$PROJECT_ROOT/evaluate_variable_rate.py" ]] || die "evaluate_variable_rate.py is missing"

FAMILY="${SIMVQ_EXP_FAMILY:-single_teacher_variable_rate_raq}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/checkpoints/${FAMILY}_stage5_channel_finetune/best_variable_rate_raq.pth}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/${FAMILY}_stage1_src_teacher/best_src_teacher.pth}"
DATASET="${DATASET:-/workspace/yi/work/Kodak-256-transform-resize}"
FIXED_TEST_RESIZE="256x256"

[[ -f "$CHECKPOINT" ]] || die "student checkpoint is missing: ${CHECKPOINT}"
[[ -f "$TEACHER_CHECKPOINT" ]] || die "SRC teacher checkpoint is missing: ${TEACHER_CHECKPOINT}"
[[ -d "$DATASET" ]] || die "test dataset is missing: ${DATASET}"
for local_checkpoint in "$CHECKPOINT" "$TEACHER_CHECKPOINT"; do
  resolved="$(realpath -e "$local_checkpoint")"
  case "$resolved" in
    "${PROJECT_ROOT}"/checkpoints/*) ;;
    *) die "checkpoints must be under ${PROJECT_ROOT}/checkpoints: ${resolved}" ;;
  esac
done

ALL_PROFILES="${ALL_PROFILES:-0}"
[[ "$ALL_PROFILES" == "0" || "$ALL_PROFILES" == "1" ]] || die "ALL_PROFILES must be 0 or 1"
FIXED_PROFILES="2048x2048;32x16;128x16;8x16;2x16;64x16"
RUN_NAME="${EVAL_RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
if [[ "$RUN_NAME" == *"/"* || "$RUN_NAME" == *".."* ]]; then
  die "EVAL_RUN_NAME must be a plain directory name"
fi
OUTPUT_DIR="${PROJECT_ROOT}/experiments/eval/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR/per_profile"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --teacher-checkpoint "$TEACHER_CHECKPOINT"
  --dataset "$DATASET"
  --test-resize "$FIXED_TEST_RESIZE"
  --device cuda:0
  --batch-size "${BATCH_SIZE:-1}"
  --num-workers "${NUM_WORKERS:-2}"
  --profiles "$FIXED_PROFILES"
  --csv "$OUTPUT_DIR/profiles.csv"
  --per-profile-csv-dir "$OUTPUT_DIR/per_profile"
  --json "$OUTPUT_DIR/results.json"
  --max-teacher-drop-db "${MAX_TEACHER_DROP_DB:-0.30}"
)

if [[ "$ALL_PROFILES" == "1" ]]; then
  ARGS+=(--all-profiles)
fi
if [[ -n "${MAX_BATCHES:-}" ]]; then
  ARGS+=(--max-batches "$MAX_BATCHES")
fi
if [[ -n "${SRC_REFERENCE_PSNR:-}" ]]; then
  ARGS+=(--src-reference-psnr "$SRC_REFERENCE_PSNR")
fi
if [[ -n "${PROFILE_WEIGHTS:-}" ]]; then
  ARGS+=(--profile-weights "$PROFILE_WEIGHTS")
fi
echo "Evaluating on physical GPU ${GPU_ID}; ALL_PROFILES=${ALL_PROFILES}; TEST_RESIZE=${FIXED_TEST_RESIZE}."
"$PYTHON_BIN" -u evaluate_variable_rate.py "${ARGS[@]}"
echo "Evaluation outputs: ${OUTPUT_DIR}"
