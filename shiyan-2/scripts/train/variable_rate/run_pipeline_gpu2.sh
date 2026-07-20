#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/workspace/yi/work/shiyan-2"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/train/variable_rate"
cd "$PROJECT_ROOT"

GPU_ID="${GPU_ID:-2}"
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "ERROR: GPU_ID must be one physical GPU index" >&2; exit 1; }
export GPU_ID

FAMILY="${SIMVQ_EXP_FAMILY:-single_teacher_variable_rate_raq}"
if [[ "$FAMILY" == *"/"* || "$FAMILY" == *".."* ]]; then
  echo "ERROR: SIMVQ_EXP_FAMILY must be a plain name" >&2
  exit 1
fi
export SIMVQ_EXP_FAMILY="$FAMILY"

STAGE1_NAME="${FAMILY}_stage1_src_teacher"
STAGE2_NAME="${FAMILY}_stage2_identity_warmup"
STAGE3_NAME="${FAMILY}_stage3_variable_rate"
STAGE4_NAME="${FAMILY}_stage4_joint_lite"
STAGE5_NAME="${FAMILY}_stage5_channel_finetune"

TEACHER="${PROJECT_ROOT}/checkpoints/${STAGE1_NAME}/best_src_teacher.pth"
STAGE2_STUDENT="${PROJECT_ROOT}/checkpoints/${STAGE2_NAME}/best_variable_rate_raq.pth"
STAGE3_STUDENT="${PROJECT_ROOT}/checkpoints/${STAGE3_NAME}/best_variable_rate_raq.pth"
STAGE4_STUDENT="${PROJECT_ROOT}/checkpoints/${STAGE4_NAME}/best_variable_rate_raq.pth"
STAGE5_STUDENT="${PROJECT_ROOT}/checkpoints/${STAGE5_NAME}/best_variable_rate_raq.pth"

require_checkpoint() {
  local path="$1"
  local stage="$2"
  [[ -f "$path" ]] || {
    echo "ERROR: ${stage} did not produce its required checkpoint: ${path}" >&2
    exit 1
  }
}

echo "Starting the five-stage single-teacher variable-rate RAQ pipeline on physical GPU ${GPU_ID}."

SIMVQ_EXPERIMENT_NAME="$STAGE1_NAME" \
  bash "${SCRIPT_DIR}/run_stage1_src_teacher_gpu2.sh"
require_checkpoint "$TEACHER" "Stage 1"

SIMVQ_EXPERIMENT_NAME="$STAGE2_NAME" \
SIMVQ_SRC_TEACHER_CHECKPOINT="$TEACHER" \
  bash "${SCRIPT_DIR}/run_stage2_identity_warmup_gpu2.sh"
require_checkpoint "$STAGE2_STUDENT" "Stage 2"

SIMVQ_EXPERIMENT_NAME="$STAGE3_NAME" \
SIMVQ_SRC_TEACHER_CHECKPOINT="$TEACHER" \
SIMVQ_RAQ_STUDENT_CHECKPOINT="$STAGE2_STUDENT" \
  bash "${SCRIPT_DIR}/run_stage3_variable_rate_gpu2.sh"
require_checkpoint "$STAGE3_STUDENT" "Stage 3"

SIMVQ_EXPERIMENT_NAME="$STAGE4_NAME" \
SIMVQ_SRC_TEACHER_CHECKPOINT="$TEACHER" \
SIMVQ_RAQ_STUDENT_CHECKPOINT="$STAGE3_STUDENT" \
  bash "${SCRIPT_DIR}/run_stage4_joint_lite_gpu2.sh"
require_checkpoint "$STAGE4_STUDENT" "Stage 4"

SIMVQ_EXPERIMENT_NAME="$STAGE5_NAME" \
SIMVQ_SRC_TEACHER_CHECKPOINT="$TEACHER" \
SIMVQ_RAQ_STUDENT_CHECKPOINT="$STAGE4_STUDENT" \
  bash "${SCRIPT_DIR}/run_stage5_channel_finetune_gpu2.sh"
require_checkpoint "$STAGE5_STUDENT" "Stage 5"

echo "Pipeline complete. Final checkpoint: ${STAGE5_STUDENT}"
