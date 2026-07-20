#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

FAMILY="${SIMVQ_EXP_FAMILY:-single_teacher_variable_rate_raq}"
export SIMVQ_EXP_FAMILY="$FAMILY"
export SIMVQ_RAQ_STAGE="variable_rate"
export SIMVQ_RAQ_STAGE3_EPOCHS="${STAGE3_EPOCHS:-${NUM_EPOCHS:-120}}"
export NUM_EPOCHS="$SIMVQ_RAQ_STAGE3_EPOCHS"
export SIMVQ_RAQ_STAGE3_RAQ_LR="${SIMVQ_RAQ_STAGE3_RAQ_LR:-1e-4}"
export SIMVQ_RAQ_TARGET_PROFILES="all"
export SIMVQ_RAQ_MIN_PROFILE="2x2"
export SIMVQ_RAQ_SANDWICH_NUM_RANDOM="${SIMVQ_RAQ_SANDWICH_NUM_RANDOM:-1}"
export SIMVQ_SRC_TEACHER_CHECKPOINT="${SIMVQ_SRC_TEACHER_CHECKPOINT:-/workspace/yi/work/shiyan-2/checkpoints/${FAMILY}_stage1_src_teacher/best_src_teacher.pth}"
export SIMVQ_RAQ_STUDENT_CHECKPOINT="${SIMVQ_RAQ_STUDENT_CHECKPOINT:-/workspace/yi/work/shiyan-2/checkpoints/${FAMILY}_stage2_identity_warmup/best_variable_rate_raq.pth}"
export SIMVQ_BEST_CHECKPOINT_NAME="best_variable_rate_raq.pth"

vr_init "${FAMILY}_stage3_variable_rate"
vr_require_local_checkpoint "$SIMVQ_SRC_TEACHER_CHECKPOINT" "Stage-1 SRC teacher checkpoint"
vr_require_local_checkpoint "$SIMVQ_RAQ_STUDENT_CHECKPOINT" "Stage-2 student checkpoint"
vr_run_train "$@"
