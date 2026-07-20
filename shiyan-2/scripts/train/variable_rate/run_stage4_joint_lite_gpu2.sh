#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

FAMILY="${SIMVQ_EXP_FAMILY:-single_teacher_variable_rate_raq}"
export SIMVQ_EXP_FAMILY="$FAMILY"
export SIMVQ_RAQ_STAGE="joint_lite"
export SIMVQ_RAQ_STAGE4_EPOCHS="${STAGE4_EPOCHS:-${NUM_EPOCHS:-40}}"
export NUM_EPOCHS="$SIMVQ_RAQ_STAGE4_EPOCHS"
export SIMVQ_RAQ_STAGE4_RAQ_LR="${SIMVQ_RAQ_STAGE4_RAQ_LR:-5e-5}"
export SIMVQ_RAQ_STAGE4_DECODER_LR="${SIMVQ_RAQ_STAGE4_DECODER_LR:-1e-5}"
export SIMVQ_RAQ_STAGE4_ENCODER_LR="${SIMVQ_RAQ_STAGE4_ENCODER_LR:-1e-6}"
export SIMVQ_RAQ_STAGE4_TRAIN_ENCODER="${SIMVQ_RAQ_STAGE4_TRAIN_ENCODER:-0}"
export SIMVQ_RAQ_TARGET_PROFILES="all"
export SIMVQ_RAQ_MIN_PROFILE="2x2"
export SIMVQ_RAQ_SANDWICH_NUM_RANDOM="${SIMVQ_RAQ_SANDWICH_NUM_RANDOM:-1}"
export SIMVQ_SRC_TEACHER_CHECKPOINT="${SIMVQ_SRC_TEACHER_CHECKPOINT:-/workspace/yi/work/shiyan-2/checkpoints/${FAMILY}_stage1_src_teacher/best_src_teacher.pth}"
export SIMVQ_RAQ_STUDENT_CHECKPOINT="${SIMVQ_RAQ_STUDENT_CHECKPOINT:-/workspace/yi/work/shiyan-2/checkpoints/${FAMILY}_stage3_variable_rate/best_variable_rate_raq.pth}"
export SIMVQ_BEST_CHECKPOINT_NAME="best_variable_rate_raq.pth"

vr_init "${FAMILY}_stage4_joint_lite"
vr_require_local_checkpoint "$SIMVQ_SRC_TEACHER_CHECKPOINT" "Stage-1 SRC teacher checkpoint"
vr_require_local_checkpoint "$SIMVQ_RAQ_STUDENT_CHECKPOINT" "Stage-3 student checkpoint"
vr_run_train "$@"
