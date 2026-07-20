#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

FAMILY="${SIMVQ_EXP_FAMILY:-single_teacher_variable_rate_raq}"
export SIMVQ_EXP_FAMILY="$FAMILY"
export SIMVQ_RAQ_STAGE="channel_finetune"
export SIMVQ_RAQ_STAGE5_EPOCHS="${STAGE5_EPOCHS:-${NUM_EPOCHS:-40}}"
export NUM_EPOCHS="$SIMVQ_RAQ_STAGE5_EPOCHS"
export SIMVQ_RAQ_STAGE5_RAQ_LR="${SIMVQ_RAQ_STAGE5_RAQ_LR:-2e-5}"
export SIMVQ_RAQ_STAGE5_DECODER_LR="${SIMVQ_RAQ_STAGE5_DECODER_LR:-5e-6}"
export SIMVQ_RAQ_STAGE5_ENCODER_LR="${SIMVQ_RAQ_STAGE5_ENCODER_LR:-5e-7}"
export SIMVQ_RAQ_STAGE5_TRAIN_ENCODER="${SIMVQ_RAQ_STAGE5_TRAIN_ENCODER:-0}"
export SIMVQ_RAQ_TARGET_PROFILES="all"
export SIMVQ_RAQ_MIN_PROFILE="2x2"
export SIMVQ_RAQ_SANDWICH_NUM_RANDOM="${SIMVQ_RAQ_SANDWICH_NUM_RANDOM:-1}"
export SIMVQ_CHANNEL_TYPE="${SIMVQ_CHANNEL_TYPE:-AWGN}"
export SIMVQ_SNR_RANGE_DB="${SIMVQ_SNR_RANGE_DB:-0,15}"
export SIMVQ_RAQ_CHANNEL_RAMP_EPOCHS="${SIMVQ_RAQ_CHANNEL_RAMP_EPOCHS:-10}"
export SIMVQ_SRC_TEACHER_CHECKPOINT="${SIMVQ_SRC_TEACHER_CHECKPOINT:-/workspace/yi/work/shiyan-2/checkpoints/${FAMILY}_stage1_src_teacher/best_src_teacher.pth}"
export SIMVQ_RAQ_STUDENT_CHECKPOINT="${SIMVQ_RAQ_STUDENT_CHECKPOINT:-/workspace/yi/work/shiyan-2/checkpoints/${FAMILY}_stage4_joint_lite/best_variable_rate_raq.pth}"
export SIMVQ_BEST_CHECKPOINT_NAME="best_variable_rate_raq.pth"

vr_init "${FAMILY}_stage5_channel_finetune"
vr_require_local_checkpoint "$SIMVQ_SRC_TEACHER_CHECKPOINT" "Stage-1 SRC teacher checkpoint"
vr_require_local_checkpoint "$SIMVQ_RAQ_STUDENT_CHECKPOINT" "Stage-4 student checkpoint"
vr_run_train "$@"
