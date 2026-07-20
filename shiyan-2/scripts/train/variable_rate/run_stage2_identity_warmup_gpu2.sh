#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

FAMILY="${SIMVQ_EXP_FAMILY:-single_teacher_variable_rate_raq}"
export SIMVQ_EXP_FAMILY="$FAMILY"
export SIMVQ_RAQ_STAGE="identity_warmup"
export SIMVQ_RAQ_STAGE2_EPOCHS="${STAGE2_EPOCHS:-${NUM_EPOCHS:-20}}"
export NUM_EPOCHS="$SIMVQ_RAQ_STAGE2_EPOCHS"
export SIMVQ_RAQ_STAGE2_RAQ_LR="${SIMVQ_RAQ_STAGE2_RAQ_LR:-2e-4}"
# The all-max profile is an exact structural bypass and gives the generator no
# gradient.  Near-max profiles are therefore mandatory in this stage.
export SIMVQ_RAQ_TARGET_PROFILES="2048x2048;2048x1024;1024x2048;1024x1024"
export SIMVQ_RAQ_MIN_PROFILE="1024x1024"
export SIMVQ_RAQ_SANDWICH_NUM_RANDOM="${SIMVQ_RAQ_SANDWICH_NUM_RANDOM:-1}"
export SIMVQ_SRC_TEACHER_CHECKPOINT="${SIMVQ_SRC_TEACHER_CHECKPOINT:-/workspace/yi/work/shiyan-2/checkpoints/${FAMILY}_stage1_src_teacher/best_src_teacher.pth}"
export SIMVQ_BEST_CHECKPOINT_NAME="best_variable_rate_raq.pth"

vr_init "${FAMILY}_stage2_identity_warmup"
vr_require_local_checkpoint "$SIMVQ_SRC_TEACHER_CHECKPOINT" "Stage-1 SRC teacher checkpoint"
vr_run_train "$@"
