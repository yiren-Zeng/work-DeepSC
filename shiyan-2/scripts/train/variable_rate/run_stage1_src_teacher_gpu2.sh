#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

FAMILY="${SIMVQ_EXP_FAMILY:-single_teacher_variable_rate_raq}"
export SIMVQ_EXP_FAMILY="$FAMILY"
export SIMVQ_RAQ_STAGE="src_teacher"
export SIMVQ_RAQ_STAGE1_EPOCHS="${STAGE1_EPOCHS:-${NUM_EPOCHS:-200}}"
export NUM_EPOCHS="$SIMVQ_RAQ_STAGE1_EPOCHS"
export SIMVQ_RAQ_STAGE1_SRC_LR="${SIMVQ_RAQ_STAGE1_SRC_LR:-5e-5}"
export SIMVQ_BEST_CHECKPOINT_NAME="best_src_teacher.pth"

vr_init "${FAMILY}_stage1_src_teacher"
vr_run_train "$@"
