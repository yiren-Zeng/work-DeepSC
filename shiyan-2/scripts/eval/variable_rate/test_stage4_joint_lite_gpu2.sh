#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
vr_eval_init

CHECKPOINT="${CHECKPOINT:-${VR_EVAL_PROJECT_ROOT}/checkpoints/${VR_EVAL_FAMILY}_stage4_joint_lite/best_variable_rate_raq.pth}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${VR_EVAL_PROJECT_ROOT}/checkpoints/${VR_EVAL_FAMILY}_stage1_src_teacher/best_src_teacher.pth}"
PROFILES="${PROFILES:-2048x2048;2048x16;16x2;1024x256;512x64;64x16}"

vr_eval_prepare "stage4_joint_lite" "$CHECKPOINT" "joint_lite" "$PROFILES"
vr_eval_attach_teacher "$TEACHER_CHECKPOINT"
vr_eval_execute
