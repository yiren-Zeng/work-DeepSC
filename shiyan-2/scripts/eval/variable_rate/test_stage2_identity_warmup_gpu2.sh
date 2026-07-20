#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
vr_eval_init

CHECKPOINT="${CHECKPOINT:-${VR_EVAL_PROJECT_ROOT}/checkpoints/${VR_EVAL_FAMILY}_stage2_identity_warmup/best_variable_rate_raq.pth}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${VR_EVAL_PROJECT_ROOT}/checkpoints/${VR_EVAL_FAMILY}_stage1_src_teacher/best_src_teacher.pth}"
PROFILES="${PROFILES:-2048x2048;2048x1024;1024x2048;1024x1024}"

vr_eval_prepare "stage2_identity_warmup" "$CHECKPOINT" "identity_warmup" "$PROFILES"
vr_eval_attach_teacher "$TEACHER_CHECKPOINT"
vr_eval_execute
