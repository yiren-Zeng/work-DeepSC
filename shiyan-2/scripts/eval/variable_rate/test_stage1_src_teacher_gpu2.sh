#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
vr_eval_init

CHECKPOINT="${CHECKPOINT:-${VR_EVAL_PROJECT_ROOT}/checkpoints/${VR_EVAL_FAMILY}_stage1_src_teacher/best_src_teacher.pth}"
PROFILES="${PROFILES:-2048x2048}"

vr_eval_prepare "stage1_src_teacher" "$CHECKPOINT" "src_teacher" "$PROFILES"
VR_EVAL_ARGS+=(--src-teacher-only --no-teacher)
vr_eval_execute
