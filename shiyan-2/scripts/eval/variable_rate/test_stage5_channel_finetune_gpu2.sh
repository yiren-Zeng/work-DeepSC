#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
vr_eval_init

CHECKPOINT="${CHECKPOINT:-${VR_EVAL_PROJECT_ROOT}/checkpoints/${VR_EVAL_FAMILY}_stage5_channel_finetune/best_variable_rate_raq.pth}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${VR_EVAL_PROJECT_ROOT}/checkpoints/${VR_EVAL_FAMILY}_stage1_src_teacher/best_src_teacher.pth}"
PROFILES="${PROFILES:-2048x2048;2048x16;16x2;1024x256;512x64;64x16}"
RUN_CLEAN="${RUN_CLEAN:-1}"
RUN_CHANNEL="${RUN_CHANNEL:-1}"
CHANNEL_SNRS="${CHANNEL_SNRS:-0 3 6 9 12 15}"
CHANNEL_CODING_RATE="${CHANNEL_CODING_RATE:-0.5}"
CHANNEL_MOD_BITS="${CHANNEL_MOD_BITS:-1}"

vr_eval_require_bool "$RUN_CLEAN" "RUN_CLEAN"
vr_eval_require_bool "$RUN_CHANNEL" "RUN_CHANNEL"
[[ "$RUN_CLEAN" == "1" || "$RUN_CHANNEL" == "1" ]] \
  || vr_eval_die "At least one of RUN_CLEAN or RUN_CHANNEL must be 1"
[[ "$CHANNEL_MOD_BITS" == "1" || "$CHANNEL_MOD_BITS" == "2" || "$CHANNEL_MOD_BITS" == "4" ]] \
  || vr_eval_die "CHANNEL_MOD_BITS must be 1, 2, or 4"

if [[ "$RUN_CLEAN" == "1" ]]; then
  vr_eval_prepare \
    "stage5_channel_finetune" \
    "$CHECKPOINT" \
    "channel_finetune" \
    "$PROFILES" \
    "_clean"
  vr_eval_attach_teacher "$TEACHER_CHECKPOINT"
  vr_eval_execute
fi

if [[ "$RUN_CHANNEL" == "1" ]]; then
  read -r -a SNR_VALUES <<< "$CHANNEL_SNRS"
  [[ "${#SNR_VALUES[@]}" -gt 0 ]] || vr_eval_die "CHANNEL_SNRS must not be empty"
  for snr in "${SNR_VALUES[@]}"; do
    [[ "$snr" =~ ^-?[0-9]+([.][0-9]+)?$ ]] \
      || vr_eval_die "Invalid SNR value: ${snr}"
    snr_tag="${snr//-/m}"
    snr_tag="${snr_tag//./p}"
    vr_eval_prepare \
      "stage5_channel_finetune" \
      "$CHECKPOINT" \
      "channel_finetune" \
      "$PROFILES" \
      "_snr_${snr_tag}db"
    VR_EVAL_ARGS+=(
      --no-teacher
      --use-channel
      --snr-db "$snr"
      --channel-coding-rate "$CHANNEL_CODING_RATE"
      --mod-bits "$CHANNEL_MOD_BITS"
      --channel-prob 1.0
    )
    vr_eval_execute
  done
fi
