#!/bin/bash
# Exact-1/32 four-K Bandit search with one combined payload stream.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CHANNEL_PROFILE="${CHANNEL_PROFILE:-ldpc12_16qam}"
export STREAM_PACKING="${STREAM_PACKING:-combined}"
export TARGET_RATIO="${TARGET_RATIO:-1/48}"
export OUTPUT_DIR="${OUTPUT_DIR:-experiments/eval/independent_raq_rvq_rate1over32_bandit_psnr_ldpc34_qpsk_combined}"
export SNRS="${SNRS:-10}"

exec bash "$SCRIPT_DIR/test_independent_raq_rvq_src64_64_rate3over32_bandit_psnr_ldpc.sh" "$@"
