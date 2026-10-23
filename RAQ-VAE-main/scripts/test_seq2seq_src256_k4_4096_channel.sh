#!/usr/bin/env bash
# Same PHY defaults as test_seq2seq_channel.sh; choose K in {4,8,...,4096}.
set -euo pipefail
raq_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL="3"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-4}"
cd -- "$raq_root"
exec "${RAQ_PYTHON:-/home/yi/.conda/envs/work/bin/python}" -B eval_seq2seq_src256_k4_4096_channel.py \
  --run-dir "${RAQ_RUN_DIR:-$raq_root/runs/latest_seq2seq_src256_k4_4096}" \
  --dataset "${RAQ_TEST_DATASET:-/workspace/yi/work/Kodak}" \
  --target-k 4096 \
  --ldpc-rate 1/2 \
  --modulation 16QAM \
  --snr 12 \
  --concise \
  "$@"
