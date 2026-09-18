#!/usr/bin/env bash
# 修改下面四项即可测试新 1/8 + 1/16 模型；默认读取新训练当前最佳权重。
set -euo pipefail
raq_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

ds8_run_dir="$raq_root/runs/seq2seq_ds8_src64_20260910_005420_629332"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL="3"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-3}"
cd -- "$raq_root"
exec "${RAQ_PYTHON:-/home/yi/.conda/envs/work/bin/python}" -B eval_seq2seq_ds8_channel.py \
  --run-dir "$ds8_run_dir" \
  --target-k 8 \
  --ldpc-rate 1/2 \
  --modulation QPSK \
  --snr 3 \
  --concise \
  "$@"
