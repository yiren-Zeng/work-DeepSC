#!/usr/bin/env bash
# 直接修改下面的 --target-k、--ldpc-rate、--modulation、--snr，然后运行本脚本。
# 默认使用当前记录的最佳权重、物理 GPU1；结果自动保存到本项目 runs/ 下。
set -euo pipefail
raq_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL="3"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-1}"
cd -- "$raq_root"
exec "${RAQ_PYTHON:-/home/yi/.conda/envs/work/bin/python}" -B eval_seq2seq_channel.py \
  --run-dir "$raq_root/runs/seq2seq_src64_20260909_173605_993462" \
  --target-k 2 \
  --ldpc-rate 3/4 \
  --modulation QPSK \
  --snr 6 \
  --concise \
  "$@"
