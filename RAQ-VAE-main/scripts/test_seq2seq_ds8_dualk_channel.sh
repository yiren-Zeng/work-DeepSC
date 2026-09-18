#!/usr/bin/env bash
# 修改下面五项即可测试双 K 模型；正式训练启动后会写入对应运行目录。
set -euo pipefail
raq_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dualk_run_dir="$raq_root/runs/seq2seq_ds8_dualk_src64_20260910_113736_956705"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL="3"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-3}"
cd -- "$raq_root"
exec "${RAQ_PYTHON:-/home/yi/.conda/envs/work/bin/python}" -B eval_seq2seq_ds8_dualk_channel.py \
  --run-dir "$dualk_run_dir" \
  --bottom-k 32 \
  --top-k 16 \
  --ldpc-rate 1/2 \
  --modulation 16QAM \
  --snr 10 \
  --concise \
  "$@"
