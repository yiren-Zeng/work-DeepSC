#!/usr/bin/env bash
# Test the source-K=16, target-K<=128 dual-K model after training completes.
set -euo pipefail
raq_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dualk_run_dir="${DUALK_RUN_DIR:-$raq_root/runs/seq2seq_ds8_dualk_src16_kmax128_20260926_224222_564841}"

if [[ ! -f "$dualk_run_dir/config.json" ]]; then
  echo "Training run not found: $dualk_run_dir" >&2
  echo "Set DUALK_RUN_DIR to a completed src16/kmax128 run directory." >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL="3"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
cd -- "$raq_root"
exec "${RAQ_PYTHON:-/home/yi/.conda/envs/work/bin/python}" -B \
  eval_seq2seq_ds8_dualk_src16_kmax128_channel.py \
  --run-dir "$dualk_run_dir" \
  --bottom-k 8 \
  --top-k 16 \
  --ldpc-rate 1/2 \
  --modulation QPSK \
  --snr 3 \
  --concise \
  "$@"
