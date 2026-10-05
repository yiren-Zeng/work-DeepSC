#!/usr/bin/env bash
# 修改下面五项即可测试双 K 模型；正式训练启动后会写入对应运行目录。
set -euo pipefail
raq_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dualk_run_dir="${DUALK_RUN_DIR:-$raq_root/runs/seq2seq_ds8_dualk_src64_20260910_113736_956705}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL="3"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
cd -- "$raq_root"

channel_types="${CHANNEL_TYPES:-awgn rician}"
rician_k_factor="${RICIAN_K_FACTOR:-10}"
bottom_k="${BOTTOM_K:-8}"
top_k="${TOP_K:-8}"
ldpc_rate="${LDPC_RATE:-5/8}"
modulation="${MODULATION:-QPSK}"
snr="${SNR:-5}"
ratio_denominator="${RATIO_DENOMINATOR:-64}"
output_root="${OUTPUT_ROOT:-}"
if [[ -n "${CHANNEL_SEEDS:-}" ]]; then
  channel_seeds="$CHANNEL_SEEDS"
elif [[ -n "${SEED:-}" ]]; then
  channel_seeds="$SEED"
else
  channel_seeds="42 43 44 45 46 47 48 49 50 51"
fi
read -r -a channel_args <<< "$channel_types"
read -r -a channel_seed_args <<< "$channel_seeds"

if [[ -z "$output_root" ]]; then
  output_root="$raq_root/runs/monte_carlo_seq2seq_ds8_dualk_$(date +%Y%m%d_%H%M%S_%N)"
fi
mkdir -p -- "$output_root"

common_args=(
  --run-dir "$dualk_run_dir"
  --bottom-k "$bottom_k"
  --top-k "$top_k"
  --ldpc-rate "$ldpc_rate"
  --modulation "$modulation"
  --snr "$snr"
  --ratio-denominator "$ratio_denominator"
  --model-seed 42
  --concise
)

for channel_type in "${channel_args[@]}"; do
  channel_output="$output_root/$channel_type"
  if [[ "$channel_type" == "rician" ]]; then
    channel_output="${channel_output}_k${rician_k_factor}"
  fi
  mkdir -p -- "$channel_output"
  trial_results=()
  for channel_seed in "${channel_seed_args[@]}"; do
    trial_output="$channel_output/seed_${channel_seed}"
    "${RAQ_PYTHON:-/home/yi/.conda/envs/work/bin/python}" -B \
      eval_seq2seq_ds8_dualk_channel.py \
      "${common_args[@]}" \
      --output "$trial_output" \
      --seed "$channel_seed" \
      "$@" \
      --channel "$channel_type" \
      --rician-k-factor "$rician_k_factor"
    trial_results+=("$trial_output/results.json")
  done
  "${RAQ_PYTHON:-/home/yi/.conda/envs/work/bin/python}" -B \
    aggregate_seq2seq_channel_trials.py \
    --trial-results "${trial_results[@]}" \
    --expected-seeds "${channel_seed_args[@]}" \
    --output "$channel_output/monte_carlo_results.json"
done
