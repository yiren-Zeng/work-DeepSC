#!/usr/bin/env bash
# 修改下面参数即可测试双 K 模型；SNR 支持空格分隔的多个值。
# USE_CSCR=0 保持原方案；USE_CSCR=1 对 bottom/top 启用余弦重排 + Gray 标签。
# 仅终端输出：每组配置一次、每个种子的 PSNR/MS-SSIM，最后输出均值；不保存测试文件。
set -euo pipefail
raq_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dualk_run_dir="${DUALK_RUN_DIR:-$raq_root/runs/seq2seq_ds8_dualk_src64_20260910_113736_956705}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL="3"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
cd -- "$raq_root"

channel_types="${CHANNEL_TYPES:-awgn rician}"
rician_k_factor="${RICIAN_K_FACTOR:-1}"
bottom_k="${BOTTOM_K:-4}"
top_k="${TOP_K:-16}"
ldpc_rate="${LDPC_RATE:-1/2}"
modulation="${MODULATION:-QPSK}"
use_cscr="${USE_CSCR:-1}"
snr_values="${SNR:-10}"
ratio_denominator="${RATIO_DENOMINATOR:-64}"
if [[ -n "${CHANNEL_SEEDS:-}" ]]; then
  channel_seeds="$CHANNEL_SEEDS"
elif [[ -n "${SEED:-}" ]]; then
  channel_seeds="$SEED"
else
  channel_seeds="42 43 44 45 46 47 48 49 50 51"
fi
read -r -a channel_args <<< "$channel_types"
read -r -a channel_seed_args <<< "$channel_seeds"
read -r -a snr_args <<< "$snr_values"

case "$use_cscr" in
  0|1) ;;
  *) printf 'USE_CSCR 必须为 0 或 1，当前值：%s\n' "$use_cscr" >&2; exit 2 ;;
esac

common_args=(
  --run-dir "$dualk_run_dir"
  --bottom-k "$bottom_k"
  --top-k "$top_k"
  --ldpc-rate "$ldpc_rate"
  --modulation "$modulation"
  --ratio-denominator "$ratio_denominator"
  --model-seed 42
  --concise
)
if [[ "$use_cscr" == "1" ]]; then
  common_args+=(--cscr)
fi

for channel_type in "${channel_args[@]}"; do
  for snr in "${snr_args[@]}"; do
    "${RAQ_PYTHON:-/home/yi/.conda/envs/work/bin/python}" -B \
      eval_seq2seq_ds8_dualk_console.py \
      "${common_args[@]}" \
      --snr "$snr" \
      "$@" \
      --channel "$channel_type" \
      --rician-k-factor "$rician_k_factor" \
      --channel-seeds "${channel_seed_args[@]}"
  done
done
