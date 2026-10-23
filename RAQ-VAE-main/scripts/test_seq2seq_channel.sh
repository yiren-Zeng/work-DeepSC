# 这是是官方原版的 RAQVAE_TWO，对应 data-driven／Seq2Seq RAQ，基于 VQ-VAE-2，bottom、top 分别下采样 4 倍、8 倍。

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
