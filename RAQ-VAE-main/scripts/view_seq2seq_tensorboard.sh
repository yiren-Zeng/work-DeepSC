#!/usr/bin/env bash
# Read live training logs; keep TensorBoard caches and temporary files in this project.
set -euo pipefail
raq_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$raq_root/.runtime/tmp"
export XDG_CACHE_HOME="$raq_root/.runtime/cache"
export MPLCONFIGDIR="$raq_root/.runtime/cache/matplotlib"
export CUDA_VISIBLE_DEVICES=""
exec /home/yi/.conda/envs/work/bin/python -B -m tensorboard.main \
  --logdir "$raq_root/runs/seq2seq_src64_20260909_173605_993462/tensorboard" \
  --host 127.0.0.1 --port 6006 "$@"
