#!/usr/bin/env bash
# Single-scale LSTM-RAQ baseline aligned to the Ours-RAQ ds2/source-K256 setup.
set -euo pipefail
raq_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
raq_python="${RAQ_PYTHON:-/home/yi/.conda/envs/work/bin/python}"

run_dir="${ONE_DS2_RUN_DIR:-}"
if [[ -z "$run_dir" ]]; then
  run_dir="$("$raq_python" - "$raq_root/runs" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for candidate in sorted(
    root.glob("seq2seq_one_ds2_src256_k32_2048_original_*"), reverse=True
):
    config_path = candidate / "config.json"
    status_path = candidate / "status.json"
    if not config_path.is_file() or not status_path.is_file():
        continue
    config = json.loads(config_path.read_text())
    status = json.loads(status_path.read_text())
    checkpoint = Path(status.get("best_model_path", ""))
    if (
        config.get("variant") == "RAQVAE_ONE_DS2_SRC256_K32_2048_ORIGINAL"
        and status.get("state") == "completed"
        and checkpoint.is_file()
    ):
        print(candidate)
        break
PY
)"
fi
if [[ -z "$run_dir" || ! -f "$run_dir/config.json" || ! -f "$run_dir/status.json" ]]; then
  echo "No completed single-scale ds2 run found. Set ONE_DS2_RUN_DIR explicitly." >&2
  exit 1
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL="3"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
cd -- "$raq_root"
exec "$raq_python" -B \
  eval_seq2seq_one_ds2_src256_k32_2048_channel.py \
  --run-dir "$run_dir" \
  --target-k 256 \
  --ldpc-rate 1/2 \
  --modulation BPSK \
  --channel awgn \
  --snr 0 \
  --concise \
  "$@"
