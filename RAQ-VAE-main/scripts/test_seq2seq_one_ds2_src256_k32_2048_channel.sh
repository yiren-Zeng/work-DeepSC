#!/usr/bin/env bash
# Sweep the strict-original single-scale x2 RAQVAE_ONE target codebooks.
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

targets="${TARGETS:-32}"
no_channel="${NO_CHANNEL:-0}"
channel="${CHANNEL:-awgn}"
rician_k_factor="${RICIAN_K_FACTOR:-1}"
ldpc_rate="${LDPC_RATE:-1/2}"
modulation="${MODULATION:-BPSK}"
snr="${SNR:-10}"
seed="${SEED:-42}"
ratio_denominator="${RATIO_DENOMINATOR:-48}"
dataset="${DATASET:-/workspace/yi/work/Kodak}"
output_root="${OUTPUT_ROOT:-$raq_root/generated_images}"

if [[ "$no_channel" != "0" && "$no_channel" != "1" ]]; then
  echo "NO_CHANNEL must be 0 or 1, got: $no_channel" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL="3"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
cd -- "$raq_root"

run_stamp="$("$raq_python" -B - <<'PY'
import datetime
import uuid
from zoneinfo import ZoneInfo
print(datetime.datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:8])
PY
)"
image_run="$output_root/$run_stamp"

for target in $targets; do
  case " 32 64 128 256 512 1024 2048 " in
    *" $target "*) ;;
    *)
      echo "Unsupported target K=$target; expected one of: 32 64 128 256 512 1024 2048" >&2
      exit 2
      ;;
  esac

  args=(
    --run-dir "$run_dir"
    --target-k "$target"
    --ldpc-rate "$ldpc_rate"
    --modulation "$modulation"
    --channel "$channel"
    --rician-k-factor "$rician_k_factor"
    --snr "$snr"
    --seed "$seed"
    --model-seed 42
    --ratio-denominator "$ratio_denominator"
    --dataset "$dataset"
    --images-only
    --concise
  )
  if [[ "$no_channel" == "1" ]]; then
    args+=(--no-channel)
    condition="no_channel"
  else
    condition="${channel}_snr${snr}"
  fi
  args+=(--output "$image_run/k${target}/$condition")

  "$raq_python" -B \
    eval_seq2seq_one_ds2_src256_k32_2048_channel.py \
    "${args[@]}" \
    "$@"
done
