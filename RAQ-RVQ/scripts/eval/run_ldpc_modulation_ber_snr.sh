#!/bin/bash
# Pure physical-layer Monte Carlo BER-SNR simulation. No model, dataset,
# RAQ/RVQ index, or codebook is involved.
set -euo pipefail

eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/RAQ-RVQ

export GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::UserWarning}"

SNRS="${SNRS:-0 2 4 6 8 10 12 14 16 18 20}"
LDPC_N="${LDPC_N:-256}"
RICIAN_K_FACTOR="${RICIAN_K_FACTOR:-10}"
MIN_ERRORS="${MIN_ERRORS:-200}"
MAX_BITS="${MAX_BITS:-3000000}"
BATCH_BLOCKS="${BATCH_BLOCKS:-256}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
RESUME="${RESUME:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/eval/ldpc_modulation_ber_snr}"

read -r -a SNR_ARGS <<< "$SNRS"
ARGS=(
  --snrs "${SNR_ARGS[@]}"
  --channels awgn rician
  --ldpc-n "$LDPC_N"
  --rician-k-factor "$RICIAN_K_FACTOR"
  --min-errors "$MIN_ERRORS"
  --max-bits "$MAX_BITS"
  --batch-blocks "$BATCH_BLOCKS"
  --seed "$SEED"
  --device "$DEVICE"
  --output-dir "$OUTPUT_DIR"
)
if [[ "$RESUME" == "1" ]]; then
  ARGS+=(--resume)
fi

mkdir -p "$OUTPUT_DIR"

echo "Pure LDPC + modulation BER-SNR simulation"
echo "  profiles: 1/2-QPSK, 5/8-QPSK, 3/4-QPSK, 7/16-16QAM, 1/2-16QAM"
echo "  channels: AWGN, Rician(K=$RICIAN_K_FACTOR, perfect CSI)"
echo "  SNRs: $SNRS dB"
echo "  stop rule: at least $MIN_ERRORS errors or at most $MAX_BITS information bits"
echo "  output: $OUTPUT_DIR"

python -u scripts/eval/simulate_ldpc_modulation_ber_snr.py \
  "${ARGS[@]}" \
  2> >(grep -v -E \
    '^(WARNING: All log messages before absl::InitializeLog|I[0-9]+ .*oneDNN custom operations are on\.)' \
    >&2) \
  | tee "$OUTPUT_DIR/run.log"
