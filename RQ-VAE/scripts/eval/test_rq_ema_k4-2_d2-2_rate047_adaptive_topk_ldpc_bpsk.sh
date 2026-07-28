#!/bin/bash
# Per-image/per-scale exact Top-K rq_ema evaluation over LDPC 1/2+BPSK+AWGN.
set -euo pipefail

eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/RQ-VAE

export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="quality_v2_B_larger_rate047"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="6"
export SIMVQ_DECODER_RES_BLOCKS="6"
export SIMVQ_QUANTIZER_TYPE="rq_ema"
export SIMVQ_QUANTIZER_AXIS_LIST="patch,patch"
export SIMVQ_CVQ_CODEWORD_SHAPES="patch,patch"
export SIMVQ_NUM_EMBEDDINGS_LIST="4,2"
export SIMVQ_RQ_DEPTH_LIST="2,2"
export SIMVQ_RQ_EMA_DECAY="0.99"
export SIMVQ_RQ_RESTART_UNUSED_CODES="1"
export SIMVQ_RQ_SHARED_CODEBOOK="1"
export SIMVQ_LAYER_LOSS_WEIGHTS_INIT="1,1"
export SIMVQ_LAYER_LOSS_WEIGHTS_FINAL="1,1"
export SIMVQ_LPIPS_WEIGHT="0"
export SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA="0.0"
export SIMVQ_RESUME="0"
unset SIMVQ_PRETRAINED_CHECKPOINT

export SIMVQ_TEST_DATASET_PATH="${SIMVQ_TEST_DATASET_PATH:-/workspace/yi/work/Kodak-256-transform-resize}"
export SIMVQ_TEST_NO_RESIZE="${SIMVQ_TEST_NO_RESIZE:-1}"
export GPU_ID="${GPU_ID:-3}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

OUTPUT_DIR="experiments/adaptive_eval/quality_v2_B_larger_rate047_rq_ema_unet2_ds8x2_k4-2_d2-2/ldpc_bpsk_explicit_mask_per_image_topk"
mkdir -p "$OUTPUT_DIR" experiments/logs
RUN_ID="rq_ema_adaptive_topk_ldpc_bpsk_gpu${GPU_ID}-$(date +%Y%m%d-%H%M%S)"
LOG_PATH="experiments/logs/eval_${RUN_ID}.log"

echo "Physical GPU: $GPU_ID"
echo "Selection: per-image/per-scale exact Top-K with stable raster tie-break"
echo "Threshold count: reported from the actual image and scale counts"
echo "Channel: Sionna 5G LDPC k=128 n=256 R=1/2 + BPSK + AWGN"
echo "Activation targets: 0%,10%,...,100%"
echo "SNRs: 0,3,6,9,12 dB"
echo "Log: $LOG_PATH"

python -u test_adaptive_topk_ldpc.py \
  --dataset "$SIMVQ_TEST_DATASET_PATH" \
  --target-active-rates 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --snrs 0 3 6 9 12 \
  --json-output "$OUTPUT_DIR/results.json" \
  --csv-output "$OUTPUT_DIR/results.csv" \
  --thresholds-csv-output "$OUTPUT_DIR/per_image_thresholds.csv" \
  "$@" 2>&1 | tee "$LOG_PATH"
