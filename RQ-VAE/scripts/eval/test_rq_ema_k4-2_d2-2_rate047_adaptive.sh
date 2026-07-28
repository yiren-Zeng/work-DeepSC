#!/bin/bash
# Adaptive second-stage EMA-RQ rate/distortion scan on Kodak-256.
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
export GPU_ID="${GPU_ID:-2}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

OUTPUT_DIR="experiments/adaptive_eval/quality_v2_B_larger_rate047_rq_ema_unet2_ds8x2_k4-2_d2-2"
python -u test_adaptive.py \
  --dataset "$SIMVQ_TEST_DATASET_PATH" \
  --json-output "$OUTPUT_DIR/adaptive_scan.json" \
  --csv-output "$OUTPUT_DIR/adaptive_scan.csv" \
  --plot-output "$OUTPUT_DIR/adaptive_scan.png" \
  "$@"
