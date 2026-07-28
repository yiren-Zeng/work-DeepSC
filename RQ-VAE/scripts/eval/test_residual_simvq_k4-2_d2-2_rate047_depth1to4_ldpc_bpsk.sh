#!/bin/bash
# Zero-training Residual-SimVQ depth 1-4 sweep, including LDPC 1/2 + BPSK.
set -euo pipefail

eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/RQ-VAE

# These values describe the trained source checkpoint and remain depth [2,2].
export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="quality_v2_B_larger_rate047"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="6"
export SIMVQ_DECODER_RES_BLOCKS="6"
export SIMVQ_QUANTIZER_TYPE="residual_simvq"
export SIMVQ_QUANTIZER_AXIS_LIST="patch,patch"
export SIMVQ_CVQ_CODEWORD_SHAPES="patch,patch"
export SIMVQ_NUM_EMBEDDINGS_LIST="4,2"
export SIMVQ_RQ_DEPTH_LIST="2,2"
export SIMVQ_RQ_SHARED_CODEBOOK="1"
export SIMVQ_RQ_RESTART_UNUSED_CODES="0"
export SIMVQ_LAYER_LOSS_WEIGHTS_INIT="0.25,0.50"
export SIMVQ_LAYER_LOSS_WEIGHTS_FINAL="0.25,0.25"
export SIMVQ_LPIPS_WEIGHT="0"
export SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA="0.0"
export SIMVQ_MODEL_PARALLEL="0"
export SIMVQ_RESUME="0"
unset SIMVQ_PRETRAINED_CHECKPOINT

export SIMVQ_TEST_DATASET_PATH="${SIMVQ_TEST_DATASET_PATH:-/workspace/yi/work/Kodak-256-transform-resize}"
export SIMVQ_TEST_NO_RESIZE="${SIMVQ_TEST_NO_RESIZE:-1}"
export SIMVQ_NUM_WORKERS="${SIMVQ_NUM_WORKERS:-0}"
export GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

CHECKPOINT_PATH="checkpoints/quality_v2_B_larger_rate047_residual_simvq_unet2_ds8x2_k4-2_d2-2/best_vq_deepsc.pth"
OUTPUT_DIR="experiments/depth_extension_eval/quality_v2_B_larger_rate047_residual_simvq_unet2_ds8x2_k4-2_d2-2/zero_training_shared_codebook_d1to4_all_depths_ldpc_bpsk"
mkdir -p "$OUTPUT_DIR" experiments/logs
LOG_PATH="experiments/logs/eval_residual_simvq_depth1to4_ldpc_bpsk_gpu${GPU_ID}_$(date +%Y%m%d-%H%M%S).log"

PYTHON_CMD=(python -u)
if [ "${DEBUG_PDB:-0}" = "1" ]; then
  PYTHON_CMD=(python -m pdb)
fi

if [ "$#" -eq 0 ]; then
  "${PYTHON_CMD[@]}" test_residual_simvq_depth1to4.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --depths 1 2 3 4 \
    --channel-depths 1 2 3 4 \
    --snrs 0 3 6 9 12 \
    --modulation bpsk \
    --json-output "$OUTPUT_DIR/results.json" \
    --csv-output "$OUTPUT_DIR/results.csv" \
    2>&1 | tee "$LOG_PATH"
else
  "${PYTHON_CMD[@]}" test_residual_simvq_depth1to4.py "$@" \
    2>&1 | tee "$LOG_PATH"
fi

echo "Log: $LOG_PATH"
