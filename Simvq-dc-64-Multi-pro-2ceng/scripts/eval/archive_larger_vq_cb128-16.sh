#!/bin/bash
set -euo pipefail

export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="quality_v2_B_larger_cb128-16_VQ"
export SIMVQ_NUM_EMBEDDINGS_LIST="128,16"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="4"
export SIMVQ_DECODER_RES_BLOCKS="4"
export SIMVQ_QUANTIZER_TYPE="vq"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/_eval_common.sh" \
  "checkpoints/quality_v2_B_larger_cb128-16_VQ_unet2_ds8x2_k128-16/best_vq_deepsc.pth" "$@"
