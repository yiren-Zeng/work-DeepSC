#!/bin/bash
set -euo pipefail

export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="quality_v2_B_larger_ViTvqNoCompress"
export SIMVQ_NUM_EMBEDDINGS_LIST="64,256"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="4"
export SIMVQ_DECODER_RES_BLOCKS="4"
export SIMVQ_QUANTIZER_TYPE="vitvq_nocompress"
export SIMVQ_VITVQ_QBRIDGE_TYPE="QBridgeNoCompress-S"
export SIMVQ_VITVQ_EMB_NOGRAD="0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/_eval_common.sh" \
  "checkpoints/quality_v2_B_larger_ViTvqNoCompress_unet2_ds8x2_k64-256/best_vq_deepsc.pth" "$@"
