#!/bin/bash
set -euo pipefail

export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="quality_v2_B_DynSwinEnhance"
export SIMVQ_NUM_EMBEDDINGS_LIST="64,256"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="96"
export SIMVQ_QUANTIZER_TYPE="simvq"
export SIMVQ_USE_SWINIR_ENHANCE="1"
export SIMVQ_SWINIR_ENHANCE_BLOCKS="6"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/_eval_common.sh" \
  "checkpoints/quality_v2_B_DynSwinEnhance_unet2_ds8x2_k64-256/best_vq_deepsc.pth" "$@"
