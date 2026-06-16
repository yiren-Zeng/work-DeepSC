#!/bin/bash
set -euo pipefail

export SIMVQ_EXPERIMENT_STAGE="C"
export SIMVQ_EXP_FAMILY="quality_v2_C_full"
export SIMVQ_NUM_EMBEDDINGS_LIST="64,256"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="64"
export SIMVQ_QUANTIZER_TYPE="simvq"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/_eval_common.sh" \
  "checkpoints/quality_v2_C_full_unet2_ds8x2_k64-256/best_vq_deepsc.pth" "$@"
