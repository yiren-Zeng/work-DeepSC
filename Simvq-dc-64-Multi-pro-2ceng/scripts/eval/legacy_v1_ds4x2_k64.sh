#!/bin/bash
set -euo pipefail

export SIMVQ_EXPERIMENT_STAGE="A"
export SIMVQ_EXP_FAMILY="quality_v1"
export SIMVQ_NUM_EMBEDDINGS_LIST="64,64"
export SIMVQ_DOWNSAMPLE_STRIDES="4,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="64"
export SIMVQ_QUANTIZER_TYPE="simvq"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/_eval_common.sh" \
  "checkpoints/quality_v1_unet2_ds4x2_k64/best_vq_deepsc.pth" "$@"
