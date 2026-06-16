#!/bin/bash
set -euo pipefail

export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="quality_v2_B_larger_cb4096-65536"
export SIMVQ_NUM_EMBEDDINGS_LIST="4096,65536"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="4"
export SIMVQ_DECODER_RES_BLOCKS="4"
export SIMVQ_QUANTIZER_TYPE="simvq"


# 下面一句代表获取当前这个脚本所在的目录，如/workspace/yi/work/Simvq-dc-64-Multi-pro-2ceng/scripts/eval
# "$@" 代表的是将参数传入进.sh或者.py里面
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/_eval_common.sh" \
  "checkpoints/quality_v2_B_larger_cb4096-65536_unet2_ds8x2_k4096-65536/best_vq_deepsc.pth" "$@"
