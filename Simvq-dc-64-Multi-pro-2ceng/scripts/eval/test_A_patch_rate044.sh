#!/bin/bash
set -euo pipefail

cd /workspace/yi/work/Simvq-dc-64-Multi-pro-2ceng

export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="quality_v2_B_larger_rate044_A_patch_cb16-2"
export SIMVQ_NUM_EMBEDDINGS_LIST="16,2"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="4"
export SIMVQ_DECODER_RES_BLOCKS="4"
export SIMVQ_QUANTIZER_TYPE="simvq"
export SIMVQ_QUANTIZER_AXIS_LIST="patch,patch"
export SIMVQ_CVQ_CODEWORD_SHAPES="patch,patch"
export SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA="0.0"

if [ "$#" -eq 0 ]; then
  python -u test_real.py \
    --checkpoint checkpoints/quality_v2_B_larger_rate044_A_patch_cb16-2_unet2_ds8x2_k16-2/best_vq_deepsc.pth \
    --snrs 0 \
    --modulation bpsk
else
  python -u test_real.py "$@"
fi
