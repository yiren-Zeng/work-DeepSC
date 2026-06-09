#!/bin/bash
# ============================================================
# 方案1: LPIPS (VGG感知损失) - 基于 Stage B
# 预期收益: +1~2 dB PSNR
# GPU: 0
# 压缩率: 0.083 bits/value (保持码本[64,256])
# ============================================================
set -euo pipefail

eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/Simvq-dc-64-Multi-pro-2ceng
mkdir -p checkpoints experiments/logs

# ---- 实验标识 ----
export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="quality_v2_B_LPIPS"
export SIMVQ_NUM_EMBEDDINGS_LIST="64,256"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"

# ---- 方案1 特有: LPIPS感知损失 ----
export SIMVQ_LPIPS_WEIGHT="0.1"

# ---- 预训练权重 (Stage B best checkpoint) ----
export SIMVQ_PRETRAINED_CHECKPOINT="checkpoints/quality_v2_B_backbone_unet2_ds8x2_k64-256/best_vq_deepsc.pth"

# ---- GPU ----
export GPU_ID="0"

# ---- 运行 ----
RUN_ID="exp1_LPIPS-$(date +%Y%m%d-%H%M%S)"
export EXPERIMENT_RUN_ID="$RUN_ID"
export EXPERIMENT_NAME="$SIMVQ_EXP_FAMILY"
export PYTHONUNBUFFERED=1

echo "========================================"
echo " 方案1: LPIPS VGG感知损失"
echo " Experiment: $SIMVQ_EXP_FAMILY"
echo " Run ID: $RUN_ID"
echo " GPU: $GPU_ID"
echo " LPIPS Weight: $SIMVQ_LPIPS_WEIGHT"
echo "========================================"

CUDA_VISIBLE_DEVICES="$GPU_ID" python -u train.py 2>&1 | tee "experiments/logs/train_${RUN_ID}.log"
