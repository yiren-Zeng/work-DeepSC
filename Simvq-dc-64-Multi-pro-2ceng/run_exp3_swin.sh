#!/bin/bash
# ============================================================
# 方案3: Swin Transformer Hyperprior / Patched-based Swin - 基于 Stage B
# 改动: 解码器输出后添加轻量级 SwinIR 质量增强子网络
# 参考: Patched-based Swin Transformer Hyperprior (Journal of Imaging, 2025)
#       在更低 bpp 下达到 27.285 dB vs CNN Hyperprior 26.231 dB (提升1.054 dB)
# 注意: SwinIR 增强网络不改变压缩率，仅做后处理质量增强
# GPU: 2
# 压缩率: 0.083 bits/value (保持码本[64,256])
# ============================================================
set -euo pipefail

eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/Simvq-dc-64-Multi-pro-2ceng
mkdir -p checkpoints experiments/logs

# ---- 实验标识 ----
export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="quality_v2_B_SwinEnhance"
export SIMVQ_NUM_EMBEDDINGS_LIST="64,256"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"

# ---- 方案3 特有: SwinIR 后处理增强 ----
export SIMVQ_USE_SWINIR_ENHANCE="1"
export SIMVQ_SWINIR_ENHANCE_BLOCKS="4"
export SIMVQ_TOTAL_BATCH_SIZE="8"
export SIMVQ_MICRO_BATCH_SIZE="8"

# ---- 预训练权重 (Stage B best checkpoint) ----
export SIMVQ_PRETRAINED_CHECKPOINT="checkpoints/quality_v2_B_backbone_unet2_ds8x2_k64-256/best_vq_deepsc.pth"

# ---- GPU ----
export GPU_ID="2"

# ---- 运行 ----
RUN_ID="exp3_SwinEnhance-$(date +%Y%m%d-%H%M%S)"
export EXPERIMENT_RUN_ID="$RUN_ID"
export EXPERIMENT_NAME="$SIMVQ_EXP_FAMILY"
export PYTHONUNBUFFERED=1

echo "========================================"
echo " 方案3: SwinIR 后处理质量增强"
echo " Experiment: $SIMVQ_EXP_FAMILY"
echo " Run ID: $RUN_ID"
echo " GPU: $GPU_ID"
echo " SwinIR Blocks: $SIMVQ_SWINIR_ENHANCE_BLOCKS"
echo "========================================"

CUDA_VISIBLE_DEVICES="$GPU_ID" python -u train.py 2>&1 | tee "experiments/logs/train_${RUN_ID}.log"

# 训练完成后自动补测 no-channel 与 SNR=0 dB，并刷新 Excel 台账。
AUTO_EVAL_GPU_ID="$GPU_ID" python -u tools/auto_evaluate_completed_experiments.py \
  --experiment "quality_v2_B_SwinEnhance_unet2_ds8x2_k64-256" \
  --gpu "$GPU_ID"
