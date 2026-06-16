#!/bin/bash
# ============================================================
# 方案4: Dynamic Window Swin Transformer 质量增强 - 基于 Stage B
# 改动: 解码器输出后添加更重的 SwinIR 增强网络 (6 RSTB blocks)
#       同时扩大 base_channels 到 96 以增强主干网络
# 参考: Dynamic Window Swin Transformer (IEEE, 2024)
#       质量增强网络提升 2.68 dB
# Default GPU: 3. Override with GPU_ID when resuming on another device.
# 压缩率: 0.083 bits/value (保持码本[64,256])
# ============================================================
set -euo pipefail

eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/Simvq-dc-64-Multi-pro-2ceng
mkdir -p checkpoints experiments/logs

# ---- 实验标识 ----
export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="quality_v2_B_DynSwinEnhance"
export SIMVQ_NUM_EMBEDDINGS_LIST="64,256"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"

# ---- 方案4 特有: 更重的 SwinIR 增强 + 稍大主干 ----
export SIMVQ_USE_SWINIR_ENHANCE="1"
export SIMVQ_SWINIR_ENHANCE_BLOCKS="6"   # 更多 RSTB blocks
export SIMVQ_BASE_CHANNELS="96"          # 64->96 适度扩大主干
export SIMVQ_TOTAL_BATCH_SIZE="8"
export SIMVQ_MICRO_BATCH_SIZE="8"

# ---- 预训练权重 (Stage B best checkpoint) ----
export SIMVQ_PRETRAINED_CHECKPOINT="checkpoints/quality_v2_B_backbone_unet2_ds8x2_k64-256/best_vq_deepsc.pth"

# ---- GPU ----
export GPU_ID="${GPU_ID:-3}"

# ---- 运行 ----
RUN_ID="exp4_DynSwin-$(date +%Y%m%d-%H%M%S)"
export EXPERIMENT_RUN_ID="$RUN_ID"
export EXPERIMENT_NAME="$SIMVQ_EXP_FAMILY"
export PYTHONUNBUFFERED=1

echo "========================================"
echo " 方案4: Dynamic Window Swin 质量增强"
echo " Experiment: $SIMVQ_EXP_FAMILY"
echo " Run ID: $RUN_ID"
echo " GPU: $GPU_ID"
echo " SwinIR Blocks: $SIMVQ_SWINIR_ENHANCE_BLOCKS"
echo " Base Channels: $SIMVQ_BASE_CHANNELS"
echo "========================================"

CUDA_VISIBLE_DEVICES="$GPU_ID" python -u train.py 2>&1 | tee "experiments/logs/train_${RUN_ID}.log"

# 训练完成后自动补测 no-channel 与 SNR=0 dB，并刷新 Excel 台账。
AUTO_EVAL_GPU_ID="$GPU_ID" python -u tools/auto_evaluate_completed_experiments.py \
  --experiment "quality_v2_B_DynSwinEnhance_unet2_ds8x2_k64-256" \
  --gpu "$GPU_ID"
