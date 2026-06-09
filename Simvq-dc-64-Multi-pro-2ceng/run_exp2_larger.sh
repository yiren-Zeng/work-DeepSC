#!/bin/bash
# ============================================================
# 方案2: 扩大模型容量 - 基于 Stage B
# 改动: base_channels 64->128, encoder/decoder res_blocks 2->4
# 预期收益: +1~1.5 dB PSNR
# GPU: 1
# 压缩率: 0.083 bits/value (保持码本[64,256], 不改变BPP)
# ============================================================
set -euo pipefail

eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/Simvq-dc-64-Multi-pro-2ceng
mkdir -p checkpoints experiments/logs

# ---- 实验标识 ----
export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="quality_v2_B_larger"
export SIMVQ_NUM_EMBEDDINGS_LIST="64,256"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"

# ---- 方案2 特有: 扩大模型容量 ----
export SIMVQ_BASE_CHANNELS="128"       # 64 -> 128
export SIMVQ_ENCODER_RES_BLOCKS="4"    # 2 -> 4
export SIMVQ_DECODER_RES_BLOCKS="4"    # 2 -> 4

# ---- 预训练权重 (Stage B best checkpoint) ----
# 注意: 由于模型结构变化(通道数和res_blocks不同)，预训练权重仅部分匹配
# load_pretrained_weights 会自动跳过形状不匹配的参数
export SIMVQ_PRETRAINED_CHECKPOINT="checkpoints/quality_v2_B_backbone_unet2_ds8x2_k64-256/best_vq_deepsc.pth"

# ---- GPU ----
export GPU_ID="1"

# ---- 运行 ----
RUN_ID="exp2_larger-$(date +%Y%m%d-%H%M%S)"
export EXPERIMENT_RUN_ID="$RUN_ID"
export EXPERIMENT_NAME="$SIMVQ_EXP_FAMILY"
export PYTHONUNBUFFERED=1

echo "========================================"
echo " 方案2: 扩大模型容量"
echo " Experiment: $SIMVQ_EXP_FAMILY"
echo " Run ID: $RUN_ID"
echo " GPU: $GPU_ID"
echo " Base Channels: $SIMVQ_BASE_CHANNELS"
echo " Res Blocks: encoder=$SIMVQ_ENCODER_RES_BLOCKS, decoder=$SIMVQ_DECODER_RES_BLOCKS"
echo "========================================"

CUDA_VISIBLE_DEVICES="$GPU_ID" python -u train.py 2>&1 | tee "experiments/logs/train_${RUN_ID}.log"

# 训练完成后自动补测 no-channel 与 SNR=0 dB，并刷新 Excel 台账。
AUTO_EVAL_GPU_ID="$GPU_ID" python -u tools/auto_evaluate_completed_experiments.py \
  --experiment "quality_v2_B_larger_unet2_ds8x2_k64-256" \
  --gpu "$GPU_ID"
