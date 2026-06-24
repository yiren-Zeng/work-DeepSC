# Variant: RAQ source K [65536,64], no codebook repulsion

这个变体基于当前正在训练的：

```text
shiyan_raq_quality_v2_B_larger_rate044_A_patch_cb16-2_ch512-1024
```

新实验名为：

```text
shiyan_raq_src65536-64_norepulse_rate044_A_patch_ch512-1024_unet2_ds8x2_k65536-64
```

## 改动点

| 项目 | 原可变速率模型 | 新模型 |
| --- | --- | --- |
| 主干 | Stage B, 2-layer U-Net | 不变 |
| base channels | 256 | 不变 |
| 特征维度 | [512,1024] | 不变 |
| 下采样 | [8,2] | 不变 |
| 量化方式 | patch-wise SimVQ + RAQ | 不变 |
| 源码本 K | [16,2] | [65536,64] |
| RAQ 测试/传输目标 K | [16,2] | [16,2] |
| RAQ 训练目标 K 范围 | [2,16] | [2,16] |
| RAQ codebook repulsion | 0.05 | 0.0 |
| 预训练 | 禁用 | 禁用 |

## RAQ 源码本输入

RAQ 的 Transformer 直接读取完整源码本权重 `W_src` 来生成目标码本 `W_trg`。源码本输入不再提供
截断环境变量；脚本无需传入源码本输入上限。

## 训练命令

```bash
cd /workspace/yi/work/shiyan
bash scripts/train/current/run_exp15_raq_src65536_64_norepulse_ch512_1024.sh
```

默认设置：

```bash
GPU_ID=3
SIMVQ_TOTAL_BATCH_SIZE=24
SIMVQ_MICRO_BATCH_SIZE=2
SIMVQ_RAQ_REPULSION_WEIGHT=0.0
SIMVQ_RESUME=0
```

如果显存仍然不够，可以降低 micro batch：

```bash
GPU_ID=3 SIMVQ_MICRO_BATCH_SIZE=1 bash scripts/train/current/run_exp15_raq_src65536_64_norepulse_ch512_1024.sh
```

## 输出位置

```text
checkpoints/shiyan_raq_src65536-64_norepulse_rate044_A_patch_ch512-1024_unet2_ds8x2_k65536-64/
experiments/shiyan_raq_src65536-64_norepulse_rate044_A_patch_ch512-1024_unet2_ds8x2_k65536-64_epoch_metrics.csv
experiments/shiyan_raq_src65536-64_norepulse_rate044_A_patch_ch512-1024_unet2_ds8x2_k65536-64_codebook_metrics.csv
experiments/logs/train_exp15_raq_src65536-64_norepulse_ch512-1024-*.log
```

## 测试时如何调节码本大小

测试时调的是 RAQ 目标码本大小，不是源码本大小。源码本大小由 checkpoint 固定为 `[65536,64]`。

默认测试目标 K：

```bash
SIMVQ_RAQ_TARGET_LIST="16,2"
```

例如测试更小的目标码本：

```bash
cd /workspace/yi/work/shiyan
SIMVQ_EXP_FAMILY="shiyan_raq_src65536-64_norepulse_rate044_A_patch_ch512-1024" \
SIMVQ_NUM_EMBEDDINGS_LIST="65536,64" \
SIMVQ_RAQ_TARGET_LIST="8,2" \
python -u test_real.py \
  --checkpoint checkpoints/shiyan_raq_src65536-64_norepulse_rate044_A_patch_ch512-1024_unet2_ds8x2_k65536-64/best_vq_deepsc.pth \
  --snrs 0 3 6 9 12 \
  --modulation bpsk
```

再比如：

```bash
SIMVQ_RAQ_TARGET_LIST="4,2"
SIMVQ_RAQ_TARGET_LIST="16,4"
```

注意：当前训练范围是 `SIMVQ_RAQ_MIN_TRG=2`、`SIMVQ_RAQ_MAX_TRG=16`，所以测试目标 K 必须在
`[2,16]` 范围内。如果想测试大于 16 的目标码本，需要新训练时把 `SIMVQ_RAQ_MAX_TRG` 同步调大。

目标码本越大，传输比特数越高；目标码本越小，码率越低但重建质量通常会下降。
