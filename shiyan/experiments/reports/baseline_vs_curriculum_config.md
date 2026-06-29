# Baseline 与 Baseline + Curriculum 配置说明

本文档记录两组 `[64,64] / RAQ [2,64]` 方案的完整训练配置。两组方案的网络结构、量化模块、损失函数主体、优化器、batch size、训练轮数等保持一致；唯一核心变量是 **RAQ 目标码本 K 的训练采样策略**。

## 方案总览

| 项目 | Baseline | Baseline + Curriculum |
|---|---|---|
| 训练脚本 | `scripts/train/current/run_src64_64_raq2_64_ch64_128.sh` | `scripts/train/current/run_src64_64_raq2_64_curriculum_ch64_128_gpu0.sh` |
| 实验名 | `shiyan_raq_src64-64_raq2-64_rate044_A_patch_ch64-128` | `shiyan_raq_src64-64_raq2-64_curriculum_rate044_A_patch_ch64-128` |
| Checkpoint 目录 | `checkpoints/shiyan_raq_src64-64_raq2-64_rate044_A_patch_ch64-128_unet2_ds8x2_k64` | `checkpoints/shiyan_raq_src64-64_raq2-64_curriculum_rate044_A_patch_ch64-128_unet2_ds8x2_k64` |
| 日志文件 | `experiments/logs/train_src64-64_raq2-64_ch64-128-20260625-021929.log` | `experiments/logs/train_src64-64_raq2-64_curriculum_ch64-128_gpu0-20260625-182007.log` |
| Epoch metrics | `experiments/shiyan_raq_src64-64_raq2-64_rate044_A_patch_ch64-128_unet2_ds8x2_k64_epoch_metrics.csv` | `experiments/shiyan_raq_src64-64_raq2-64_curriculum_rate044_A_patch_ch64-128_unet2_ds8x2_k64_epoch_metrics.csv` |
| SRC 码本 | `[64,64]` | `[64,64]` |
| RAQ 训练 K 范围 | `[2,64]` | `[2,64]` |
| RAQ K 采样 | 全程从 `{2,4,8,16,32,64}` 均匀随机采样 | 课程采样：early `{32,64}`，middle `{8,16,32,64}`，late `{2,4,8,16,32,64}` |
| latent distill | 关闭，`0.00` | 关闭，`0.00` |
| SRC codebook repulsion | 关闭，`0.00` | 关闭，`0.00` |
| RAQ repulsion | 关闭，`0.00` | 关闭，`0.00` |

## 共同基础配置

### 训练基本参数

| 参数 | 值 |
|---|---|
| `NUM_EPOCHS` | `200` |
| `SIMVQ_TOTAL_BATCH_SIZE` | `24` |
| `SIMVQ_MICRO_BATCH_SIZE` | `24` |
| 梯度累积步数 | `1` |
| 随机种子 | `42` |
| 训练方式 | 从零开始训练，`SIMVQ_PRETRAINED_CHECKPOINT` unset |
| 断点续训默认 | `SIMVQ_RESUME=0` |
| 训练数据 | `TRAIN_DATASET_PATH=/workspace/yi/work/Cars196/train_data` |
| 验证数据 | `VAL_DATASET_PATH=/workspace/yi/work/Cars196/val_data` |
| 测试数据默认 | `TEST_DATASET_PATH=/workspace/yi/work/Kodak-256-transform-resize` |

### U-Net 主干结构

| 参数 | 值 |
|---|---|
| 实验阶段 | `B` |
| 输入通道 | `3` |
| 输出通道 | `3` |
| U-Net 层数 | `2` |
| 下采样步幅 | `[8,2]` |
| 总下采样倍率 | `16x` |
| `base_channels` | `32` |
| 每层特征维度 | `[64,128]` |
| Encoder residual blocks | `4` |
| Decoder residual blocks | `4` |
| Norm | `group` |
| Activation | `silu` |
| Upsample mode | `bilinear` |
| Cascade downsample | `False` |
| Bottleneck attention | `False` |
| SwinIR enhance | `False` |
| Swin backbone | `False` |

### Skip Dropout 与训练阶段

训练阶段由 `training/schedules.py` 控制：

| 阶段 | Epoch 范围 | Skip dropout | VQ 层权重 |
|---|---:|---|---|
| Phase1-拓荒 | `0 <= epoch < 20` | `[0.1]` | `[0.25,0.5]` |
| Phase2-退火 | `20 <= epoch < 80` | 从 `[0.1]` 线性退火到 `[0.0]` | 从 `[0.25,0.5]` 线性退火到 `[0.25,0.25]` |
| Phase3-微调 | `80 <= epoch < 200` | `[0.0]` | `[0.25,0.25]` |

说明：这里没有启用额外的深层 VQ weighting，例如 `[1,5]`。当前两组方案使用的是项目默认的分阶段 VQ loss 权重。

### 信道训练课程

| Epoch 范围 | `channel_prob` |
|---|---:|
| `epoch < 80` | `0.0` |
| `80 <= epoch < 120` | 从 `0.0` 线性升到 `1.0` |
| `epoch >= 120` | `1.0` |

训练阶段的信道扰动模块为 `FiniteBlocklengthChannel`。它**没有执行真实 LDPC 信道编码/译码**，而是用有限块长公式根据 `channel_coding_rate=0.5`、`block_length=256`、SNR 和调制 bit 数估计 BER，然后直接对量化索引的二进制位做随机翻转。真实的 LDPC 编码、调制、AWGN、LLR 和 LDPC 译码链路只在 `test_real.py` / `evaluation/quality.py::evaluate_ldpc_channel` 测试阶段使用。

因此更准确的表述是：

| 阶段 | 信道处理方式 |
|---|---|
| 训练 | `FiniteBlocklengthChannel` 估计 BER，并随机翻转索引 bit；无真实 LDPC 编码/译码 |
| 测试 | `ldpc_encode -> BPSK/QPSK -> AWGN -> LLR -> ldpc_decode -> reconstruct` |

## 量化模块配置

### SRC 分支

| 参数 | 值 |
|---|---|
| Quantizer type | `simvq` |
| Quantizer axis | `patch,patch` |
| SRC codebook size | `[64,64]` |
| Codebook embedding dim | `[64,128]` |
| Commitment cost | `0.25` |
| Nested channel dropout | `0.0` |

SRC 量化器使用 `VectorQuantizer`。其内部 codebook 为 `ProjectedEmbedding`：底层 `nn.Embedding` 冻结，只训练线性投影 `proj`。训练时每层 encoder feature 做最近邻量化，产生 `z_q_src` 与 SRC VQ loss。

### RAQ 分支

| 参数 | 值 |
|---|---|
| `SIMVQ_USE_RAQ` | `1` |
| RAQ min target K | `2` |
| RAQ max target K | `64` |
| RAQ target list during training | 动态采样 |
| RAQ target list during eval/test | 需要由 `SIMVQ_RAQ_TARGET_LIST` 指定；测试脚本默认 `64,64` |
| RAQ repulsion | `0.00` |

RAQ 码本由 `models/raq.py` 中的 `RAQ` 模块生成。每层 RAQ 包含：

| 参数 | 值 |
|---|---|
| Target embedding pool | `ProjectedEmbedding(n_embed_max_trg=64, embedding_dim=D)` |
| Transformer d_model | 当前层 `embedding_dim`，即第 1 层 `64`、第 2 层 `128` |
| `nhead` | `8` |
| Encoder layers | `3` |
| Decoder layers | `3` |
| Feed-forward dim | `4 * embedding_dim` |
| Dropout | `0.1` |

Transformer codebook generator 使用 SRC 投影后码本 `vector_quantizers[i].transformed_weight()` 作为 source tokens，并使用目标码本位置 id 对应的 target embedding 作为 target tokens，输出动态目标码本 `W_trg`。之后 RAQ 分支用 `W_trg` 对同一 encoder feature 做最近邻量化。

## 损失函数配置

两组方案都使用 `DeepSCLoss`，但只启用重建损失与 VQ loss。

### 总损失结构

训练中实际优化：

```text
L_total = L_recon_src + L_recon_raq + L_vq_src + L_vq_raq
```

其中：

| 项 | Baseline | Baseline + Curriculum |
|---|---:|---:|
| SRC reconstruction MSE | 启用 | 启用 |
| RAQ reconstruction MSE | 启用 | 启用 |
| SRC VQ loss | 启用 | 启用 |
| RAQ VQ loss | 启用 | 启用 |
| MS-SSIM loss | `0.0`，关闭 | `0.0`，关闭 |
| LPIPS/VGG perceptual loss | `0.0`，关闭 | `0.0`，关闭 |
| Latent distillation | `0.0`，关闭 | `0.0`，关闭 |
| SRC codebook repulsion | `0.0`，关闭 | `0.0`，关闭 |
| RAQ codebook repulsion | `0.0`，关闭 | `0.0`，关闭 |

### VQ loss 分层权重

VQ loss 使用训练阶段调度：

```text
Phase1: [0.25, 0.5]
Phase2: [0.25, 0.5] -> [0.25, 0.25]
Phase3: [0.25, 0.25]
```

## 优化器配置

| 参数 | 值 |
|---|---|
| Optimizer | `Adam` |
| 普通参数 learning rate | `5e-5` |
| Codebook projection learning rate | `2e-4` |
| Adam betas | `(0.5,0.999)` |
| Scheduler | `StepLR(step_size=100, gamma=0.5)` |
| Gradient clipping | `clip_grad_norm_=1.0` |

参数分组规则：

| 参数名包含 | LR |
|---|---:|
| `codebook.proj` | `2e-4` |
| `trg_embed.proj` | `2e-4` |
| `.qbridge.` | `2e-4` |
| 其他可训练参数 | `5e-5` |

## 两组方案的唯一区别：RAQ K 采样策略

### Baseline：全程均匀幂次采样

训练脚本不设置 `SIMVQ_RAQ_USE_CURRICULUM`，因此训练过程中每层目标码本 K 使用：

```text
K_trg ~ Uniform({2,4,8,16,32,64})
```

两层独立采样。例如日志中可以看到：

```text
RAQ target K this accumulation: [64, 2]
RAQ target K this accumulation: [16, 32]
RAQ target K this accumulation: [2, 2]
```

### Baseline + Curriculum：分阶段课程采样

训练脚本设置：

```text
SIMVQ_RAQ_USE_CURRICULUM=1
SIMVQ_RAQ_CURRICULUM_EARLY_LIST=32,64
SIMVQ_RAQ_CURRICULUM_MIDDLE_LIST=8,16,32,64
SIMVQ_RAQ_CURRICULUM_LATE_LIST=2,4,8,16,32,64
```

阶段边界复用训练阶段：

| 阶段 | Epoch 范围 | K 采样集合 |
|---|---:|---|
| early | `0 <= epoch < 20` | `{32,64}` |
| middle | `20 <= epoch < 80` | `{8,16,32,64}` |
| late | `80 <= epoch < 200` | `{2,4,8,16,32,64}` |

两层仍然独立采样。例如日志中 early 阶段可以看到：

```text
RAQ target K this accumulation: [32, 32] (sampling=early)
RAQ target K this accumulation: [64, 32] (sampling=early)
RAQ target K this accumulation: [32, 64] (sampling=early)
```

课程采样的目的：训练初期避免过早采到极低容量的 `K=2/4`，先让 RAQ Transformer 与 decoder 在较高容量目标码本上稳定，再逐步引入更低码率目标。

## BPP 与测试压缩率估算

两组训练源端 BPP 相同：

```text
Source BPP = log2(64) / 8^2 + log2(64) / 16^2
           = 6 / 64 + 6 / 256
           = 0.1171875
```

日志中显示：

```text
估算训练源端BPP: 0.117188
估算测试源端BPP: 0.117188
估算测试传输压缩率(LDPC1/2+BPSK): 0.07812500
```

注意：训练中的 RAQ K 是动态采样；测试时需要固定 `SIMVQ_RAQ_TARGET_LIST`。当前六个测试脚本默认 `SIMVQ_RAQ_TARGET_LIST=64,64`，如果要比较低码率，需要统一指定同一个测试 K，例如：

```bash
SIMVQ_RAQ_TARGET_LIST=16,2 bash scripts/eval/test_src64_64_raq2_64_baseline_ch64_128.sh
SIMVQ_RAQ_TARGET_LIST=16,2 bash scripts/eval/test_src64_64_raq2_64_curriculum_ch64_128.sh
```

## 监控指标

每 10 个 epoch 会统计 codebook utilization，并写入对应 `*_codebook_metrics.csv` 与 TensorBoard。

记录内容包括：

| 分支 | 指标 |
|---|---|
| SRC | `active_ratio`、`active_count`、`dead_count`、`perplexity`、`min_l2_dist`、`collapse_count`、`collapse_ratio` |
| RAQ | `active_ratio`、`active_count`、`dead_count`、`perplexity`、`min_l2_dist`、`collapse_count`、`collapse_ratio` |

相关文件：

```text
experiments/shiyan_raq_src64-64_raq2-64_rate044_A_patch_ch64-128_unet2_ds8x2_k64_codebook_metrics.csv
experiments/shiyan_raq_src64-64_raq2-64_curriculum_rate044_A_patch_ch64-128_unet2_ds8x2_k64_codebook_metrics.csv
```

## 当前训练状态快照

截至本文档生成时：

| 方案 | 状态 |
|---|---|
| Baseline | 已完成 `200` epochs |
| Baseline + Curriculum | 仍在训练中；已记录到约 `epoch 170`，训练进程仍在运行 |

该状态只用于记录当前实验进度，不属于方案配置本身。
