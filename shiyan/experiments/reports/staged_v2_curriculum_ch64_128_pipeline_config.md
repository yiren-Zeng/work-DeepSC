# staged_v2 curriculum ch64-128 训练方案与配置说明

本文档对应训练入口：

```bash
scripts/train/current/run_src64_64_raq2_64_staged_v2_curriculum_ch64_128_pipeline.sh
```

该方案是一个 4 阶段串行训练 pipeline，用同一张物理 GPU 依次训练 source teacher、RAQ warmup、RAQ fine-tune、RAQ channel 四个阶段。

## 训练逻辑总览

这个方案的核心思路是：先训练一个稳定的 `64,64` SimVQ 源码本 teacher，再用它作为基础训练 RAQ 动态码本分支，最后把信道扰动加进来做信道适配。

整体流程：

```text
Stage 1: 训练 source teacher
  -> 从零训练普通 SimVQ 源分支，不启用 RAQ，不加信道扰动

Stage 2: 训练 RAQ warmup
  -> 加载 Stage 1 best checkpoint
  -> 冻结 source 主体，只训练 RAQ 生成器
  -> RAQ 目标 K 按课程动态采样
  -> 不加信道扰动

Stage 3: RAQ fine-tune
  -> 加载 Stage 2 best checkpoint
  -> 训练 RAQ + encoder + decoder
  -> 继续使用 RAQ 动态 K 课程
  -> latent distill 从 0.25 衰减到 0
  -> 不加信道扰动

Stage 4: RAQ channel
  -> 加载 Stage 3 best checkpoint
  -> 训练 RAQ + encoder + decoder
  -> 继续使用 RAQ 动态 K 课程
  -> latent distill 从 0.10 衰减到 0
  -> 开启信道课程，逐步引入 LDPC/BPSK 链路扰动
```

训练时每个 batch 的主要路径如下：

```text
输入图像
  -> semantic encoder 得到多尺度特征
  -> source SimVQ 码本先做源分支量化，得到 source latent/reference
  -> 如果 USE_RAQ=1：
       RAQ 根据 source codebook 和采样到的目标 K 生成目标码本 W_trg
       使用 W_trg 做 RAQ 分支量化
       decoder 重建 RAQ 输出图像
       loss 使用 RAQ reconstruction + RAQ VQ + latent distill
     否则：
       decoder 重建 source 输出图像
       loss 使用 source reconstruction + source VQ
```

各阶段训练/冻结关系：

| Stage | 分支 | 训练模块 | 冻结模块 | 损失主体 |
| --- | --- | --- | --- | --- |
| Stage 1 | `src` | `semantic_encoder`、`vector_quantizers`、`semantic_decoder` 等默认可训练参数 | 无特殊冻结 | source reconstruction + source VQ |
| Stage 2 | `raq_warmup` | `raqs` | `semantic_encoder`、`vector_quantizers`、`semantic_decoder` 等主干模块 | RAQ reconstruction + RAQ VQ + latent distill |
| Stage 3 | `raq_finetune` | `raqs`、`semantic_encoder`、`semantic_decoder` | `vector_quantizers` 源码本主体 | RAQ reconstruction + RAQ VQ + latent distill |
| Stage 4 | `raq_channel` | `raqs`、`semantic_encoder`、`semantic_decoder` | `vector_quantizers` 源码本主体 | 带信道扰动的 RAQ reconstruction + RAQ VQ + latent distill |

RAQ 分支中的 source path 不是主要优化目标，而是提供 source codebook 和 source latent，作为 RAQ 动态码本生成和 latent distill 的参考。

信道训练策略：

| Stage | 信道扰动 |
| --- | --- |
| Stage 1 | 关闭 |
| Stage 2 | 关闭 |
| Stage 3 | 关闭 |
| Stage 4 | 开启课程：epoch `<10` 概率 0，`10-40` 线性升到 1，`>=40` 概率 1 |

因此，这个 staged_v2 不是一开始就端到端上信道训练，而是先让 source 与 RAQ 的表示学稳，再最后做 channel adaptation。

## 运行方式

```bash
cd /workspace/yi/work/shiyan
GPU_ID=0 bash scripts/train/current/run_src64_64_raq2_64_staged_v2_curriculum_ch64_128_pipeline.sh
```

pipeline 顶层默认值：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `GPU_ID` | `0` | 物理 GPU 编号，传给各 stage 的 `CUDA_VISIBLE_DEVICES` |
| `SIMVQ_TOTAL_BATCH_SIZE` | `24` | 总 batch size |
| `SIMVQ_MICRO_BATCH_SIZE` | `24` | 单次 micro batch size |

默认阶段长度为 `200 + 100 + 100 + 100` epochs。各 stage 在独立子 shell 中执行，所以如果外部没有预先设置 `NUM_EPOCHS`，stage1 使用 200，stage2-4 使用 100；如果在运行 pipeline 前显式设置 `NUM_EPOCHS`，四个 stage 都会继承这个外部值。

## 公共模型配置

四个 stage 共享以下主体结构：

| 配置项 | 值 |
| --- | --- |
| `SIMVQ_EXPERIMENT_STAGE` | `B` |
| `SIMVQ_NUM_EMBEDDINGS_LIST` | `64,64` |
| `SIMVQ_DOWNSAMPLE_STRIDES` | `8,2` |
| `SIMVQ_UNET_DEPTH` | `2` |
| `SIMVQ_BASE_CHANNELS` | `32` |
| 推导特征维度 | `[64, 128]`，对应脚本名里的 `ch64-128` |
| 总下采样倍率 | `16x` |
| `SIMVQ_ENCODER_RES_BLOCKS` | `4` |
| `SIMVQ_DECODER_RES_BLOCKS` | `4` |
| `SIMVQ_QUANTIZER_TYPE` | `simvq` |
| `SIMVQ_QUANTIZER_AXIS_LIST` | `patch,patch` |
| `SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA` | `0.0` |
| Norm / Activation | `group` / `silu` |
| Upsample | `bilinear` |
| Bottleneck Attention | `False` |
| SwinIR Enhance | `False` |
| Swin Backbone | `False` |
| 重建损失 | `MSE * 1.0 + MS-SSIM * 0.0 + LPIPS * 0.0` |

数据路径来自 `config.py` 默认值，脚本未覆盖：

| 数据集 | 路径 |
| --- | --- |
| Train | `/workspace/yi/work/Cars196/train_data` |
| Val | `/workspace/yi/work/Cars196/val_data` |
| Test 默认 | `/workspace/yi/work/Kodak-256-transform-resize` |

源端码率估算：

| 项 | 值 |
| --- | --- |
| 源端 bpp | `0.1171875` |
| LDPC1/2 + BPSK 估算传输比例 | `0.078125` |

注意：实验名中保留了 `rate044_A_patch`，但对 `64,64` 源码本来说，按当前 `config.py` 的计算公式，估算传输比例是 `0.078125`。

## 训练调度和优化器

公共训练设置：

| 配置项 | 值 |
| --- | --- |
| Optimizer | `Adam` |
| Betas | `(0.5, 0.999)` |
| LR scheduler | `StepLR(step_size=100, gamma=0.5)` |
| 梯度裁剪 | `max_norm=1.0` |
| `SIMVQ_RESUME` | 默认 `0`，即不从 `last_checkpoint.pth` 断点续训 |
| 随机种子 | `42` |

Phase 调度来自 stage B：

| 调度项 | 值 |
| --- | --- |
| `PHASE1_END` | `0.1` |
| `PHASE2_END` | `0.4` |
| VQ 层权重初始 | `[0.25, 0.5]` |
| VQ 层权重最终 | `[0.25, 0.25]` |
| Skip dropout 初始 | `[0.1]` |
| Skip dropout 最终 | `[0.0]` |

对应到 epoch：

| Stage epochs | Phase1 | Phase2 | Phase3 |
| --- | --- | --- | --- |
| `200` | `[0, 20)` | `[20, 80)` | `[80, 200]` |
| `100` | `[0, 10)` | `[10, 40)` | `[40, 100]` |

## Stage 1: SRC teacher

脚本：

```bash
scripts/train/current/run_src64_64_raq2_64_staged_v2_curriculum_stage1_src_teacher_ch64_128.sh
```

核心配置：

| 配置项 | 值 |
| --- | --- |
| `SIMVQ_EXP_FAMILY` | `shiyan_raq_src64-64_raq2-64_staged_v2_curriculum_stage1_src_teacher_rate044_A_patch_ch64-128` |
| `SIMVQ_TRAIN_BRANCH` | `src` |
| `SIMVQ_USE_RAQ` | `0` |
| `SIMVQ_RAQ_LATENT_DISTILL_WEIGHT` | `0.00` |
| `SIMVQ_SRC_CODEBOOK_REPULSION_WEIGHT` | `0.00` |
| `SIMVQ_RAQ_RECON_GRAD_MODE` | `ste`，但本阶段 RAQ 关闭 |
| `SIMVQ_RAQ_TRAIN_ENCODER` | `0` |
| `NUM_EPOCHS` | 默认 `200` |
| `SIMVQ_LEARNING_RATE_G` | 默认 `5e-5` |
| `SIMVQ_CODEBOOK_PROJ_LR` | 默认 `2e-4` |
| 预训练 | 显式 `unset SIMVQ_PRETRAINED_CHECKPOINT` 和 `unset SIMVQ_ALLOW_PRETRAINED` |

信道课程：

| 配置项 | 值 |
| --- | --- |
| `SIMVQ_CHANNEL_PROB_START_EPOCH` | `1000000` |
| `SIMVQ_CHANNEL_PROB_END_EPOCH` | `1000001` |

含义：Stage 1 实际禁用信道扰动，训练干净 source teacher。

输出：

```text
checkpoints/shiyan_raq_src64-64_raq2-64_staged_v2_curriculum_stage1_src_teacher_rate044_A_patch_ch64-128_unet2_ds8x2_k64/
experiments/shiyan_raq_src64-64_raq2-64_staged_v2_curriculum_stage1_src_teacher_rate044_A_patch_ch64-128_unet2_ds8x2_k64_epoch_metrics.csv
experiments/shiyan_raq_src64-64_raq2-64_staged_v2_curriculum_stage1_src_teacher_rate044_A_patch_ch64-128_unet2_ds8x2_k64_codebook_metrics.csv
```

## Stage 2: RAQ warmup

脚本：

```bash
scripts/train/current/run_src64_64_raq2_64_staged_v2_curriculum_stage2_raq_warmup_ch64_128.sh
```

核心配置：

| 配置项 | 值 |
| --- | --- |
| `SIMVQ_EXP_FAMILY` | `shiyan_raq_src64-64_raq2-64_staged_v2_curriculum_stage2_raq_warmup_rate044_A_patch_ch64-128` |
| `SIMVQ_TRAIN_BRANCH` | `raq_warmup` |
| `SIMVQ_USE_RAQ` | `1` |
| `SIMVQ_RAQ_TARGET_LIST` | unset，训练时动态采样目标 K |
| `SIMVQ_RAQ_MIN_TRG` | `2` |
| `SIMVQ_RAQ_MAX_TRG` | `64` |
| `SIMVQ_RAQ_REPULSION_WEIGHT` | `0.00` |
| `SIMVQ_RAQ_LATENT_DISTILL_WEIGHT` | 默认 `1.00` |
| `SIMVQ_RAQ_RECON_GRAD_MODE` | `dual` |
| `SIMVQ_RAQ_TRAIN_ENCODER` | `0` |
| `NUM_EPOCHS` | 默认 `100` |
| `SIMVQ_LEARNING_RATE_G` | 默认 `5e-5` |
| `SIMVQ_CODEBOOK_PROJ_LR` | 默认 `2e-4` |

预训练来源：

```text
checkpoints/shiyan_raq_src64-64_raq2-64_staged_v2_curriculum_stage1_src_teacher_rate044_A_patch_ch64-128_unet2_ds8x2_k64/best_vq_deepsc.pth
```

训练参数范围：

| 模块 | 是否训练 |
| --- | --- |
| `raqs` / RAQ 动态码本生成器 | 是 |
| `semantic_encoder` / 图像编码器 | 否 |
| `vector_quantizers` / source SimVQ 源码本量化器 | 否 |
| `semantic_decoder` / 图像解码器 | 否 |
| 其他主干模块 | 否 |

信道课程仍禁用：

| 配置项 | 值 |
| --- | --- |
| `SIMVQ_CHANNEL_PROB_START_EPOCH` | `1000000` |
| `SIMVQ_CHANNEL_PROB_END_EPOCH` | `1000001` |

输出 checkpoint 目录：

```text
checkpoints/shiyan_raq_src64-64_raq2-64_staged_v2_curriculum_stage2_raq_warmup_rate044_A_patch_ch64-128_unet2_ds8x2_k64/
```

## Stage 3: RAQ fine-tune

脚本：

```bash
scripts/train/current/run_src64_64_raq2_64_staged_v2_curriculum_stage3_raq_finetune_ch64_128.sh
```

核心配置：

| 配置项 | 值 |
| --- | --- |
| `SIMVQ_EXP_FAMILY` | `shiyan_raq_src64-64_raq2-64_staged_v2_curriculum_stage3_raq_finetune_rate044_A_patch_ch64-128` |
| `SIMVQ_TRAIN_BRANCH` | `raq_finetune` |
| `SIMVQ_USE_RAQ` | `1` |
| `SIMVQ_RAQ_TARGET_LIST` | unset，训练时动态采样目标 K |
| `SIMVQ_RAQ_MIN_TRG` | `2` |
| `SIMVQ_RAQ_MAX_TRG` | `64` |
| `SIMVQ_RAQ_REPULSION_WEIGHT` | `0.00` |
| `SIMVQ_RAQ_LATENT_DISTILL_WEIGHT` | 默认 `0.25` |
| `SIMVQ_RAQ_LATENT_DISTILL_FINAL_WEIGHT` | 默认 `0.0` |
| `SIMVQ_RAQ_LATENT_DISTILL_DECAY_START_EPOCH` | 默认 `0` |
| `SIMVQ_RAQ_LATENT_DISTILL_DECAY_END_EPOCH` | 默认 `100` |
| `SIMVQ_RAQ_RECON_GRAD_MODE` | `dual` |
| `SIMVQ_RAQ_TRAIN_ENCODER` | `1` |
| `NUM_EPOCHS` | 默认 `100` |
| `SIMVQ_LEARNING_RATE_G` | 默认 `1e-5` |
| `SIMVQ_CODEBOOK_PROJ_LR` | 默认 `5e-5` |

预训练来源：

```text
checkpoints/shiyan_raq_src64-64_raq2-64_staged_v2_curriculum_stage2_raq_warmup_rate044_A_patch_ch64-128_unet2_ds8x2_k64/best_vq_deepsc.pth
```

训练参数范围：

| 模块 | 是否训练 |
| --- | --- |
| RAQ 模块 | 是 |
| Encoder | 是 |
| Decoder | 是 |
| Bottleneck attention | 是，但当前 stage B 配置下 attention 实际关闭 |
| SwinIR enhance | 是，但当前配置下 SwinIR 实际关闭 |
| Source quantizer 码本主体 | 否 |

信道课程仍禁用：

| 配置项 | 值 |
| --- | --- |
| `SIMVQ_CHANNEL_PROB_START_EPOCH` | `1000000` |
| `SIMVQ_CHANNEL_PROB_END_EPOCH` | `1000001` |

输出 checkpoint 目录：

```text
checkpoints/shiyan_raq_src64-64_raq2-64_staged_v2_curriculum_stage3_raq_finetune_rate044_A_patch_ch64-128_unet2_ds8x2_k64/
```

## Stage 4: RAQ channel

脚本：

```bash
scripts/train/current/run_src64_64_raq2_64_staged_v2_curriculum_stage4_raq_channel_ch64_128.sh
```

核心配置：

| 配置项 | 值 |
| --- | --- |
| `SIMVQ_EXP_FAMILY` | `shiyan_raq_src64-64_raq2-64_staged_v2_curriculum_stage4_raq_channel_rate044_A_patch_ch64-128` |
| `SIMVQ_TRAIN_BRANCH` | `raq_channel` |
| `SIMVQ_USE_RAQ` | `1` |
| `SIMVQ_RAQ_TARGET_LIST` | unset，训练时动态采样目标 K |
| `SIMVQ_RAQ_MIN_TRG` | `2` |
| `SIMVQ_RAQ_MAX_TRG` | `64` |
| `SIMVQ_RAQ_REPULSION_WEIGHT` | `0.00` |
| `SIMVQ_RAQ_LATENT_DISTILL_WEIGHT` | 默认 `0.10` |
| `SIMVQ_RAQ_LATENT_DISTILL_FINAL_WEIGHT` | 默认 `0.0` |
| `SIMVQ_RAQ_LATENT_DISTILL_DECAY_START_EPOCH` | 默认 `0` |
| `SIMVQ_RAQ_LATENT_DISTILL_DECAY_END_EPOCH` | 默认 `100` |
| `SIMVQ_RAQ_RECON_GRAD_MODE` | `dual` |
| `SIMVQ_RAQ_TRAIN_ENCODER` | `1` |
| `NUM_EPOCHS` | 默认 `100` |
| `SIMVQ_LEARNING_RATE_G` | 默认 `1e-5` |
| `SIMVQ_CODEBOOK_PROJ_LR` | 默认 `5e-5` |

预训练来源：

```text
checkpoints/shiyan_raq_src64-64_raq2-64_staged_v2_curriculum_stage3_raq_finetune_rate044_A_patch_ch64-128_unet2_ds8x2_k64/best_vq_deepsc.pth
```

训练参数范围与 Stage 3 一致：

| 模块 | 是否训练 |
| --- | --- |
| RAQ 模块 | 是 |
| Encoder | 是 |
| Decoder | 是 |
| Source quantizer 码本主体 | 否 |

信道课程开启：

| 区间 | 信道概率 |
| --- | --- |
| `epoch < 10` | `0` |
| `10 <= epoch < 40` | 线性升至 `1` |
| `epoch >= 40` | `1` |

输出 checkpoint 目录：

```text
checkpoints/shiyan_raq_src64-64_raq2-64_staged_v2_curriculum_stage4_raq_channel_rate044_A_patch_ch64-128_unet2_ds8x2_k64/
```

## RAQ 动态码本课程

Stage 2-4 都启用：

```bash
SIMVQ_RAQ_USE_CURRICULUM=1
SIMVQ_RAQ_CURRICULUM_EARLY_LIST="32,64"
SIMVQ_RAQ_CURRICULUM_MIDDLE_LIST="8,16,32,64"
SIMVQ_RAQ_CURRICULUM_LATE_LIST="2,4,8,16,32,64"
```

训练时每个 accumulation 起点会动态采样每层目标 K。因为 `SIMVQ_RAQ_TARGET_LIST` 被 unset，训练目标不是固定列表，而是从对应课程阶段的集合中抽样。

对应到默认 100 epoch 的 RAQ stage：

| Epoch 区间 | 采样集合 |
| --- | --- |
| `[0, 10)` | `32,64` |
| `[10, 40)` | `8,16,32,64` |
| `[40, 100]` | `2,4,8,16,32,64` |

## Checkpoint 串联关系

```text
Stage 1 SRC teacher
  -> Stage 2 RAQ warmup
  -> Stage 3 RAQ fine-tune
  -> Stage 4 RAQ channel
```

具体路径：

| Stage | 加载来源 | 保存目录 |
| --- | --- | --- |
| Stage 1 | 无预训练，从头训练 | `checkpoints/shiyan_raq_src64-64_raq2-64_staged_v2_curriculum_stage1_src_teacher_rate044_A_patch_ch64-128_unet2_ds8x2_k64/` |
| Stage 2 | Stage 1 `best_vq_deepsc.pth` | `checkpoints/shiyan_raq_src64-64_raq2-64_staged_v2_curriculum_stage2_raq_warmup_rate044_A_patch_ch64-128_unet2_ds8x2_k64/` |
| Stage 3 | Stage 2 `best_vq_deepsc.pth` | `checkpoints/shiyan_raq_src64-64_raq2-64_staged_v2_curriculum_stage3_raq_finetune_rate044_A_patch_ch64-128_unet2_ds8x2_k64/` |
| Stage 4 | Stage 3 `best_vq_deepsc.pth` | `checkpoints/shiyan_raq_src64-64_raq2-64_staged_v2_curriculum_stage4_raq_channel_rate044_A_patch_ch64-128_unet2_ds8x2_k64/` |

## 日志和指标输出

每个 stage 的终端日志写入：

```text
experiments/logs/train_${RUN_ID}.log
```

`RUN_ID` 格式：

```text
staged_v2_src64-64_stage{N}_{stage_name}_ch64-128_gpu${GPU_ID}-YYYYMMDD-HHMMSS
```

每个实验还会写入：

```text
experiments/{EXPERIMENT_NAME}_epoch_metrics.csv
experiments/{EXPERIMENT_NAME}_codebook_metrics.csv
experiments/tensorboard/{EXPERIMENT_NAME}/...
experiments/snapshots/{EXPERIMENT_NAME}/...
```

其中 `EXPERIMENT_NAME` 的规则是：

```text
${SIMVQ_EXP_FAMILY}_unet2_ds8x2_k64
```

## 可覆盖变量提醒

这些变量在脚本中使用 `${VAR:-default}`，可以在运行 pipeline 前外部覆盖：

| 变量 | 默认值 | 影响范围 |
| --- | --- | --- |
| `GPU_ID` | `0` | 所有 stage |
| `SIMVQ_TOTAL_BATCH_SIZE` | `24` | 所有 stage |
| `SIMVQ_MICRO_BATCH_SIZE` | `24` | 所有 stage |
| `NUM_EPOCHS` | stage1: `200`, stage2-4: `100` | 若外部设置，则所有 stage 都会继承同一个值 |
| `SIMVQ_RESUME` | `0` | 所有 stage |
| `SIMVQ_LEARNING_RATE_G` | stage1-2: `5e-5`, stage3-4: `1e-5` | 对设置它的 stage 生效 |
| `SIMVQ_CODEBOOK_PROJ_LR` | stage1-2: `2e-4`, stage3-4: `5e-5` | 对设置它的 stage 生效 |
| `SIMVQ_RAQ_LATENT_DISTILL_WEIGHT` | stage2: `1.00`, stage3: `0.25`, stage4: `0.10` | RAQ stage |
| `SIMVQ_RAQ_LATENT_DISTILL_FINAL_WEIGHT` | stage3-4: `0.0` | RAQ distill 衰减终值 |
| `SIMVQ_RAQ_LATENT_DISTILL_DECAY_END_EPOCH` | stage3-4: `100` | RAQ distill 衰减终点 |

## 方案一句话总结

该方案先用 `64,64` 源码本训练一个无信道扰动的 SimVQ source teacher；然后加载它训练 RAQ 生成器；再解冻 encoder/decoder 低学习率微调 RAQ 重建；最后在同样低学习率下加入信道课程，让 RAQ 分支适配真实 LDPC/BPSK 信道训练。
