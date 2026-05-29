# 实验总览与测试说明

日期：2026-05-29

## 1. 当前项目里有哪些实验

### `observed_001_baseline`

旧 baseline 归档。

```text
checkpoint: checkpoints/observed_001_baseline/
BPP: 0.4688
结构: 旧二层 SimVQ
用途: 作为最早观察到的 baseline，不再继续训练
```

### `quality_v1_unet2_ds4x2_k64`

上一轮二层高码率实验。

```text
checkpoint: checkpoints/quality_v1_unet2_ds4x2_k64/
UNET_DEPTH = 2
DOWNSAMPLE_STRIDES = [4, 2]
NUM_EMBEDDINGS_LIST = [64, 64]
BPP = 0.46875
用途: 对比高码率二层模型
```

### `quality_v2_A_curriculum_unet2_ds8x2_k16-32`

A 消融实验：只验证课程学习和低码率配置。

```text
checkpoint: checkpoints/quality_v2_A_curriculum_unet2_ds8x2_k16-32/
UNET_DEPTH = 2
DOWNSAMPLE_STRIDES = [8, 2]
NUM_EMBEDDINGS_LIST = [16, 32]
BPP = 0.08203125
归一化: BatchNorm
激活: PReLU
下采样: stride=8 一步卷积下采样
残差块: encoder=1, decoder=1
上采样: nearest
Bottleneck Attention: 关闭
损失: 纯 MSE
课程学习: 开启
退火阶段: PHASE1_END=0.1, PHASE2_END=0.4
```

当前 no-channel 最好筛选结果：

```text
PSNR = 24.6941
MS-SSIM = 0.8940
```

### `quality_v2_B_backbone_unet2_ds8x2_k16-32`

B 消融实验：A + 主干网络升级。

```text
checkpoint: checkpoints/quality_v2_B_backbone_unet2_ds8x2_k16-32/
UNET_DEPTH = 2
DOWNSAMPLE_STRIDES = [8, 2]
NUM_EMBEDDINGS_LIST = [16, 32]
BPP = 0.08203125
归一化: GroupNorm
激活: SiLU
残差块: encoder=2, decoder=2
下采样: stride=8 一步卷积下采样
上采样: bilinear
Bottleneck Attention: 关闭
损失: 纯 MSE
课程学习: 开启
退火阶段: PHASE1_END=0.1, PHASE2_END=0.4
```

当前 no-channel 最好筛选结果：

```text
PSNR = 25.2496
MS-SSIM = 0.9064
```

### `quality_v2_C_full_unet2_ds8x2_k16-32`

严格命名的 C 消融实验，正在进行。

```text
checkpoint: checkpoints/quality_v2_C_full_unet2_ds8x2_k16-32/
UNET_DEPTH = 2
DOWNSAMPLE_STRIDES = [8, 2]
NUM_EMBEDDINGS_LIST = [16, 32]
BPP = 0.08203125
归一化: GroupNorm
激活: SiLU
残差块: encoder=2, decoder=2
下采样: stride=8 一步卷积下采样
上采样: bilinear
Bottleneck Attention: 开启
损失: 纯 MSE
课程学习: 开启
退火阶段: PHASE1_END=0.1, PHASE2_END=0.4
```

当前训练状态：

```text
从 last checkpoint 恢复后继续训练
当前约在 Epoch 107/200
当前阶段: Phase2 信道噪声过渡期
channel_prob 约 0.65
```

## 2. A/B/C 的区别

```text
A: 只验证课程学习 + 0.082 BPP 低码率方案
B: A + 主干网络升级
C: B + MS-SSIM 混合损失 + Bottleneck Attention
```

可以这样看：

```text
B - A = 主干网络升级带来的收益
C - B = MS-SSIM 损失和 Attention 带来的收益
```

## 3. 如何用 test_real.py 测不同方案权重

### 测 v2 A/B/C 低码率权重

这些实验都是：

```text
DOWNSAMPLE_STRIDES = [8, 2]
NUM_EMBEDDINGS_LIST = [16, 32]
```

所以直接指定对应 checkpoint 即可。

#### A

```bash
SIMVQ_EXPERIMENT_STAGE=A \
python test_real.py \
  --checkpoint checkpoints/quality_v2_A_curriculum_unet2_ds8x2_k16-32/best_vq_deepsc.pth \
  --no-channel
```

#### B

```bash
SIMVQ_EXPERIMENT_STAGE=B \
python test_real.py \
  --checkpoint checkpoints/quality_v2_B_backbone_unet2_ds8x2_k16-32/best_vq_deepsc.pth \
  --no-channel
```

#### C

```bash
SIMVQ_EXPERIMENT_STAGE=C \
python test_real.py \
  --checkpoint checkpoints/quality_v2_C_full_unet2_ds8x2_k16-32/best_vq_deepsc.pth \
  --no-channel
```

### 测真实链路 LDPC + BPSK + AWGN

去掉 `--no-channel`，并指定 SNR：

```bash
SIMVQ_EXPERIMENT_STAGE=B \
python test_real.py \
  --checkpoint checkpoints/quality_v2_B_backbone_unet2_ds8x2_k16-32/best_vq_deepsc.pth \
  --snrs 0 3 6 9 12
```

### 测旧 v1 `[4,2] / [64,64]` 权重

旧 v1 的下采样和码本与 v2 不同，所以测试时要加两个环境变量：

```bash
SIMVQ_DOWNSAMPLE_STRIDES=4,2 \
SIMVQ_NUM_EMBEDDINGS_LIST=64,64 \
python test_real.py \
  --checkpoint checkpoints/quality_v1_unet2_ds4x2_k64/best_vq_deepsc.pth \
  --no-channel
```

真实链路：

```bash
SIMVQ_DOWNSAMPLE_STRIDES=4,2 \
SIMVQ_NUM_EMBEDDINGS_LIST=64,64 \
python test_real.py \
  --checkpoint checkpoints/quality_v1_unet2_ds4x2_k64/best_vq_deepsc.pth \
  --snrs 0 3 6 9 12
```

## 4. checkpoint 保留规则

现在已经改成只保留：

```text
best_vq_deepsc.pth
last_checkpoint.pth
```

含义：

```text
best_vq_deepsc.pth: 当前实验验证集表现最好的权重，用来测试/对比
last_checkpoint.pth: 最近一次训练断点，用来中断后恢复训练
```

已经删除：

```text
vq_deepsc_epoch_10.pth
vq_deepsc_epoch_20.pth
vq_deepsc_epoch_30.pth
...
vq_deepsc_epoch_200.pth
```

后续训练代码也不再保存这些周期性 epoch 权重。

## 5. experiments 目录说明

```text
experiments/
```

只放实验输出，不放模型权重。

### 顶层 CSV/JSON/Markdown

```text
experiment_log.md
```

总实验记录，包含启动了哪些实验、在哪张 GPU、PID、修改过程和原因。

```text
*_epoch_metrics.csv
```

每个实验的逐 epoch 训练/验证指标。

例如：

```text
quality_v2_A_curriculum_unet2_ds8x2_k16-32_epoch_metrics.csv
quality_v2_B_backbone_unet2_ds8x2_k16-32_epoch_metrics.csv
quality_v2_C_full_unet2_ds8x2_k16-32_epoch_metrics.csv
```

```text
*_screening.csv
```

监控脚本定期跑 no-channel 上限评估后写入的 PSNR/MS-SSIM。

```text
baseline_best_epoch079_nochannel.json
```

旧 baseline 的评估结果。

### `experiments/logs/`

训练、监控、测试脚本的 stdout 日志。

常见文件：

```text
train_<实验名>.log
monitor_<实验名>.log
<实验名>_screen_epoch_XXX.log
```

### `experiments/snapshots/`

监控脚本输出的 JSON 评估结果。

这里保存的是指标快照，不保存 `.pth` 权重。

### `experiments/tensorboard/`

TensorBoard 事件文件。

用于看 loss 曲线、码本利用率、channel_prob 等训练曲线。

### `experiments/variants/`

历史配置归档。

例如：

```text
preexisting_k128_config.py
```

这是旧配置备份，不是当前训练入口。
