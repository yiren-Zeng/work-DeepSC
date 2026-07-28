# A / B / C 三组方案详细配置与当前结果分析

生成时间：2026-06-09  
代码目录：`/workspace/yi/work/Simvq-dc-64-Multi-pro-2ceng`  
基准来源：`quality_v2_B_larger_cb128-16_unet2_ds8x2_k128-16`  
当前三组训练仍在进行中，因此本文档是生成时刻的阶段性快照。

## 1. 日志清理说明

已清理 `experiments/logs` 目录：

- 保留当前正在训练的三条日志：
  - `train_exp12_rate044_A_patch_cb16-2-20260608-191100.log`
  - `train_exp13_rate041_B_hybridcvq_cb65536-8192-20260608-191100.log`
  - `train_exp14_rate041_C_hybridcvq_nested_cb65536-8192-20260608-191100.log`
- 其它历史日志没有删除，已移动到：
  - `experiments/logs/archive/`

这不会影响后续生成对比文件。后续对比主要依赖：

- `experiments/*_epoch_metrics.csv`
- `experiments/*_codebook_metrics.csv`
- `checkpoints/*/best_vq_deepsc.pth`
- `experiments/eval_best_20260609_rate041_large_to_small/*.json`

旧日志移动到 `archive` 只影响日志目录的整洁度，不影响 checkpoint、CSV 指标和测试 JSON。

## 2. 三组方案总览

| 方案 | 名称 | 量化结构 | 码本大小 | 测试压缩率口径 | 当前 best epoch | 当前 best val_recon |
|---|---|---|---:|---:|---:|---:|
| A | SimVQ patch-wise baseline | 两层均为空间 patch-wise SimVQ | `K1=16, K2=2` | `0.0442708` | 113 | `0.01639258` |
| B | SimVQ + CVQ | 第 1 层 channel-wise，第 2 层 patch-wise | `K1=65536, K2=8192` | `0.0407986` | 101 | `0.01229034` |
| C | SimVQ + CVQ + nested channel dropout | 第 1 层 channel-wise，第 2 层 patch-wise，第 1 层训练时 nested dropout | `K1=65536, K2=8192` | `0.0407986` | 101 | `0.01249325` |

注意：A 的压缩率是 `0.04427`，B/C 是 `0.04080`，不是完全相同；B/C 的码率略低一些。这会给 B/C 带来一点码率劣势，但不足以解释全部性能差距。

## 3. 压缩率计算口径

本实验按你的要求，压缩率必须带上信道编码率和调制阶数。本文使用：

- 信道编码：LDPC，码率 `R=1/2`
- 调制：BPSK，调制阶数对应每符号比特数 `M=1`
- 图像通道数：RGB 三通道，即 `3`
- 测试分辨率：`768 x 512`
- 分母：`R * M * 3 * H * W = 0.5 * 1 * 3 * 768 * 512 = 589824`

### 3.1 A 方案压缩率

A 是两层 patch-wise：

- 第 1 层总下采样倍率：`8`
  - 测试 token 数：`(768/8) * (512/8) = 96 * 64 = 6144`
  - `K1=16`，每个 index 需要 `log2(16)=4` bit
  - 第 1 层 bit 数：`6144 * 4 = 24576`
- 第 2 层总下采样倍率：`8*2=16`
  - 测试 token 数：`(768/16) * (512/16) = 48 * 32 = 1536`
  - `K2=2`，每个 index 需要 `log2(2)=1` bit
  - 第 2 层 bit 数：`1536 * 1 = 1536`

总 source bit：

```text
24576 + 1536 = 26112
```

带 LDPC 1/2 + BPSK 的测试压缩率：

```text
26112 / (0.5 * 1 * 3 * 768 * 512) = 0.0442708
```

因为 A 是 patch-wise，token 数随图像面积同比例变化，所以训练 `256x256` 和测试 `768x512` 下的该口径压缩率一致。

### 3.2 B/C 方案压缩率

B/C 是混合结构：

- 第 1 层 channel-wise CVQ
  - 第 1 层特征通道数：`256`
  - 每个通道一个 index，因此 token 数固定为 `256`
  - `K1=65536`，每个 index 需要 `log2(65536)=16` bit
  - 第 1 层 bit 数：`256 * 16 = 4096`
- 第 2 层 patch-wise SimVQ
  - 第 2 层测试 token 数：`48 * 32 = 1536`
  - `K2=8192`，每个 index 需要 `log2(8192)=13` bit
  - 第 2 层 bit 数：`1536 * 13 = 19968`

总 source bit：

```text
4096 + 19968 = 24064
```

带 LDPC 1/2 + BPSK 的测试压缩率：

```text
24064 / (0.5 * 1 * 3 * 768 * 512) = 0.0407986
```

这里有一个关键点：channel-wise 的第 1 层 token 数是通道数 `256`，不随空间分辨率同比例增长。因此 B/C 在训练分辨率 `256x256` 下的压缩率并不等于测试分辨率下的压缩率。

B/C 在训练分辨率 `256x256` 下：

- 第 1 层 channel-wise bit：仍然是 `256 * 16 = 4096`
- 第 2 层 patch-wise token：`16 * 16 = 256`
- 第 2 层 bit：`256 * 13 = 3328`
- 总 bit：`7424`
- 训练口径压缩率：`7424 / (0.5 * 1 * 3 * 256 * 256) = 0.0755208`

也就是说，B/C 是按测试分辨率约 `0.04080` 设计的，但训练时实际看到的是更高的等效传输开销。

## 4. 共同训练配置

三组方案的主干配置保持一致，来自 `quality_v2_B_larger` 风格：

| 项目 | 配置 |
|---|---|
| 输入通道 | RGB，`in_channels=3` |
| 输出通道 | RGB，`out_channels=3` |
| 训练数据 | `/workspace/yi/work/Cars196/train_data` |
| 验证数据 | `/workspace/yi/work/Cars196/val_data` |
| 测试数据 | `/workspace/yi/work/Kodak` |
| 训练 resize | `256 x 256` |
| 验证 resize | 默认验证 transform：`Resize(256) + CenterCrop(256)`，除非环境变量覆盖 |
| 测试 resize | `768 x 512` |
| batch | `SIMVQ_TOTAL_BATCH_SIZE=24`, `SIMVQ_MICRO_BATCH_SIZE=24` |
| 总 epoch | `200` |
| resume | `True`，从 `last_checkpoint.pth` 恢复 |
| 主学习率 | `LEARNING_RATE_G=5e-5` |
| SimVQ 投影层学习率 | `CODEBOOK_PROJ_LR=2e-4` |
| Adam betas | `(0.5, 0.999)` |
| VQ commitment cost | `0.25` |
| 训练信道码率 | `0.5` |
| 验证信道码率 | `0.5` |
| 信道 block length | `256` |
| 训练 SNR 范围 | `[0, 15] dB` |
| 训练中调制 bit 抽样 | SNR < 4 时从 `[1,2]` 抽，4-8 时从 `[1,2,4]` 抽，>=8 时从 `[2,4]` 抽 |
| channel_prob 调度 | epoch 80 到 120 逐步引入/增强信道扰动 |
| 重建损失 | `MSELoss`，权重 `1.0` |
| MS-SSIM loss | 权重 `0.0`，未启用 |
| VGG/LPIPS 感知损失 | 权重 `0.0`，未启用 |

训练图像 transform：

```text
Resize(256,256)
RandomHorizontalFlip()
ToTensor()
Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
```

说明：

- `RandomHorizontalFlip()` 是训练阶段的数据增强，含义是以默认概率 `p=0.5` 将输入图像做左右水平翻转。
- 它只用于 `mode="train"`，验证和测试不会做随机翻转。
- 它的作用是增加训练样本的视角变化，让模型不要过度依赖物体固定朝向，从而改善泛化。
- 这不是 A/B/C 新方案额外加入的改动，而是当前 `data/datasets.py` 中训练 dataloader 本来就有的逻辑。
- 原始基准 `quality_v2_B_larger_cb128-16_unet2_ds8x2_k128-16` 也是通过同一个 `get_dataloader(..., mode="train")` 进入训练，因此它同样使用该训练增强。
- 如果设置了 `SIMVQ_TRAIN_RESIZE=H,W`，训练 transform 是 `Resize(H,W) + RandomHorizontalFlip + ToTensor + Normalize`；如果没有设置 `SIMVQ_TRAIN_RESIZE`，代码路径会使用 `Resize(256) + RandomCrop(256) + ToTensor + Normalize`。本次 A/B/C 文档中写 `Resize(256,256)`，是因为实验讨论和 CVQ codeword shape 都按训练尺寸 `256x256` 记录。

测试图像 transform：

```text
Resize(768,512)
ToTensor()
Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
```

## 5. U-Net 主干结构

三组方案都使用两层 U-Net：

```text
UNET_DEPTH = 2
DOWNSAMPLE_STRIDES = [8, 2]
BASE_CHANNELS = 128
EMBEDDING_DIM_LIST = [256, 512]
ENCODER_RES_BLOCKS = 4
DECODER_RES_BLOCKS = 4
```

### 5.1 编码器 SemanticEncoder

编码器来自 `models/semantic_encoder.py`。

初始层：

```text
nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
nn.SiLU(inplace=True)
```

每个 `ResidualBlock`：

```text
nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1)
GroupNorm 或 BatchNorm2d
SiLU 或 PReLU
nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1)
残差相加
```

本实验阶段为 `SIMVQ_EXPERIMENT_STAGE=B`，所以归一化和激活为：

```text
NORM_TYPE = group
GROUP_NORM_GROUPS = 32
ACTIVATION = silu
```

也就是实际使用：

```text
nn.GroupNorm(num_groups=32, num_channels=C)
nn.SiLU(inplace=True)
```

下采样块 `DownSampleBlock`：

- 下采样前：4 个 residual block
- 下采样：`nn.Conv2d(..., kernel_size=3, stride=stride, padding=1)`
- 下采样后：4 个 residual block
- 尾部：`GroupNorm + SiLU + nn.Conv2d(..., kernel_size=3, stride=1, padding=1)`

关于这里的 4 个 residual block：

- 这 4 个 residual block 由 `SIMVQ_ENCODER_RES_BLOCKS=4` 控制；原始基准脚本 `run_exp9_larger_cb128_16.sh` 中已经明确设置为 `4`，所以 A/B/C 是继承原始 `quality_v2_B_larger_cb128-16` 的主干配置，不是这次 CVQ 实验临时新增的。
- 你记得“之前是 1 或 2 个”也是对的：更早的 stage 默认配置里，Stage A 默认是 1 个 residual block，Stage B 默认是 2 个 residual block；后来 `larger` 系列脚本将 encoder/decoder residual blocks 都显式提高到 4，性能也确实随主干容量增强而提升。
- residual block 的作用是提高每个尺度内的特征提取能力。下采样前的 4 个 block 先在原尺度上做局部纹理、边缘、颜色结构的融合；下采样后的 4 个 block 再在新尺度和新通道数上重整语义特征。
- 残差连接让网络可以学习“修正量”，而不是每层都从头重构特征；这通常让更深的卷积堆叠更容易训练，梯度传播更稳定。
- 对这个任务来说，量化器会把连续特征压成离散 index。量化前特征越规整、越有表达力，码本越容易匹配；因此把 residual block 从 1/2 增加到 4，往往会提升重建质量。
- 代价是训练和推理更慢、显存更多，但相比直接增大码本，它提升的是 encoder/decoder 的连续表征能力，通常更稳定。

由于 `USE_CASCADE_DOWNSAMPLE=False`，第 1 层下采样 `stride=8` 是单个 stride=8 的 `Conv2d`，不是 2x 级联下采样。

编码器输出尺度：

| 层 | 输入测试尺度 | 下采样 | 输出尺度 | 输出通道 |
|---|---:|---:|---:|---:|
| Layer 1 | `768 x 512` | `/8` | `96 x 64` | `256` |
| Layer 2 | `96 x 64` | `/2` | `48 x 32` | `512` |

训练 `256x256` 时：

| 层 | 输出尺度 | 输出通道 |
|---|---:|---:|
| Layer 1 | `32 x 32` | `256` |
| Layer 2 | `16 x 16` | `512` |

### 5.2 解码器 SemanticDecoder

解码器来自 `models/semantic_decoder.py`。

起始层：

```text
nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
nn.SiLU(inplace=True)
```

每个 `UpSampleBlock`：

```text
4 个 ResidualBlock
F.interpolate(..., mode="bilinear", scale_factor=对应倍率)
nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
GroupNorm
SiLU
```

关于 decoder 中“每个 UpSampleBlock 只有一组 4 个 ResidualBlock”：

- 这里不是整个解码器只有 4 个 residual block，而是每一个 `UpSampleBlock` 内部都有一组 4 个 residual block。
- 当前两层 U-Net 有两个上采样块，因此 decoder 共有两组 residual block：先在最深层特征上处理一组，再在 concat 了浅层 skip 后继续处理/上采样下一组。
- encoder 的 `DownSampleBlock` 看起来有“下采样前 4 个 + 下采样后 4 个”，是因为它需要同时完成两个任务：先在输入尺度提取特征，再通过 stride conv 改变分辨率和通道数，最后在新尺度继续重整特征。
- decoder 的 `UpSampleBlock` 当前设计是先在当前尺度用 4 个 residual block 整理特征，然后用 `F.interpolate` 放大，再用 `Conv2d + GroupNorm + SiLU` 做通道映射和局部融合。随后如果还有 skip connection，会与浅层特征 concat，进入下一个 up block 继续处理。
- 也就是说，decoder 把“上采样后的进一步融合”交给了后续 block 和 skip concat 后的处理，而不是在同一个 up block 内再额外放一组 4 个 residual block。
- 这种设计在计算量和效果之间比较折中：如果每个 up block 也做“上采样前 4 个 + 上采样后 4 个”，decoder 计算量会明显增加，而且上采样后的高分辨率特征最耗显存。
- 目前 `quality_v2_B_larger` 系列性能提升的关键不是 decoder 只有一组就天然最优，而是“每个上采样尺度有 4 个 residual block + skip connection + bilinear upsample”已经能提供足够的解码容量。继续加深 decoder 可能还能提高性能，但需要重新验证显存、速度和过拟合风险。

上采样倍率是编码器 stride 的反序：

```text
upsample_scales = [2, 8]
```

解码流程：

1. 从最深层量化特征 `Layer 2` 开始。
2. 先上采样 `x2` 到第 1 层尺度。
3. 与第 1 层量化特征 concat。
4. 再上采样 `x8` 回到图像尺度。
5. 末端使用：

```text
nn.ConvTranspose2d(in_ch, 3, kernel_size=3, stride=1, padding=1)
```

这里的 `ConvTranspose2d` 只做通道映射，不改变空间尺寸。

### 5.3 没有启用的模块

本次 A/B/C 未启用：

- Bottleneck self-attention：`USE_BOTTLENECK_ATTENTION=False`
- SwinIR enhancement：`USE_SWINIR_ENHANCE=False`
- ViT-VQ/QBridge：未使用
- Vanilla VQ：未使用
- LPIPS/VGG perceptual loss：未使用
- MS-SSIM loss：未使用

## 6. 三组量化器配置

### 6.1 A: SimVQ patch-wise

环境变量核心配置：

```text
SIMVQ_EXP_FAMILY=quality_v2_B_larger_rate044_A_patch_cb16-2
SIMVQ_NUM_EMBEDDINGS_LIST=16,2
SIMVQ_QUANTIZER_TYPE=simvq
SIMVQ_QUANTIZER_AXIS_LIST=patch,patch
SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA=0.0
```

量化器：

- 第 1 层：`VectorQuantizer(K=16, embedding_dim=256)`
- 第 2 层：`VectorQuantizer(K=2, embedding_dim=512)`

`VectorQuantizer` 是 SimVQ 风格：

```text
ProjectedEmbedding:
  nn.Embedding(K, D)       # 冻结
  nn.Linear(D, D, bias=False)  # 可训练投影
```

每个空间位置一个 index：

```text
Layer 1 index shape:
  train: [B, 32, 32]
  test:  [B, 96, 64]

Layer 2 index shape:
  train: [B, 16, 16]
  test:  [B, 48, 32]
```

优点：

- 每个空间位置独立选码字，局部纹理和边缘保留更直接。
- 训练 `256x256` 和测试 `768x512` 时 token 结构保持同构，只是网格变大。
- 码本虽然小，但 token 数很多，尤其第 1 层有大量空间 token。

缺点：

- 码本很小，单 token 表达能力有限。
- 对单个 patch 的语义表达较粗。

### 6.2 B: SimVQ + CVQ

环境变量核心配置：

```text
SIMVQ_EXP_FAMILY=quality_v2_B_larger_rate041_B_hybridcvq_cb65536-8192
SIMVQ_NUM_EMBEDDINGS_LIST=65536,8192
SIMVQ_QUANTIZER_TYPE=simvq
SIMVQ_QUANTIZER_AXIS_LIST=channel,patch
SIMVQ_CVQ_CODEWORD_SHAPES=32x32,patch
SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA=0.0
```

量化器：

- 第 1 层：`ChannelwiseVectorQuantizer(K=65536, codeword_shape=32x32)`
- 第 2 层：`VectorQuantizer(K=8192, embedding_dim=512)`

第 1 层 channel-wise CVQ 的逻辑：

```text
输入特征: [B, C, H, W]
每个通道图作为一个 token
训练时 Layer 1 为 [B, 256, 32, 32]
每个 channel map 展平为 1024 维向量
用 K=65536 的码本做最近邻
输出 index shape: [B, 256]
```

测试时 Layer 1 是 `[B, 256, 96, 64]`，代码会先将每个通道图 resize 到训练码字大小 `32x32` 做查表，再把量化后的码字插值回 `96x64`：

```text
F.interpolate(input, size=(32,32), mode="bilinear")
nearest codebook lookup
F.interpolate(quantized, size=(96,64), mode="bilinear")
```

这一步是 B/C 与 A 的核心差异之一。

### 6.3 C: SimVQ + CVQ + nested channel dropout

环境变量核心配置：

```text
SIMVQ_EXP_FAMILY=quality_v2_B_larger_rate041_C_hybridcvq_nested_cb65536-8192
SIMVQ_NUM_EMBEDDINGS_LIST=65536,8192
SIMVQ_QUANTIZER_TYPE=simvq
SIMVQ_QUANTIZER_AXIS_LIST=channel,patch
SIMVQ_CVQ_CODEWORD_SHAPES=32x32,patch
SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA=0.25
```

C 与 B 的结构相同，区别在训练阶段对 channel-wise 层使用 nested channel dropout：

```text
if training and random.random() < alpha:
    c_keep = random.randint(1, C)
    feat[:, c_keep:, :, :] = 0
```

本实验中：

```text
alpha = 0.25
```

含义：

- 只作用于第 1 层 channel-wise CVQ。
- 每次触发时随机保留前 `c_keep` 个通道，其后的通道置零。
- 目的是让低序号通道承担更稳定的信息，形成一种 nested / progressive channel 表达。

潜在副作用：

- 如果解码器和通道排序没有被额外约束，简单按通道前缀保留未必能自然形成最优重要性排序。
- 25% 的训练 step 会给第 1 层特征引入强扰动，可能降低高分辨率 skip 特征的精细重建能力。
- 当前损失只用 MSE，不直接奖励感知质量或结构相似度，因此 nested dropout 的鲁棒性收益未必能转化成 MS-SSIM/PSNR 提升。

## 7. 测试配置

当前已经执行的关键测试：

| 测试项 | 配置 |
|---|---|
| 测试集 | Kodak |
| 测试 resize | `768 x 512` |
| batch size | `1` |
| 信道编码 | 5G NR LDPC via Sionna |
| LDPC 参数 | `k=128`, `n=256`, `R=0.5` |
| 调制 | BPSK |
| 信道 | AWGN |
| SNR | `0 dB` |
| LDPC 运行设备 | TensorFlow GPU 被屏蔽，LDPC 在 CPU 上运行 |
| 评价指标 | MS-SSIM, PSNR |
| 随机种子 | `42` |

测试流程：

1. 图像输入模型，执行 `forward_test` 得到各层 index。
2. 将 index 转 bit。
3. LDPC 编码。
4. BPSK 调制。
5. AWGN 信道，目标 SNR=0 dB。
6. BPSK LLR。
7. LDPC 解码。
8. bit 转回 index。
9. `reconstruct_from_indices` 还原图像。
10. 计算 MS-SSIM 和 PSNR。

## 8. 当前测试结果

| 方案 | 条件 | MS-SSIM | PSNR |
|---|---|---:|---:|
| A | no-channel 上界，旧 best | `0.8923` | `24.7390 dB` |
| A | LDPC 1/2 + BPSK，SNR=0，best epoch 113 | `0.9033` | `25.2304 dB` |
| B | no-channel 上界，早期 best | `0.8746` | `24.4243 dB` |
| B | LDPC 1/2 + BPSK，SNR=0，中断前控制台输出 | `0.8701` | `24.2549 dB` |
| C | no-channel 上界，早期 best | `0.8736` | `24.4821 dB` |
| C | LDPC 1/2 + BPSK，SNR=0，best epoch 42 | `0.8708` | `24.3090 dB` |
| C | LDPC 1/2 + BPSK，SNR=0，best epoch 101 | `0.8774` | `24.6618 dB` |

阶段性客观结论：

- A 当前明显优于 C。
- A 相比 C epoch 101：
  - MS-SSIM 高 `0.9033 - 0.8774 = 0.0259`
  - PSNR 高 `25.2304 - 24.6618 = 0.5686 dB`
- B 的 no-channel 上界已经低于 A，说明问题不是单纯来自 LDPC/BPSK 链路，而是源重建能力本身就落后。

## 9. 为什么 C 的大码本没有带来更好效果

表面上看，C 的码本非常大：`K1=65536, K2=8192`，而 A 只有 `K1=16, K2=2`。直觉上会觉得 C 应该更强。但这个直觉在当前结构下不成立，原因如下。

### 9.1 码本大不等于传输信息多

真正传输的信息量由：

```text
token 数量 * 每个 token 的 bit 数
```

决定，而不是单看 `K`。

A 的第 1 层虽然 `K1=16` 很小，但测试时有 `6144` 个第 1 层空间 token：

```text
6144 * 4 = 24576 bit
```

C 的第 1 层虽然 `K1=65536` 很大，但只有 `256` 个 channel token：

```text
256 * 16 = 4096 bit
```

也就是说，A 的高分辨率第 1 层实际传了 `24576 bit`，C 的高分辨率第 1 层只传了 `4096 bit`。C 第 1 层码本更大，但第 1 层总 bit 反而只有 A 的约 `1/6`。

这对图像重建很关键，因为第 1 层是高分辨率 skip 特征，最负责边缘、纹理、局部颜色和空间细节。A 在这一层保留了大量空间 token，C 则把每个通道整张特征图压成一个 index，空间自由度大幅减少。

### 9.2 CVQ 的第 1 层是“整通道图”量化，不是“每个位置”量化

A 的 patch-wise 第 1 层：

```text
[B, 256, 96, 64] -> [B, 96, 64] indices
```

每个空间位置可以独立选择码字。

C 的 channel-wise 第 1 层：

```text
[B, 256, 96, 64] -> [B, 256] indices
```

每个通道整张 `96x64` 特征图最终只对应一个码字。虽然码字本身是二维图，但每个通道只有一个选择。这种表达方式更像“从 65536 个完整 channel pattern 里选一个”，而不是对局部区域逐点适配。

对于自然图像，局部纹理、边缘方向、物体位置变化非常强。整通道图量化很容易出现：

- 局部细节无法逐位置修正；
- 码字需要同时解释整张通道图；
- 码字匹配主要拟合全局形状，牺牲局部误差；
- 解码器拿到的高分辨率 skip 特征不够细。

这解释了为什么大码本仍然可能输给小码本 patch-wise。

### 9.3 训练分辨率与测试分辨率存在 CVQ 特有的尺度错配

C 的第 1 层 CVQ codeword shape 固定为：

```text
32 x 32
```

这是训练 `256x256` 时第 1 层特征的空间大小。

但测试 `768x512` 时第 1 层特征是：

```text
96 x 64
```

当前实现为了让同一个 CVQ 码本能测试，会做：

```text
96x64 -> bilinear resize 到 32x32 -> 查码本 -> bilinear resize 回 96x64
```

这带来两个问题：

1. 查码本前已经把高分辨率特征压到 `32x32`，高频细节会被平滑。
2. 查完后再插值回 `96x64`，恢复的是平滑后的 codeword，不可能凭空恢复原始高频结构。

A 没有这个问题。A 的 patch-wise token 网格会自然从训练的 `32x32` 扩展到测试的 `96x64`，没有把整层特征 resize 回训练尺度再查码本。

因此，C 在测试分辨率下的高分辨率层实际被引入了额外的信息瓶颈。

### 9.4 第 2 层承担了过多 bit，但它是低分辨率层

C 的总 bit 中，第 2 层占主导：

```text
Layer 1: 4096 bit
Layer 2: 19968 bit
```

第 2 层是 `48x32` 的低分辨率特征，主要负责更抽象、更大感受野的信息。它有助于整体结构，但不如第 1 层适合恢复局部高频细节。

A 的 bit 分配相反，更偏向高分辨率第 1 层：

```text
Layer 1: 24576 bit
Layer 2: 1536 bit
```

对于 PSNR/MS-SSIM 这类重建指标，高分辨率 skip 特征通常非常重要。当前结果说明：把更多 bit 放在第 1 层空间 token，比把大码本放到低分辨率或整通道 token 更有效。

### 9.5 nested channel dropout 可能让 C 的第 1 层更难训练

C 的 nested dropout 是一种合理的鲁棒性尝试，但当前实现比较硬：

```text
随机触发时，将 c_keep 之后所有通道置零
```

这要求网络自然学会“前面的通道更重要”。但模型中没有显式排序损失，也没有逐级码率训练目标，所以通道重要性未必会自动按编号排列。

可能结果是：

- 第 1 层高分辨率特征被扰动；
- 解码器更依赖第 2 层低分辨率特征；
- no-channel 上界也下降；
- 在 SNR=0 时虽然有一点鲁棒性，但不足以弥补源重建损失。

这与当前结果一致：C 的 SNR=0 比早期有提升，但仍低于 A。

### 9.6 SimVQ 的大码本也可能存在有效利用不足

SimVQ 使用冻结底层 embedding + 可训练线性投影：

```text
nn.Embedding(K,D) frozen
nn.Linear(D,D,bias=False) trainable
```

这种设计对稳定训练有帮助，但当 `K=65536` 且每个样本第 1 层只有 `256` 个 channel token 时，每个 batch 对第 1 层码本的实际访问非常稀疏。

以 batch 24 估算：

```text
24 * 256 = 6144 个第 1 层 token / batch
```

面对 `65536` 个码字，单个 epoch 内每个码字被充分比较和有效使用的机会不一定高。大码本如果没有高利用率，就只是增加候选集合，不等价于更高有效表达能力。

建议后续必须导出并检查：

- 第 1 层 active code count
- 第 1 层 active ratio
- 第 1 层 perplexity
- 第 1 层 dead code count
- 第 2 层对应指标

如果 C 的第 1 层 active ratio 很低，就能进一步证明“大码本没有被有效用起来”。

### 9.7 训练目标只优化 MSE，不一定鼓励 CVQ 学到感知更好的通道模式

当前损失：

```text
recon_loss = MSE(x_hat, x)
total_loss = recon_loss + weighted_vq_loss
```

没有启用：

- MS-SSIM loss
- LPIPS/VGG perceptual loss
- adversarial loss
- codebook usage regularization
- channel-order regularization

CVQ 的整通道码字更像学习全局 feature basis。如果只用 MSE，它可能优先学平均、平滑、低频更稳定的模式，而不是局部锐利纹理。这也会导致 PSNR/MS-SSIM 不如 patch-wise 高分辨率 token。

## 10. 对当前结果的客观判断

当前不能简单说“CVQ 方法无效”，更准确的结论是：

```text
在当前 K1=65536, K2=8192、Layer1 channel-wise + Layer2 patch-wise、
训练 256x256 / 测试 768x512、MSE-only loss 的实现条件下，
CVQ/C 方案没有超过 A，主要瓶颈在高分辨率第 1 层空间信息不足和尺度错配。
```

A 强的核心原因不是码本大，而是它把更多 bit 用在高分辨率空间 token 上。C 码本大，但它把第 1 层从 `6144` 个空间 token 变成 `256` 个通道 token，实际空间表达自由度降低了。

## 11. 后续改进建议

### 11.1 最优先：不要把第 1 层完全改成 channel-wise

从当前结果看，第 1 层是高分辨率关键层。建议下一版不要用纯 channel-wise 代替第 1 层 patch-wise，可以考虑：

```text
Layer 1: patch-wise 或 patch-wise + channel-wise side information
Layer 2: channel-wise / patch-wise 混合
```

也就是把 CVQ 放在更低分辨率层，或者作为辅助分支，而不是替代第 1 层主路径。

### 11.2 如果继续做 CVQ，第 1 层要分块 channel-wise

当前第 1 层是整通道图一个 index。更合理的是 block-channel CVQ：

```text
[C, H, W] -> 按空间分块 -> 每个 channel block 一个 index
```

例如第 1 层测试 `96x64` 可分为多个 `16x16` 或 `32x32` block。这样既保留 CVQ 的通道建模，又不彻底丢掉空间自由度。

### 11.3 让训练和测试的 CVQ codeword shape 一致

当前训练 codeword 是 `32x32`，测试第 1 层是 `96x64`，需要 resize。后续可以考虑：

- 直接用 `768x512` 或接近测试分辨率训练；
- multi-scale CVQ 训练；
- 为 `32x32`、`64x64`、`96x64` 建多尺度 codeword；
- 测试时避免先下采样整张 channel map 再查码本。

### 11.4 调整 bit 分配，让高分辨率层拿到更多空间 token

如果目标仍然是约 `0.04` 压缩率，建议优先保证第 1 层空间 token，而不是只增大 channel-wise K。

可能方向：

```text
方案 D:
Layer 1 patch-wise: 较小 K，但保留 96x64 空间 token
Layer 2 channel-wise 或大 K patch-wise

方案 E:
Layer 1 block-CVQ: 每个 channel 的局部分块 token
Layer 2 patch-wise
```

### 11.5 检查 codebook usage

建议立刻对 A/B/C best checkpoint 导出码本利用率：

- active ratio
- perplexity
- dead code count
- min L2 distance
- collapse ratio

如果 C 的第 1 层利用率低，就应该降低 `K1` 或改变训练策略，而不是继续增大码本。

### 11.6 C 的 nested dropout 需要配套约束

如果要保留 nested channel dropout，建议增加：

- 通道重要性排序约束；
- 多码率训练目标；
- 按前缀通道重建的辅助 loss；
- 不同 `c_keep` 下的 reconstruction consistency loss。

否则只是随机置零后半通道，未必能形成真正可嵌套的表示。

## 12. 当前文件路径

训练日志：

- `experiments/logs/train_exp12_rate044_A_patch_cb16-2-20260608-191100.log`
- `experiments/logs/train_exp13_rate041_B_hybridcvq_cb65536-8192-20260608-191100.log`
- `experiments/logs/train_exp14_rate041_C_hybridcvq_nested_cb65536-8192-20260608-191100.log`

历史日志：

- `experiments/logs/archive/`

测试结果：

- `experiments/eval_best_20260609_rate041_large_to_small/A_patch_rate044_snr0_ldpc12_bpsk_retest_epoch113.json`
- `experiments/eval_best_20260609_rate041_large_to_small/C_hybrid_nested_rate041_snr0_ldpc12_bpsk_retest_epoch101.json`
- `experiments/eval_best_20260609_rate041_large_to_small/A_patch_rate044_no_channel.json`
- `experiments/eval_best_20260609_rate041_large_to_small/B_hybrid_rate041_no_channel.json`
- `experiments/eval_best_20260609_rate041_large_to_small/C_hybrid_nested_rate041_no_channel.json`

当前最重要的结论：

```text
C 的码本大，但第 1 层 token 数太少，并且存在训练/测试空间尺度错配；
A 的码本小，但第 1 层保留了大量高分辨率空间 token。
因此当前实验中 A 更强是合理的，不是测试异常。
```
