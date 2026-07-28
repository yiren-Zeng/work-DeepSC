# CVQ 混合量化 0.083 压缩率实验方案

日期：2026-06-08

本文档集中分析并记录以下原始方案：

`quality_v2_B_larger_cb128-16_unet2_ds8x2_k128-16`

并基于该方案推导三组实验：

```text
1. Baseline: SimVQ patch-wise
2. Variant B: SimVQ + CVQ
3. Variant C: SimVQ + CVQ + nested channel dropout
```

其中 Variant B 和 Variant C 按用户要求采用混合量化方式：

```text
第一层：CVQ，按通道量化
第二层：patch-wise，按空间位置量化，保持原方案方式
```

所有方案均保持：

```text
训练尺寸：256 x 256
测试尺寸：768 x 512
信道编码：1/2 LDPC
调制方式：BPSK
U-Net 层数：2
下采样倍率：[8, 2]
训练损失函数：保持原 quality_v2_B_larger_cb128-16 方案不变
```

## 1. 压缩率计算口径

本文档中的压缩率不是单纯的信源 BPP，而是带上信道编码率和调制阶数后的真实传输符号开销。

定义：

```text
compression_ratio =
    transmitted_channel_symbols / original_rgb_values
```

对于 BPSK：

```text
log2(M) = 1
```

对于 1/2 LDPC：

```text
R_c = 0.5
```

若离散索引总比特数为 `source_bits`，则：

```text
LDPC 编码后比特数 = source_bits / R_c
BPSK 传输符号数 = source_bits / (R_c * log2(M))
```

因此压缩率为：

```text
compression_ratio =
    source_bits / (R_c * log2(M) * 3 * H * W)
```

代入 `R_c = 0.5`、`log2(M)=1`：

```text
compression_ratio =
    source_bits / (0.5 * 1 * 3 * H * W)
```

目标压缩率为：

```text
compression_ratio = 0.0833333333
```

对应的信源侧 BPP 为：

```text
source_bpp = compression_ratio * R_c * log2(M) * 3
           = 0.0833333333 * 0.5 * 1 * 3
           = 0.125 bits/pixel
```

测试分辨率为 `768 x 512`：

```text
H * W = 768 * 512 = 393216
target_source_bits = 0.125 * 393216 = 49152 bits/image
```

也就是说，在测试条件下，若要满足压缩率 `0.0833333333`，所有量化索引加起来必须正好对应：

```text
49152 source bits / image
```

## 2. 原始方案核实结果

原始方案脚本：

`run_exp9_larger_cb128_16.sh`

脚本中的关键配置为：

```bash
export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="quality_v2_B_larger_cb128-16"
export SIMVQ_NUM_EMBEDDINGS_LIST="128,16"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="4"
export SIMVQ_DECODER_RES_BLOCKS="4"
export SIMVQ_QUANTIZER_TYPE="simvq"
```

通过 `config.py` 实际核实得到的结构为：

```text
experiment_name: quality_v2_B_larger_cb128-16_unet2_ds8x2_k128-16
experiment_stage: B
unet_depth: 2
downsample_strides: [8, 2]
total_downsample: 16
embedding_dim_list: [256, 512]
num_embeddings_list: [128, 16]
quantizer_type: simvq
norm_type: group
activation: silu
encoder_res_blocks: 4
decoder_res_blocks: 4
upsample_mode: bilinear
use_cascade_downsample: False
use_bottleneck_attention: False
mse_loss_weight: 1.0
ms_ssim_loss_weight: 0.0
lpips_loss_weight: 0.0
```

因此，用户描述完全正确：

```text
两层 U-Net：是
第一层码本：128
第二层码本：16
量化器：SimVQ
第一层 embedding dim：256
第二层 embedding dim：512
```

## 3. 原始方案压缩率核实

原始方案是两层 patch-wise SimVQ。

下采样结构：

```text
第一层累计下采样倍率 = 8
第二层累计下采样倍率 = 8 * 2 = 16
```

码本大小：

```text
K1 = 128, log2(K1) = 7 bits/index
K2 = 16,  log2(K2) = 4 bits/index
```

### 3.1 训练尺寸 256 x 256

第一层 token 数：

```text
256 / 8 = 32
32 * 32 = 1024 tokens
```

第二层 token 数：

```text
256 / 16 = 16
16 * 16 = 256 tokens
```

信源比特数：

```text
source_bits = 1024 * 7 + 256 * 4
            = 7168 + 1024
            = 8192 bits
```

信源 BPP：

```text
source_bpp = 8192 / (256 * 256)
           = 8192 / 65536
           = 0.125
```

带 1/2 LDPC + BPSK 后的压缩率：

```text
compression_ratio =
    8192 / (0.5 * 1 * 3 * 256 * 256)
  = 8192 / 98304
  = 0.0833333333
```

### 3.2 测试尺寸 768 x 512

第一层 token 数：

```text
768 / 8 = 96
512 / 8 = 64
96 * 64 = 6144 tokens
```

第二层 token 数：

```text
768 / 16 = 48
512 / 16 = 32
48 * 32 = 1536 tokens
```

信源比特数：

```text
source_bits = 6144 * 7 + 1536 * 4
            = 43008 + 6144
            = 49152 bits
```

信源 BPP：

```text
source_bpp = 49152 / (768 * 512)
           = 49152 / 393216
           = 0.125
```

带 1/2 LDPC + BPSK 后的压缩率：

```text
compression_ratio =
    49152 / (0.5 * 1 * 3 * 768 * 512)
  = 49152 / 589824
  = 0.0833333333
```

结论：

```text
原始方案在训练 256x256 和测试 768x512 下，压缩率均为 0.0833333333。
```

这是因为 patch-wise token 数会随图像面积等比例增长，所以信源 BPP 保持不变。

## 4. Baseline 方案

### 4.1 方案定义

```text
Baseline = SimVQ patch-wise
```

即保持原始方案不变：

```text
第一层：patch-wise SimVQ
第二层：patch-wise SimVQ
```

### 4.2 码本大小

测试分辨率 `768 x 512` 下：

```text
第一层 token 数 = 6144
第二层 token 数 = 1536
```

目标信源比特数：

```text
49152 bits
```

设：

```text
b1 = log2(K1)
b2 = log2(K2)
```

则：

```text
6144 * b1 + 1536 * b2 = 49152
```

两边除以 `1536`：

```text
4 * b1 + b2 = 32
```

原始码本：

```text
K1 = 128 => b1 = 7
K2 = 16  => b2 = 4
```

代入：

```text
4 * 7 + 4 = 32
```

所以 Baseline 的码本大小为：

```text
K1 = 128
K2 = 16
```

### 4.3 可训练性

Baseline 可训练，并且已经完成训练。

已有训练记录：

```text
run_id: exp9_larger_cb128-16-20260602-223140
GPU: physical GPU 1
epochs: 200 completed
checkpoint: checkpoints/quality_v2_B_larger_cb128-16_unet2_ds8x2_k128-16/best_vq_deepsc.pth
resume checkpoint: checkpoints/quality_v2_B_larger_cb128-16_unet2_ds8x2_k128-16/last_checkpoint.pth
epoch metrics: experiments/quality_v2_B_larger_cb128-16_unet2_ds8x2_k128-16_epoch_metrics.csv
codebook metrics: experiments/quality_v2_B_larger_cb128-16_unet2_ds8x2_k128-16_codebook_metrics.csv
training log: experiments/logs/train_exp9_larger_cb128-16-20260602-223140.log
```

第 200 epoch 码本利用率：

```text
layer 0: K=128, active_ratio=100%, active_count=128/128
layer 1: K=16,  active_ratio=100%, active_count=16/16
```

### 4.4 Baseline 启动命令

如果需要重新跑 Baseline，可使用：

```bash
cd /workspace/yi/work/Simvq-dc-64-Multi-pro-2ceng

export SIMVQ_EXPERIMENT_STAGE="B"
export SIMVQ_EXP_FAMILY="quality_v2_B_larger_cb128-16"
export SIMVQ_NUM_EMBEDDINGS_LIST="128,16"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="4"
export SIMVQ_DECODER_RES_BLOCKS="4"
export SIMVQ_QUANTIZER_TYPE="simvq"
export SIMVQ_TOTAL_BATCH_SIZE="24"
export SIMVQ_MICRO_BATCH_SIZE="24"
export GPU_ID="1"

RUN_ID="baseline_simvq_patchwise_cb128-16_$(date +%Y%m%d-%H%M%S)"
export EXPERIMENT_RUN_ID="$RUN_ID"

CUDA_VISIBLE_DEVICES="$GPU_ID" python -u train.py 2>&1 \
  | tee "experiments/logs/train_${RUN_ID}.log"
```

## 5. Variant B：SimVQ + CVQ

### 5.1 方案定义

```text
Variant B = SimVQ + CVQ
```

按照用户要求：

```text
第一层：CVQ，按通道数进行量化
第二层：patch-wise，按空间位置进行量化
```

第一层特征维度为：

```text
C1 = 256
```

因此第一层 CVQ 的 token 数不是 `96 * 64 = 6144`，而是：

```text
layer 1 CVQ tokens = 256
```

第二层保持 patch-wise：

```text
layer 2 patch-wise tokens = 48 * 32 = 1536
```

### 5.2 码本大小推导

目标信源比特数仍然是：

```text
49152 bits
```

设：

```text
b1 = log2(K1)
b2 = log2(K2)
```

则 Variant B 的目标方程为：

```text
256 * b1 + 1536 * b2 = 49152
```

两边除以 `256`：

```text
b1 + 6 * b2 = 192
```

这就是 Variant B 在测试 `768 x 512` 下达到 `0.0833333333` 压缩率必须满足的码本位数约束。

可选整数解包括：

| b1 | 第一层码本 K1 | b2 | 第二层码本 K2 |
|---:|---:|---:|---:|
| 6 | 64 | 31 | 2147483648 |
| 12 | 4096 | 30 | 1073741824 |
| 18 | 262144 | 29 | 536870912 |
| 24 | 16777216 | 28 | 268435456 |
| 30 | 1073741824 | 27 | 134217728 |
| 36 | 2^36 | 26 | 67108864 |
| 42 | 2^42 | 25 | 33554432 |
| 48 | 2^48 | 24 | 16777216 |
| 54 | 2^54 | 23 | 8388608 |
| 60 | 2^60 | 22 | 4194304 |

### 5.3 可训练性判断

Variant B 在数学上可以满足压缩率要求，但在工程上不可训练。

原因：

```text
第一层 CVQ token 数从原 patch-wise 的 6144 降到 256。
为了保持同样的 49152 source bits，缺失的比特必须转移到码本 index 位宽上。
这会导致 K1 或 K2 极端巨大。
```

例如选择：

```text
b1 = 12, K1 = 4096
b2 = 30, K2 = 1073741824
```

第二层码本需要：

```text
1,073,741,824 个 codewords
```

而第二层 embedding dim 是：

```text
512
```

仅码本参数规模就已经不可接受，最近邻查找也不可行。

再比如选择：

```text
b1 = 60, K1 = 2^60
b2 = 22, K2 = 4194304
```

虽然第二层码本降到了约 419 万，但第一层码本变成 `2^60`，更不可训练。

结论：

```text
Variant B 在“第一层 CVQ + 第二层 patch-wise + 测试压缩率严格 0.0833333333”的组合约束下，不建议启动训练。
```

### 5.4 若未来放宽约束后的建议 GPU

若后续允许实际压缩率低于 0.083，或者允许改变 token 设计，则建议分配：

```text
GPU: physical GPU 2
```

## 6. Variant C：SimVQ + CVQ + Nested Channel Dropout

### 6.1 方案定义

```text
Variant C = SimVQ + CVQ + nested channel dropout
```

量化轴与 Variant B 相同：

```text
第一层：CVQ，按通道量化
第二层：patch-wise，按空间位置量化
```

额外加入：

```text
nested channel dropout
```

训练时随机保留第一层前 `c_keep` 个 channel，其余 channel 置零：

```text
保留：Z[:, :c_keep, :, :]
置零：Z[:, c_keep:, :, :]
```

这样做的目的：

```text
强制前面的 channel 学全局结构、主体轮廓、颜色等重要信息；
后面的 channel 学纹理、高频细节等补充信息。
```

参考 CVQ 论文，推荐初始 dropout ratio：

```text
alpha = 0.25
```

### 6.2 码本大小推导

Nested channel dropout 不改变 token 数，也不改变每个 index 的 bit 数。

因此 Variant C 的压缩率方程与 Variant B 完全相同：

```text
256 * b1 + 1536 * b2 = 49152
```

即：

```text
b1 + 6 * b2 = 192
```

同样会得到极端码本：

| b1 | 第一层码本 K1 | b2 | 第二层码本 K2 |
|---:|---:|---:|---:|
| 6 | 64 | 31 | 2147483648 |
| 12 | 4096 | 30 | 1073741824 |
| 18 | 262144 | 29 | 536870912 |
| 24 | 16777216 | 28 | 268435456 |
| 30 | 1073741824 | 27 | 134217728 |
| 36 | 2^36 | 26 | 67108864 |
| 42 | 2^42 | 25 | 33554432 |
| 48 | 2^48 | 24 | 16777216 |
| 54 | 2^54 | 23 | 8388608 |
| 60 | 2^60 | 22 | 4194304 |

### 6.3 可训练性判断

Variant C 同样不建议在当前硬约束下启动训练。

原因：

```text
nested channel dropout 只能改变通道信息排序；
它不能增加 token 数，也不能减少达到 0.083 所需的 index 位数。
```

因此它无法解决码本爆炸问题。

结论：

```text
Variant C 在“第一层 CVQ + 第二层 patch-wise + 测试压缩率严格 0.0833333333”的组合约束下，不建议启动训练。
```

### 6.4 若未来放宽约束后的建议 GPU

若后续允许实际压缩率低于 0.083，或者允许改变 token 设计，则建议分配：

```text
GPU: physical GPU 3
```

## 7. 为什么没有直接启动 Variant B/C 训练

本次没有启动 Variant B 和 Variant C，是因为它们在用户指定约束下虽然可以写出数学解，但工程上不可训练。

关键矛盾是：

```text
原 Baseline 第一层 patch-wise token 数：
768/8 * 512/8 = 6144

改成第一层 CVQ 后，第一层 token 数：
C1 = 256
```

第一层 token 数降低了：

```text
6144 / 256 = 24 倍
```

但压缩率要求仍然保持同样的 source bits：

```text
49152 bits/image
```

于是必须把大量 bit 压到码本 index 位宽里，导致：

```text
b1 + 6*b2 = 192
```

这会让码本大小变成 `2^30`、`2^31`、甚至 `2^60` 这种不可训练规模。

因此，直接启动训练不客观，也不符合实验质量要求。

## 8. 可行替代方案

### 8.1 替代方案 A：保持严格 0.083，但不把第一层改成纯 CVQ

也就是继续使用：

```text
第一层：patch-wise
第二层：patch-wise
K1 = 128
K2 = 16
```

这是当前 Baseline，已经完成训练。

### 8.2 替代方案 B：保留混合 CVQ/patch-wise，但接受测试压缩率低于 0.083

如果使用可训练码本：

```text
第一层 CVQ：K1 = 128, b1 = 7
第二层 patch-wise：K2 = 16, b2 = 4
```

测试 `768 x 512` 下：

```text
source_bits = 256 * 7 + 1536 * 4
            = 1792 + 6144
            = 7936 bits
```

压缩率为：

```text
compression_ratio =
    7936 / (0.5 * 1 * 3 * 768 * 512)
  = 0.0134548611
```

这个方案可以训练，但它不是同码率对比；它是更低传输开销下的 CVQ 混合方案。

### 8.3 替代方案 C：保留混合 CVQ/patch-wise，并使用较大的可训练码本

例如：

```text
第一层 CVQ：K1 = 4096, b1 = 12
第二层 patch-wise：K2 = 65536, b2 = 16
```

测试 `768 x 512` 下：

```text
source_bits = 256 * 12 + 1536 * 16
            = 3072 + 24576
            = 27648 bits
```

压缩率为：

```text
compression_ratio =
    27648 / (0.5 * 1 * 3 * 768 * 512)
  = 0.046875
```

这个方案仍然低于 0.083，但比 `K1=128,K2=16` 的混合方案更接近目标码率，并且工程上更可能训练。

### 8.4 替代方案 D：修改 CVQ token 设计，让第一层不是只有 256 个 token

如果必须同码率比较，并且又希望引入 CVQ，建议不要让第一层纯粹只有 `C=256` 个 token。

可考虑：

```text
1. 分组 CVQ：每个 channel group 产生多个 token
2. 空间分块 CVQ：每个 channel 在若干空间块内量化
3. residual CVQ：第一层引入多级残差码本
4. layer-1 CVQ + layer-1 patch residual side information
```

核心目标是：

```text
提高 CVQ 分支 token 数，避免从 6144 token 直接降到 256 token。
```

只有这样才能在不爆炸码本大小的情况下接近或保持 `0.0833333333` 压缩率。

## 9. 当前执行状态

当前已完成：

```text
1. 核实原方案结构
2. 核实原方案压缩率
3. 推导 Baseline / Variant B / Variant C 的码本大小
4. 判断 B/C 在严格约束下不可训练
5. 生成本文档
```

当前未启动：

```text
Variant B
Variant C
```

未启动原因：

```text
严格满足 0.0833333333 时，所需码本极端巨大，不具备工程可训练性。
```

建议下一步：

```text
若坚持同码率 0.0833333333：
    需要重新设计 CVQ token 数，而不是第一层纯通道量化。

若坚持第一层 CVQ + 第二层 patch-wise：
    需要接受实际测试压缩率低于 0.0833333333。
```

