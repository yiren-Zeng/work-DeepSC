# 0.042 压缩率 CVQ/SimVQ 三方案训练配置

日期：2026-06-08

本文档记录从原方案 `quality_v2_B_larger_cb128-16_unet2_ds8x2_k128-16` 修改得到的三组训练方案：

```text
A / Baseline：SimVQ patch-wise
B：SimVQ + CVQ
C：SimVQ + CVQ + nested channel dropout
```

共同约束：

```text
训练尺寸：256 x 256
测试尺寸：768 x 512
信道编码：1/2 LDPC
调制方式：BPSK
U-Net 层数：2
下采样倍率：[8, 2]
网络主干：沿用 quality_v2_B_larger_cb128-16
损失函数：沿用原方案，不额外改变
```

## 1. 压缩率计算口径

本文档中的压缩率带上信道编码率和调制阶数。

```text
compression_ratio =
    source_bits / (R_c * log2(M) * 3 * H * W)
```

其中：

```text
R_c = 0.5
BPSK: log2(M) = 1
测试尺寸: H x W = 768 x 512
RGB values = 3 * 768 * 512
```

所以：

```text
compression_ratio =
    source_bits / (0.5 * 1 * 3 * 768 * 512)
```

对于 B/C 方案，用户指定：

```text
K1 = 8192  => b1 = 13
K2 = 16384 => b2 = 14
```

B/C 的 token 数为：

```text
第一层 CVQ token 数 = C1 = 256
第二层 patch-wise token 数 = 48 * 32 = 1536
```

因此：

```text
source_bits = 256 * 13 + 1536 * 14
            = 3328 + 21504
            = 24832 bits
```

实际测试压缩率：

```text
compression_ratio =
    24832 / (0.5 * 1 * 3 * 768 * 512)
  = 0.0421006944
```

即：

```text
B/C 实际压缩率约为 0.04210
```

## 2. A / Baseline：SimVQ Patch-Wise

### 2.1 方案定义

```text
第一层：patch-wise SimVQ
第二层：patch-wise SimVQ
```

其余保持原方案：

```text
SIMVQ_EXPERIMENT_STAGE=B
SIMVQ_UNET_DEPTH=2
SIMVQ_DOWNSAMPLE_STRIDES=8,2
SIMVQ_BASE_CHANNELS=128
SIMVQ_ENCODER_RES_BLOCKS=4
SIMVQ_DECODER_RES_BLOCKS=4
SIMVQ_QUANTIZER_TYPE=simvq
SIMVQ_QUANTIZER_AXIS_LIST=patch,patch
```

### 2.2 码本选择

patch-wise 时，测试 `768 x 512` 下 token 数：

```text
第一层 token 数 = 96 * 64 = 6144
第二层 token 数 = 48 * 32 = 1536
```

为了接近 `0.042`，可选的整数 bit 解里，最接近且不超过 `0.04210` 的是：

```text
b1 = 3, b2 = 4
K1 = 2^3 = 8
K2 = 2^4 = 16
```

source bits：

```text
source_bits = 6144 * 3 + 1536 * 4
            = 18432 + 6144
            = 24576 bits
```

实际压缩率：

```text
compression_ratio =
    24576 / (0.5 * 1 * 3 * 768 * 512)
  = 0.0416666667
```

说明：

```text
patch-wise 方案由于 bit 必须是整数，无法精确等于 0.04210。
K1=8,K2=16 是最接近且不超过 0.04210 的可训练配置。
```

### 2.3 训练脚本

```text
run_exp12_rate042_A_patch_cb8_16.sh
```

默认 GPU：

```text
GPU 0
```

实验名：

```text
quality_v2_B_larger_rate042_A_patch_cb8-16_unet2_ds8x2_k8-16
```

训练日志：

```text
experiments/logs/train_exp12_rate042_A_patch_cb8-16-*.log
```

指标文件：

```text
experiments/quality_v2_B_larger_rate042_A_patch_cb8-16_unet2_ds8x2_k8-16_epoch_metrics.csv
experiments/quality_v2_B_larger_rate042_A_patch_cb8-16_unet2_ds8x2_k8-16_codebook_metrics.csv
```

checkpoint：

```text
checkpoints/quality_v2_B_larger_rate042_A_patch_cb8-16_unet2_ds8x2_k8-16/
```

## 3. Variant B：SimVQ + CVQ

### 3.1 方案定义

```text
第一层：channel-wise SimVQ / CVQ
第二层：patch-wise SimVQ
```

配置：

```text
SIMVQ_QUANTIZER_TYPE=simvq
SIMVQ_QUANTIZER_AXIS_LIST=channel,patch
SIMVQ_CVQ_CODEWORD_SHAPES=32x32,patch
SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA=0.0
```

第一层 CVQ 的含义：

```text
输入特征: [B, 256, H/8, W/8]
每个 channel 是一个 token
训练 256x256 时，第一层特征空间大小为 32x32
因此第一层 CVQ codeword dim = 32 * 32 = 1024
```

测试 `768 x 512` 时：

```text
第一层特征空间大小为 96x64
模型会先将每个 channel map 插值到 32x32 进行码本查找
再将量化后的 channel map 插值回 96x64 输入解码器
```

这个设计对应 CVQ 论文里的 variable-resolution 思路：保持固定 codeword size，同时支持不同测试分辨率。

### 3.2 码本大小

用户指定：

```text
K1 = 8192
K2 = 16384
```

对应：

```text
b1 = log2(8192) = 13
b2 = log2(16384) = 14
```

测试压缩率：

```text
source_bits = 256 * 13 + 1536 * 14
            = 24832 bits

compression_ratio = 0.0421006944
```

### 3.3 训练脚本

```text
run_exp13_rate042_B_hybridcvq_cb8192_16384.sh
```

默认 GPU：

```text
GPU 1
```

实验名：

```text
quality_v2_B_larger_rate042_B_hybridcvq_cb8192-16384_unet2_ds8x2_k8192-16384
```

训练日志：

```text
experiments/logs/train_exp13_rate042_B_hybridcvq_cb8192-16384-*.log
```

指标文件：

```text
experiments/quality_v2_B_larger_rate042_B_hybridcvq_cb8192-16384_unet2_ds8x2_k8192-16384_epoch_metrics.csv
experiments/quality_v2_B_larger_rate042_B_hybridcvq_cb8192-16384_unet2_ds8x2_k8192-16384_codebook_metrics.csv
```

checkpoint：

```text
checkpoints/quality_v2_B_larger_rate042_B_hybridcvq_cb8192-16384_unet2_ds8x2_k8192-16384/
```

## 4. Variant C：SimVQ + CVQ + Nested Channel Dropout

### 4.1 方案定义

```text
第一层：channel-wise SimVQ / CVQ
第二层：patch-wise SimVQ
额外机制：nested channel dropout
```

配置：

```text
SIMVQ_QUANTIZER_TYPE=simvq
SIMVQ_QUANTIZER_AXIS_LIST=channel,patch
SIMVQ_CVQ_CODEWORD_SHAPES=32x32,patch
SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA=0.25
```

Nested channel dropout 的训练行为：

```text
每次训练迭代中，以 alpha=0.25 的概率启用通道截断。
若启用，则随机采样 c_keep ∈ [1, C]。
只保留前 c_keep 个 channel，后面的 channel 置零。
```

目的：

```text
促使前面的 channel 学习更重要的全局结构、颜色和主体信息；
促使后面的 channel 学习纹理、高频和细节补充；
形成更适合渐进式传输的通道顺序。
```

注意：

```text
nested channel dropout 不改变 token 数，也不改变压缩率。
```

### 4.2 码本大小

与 Variant B 相同：

```text
K1 = 8192
K2 = 16384
b1 = 13
b2 = 14
```

测试压缩率：

```text
source_bits = 256 * 13 + 1536 * 14
            = 24832 bits

compression_ratio = 0.0421006944
```

### 4.3 训练脚本

```text
run_exp14_rate042_C_hybridcvq_nested_cb8192_16384.sh
```

默认 GPU：

```text
GPU 2
```

实验名：

```text
quality_v2_B_larger_rate042_C_hybridcvq_nested_cb8192-16384_unet2_ds8x2_k8192-16384
```

训练日志：

```text
experiments/logs/train_exp14_rate042_C_hybridcvq_nested_cb8192-16384-*.log
```

指标文件：

```text
experiments/quality_v2_B_larger_rate042_C_hybridcvq_nested_cb8192-16384_unet2_ds8x2_k8192-16384_epoch_metrics.csv
experiments/quality_v2_B_larger_rate042_C_hybridcvq_nested_cb8192-16384_unet2_ds8x2_k8192-16384_codebook_metrics.csv
```

checkpoint：

```text
checkpoints/quality_v2_B_larger_rate042_C_hybridcvq_nested_cb8192-16384_unet2_ds8x2_k8192-16384/
```

## 5. 代码改动说明

本次为了支持 B/C，增加了以下能力：

```text
1. 支持逐层量化轴：
   SIMVQ_QUANTIZER_AXIS_LIST=patch,patch
   SIMVQ_QUANTIZER_AXIS_LIST=channel,patch

2. 新增 ChannelwiseVectorQuantizer：
   每个 channel map 作为一个 token。
   支持固定 32x32 codeword，并在测试分辨率变化时自动插值。

3. 支持 nested channel dropout：
   SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA=0.25

4. 修改 bit_utils：
   支持 CVQ 的 [B,C] 索引形状。

5. 修改评估重建逻辑：
   forward_test 会返回 feature_shapes；
   reconstruct_from_indices 会根据 feature_shapes 恢复 CVQ 层空间尺寸。

6. 修改压缩率估算：
   patch-wise 使用空间 token 数；
   channel-wise 使用 channel token 数；
   额外打印测试尺寸下的真实传输压缩率。
```

## 6. Smoke Test 结果

已完成随机输入 smoke test。

对于 B/C 混合 CVQ 配置：

```text
输入: [1, 3, 256, 256]
索引形状: [(1, 256), (1, 16, 16)]
feature_shapes: [(32, 32), (16, 16)]
重建输出: [1, 3, 256, 256]

输入: [1, 3, 768, 512]
索引形状: [(1, 256), (1, 48, 32)]
feature_shapes: [(96, 64), (48, 32)]
重建输出: [1, 3, 768, 512]
```

说明：

```text
CVQ 混合方案已经可以同时支持训练分辨率 256x256 和测试分辨率 768x512。
```

## 7. 启动命令

如果需要手动启动：

```bash
cd /workspace/yi/work/Simvq-dc-64-Multi-pro-2ceng

./run_exp12_rate042_A_patch_cb8_16.sh
./run_exp13_rate042_B_hybridcvq_cb8192_16384.sh
./run_exp14_rate042_C_hybridcvq_nested_cb8192_16384.sh
```

若要指定 GPU：

```bash
GPU_ID=0 ./run_exp12_rate042_A_patch_cb8_16.sh
GPU_ID=1 ./run_exp13_rate042_B_hybridcvq_cb8192_16384.sh
GPU_ID=2 ./run_exp14_rate042_C_hybridcvq_nested_cb8192_16384.sh
```

## 8. 本次实际启动记录

本次最终使用 `screen` 独立会话启动，避免普通后台进程被当前执行环境清理。

### 8.1 A / Baseline

```text
screen 会话: rate042_A
GPU: 0
run_id: exp12_rate042_A_patch_cb8-16-20260608-190010
日志: experiments/logs/train_exp12_rate042_A_patch_cb8-16-20260608-190010.log
实验名: quality_v2_B_larger_rate042_A_patch_cb8-16_unet2_ds8x2_k8-16
```

启动后已确认进入训练：

```text
Epoch [1/200], Step [1/540]
Epoch [1/200], Step [11/540]
Epoch [1/200], Step [21/540]
Epoch [1/200], Step [31/540]
```

### 8.2 Variant B

```text
screen 会话: rate042_B
GPU: 1
run_id: exp13_rate042_B_hybridcvq_cb8192-16384-20260608-190010
日志: experiments/logs/train_exp13_rate042_B_hybridcvq_cb8192-16384-20260608-190010.log
实验名: quality_v2_B_larger_rate042_B_hybridcvq_cb8192-16384_unet2_ds8x2_k8192-16384
```

启动后已确认进入训练：

```text
Epoch [1/200], Step [1/540]
Epoch [1/200], Step [11/540]
Epoch [1/200], Step [21/540]
```

### 8.3 Variant C

```text
screen 会话: rate042_C
GPU: 2
run_id: exp14_rate042_C_hybridcvq_nested_cb8192-16384-20260608-190010
日志: experiments/logs/train_exp14_rate042_C_hybridcvq_nested_cb8192-16384-20260608-190010.log
实验名: quality_v2_B_larger_rate042_C_hybridcvq_nested_cb8192-16384_unet2_ds8x2_k8192-16384
```

启动后已确认进入训练：

```text
Epoch [1/200], Step [1/540]
Epoch [1/200], Step [11/540]
Epoch [1/200], Step [21/540]
```

### 8.4 运行状态检查命令

查看 screen 会话：

```bash
screen -ls
```

查看 GPU：

```bash
nvidia-smi
```

查看日志：

```bash
tail -f experiments/logs/train_exp12_rate042_A_patch_cb8-16-20260608-190010.log
tail -f experiments/logs/train_exp13_rate042_B_hybridcvq_cb8192-16384-20260608-190010.log
tail -f experiments/logs/train_exp14_rate042_C_hybridcvq_nested_cb8192-16384-20260608-190010.log
```

重新进入会话：

```bash
screen -r rate042_A
screen -r rate042_B
screen -r rate042_C
```

从会话中安全退出但不停止训练：

```text
Ctrl-a 然后按 d
```
