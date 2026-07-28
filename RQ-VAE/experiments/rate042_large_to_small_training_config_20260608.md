# 大到小码本 CVQ/SimVQ 三方案训练配置

日期：2026-06-08

本文档记录用户更新后的三组训练方案。核心变化是：码本改成“第一层大、第二层小”，因为第一层是高分辨率层，对重建图像的共享贡献更大。

## 1. 共同设置

三种方案都基于原始方案：

```text
quality_v2_B_larger_cb128-16_unet2_ds8x2_k128-16
```

共同保持：

```text
训练尺寸：256 x 256
测试尺寸：768 x 512
信道编码：1/2 LDPC
调制方式：BPSK
U-Net 层数：2
下采样倍率：[8, 2]
BASE_CHANNELS：128
Encoder residual blocks：4
Decoder residual blocks：4
量化器：SimVQ
损失函数：保持原方案不变
```

压缩率计算口径：

```text
compression_ratio =
    source_bits / (R_c * log2(M) * 3 * H * W)
```

其中：

```text
R_c = 0.5
BPSK: log2(M) = 1
测试尺寸: H x W = 768 x 512
```

所以：

```text
compression_ratio =
    source_bits / (0.5 * 1 * 3 * 768 * 512)
```

## 2. A / Baseline：Patch-Wise SimVQ

### 2.1 方案定义

```text
第一层：patch-wise SimVQ
第二层：patch-wise SimVQ
```

### 2.2 码本大小

用户指定：

```text
K1 = 16
K2 = 2
```

对应 index bit：

```text
b1 = log2(16) = 4
b2 = log2(2)  = 1
```

测试 `768 x 512` 下：

```text
第一层 token 数 = 96 * 64 = 6144
第二层 token 数 = 48 * 32 = 1536
```

source bits：

```text
source_bits = 6144 * 4 + 1536 * 1
            = 24576 + 1536
            = 26112 bits
```

实际传输压缩率：

```text
compression_ratio =
    26112 / (0.5 * 1 * 3 * 768 * 512)
  = 0.0442708333
```

### 2.3 训练信息

脚本：

```text
run_exp12_rate044_A_patch_cb16_2.sh
```

默认 GPU：

```text
GPU 0
```

实验名：

```text
quality_v2_B_larger_rate044_A_patch_cb16-2_unet2_ds8x2_k16-2
```

日志：

```text
experiments/logs/train_exp12_rate044_A_patch_cb16-2-*.log
```

checkpoint：

```text
checkpoints/quality_v2_B_larger_rate044_A_patch_cb16-2_unet2_ds8x2_k16-2/
```

## 3. Variant B：SimVQ + CVQ

### 3.1 方案定义

```text
第一层：channel-wise CVQ / SimVQ
第二层：patch-wise SimVQ
```

第一层 CVQ 设置：

```text
第一层特征通道数：256
训练时第一层空间尺寸：32 x 32
CVQ codeword shape：32 x 32
CVQ codeword dim：1024
```

测试 `768 x 512` 时：

```text
第一层特征空间尺寸：96 x 64
量化前插值到 32 x 32 查码本
量化后插值回 96 x 64 输入解码器
```

### 3.2 码本大小

用户指定：

```text
K1 = 65536
K2 = 8192
```

对应 index bit：

```text
b1 = log2(65536) = 16
b2 = log2(8192)  = 13
```

测试 `768 x 512` 下：

```text
第一层 CVQ token 数 = 256
第二层 patch-wise token 数 = 48 * 32 = 1536
```

source bits：

```text
source_bits = 256 * 16 + 1536 * 13
            = 4096 + 19968
            = 24064 bits
```

实际传输压缩率：

```text
compression_ratio =
    24064 / (0.5 * 1 * 3 * 768 * 512)
  = 0.0407986111
```

说明：

```text
该方案低于 0.04210，但满足第一层码本大于第二层码本，且工程上可训练。
```

### 3.3 训练信息

脚本：

```text
run_exp13_rate041_B_hybridcvq_cb65536_8192.sh
```

默认 GPU：

```text
GPU 1
```

实验名：

```text
quality_v2_B_larger_rate041_B_hybridcvq_cb65536-8192_unet2_ds8x2_k65536-8192
```

日志：

```text
experiments/logs/train_exp13_rate041_B_hybridcvq_cb65536-8192-*.log
```

checkpoint：

```text
checkpoints/quality_v2_B_larger_rate041_B_hybridcvq_cb65536-8192_unet2_ds8x2_k65536-8192/
```

## 4. Variant C：SimVQ + CVQ + Nested Channel Dropout

### 4.1 方案定义

```text
第一层：channel-wise CVQ / SimVQ
第二层：patch-wise SimVQ
额外机制：nested channel dropout
```

Nested channel dropout：

```text
alpha = 0.25
训练时以 25% 概率启用通道截断
随机采样 c_keep，只保留前 c_keep 个 channel
后续 channel 置零
```

目的：

```text
让前面的 channel 更倾向于承载全局结构、颜色、主体信息；
让后面的 channel 更倾向于承载纹理和细节；
更适合渐进式传输。
```

### 4.2 码本大小

与 Variant B 相同：

```text
K1 = 65536
K2 = 8192
b1 = 16
b2 = 13
```

source bits：

```text
source_bits = 256 * 16 + 1536 * 13
            = 24064 bits
```

实际传输压缩率：

```text
compression_ratio = 0.0407986111
```

### 4.3 训练信息

脚本：

```text
run_exp14_rate041_C_hybridcvq_nested_cb65536_8192.sh
```

默认 GPU：

```text
GPU 2
```

实验名：

```text
quality_v2_B_larger_rate041_C_hybridcvq_nested_cb65536-8192_unet2_ds8x2_k65536-8192
```

日志：

```text
experiments/logs/train_exp14_rate041_C_hybridcvq_nested_cb65536-8192-*.log
```

checkpoint：

```text
checkpoints/quality_v2_B_larger_rate041_C_hybridcvq_nested_cb65536-8192_unet2_ds8x2_k65536-8192/
```

## 5. 启动命令

```bash
cd /workspace/yi/work/Simvq-dc-64-Multi-pro-2ceng

GPU_ID=0 ./run_exp12_rate044_A_patch_cb16_2.sh
GPU_ID=1 ./run_exp13_rate041_B_hybridcvq_cb65536_8192.sh
GPU_ID=2 ./run_exp14_rate041_C_hybridcvq_nested_cb65536_8192.sh
```

实际训练将使用 `screen` 会话运行，便于长时间后台训练和断开终端后继续执行。

## 6. 本次启动记录

本次启动使用以下 screen 会话：

```text
A: rate044_A
B: rate041_B
C: rate041_C
```

实际 run 记录：

```text
A:
  run_id: exp12_rate044_A_patch_cb16-2-20260608-191100
  GPU: 0
  日志: experiments/logs/train_exp12_rate044_A_patch_cb16-2-20260608-191100.log
  已确认进入训练: Epoch [1/200], Step [31/540]

B:
  run_id: exp13_rate041_B_hybridcvq_cb65536-8192-20260608-191100
  GPU: 1
  日志: experiments/logs/train_exp13_rate041_B_hybridcvq_cb65536-8192-20260608-191100.log
  已确认进入训练: Epoch [1/200], Step [21/540]

C:
  run_id: exp14_rate041_C_hybridcvq_nested_cb65536-8192-20260608-191100
  GPU: 2
  日志: experiments/logs/train_exp14_rate041_C_hybridcvq_nested_cb65536-8192-20260608-191100.log
  已确认进入训练: Epoch [1/200], Step [21/540]
```

查看状态：

```bash
screen -ls
nvidia-smi
```

查看日志：

```bash
tail -f experiments/logs/train_exp12_rate044_A_patch_cb16-2-20260608-*.log
tail -f experiments/logs/train_exp13_rate041_B_hybridcvq_cb65536-8192-20260608-*.log
tail -f experiments/logs/train_exp14_rate041_C_hybridcvq_nested_cb65536-8192-20260608-*.log
```
