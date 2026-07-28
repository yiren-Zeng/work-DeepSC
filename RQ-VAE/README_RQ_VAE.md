# RQ-VAE 配置与运行说明

> **文档范围**
>
> 本文只描述本项目中 `quantizer_type=rq_ema` 的原始方案：
> **两尺度 U-Net + 官方风格 EMA Residual Quantization**。
>
> 本文不描述 `residual_simvq`、`ResidualSimVQQuantizer`、RQ-SimVQ、
> `ProjectedEmbedding` 或码本 projection 学习率。运行时请优先核对日志中的
> `Quantizer: rq_ema`，避免混用两个方案。

## 1. 项目位置与主要入口

项目根目录：

```text
/workspace/yi/work/RQ-VAE
```

原始 RQ-VAE 的主要文件如下：

| 用途 | 文件 |
|---|---|
| 正式训练入口 | `scripts/train/current/run_rq_ema_k4-2_d2-2_rate047.sh` |
| 无信道重建测试 | `scripts/eval/test_rq_ema_k4-2_d2-2_rate047_nochannel.sh` |
| 真实 LDPC + 调制链路测试 | `scripts/eval/test_rq_ema_k4-2_d2-2_rate047.sh` |
| Adaptive STOP 码率扫描 | `scripts/eval/test_rq_ema_k4-2_d2-2_rate047_adaptive.sh` |
| 通用训练程序 | `train.py` |
| 固定深度测试程序 | `test_real.py` |
| Adaptive 测试程序 | `test_adaptive.py` |
| 实验配置 | `config.py` |
| EMA-RQ 量化器 | `models/rq_ema_quantizer.py` |
| 总模型与信道接入 | `models/deepsc.py` |
| 编码器 / 解码器 | `models/semantic_encoder.py`、`models/semantic_decoder.py` |
| 损失函数 | `losses/deepsc_loss.py` |
| 训练调度 | `training/schedules.py` |

专用 shell 脚本会进入项目目录并激活 Conda 环境 `work`，因此推荐通过这些
脚本运行，不要只依赖 `config.py` 的通用默认值。

## 2. 一眼识别正确方案

本方案应同时满足以下标识：

| 项目 | 正确值 |
|---|---|
| 量化器配置 | `SIMVQ_QUANTIZER_TYPE=rq_ema` |
| Python 类 | `RQEMAQuantizer` |
| 码本更新 | EMA，无码本梯度 |
| 特征投影 | 无 |
| 实验名 | `quality_v2_B_larger_rate047_rq_ema_unet2_ds8x2_k4-2_d2-2` |
| 码本大小 | `[4, 2]` |
| 每尺度 RQ 深度 | `[2, 2]` |

如果日志中出现 `residual_simvq`、`ResidualSimVQQuantizer`、
`projection_lr` 或 `ProjectedEmbedding`，运行的就不是本文所述方案。

## 3. 模型结构

### 3.1 数据流

```text
RGB [B,3,256,256]
        │
        ▼
两尺度语义编码器
        ├── 浅尺度 F0 [B,256,32,32]
        │       └── EMA-RQ：K=4，depth=2
        │           indices [B,32,32,2]
        │
        └── 深尺度 F1 [B,512,16,16]
                └── EMA-RQ：K=2，depth=2
                    indices [B,16,16,2]
        │
        ▼
从深到浅的 U-Net 解码器
        │
        ▼
重建 RGB [B,3,256,256]
```

编码器输出的两个尺度都经过量化。解码器收到的是量化后的深层特征和量化后的
浅层 skip feature，不存在绕过量化器的原始特征旁路。

### 3.2 U-Net 精确配置

| 参数 | 值 |
|---|---|
| 实验阶段 | Stage B |
| 输入 / 输出通道 | RGB，`3 / 3` |
| 训练图像尺寸 | `256 × 256` |
| U-Net 深度 | `2` |
| Base channels | `128` |
| 两级下采样步幅 | `[8, 2]` |
| 两尺度特征维度 | `[256, 512]` |
| 两尺度空间尺寸 | `32 × 32`、`16 × 16` |
| 归一化 | GroupNorm，32 groups |
| 激活函数 | SiLU |
| 编码器残差块参数 | `6` |
| 解码器残差块参数 | `6` |
| 上采样 | bilinear，倍率按 `[2, 8]` 反向恢复 |
| Cascade downsample | 关闭 |
| Bottleneck attention | 关闭 |
| SwinIR enhance | 关闭 |
| Swin backbone | 关闭 |

编码器中的 `encoder_res_blocks=6` 不是每个尺度总共只有 6 个残差块：
每个 `DownSampleBlock` 在下采样前有 6 个残差块，下采样后还有 6 个残差块。
解码器的每个 `UpSampleBlock` 使用 6 个残差块。

## 4. EMA Residual Quantization

`models/rq_ema_quantizer.py` 由 Kakao Brain RQ-VAE 的 EMA/RQ 实现适配而来，
上游基准 commit 为
`341395e562ac347f5eb62db9f5f08b9f2cc42a60`。项目适配包括直接处理 NCHW
特征、监控统计以及独立的 adaptive 推理接口；固定深度训练仍使用官方风格的
EMA/RQ 核心。

### 4.1 每个尺度的量化过程

对一个尺度的输入特征 `z`：

1. 第一层从共享码本查找最近码字 `q1`。
2. 计算第一层残差 `r1 = z - q1`。
3. 第二层仍使用同一个码本量化 `r1`，得到 `q2`。
4. 固定深度输出为 `q1 + q2`。
5. 最终特征通过 straight-through estimator 返回，使重建梯度能传给编码器。

两个尺度各有一个独立的 `VQEmbedding`；同一尺度内的两个 RQ 深度共享同一个
码本对象。两个尺度之间不共享码本。

### 4.2 码本配置

| 参数 | 浅尺度 | 深尺度 |
|---|---:|---:|
| 特征通道 | 256 | 512 |
| 码本大小 K | 4 | 2 |
| RQ 深度 | 2 | 2 |
| 索引形状 | `[B,32,32,2]` | `[B,16,16,2]` |
| 索引取值 | `0..3` | `0..1` |

共同设置：

- 量化轴为 patch/token；
- EMA decay 为 `0.99`；
- EMA epsilon 为 `1e-5`；
- unused/dead code restart 开启；
- 每个码本额外保存一个 padding row，但最近邻搜索、EMA 统计和码率计算都不
  使用该 padding row；
- 码本参数 `requires_grad=False`，不进入优化器，由 EMA 更新；
- 不进行 projection、resize 或其他 SimVQ 特征变换。

每个深度的 commitment 都比较输入特征与“截至当前深度的累计量化结果”，一个
尺度返回两个深度 commitment 的均值。

## 5. 损失函数与优化器

固定深度 RQ-VAE 的训练损失为：

```text
L_total = MSE(x_hat, x) + 0.25 × (C_shallow + C_deep)
```

其中：

- `C_shallow` 是浅尺度两个 RQ 深度的 raw cumulative commitment 均值；
- `C_deep` 是深尺度两个 RQ 深度的 raw cumulative commitment 均值；
- 两尺度 loss weight 在整个训练过程中固定为 `[1, 1]`；
- MSE 权重为 `1.0`；
- MS-SSIM loss 权重为 `0.0`；
- LPIPS/VGG perceptual loss 权重为 `0.0`；
- 没有额外的可学习 codebook loss，码本由 EMA 更新。

PSNR 和 MS-SSIM 会在训练/验证阶段记录，但不参与反向传播，也不用于选择最佳
模型。`best_vq_deepsc.pth` 只按 validation reconstruction MSE 选择。

优化配置：

| 参数 | 值 |
|---|---|
| Optimizer | Adam |
| 主学习率 | `5e-5` |
| Adam betas | `(0.5, 0.999)` |
| Scheduler | StepLR |
| Step size | 100 epochs |
| Gamma | `0.5` |
| 梯度裁剪 | global norm `1.0` |
| AMP | 未启用 |
| 随机种子 | `42` |

`config.py` 中虽然有通用的 `CODEBOOK_PROJ_LR=2e-4`，但 `rq_ema` 没有
projection 参数，因此该学习率在本方案中不生效。

## 6. 固定码率计算

训练和普通测试的两个 RQ 深度全部发送，固定源端码率为：

```text
浅尺度：32 × 32 × 2 depths × log2(4) = 4096 bits/image
深尺度：16 × 16 × 2 depths × log2(2) =  512 bits/image
合计：                                      4608 bits/image
```

对 `256 × 256` 图像：

```text
source bpp = 4608 / (256 × 256) = 0.0703125
```

项目中的 RGB transmission ratio 定义为：

```text
transmission ratio = source_bpp / (LDPC rate × modulation bits × 3)
```

因此：

| 链路 | Transmission ratio |
|---|---:|
| LDPC 1/2 + BPSK | `0.04687500` |
| LDPC 1/2 + QPSK | `0.02343750` |
| LDPC 1/2 + 16QAM | `0.01171875` |

上述固定 `0.0703125 bpp` 是按定长索引计算的源端码率，不是熵编码后的实测文件
大小。Adaptive STOP 的理想熵码率另见第 11 节。

## 7. 数据集与预处理

正式训练脚本使用：

```text
训练集：/workspace/yi/work/Cars196/train_data
验证集：/workspace/yi/work/Cars196/val_data
测试集：/workspace/yi/work/Kodak-256-transform-resize
```

当前目录中检测到：

| 数据集 | 图像数 |
|---|---:|
| Cars196 train | 12,948 |
| Cars196 validation | 3,237 |
| Kodak-256 | 24 |

训练脚本会主动清除 `SIMVQ_TRAIN_RESIZE` 和 `SIMVQ_VAL_RESIZE`，从而固定使用
以下预处理：

- Train：`Resize(256) → RandomCrop(256) → ToTensor → Normalize(0.5,0.5)`；
- Validation：`Resize(256) → CenterCrop(256) → ToTensor → Normalize(0.5,0.5)`；
- 数值范围最终为 `[-1, 1]`。

当前默认训练分支没有 `RandomHorizontalFlip`。只有显式设置
`SIMVQ_TRAIN_RESIZE` 时才会进入另一套带水平翻转的通用分支，而正式
RQ-VAE 脚本会清除这个变量。

普通目录数据加载是**非递归**的，只读取根目录直接包含的
`.jpg`、`.jpeg`、`.png` 文件。若图像放在子目录中，程序会认为没有找到数据。

Kodak-256 测试脚本设置 `SIMVQ_TEST_NO_RESIZE=1`，因此直接测试已经准备好的
`256 × 256` 图像。此时 `config.py` 中通用的 `TEST_IMAGE_SIZE=(768,512)`
不会对图像执行 resize。

## 8. 训练阶段与信道课程

### 8.1 训练基本参数

| 参数 | 值 |
|---|---:|
| 默认 epoch 数 | 200 |
| Total batch size | 24 |
| Micro-batch size | 24 |
| 梯度累积步数 | 1 |
| DataLoader workers | 8 |
| Pin memory | 开启 |

若覆盖 batch size，代码使用：

```text
accumulation_steps = TOTAL_BATCH_SIZE // MICRO_BATCH_SIZE
```

因此 total batch 应不小于 micro-batch，最好能被 micro-batch 整除。

### 8.2 Skip dropout 与训练阶段

代码内部 epoch 从 0 开始，终端显示时加 1：

| 内部 epoch | 显示轮次 | 阶段 | 浅层 skip dropout |
|---|---|---|---|
| `0..19` | 1–20 | Phase 1 | `0.1` |
| `20..79` | 21–80 | Phase 2 | 从 `0.1` 线性降向 `0` |
| `80..199` | 81–200 | Phase 3 | `0` |

EMA-RQ 的两尺度量化损失权重在所有阶段都强制为 `[1,1]`，不随 Phase 变化。

### 8.3 训练信道

训练/验证期间使用的是 `FiniteBlocklengthChannel` 计算 BER 后进行 bit flip 的
近似，不是逐块执行真实 LDPC 编码和译码。bit flip 和整数索引本身不可微；
训练前向使用
`quantized_clean + (quantized_noisy - quantized_clean).detach()` 的 STE，
使反向梯度沿 clean quantized feature 返回。

- SNR 从 `[0,15] dB` 均匀采样；
- SNR `<4 dB` 时随机选 1 或 2 bits/symbol；
- `4≤SNR<8 dB` 时随机选 1、2 或 4 bits/symbol；
- SNR `≥8 dB` 时随机选 2 或 4 bits/symbol；
- 训练和验证 coding rate 均配置为 `0.5`；
- finite-blocklength 参数中的 coded block length 为 256 bits。

信道启用概率：

| 内部 epoch `e` | 显示轮次 | 信道概率 |
|---|---|---:|
| `e < 80` | 1–80 | `0` |
| `80 ≤ e < 120` | 81–120 | `(e-80)/40` |
| `e ≥ 120` | 121–200 | `1` |

注意显示轮次 81 的概率仍为 0；显示轮次 121 起才固定为 1。真实链路测试使用
Sionna 5G LDPC `k=128, n=256, rate=1/2`、AWGN 和指定调制方式，见第 10 节。

## 9. 环境

脚本使用 Conda 环境 `work`。当前已验证的主要版本：

```text
Python       3.10.20
PyTorch      2.7.1+cu118
torchvision  0.22.1+cu118
TensorFlow   2.21.0
Sionna       1.2.2
NumPy        2.2.6
Pillow       12.1.1
matplotlib   3.10.9
tensorboard  2.20.0
```

PyTorch 负责模型训练和推理；真实 LDPC 测试依赖 TensorFlow/Sionna；adaptive
曲线输出依赖 matplotlib。

可先验证环境：

```bash
cd /workspace/yi/work/RQ-VAE
eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 10. 固定深度模型的运行方法

### 10.1 从头训练

下面以物理 GPU 2 为例；运行前请用 `nvidia-smi` 选择空闲 GPU：

```bash
cd /workspace/yi/work/RQ-VAE

GPU_ID=2 \
SIMVQ_NUM_EPOCHS=200 \
SIMVQ_TOTAL_BATCH_SIZE=24 \
SIMVQ_MICRO_BATCH_SIZE=24 \
bash scripts/train/current/run_rq_ema_k4-2_d2-2_rate047.sh
```

`GPU_ID=2` 会设置 `CUDA_VISIBLE_DEVICES=2`，Python 内部看到的是逻辑
`cuda:0`，日志会打印它到物理 GPU 2 的映射。训练脚本本身的默认 GPU 是 3；
此前已经完成的正式 RQ-VAE 训练实际使用的是物理 GPU 2。

训练脚本强制：

```text
SIMVQ_RESUME=0
SIMVQ_PRETRAINED_CHECKPOINT 被清除
```

所以它是“从头训练”入口。即使在命令前写 `SIMVQ_RESUME=1`，也会被脚本内部
重新设回 0。

> **已有模型保护**
>
> 当前实验目录已经存在训练完成的 `best_vq_deepsc.pth` 和
> `last_checkpoint.pth`。直接重跑相同实验会从头开始，并在后续保存时覆盖同名
> checkpoint。若只是测试现有模型，不要执行训练命令；若确实要重新训练，应先
> 备份现有 checkpoint，或在脚本副本中同时修改 `SIMVQ_EXP_FAMILY` 和用于日志
> 展示的 `EXPERIMENT_NAME`，使其写入新的实验目录。

### 10.2 断点续训

专用训练脚本故意禁止 resume。若需要恢复同一个实验：

1. 在项目内部复制一份该训练脚本；
2. 将副本中的 `export SIMVQ_RESUME="0"` 改为 `"1"`；
3. 保持模型配置和实验名完全不变；
4. 确认
   `checkpoints/quality_v2_B_larger_rate047_rq_ema_unet2_ds8x2_k4-2_d2-2/last_checkpoint.pth`
   存在；
5. 运行脚本副本。

`SIMVQ_NUM_EPOCHS` 表示训练结束时的总 epoch 数，不是“额外再训练多少轮”。
恢复时会同时加载模型、optimizer、scheduler、best validation loss 和随机数
状态。

当前 `last_checkpoint.pth` 已经完成显示轮次 200。若仍以
`SIMVQ_NUM_EPOCHS=200` 恢复，`start_epoch` 会等于 200，训练循环会直接结束。
若希望从当前状态延长到总计 240 轮，应在运行修改后的 resume 脚本时显式设置：

```bash
GPU_ID=2 SIMVQ_NUM_EPOCHS=240 \
bash scripts/train/current/run_rq_ema_k4-2_d2-2_rate047_resume.sh
```

这里的 `_resume.sh` 指第 1 步创建并已将 `SIMVQ_RESUME` 改为 1 的脚本副本。

### 10.3 查看训练进度

训练日志：

```bash
tail -f experiments/logs/train_rq_ema_k4-2_d2-2_rate047-*.log
```

TensorBoard：

```bash
eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
tensorboard \
  --logdir experiments/tensorboard/quality_v2_B_larger_rate047_rq_ema_unet2_ds8x2_k4-2_d2-2 \
  --port 6006
```

每个 epoch 的标量写入：

```text
experiments/quality_v2_B_larger_rate047_rq_ema_unet2_ds8x2_k4-2_d2-2_epoch_metrics.csv
```

每 10 个 epoch 最多使用 20 个 validation batch 统计一次码本利用率，写入：

```text
experiments/quality_v2_B_larger_rate047_rq_ema_unet2_ds8x2_k4-2_d2-2_codebook_metrics.csv
```

### 10.4 无信道重建上界

选择空闲 GPU 后运行：

```bash
cd /workspace/yi/work/RQ-VAE
GPU_ID=3 \
bash scripts/eval/test_rq_ema_k4-2_d2-2_rate047_nochannel.sh
```

无参数时，脚本会自动补充正确 checkpoint 和 `--no-channel`。

若要自定义 JSON 输出，一旦传入任意参数，wrapper 就会把参数原样交给
`test_real.py`，不再由 wrapper 补 checkpoint 或 `--no-channel`。当前
`test_real.py` 仍会按 Config 推导默认 checkpoint，但为了让命令不依赖隐含
默认值，下面将 checkpoint 和 `--no-channel` 都显式写出：

```bash
GPU_ID=3 \
bash scripts/eval/test_rq_ema_k4-2_d2-2_rate047_nochannel.sh \
  --checkpoint checkpoints/quality_v2_B_larger_rate047_rq_ema_unet2_ds8x2_k4-2_d2-2/best_vq_deepsc.pth \
  --no-channel \
  --json-output experiments/rq_ema_k4-2_d2-2_rate047_kodak_nochannel_new.json
```

### 10.5 真实 LDPC + BPSK 曲线

```bash
cd /workspace/yi/work/RQ-VAE
GPU_ID=3 \
bash scripts/eval/test_rq_ema_k4-2_d2-2_rate047.sh
```

虽然脚本顶部旧注释写着“10 dB by default”，实际无参数行为由代码决定，是：

```text
BPSK，SNR = 0, 3, 6, 9, 12 dB
```

单独测试 BPSK 10 dB 并保存 JSON：

```bash
GPU_ID=3 \
bash scripts/eval/test_rq_ema_k4-2_d2-2_rate047.sh \
  --checkpoint checkpoints/quality_v2_B_larger_rate047_rq_ema_unet2_ds8x2_k4-2_d2-2/best_vq_deepsc.pth \
  --snrs 10 \
  --modulation bpsk \
  --json-output experiments/rq_ema_k4-2_d2-2_rate047_kodak_bpsk_snr10.json
```

QPSK 或 16QAM：

```bash
GPU_ID=3 \
bash scripts/eval/test_rq_ema_k4-2_d2-2_rate047.sh \
  --checkpoint checkpoints/quality_v2_B_larger_rate047_rq_ema_unet2_ds8x2_k4-2_d2-2/best_vq_deepsc.pth \
  --snrs 0 3 6 9 12 \
  --modulation qpsk \
  --json-output experiments/rq_ema_k4-2_d2-2_rate047_kodak_qpsk_snr_curve.json
```

将 `qpsk` 改成 `16qam` 即可测试 16QAM。固定链路 wrapper 只要收到任意自定义
参数，就不会由 wrapper 自动补其他默认参数，因此始终建议显式写
`--checkpoint`、`--snrs` 和 `--modulation`。

## 11. 可选：Adaptive STOP 推理

Adaptive STOP 是原始 `rq_ema` checkpoint 的独立评估接口，不改变固定深度
训练，也不需要重新训练。它只支持每尺度 RQ depth 为 2 的模型：

1. 第一层正常量化所有 token；
2. 计算每个 token 第一层之后的 channel-mean squared residual；
3. 当误差小于阈值时 STOP；
4. 第二层只量化未 STOP 的 token；
5. 接收端通过 STOP/active 信息恢复第二层是否存在。

默认扫描：

```bash
cd /workspace/yi/work/RQ-VAE
GPU_ID=2 \
bash scripts/eval/test_rq_ema_k4-2_d2-2_rate047_adaptive.sh
```

默认的第二层目标 active ratio 为：

```text
1.00, 0.75, 0.50, 0.30, 0.20, 0.10, 0.00
```

扫描 100% 到 0%、步长 10%：

```bash
GPU_ID=2 \
bash scripts/eval/test_rq_ema_k4-2_d2-2_rate047_adaptive.sh \
  --target-active-rates 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0 \
  --json-output experiments/adaptive_eval/quality_v2_B_larger_rate047_rq_ema_unet2_ds8x2_k4-2_d2-2/adaptive_scan_0to100_step10_new.json \
  --csv-output experiments/adaptive_eval/quality_v2_B_larger_rate047_rq_ema_unet2_ds8x2_k4-2_d2-2/adaptive_scan_0to100_step10_new.csv \
  --plot-output experiments/adaptive_eval/quality_v2_B_larger_rate047_rq_ema_unet2_ds8x2_k4-2_d2-2/adaptive_scan_0to100_step10_new.png
```

快速冒烟测试可加 `--max-images 2`。无 matplotlib 时可加 `--no-plot`。

输出同时给出：

- `ideal_bpp`：始终发送的第一层定长索引码率，加第二层
  `{STOP, 0, ..., K-1}` 联合符号的零阶理想熵；
- `exact_raw_bpp`：第一层定长索引码率，加 1 bit/token 的第二层活动 mask，
  再加 active token 的第二层定长索引；
- `dense_fixed_bpp`：两层全部发送时的固定码率；
- PSNR 和 MS-SSIM。

对应代码中的精确关系是：

```text
ideal_bpp
  = first_stage_fixed_bpp + joint_stop_index_entropy_bpp

exact_raw_bpp
  = first_stage_fixed_bpp + raw_mask_bpp + raw_active_index_bpp
```

理想熵统计不包含 shape、header、阈值、概率表等信令开销，因此不等于实际压缩
文件大小。

## 12. Checkpoint 与现有结果

### 12.1 Checkpoint

```text
最佳模型：
checkpoints/quality_v2_B_larger_rate047_rq_ema_unet2_ds8x2_k4-2_d2-2/best_vq_deepsc.pth

完整续训状态：
checkpoints/quality_v2_B_larger_rate047_rq_ema_unet2_ds8x2_k4-2_d2-2/last_checkpoint.pth
```

- `best_vq_deepsc.pth` 约 420 MiB，包含模型、epoch、best loss 和自描述模型配置；
- `last_checkpoint.pth` 约 1.3 GiB，额外包含 optimizer、scheduler 和 RNG 状态；
- 最佳模型 SHA256：
  `442f71c2642eb76f2609b097177e315838eba2e69ed6ceb632226a6cc476a772`。

检查文件是否仍为当前最佳模型：

```bash
sha256sum \
  checkpoints/quality_v2_B_larger_rate047_rq_ema_unet2_ds8x2_k4-2_d2-2/best_vq_deepsc.pth
```

### 12.2 已完成训练的最佳点

Checkpoint 内部 epoch 为 0-based `178`，即界面显示 Epoch 179：

| 指标 | 值 |
|---|---:|
| Validation reconstruction MSE | `0.01992665` |
| Validation PSNR | `23.04860773 dB` |
| Validation MS-SSIM | `0.90685977` |

该 validation 走训练程序的验证路径，并服从当时的信道课程；不要与下面的
Kodak 无信道结果直接当作同一数据集、同一条件比较。

### 12.3 Kodak-256 已记录结果

无信道结果文件：

```text
experiments/rq_ema_k4-2_d2-2_rate047_kodak_nochannel.json
```

| 条件 | PSNR | MS-SSIM |
|---|---:|---:|
| No channel | `23.5527 dB` | `0.871867` |

BPSK 曲线文件：

```text
experiments/rq_ema_k4-2_d2-2_rate047_kodak_bpsk_snr_curve.json
```

| SNR | PSNR | MS-SSIM |
|---:|---:|---:|
| 0 dB | `23.5479 dB` | `0.871562` |
| 3 dB | `23.5527 dB` | `0.871867` |
| 6 dB | `23.5527 dB` | `0.871867` |
| 9 dB | `23.5527 dB` | `0.871867` |
| 12 dB | `23.5527 dB` | `0.871867` |

这些是已有记录，不是每次阅读本文时重新运行得到的实时结果。

Adaptive 已有输出位于：

```text
experiments/adaptive_eval/quality_v2_B_larger_rate047_rq_ema_unet2_ds8x2_k4-2_d2-2/
```

其中包括 JSON、CSV、BPP–PSNR–MS-SSIM 曲线以及 0%–100% active ratio 图。

## 13. 测试

RQ-EMA 相关 CPU 测试文件：

```text
tests/test_rq_ema_quantizer.py
tests/test_rq_ema_integration.py
tests/test_rq_ema_project_contract.py
tests/test_rq_ema_adaptive.py
```

当前 Conda 环境没有安装 pytest。以下命令只运行上面四个 RQ-EMA 模块，
不会运行 RQ-SimVQ 测试：

```bash
cd /workspace/yi/work/RQ-VAE
eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work

python - <<'PY'
import importlib
import inspect

modules = (
    "tests.test_rq_ema_quantizer",
    "tests.test_rq_ema_integration",
    "tests.test_rq_ema_project_contract",
    "tests.test_rq_ema_adaptive",
)
passed = 0
for module_name in modules:
    module = importlib.import_module(module_name)
    for name, function in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("test_") and not inspect.signature(function).parameters:
            function()
            print("PASS", f"{module_name}::{name}")
            passed += 1
print(f"{passed} RQ-EMA tests passed")
PY
```

项目级命令 `python tests/run_cpu_tests.py` 也能运行，但它会同时执行其他量化器
的回归测试，因此不是“只测 RQ-VAE”的入口。

## 14. 常见问题

### 日志里为什么显示 `cuda:0`，明明选择了 GPU 2？

因为 `CUDA_VISIBLE_DEVICES=2` 后，进程只看到一张卡，并把物理 GPU 2 映射为
逻辑 `cuda:0`。以日志中的
`logical cuda:0 -> physical GPU 2` 为准。

### 为什么前缀设置 `SIMVQ_RESUME=1` 没有恢复训练？

专用训练脚本内部强制写了 `SIMVQ_RESUME=0`，会覆盖调用者传入的值。请按
第 10.2 节复制脚本并修改副本。

### 为什么给无信道脚本增加一个自定义参数后，却跑成了有信道测试？

两个固定测试 wrapper 都只在“完全无参数”时添加默认参数；一旦收到任何参数，
就全部原样传给 `test_real.py`。`test_real.py` 可以从 Config 推导默认
checkpoint，但不会替无信道 wrapper 自动补 `--no-channel`。因此自定义命令
应显式给出 checkpoint，并按需要给出 `--no-channel`、`--snrs` 和
`--modulation`。

### 为什么训练日志中的信道不是实际 LDPC？

训练为了速度使用 finite-blocklength BER + bit flip 近似。只有
`test_real.py` 的有信道路径执行 Sionna 5G LDPC 编解码和 AWGN 调制链路。

### 如何确认没有误用 RQ-SimVQ？

检查训练日志和 checkpoint metadata：

```text
quantizer_type = rq_ema
num_embeddings_list = [4, 2]
rq_depth_list = [2, 2]
```

同时，optimizer 中不应存在 codebook projection 参数。
