# 项目文件结构说明

生成时间：2026-06-09  
项目路径：`/workspace/yi/work/Simvq-dc-64-Multi-pro-2ceng`

这份文档用于说明当前项目中主要文件和目录的作用。重点覆盖源码、训练/测试入口、实验脚本、结果文件和容易混淆的产物目录。

## 1. 当前 A/B/C Kodak-256 测试结果

测试条件：

```text
测试集：/workspace/yi/work/Kodak-256-transform-resize
测试尺寸：256 x 256
是否 resize：不 resize，SIMVQ_TEST_NO_RESIZE=1
信道编码：LDPC R=1/2
调制方式：BPSK
SNR：0 dB
测试图像数：24
```

当前结果：

| 方案 | checkpoint best epoch | 码本 | 测试压缩率，256x256 口径 | MS-SSIM | PSNR |
|---|---:|---:|---:|---:|---:|
| A: patch-wise SimVQ | 153 | `K1=16,K2=2` | `0.0442708` | `0.8956` | `24.2625 dB` |
| B: SimVQ + CVQ | 112 | `K1=65536,K2=8192` | `0.0755208` | `0.9045` | `25.1364 dB` |
| C: SimVQ + CVQ + nested dropout | 101 | `K1=65536,K2=8192` | `0.0755208` | `0.9029` | `25.1595 dB` |

结果文件：

- `experiments/eval_best_20260609_kodak256_snr0/A_patch_rate044_best_epoch153_snr0_ldpc12_bpsk_noresize.json`
- `experiments/eval_best_20260609_kodak256_snr0/B_hybrid_rate041_best_epoch112_snr0_ldpc12_bpsk_noresize.json`
- `experiments/eval_best_20260609_kodak256_snr0/C_hybrid_nested_rate041_best_epoch101_snr0_ldpc12_bpsk_noresize.json`

客观结论：

```text
在 256x256 no-resize 测试下，B/C 明显优于 A。
这和 768x512 测试下 A 更强的结论并不矛盾：
B/C 在 256x256 下没有 CVQ 尺度错配，并且实际压缩率从 768x512 下的约 0.04080 提高到 0.07552。
```

## 2. 根目录入口文件

### `config.py`

全局配置中心。大部分实验参数都从这里读取，也支持环境变量覆盖。

负责内容：

- 实验名 `EXPERIMENT_NAME`
- U-Net 层数、下采样倍率、base channels
- 码本大小 `NUM_EMBEDDINGS_LIST`
- 量化器类型 `QUANTIZER_TYPE`
- patch-wise / channel-wise 量化轴 `QUANTIZER_AXIS_LIST`
- CVQ codeword shape
- nested channel dropout alpha
- 训练/验证/测试数据路径
- batch size、epoch 数、学习率
- LDPC 码率、SNR 范围、block length
- checkpoint、CSV、TensorBoard 路径
- 压缩率估算

注意：

- 默认不会固定某个实验，具体实验通常由 `run_exp*.sh` 通过环境变量指定。
- 我新增了 `SIMVQ_CHANNEL_PROB_START_EPOCH` 和 `SIMVQ_CHANNEL_PROB_END_EPOCH` 两个环境变量入口，默认仍是 `80` 和 `120`，所以不设置时不改变原行为。

### `train.py`

主训练入口。

流程：

1. 读取 `Config`
2. 构建 `DeepSC`
3. 可选加载预训练 checkpoint
4. 构建 train/val dataloader
5. 构建 `DeepSCLoss`
6. 训练每个 epoch
7. 验证集评估 val recon loss
8. 保存 `best_vq_deepsc.pth`
9. 保存 `last_checkpoint.pth`
10. 写入 epoch metrics CSV 和 codebook metrics CSV

核心产物：

- `checkpoints/<experiment_name>/best_vq_deepsc.pth`
- `checkpoints/<experiment_name>/last_checkpoint.pth`
- `experiments/<experiment_name>_epoch_metrics.csv`
- `experiments/<experiment_name>_codebook_metrics.csv`

### `test_real.py`

主测试入口，用于 no-channel 或真实链路评测。

支持：

- `--no-channel`：只测试源重建上界，不加 LDPC/BPSK 信道。
- `--snrs 0 3 6 ...`：指定 SNR。
- `--modulation bpsk/qpsk`：指定调制方式。
- `--json-output`：保存测试结果 JSON。

真实链路流程：

```text
图像 -> encoder/quantizer 得到 index
index -> bits
LDPC encode
BPSK/QPSK modulation
AWGN channel
LLR
LDPC decode
bits -> index
decoder reconstruct
MS-SSIM / PSNR
```

### `run_train.sh`

较早的通用训练启动脚本。现在主要实验更多使用 `run_exp*.sh`，这个脚本可以视为旧入口或通用入口。

## 3. 当前关键实验脚本

### `run_exp9_larger_cb128_16.sh`

原始重点基准：

```text
quality_v2_B_larger_cb128-16_unet2_ds8x2_k128-16
K1=128,K2=16
patch-wise SimVQ
压缩率 0.0833333
```

这是 A/B/C 修改方案的主要参照来源。

### `run_exp12_rate044_A_patch_cb16_2.sh`

当前 A 方案：

```text
quality_v2_B_larger_rate044_A_patch_cb16-2
K1=16,K2=2
quantizer_axis_list=patch,patch
```

用途：

- patch-wise SimVQ baseline
- 768x512 下压缩率约 `0.0442708`
- 256x256 下压缩率仍约 `0.0442708`

### `run_exp13_rate041_B_hybridcvq_cb65536_8192.sh`

当前 B 方案：

```text
quality_v2_B_larger_rate041_B_hybridcvq_cb65536-8192
K1=65536,K2=8192
quantizer_axis_list=channel,patch
CVQ_CODEWORD_SHAPES=32x32,patch
nested alpha=0
```

用途：

- 第 1 层 channel-wise CVQ
- 第 2 层 patch-wise SimVQ
- 768x512 下压缩率约 `0.0407986`
- 256x256 下压缩率约 `0.0755208`

### `run_exp14_rate041_C_hybridcvq_nested_cb65536_8192.sh`

当前 C 方案：

```text
quality_v2_B_larger_rate041_C_hybridcvq_nested_cb65536-8192
K1=65536,K2=8192
quantizer_axis_list=channel,patch
CVQ_CODEWORD_SHAPES=32x32,patch
nested alpha=0.25
```

用途：

- B 的基础上增加 nested channel dropout
- 训练时随机保留前若干通道，其余通道置零

### 其它 `run_exp*.sh`

这些是历史实验脚本：

- `run_exp1_lpips.sh`：LPIPS/VGG perceptual loss 相关实验。
- `run_exp2_larger.sh`：larger backbone 基础实验。
- `run_exp3_swin.sh`：Swin enhance 实验。
- `run_exp4_dynamic_swin.sh`：动态 Swin enhance 实验。
- `run_exp5_larger_cb4096_65536.sh`：码本 `[4096,65536]`。
- `run_exp6_larger_cb16384_256.sh`：码本 `[16384,256]`。
- `run_exp7_larger_vitvq_nocompress_k64_256.sh`：ViT-VQ/QBridge 不压缩对照。
- `run_exp8_larger_noquant.sh`：无量化直通对照。
- `run_exp10_larger_cb128_16_vq.sh`：vanilla VQ 对照。
- `run_exp11_larger_cb128_16_div2k_flickr2k_768x512.sh`：DIV2K/Flickr2K 且训练 resize 到 768x512 的实验。

这些脚本不要随便删除，因为它们记录了历史实验配置。

## 4. `models/` 目录

### `models/deepsc.py`

主模型 `DeepSC`。

负责把以下模块串起来：

- `SemanticEncoder`
- 多层 vector quantizer
- finite blocklength channel 训练噪声
- `SemanticDecoder`
- 可选 bottleneck attention
- 可选 SwinIR enhancement

重要方法：

- `forward_train(x)`：训练路径，可能加入信道扰动。
- `forward_val(x)`：验证路径。
- `forward_test(x)`：测试时只输出 index。
- `reconstruct_from_indices(indices)`：从 index 重建图像。
- `set_channel_prob()`：设置训练中使用信道扰动的概率。

### `models/semantic_encoder.py`

U-Net 编码器。

主要类：

- `ResidualBlock`
- `DownSampleBlock`
- `SemanticEncoder`

当前 A/B/C 的实际结构：

```text
init Conv2d(3 -> 128)
Layer1 DownSampleBlock: 128 -> 256, stride=8
Layer2 DownSampleBlock: 256 -> 512, stride=2
每个 DownSampleBlock 下采样前 4 个 residual block，下采样后 4 个 residual block
GroupNorm + SiLU
```

### `models/semantic_decoder.py`

U-Net 解码器。

主要类：

- `SkipConnectionDropout`
- `ResidualBlock`
- `UpSampleBlock`
- `SemanticDecoder`

当前 A/B/C 的实际结构：

```text
从 Layer2 量化特征开始
上采样 x2
concat Layer1 skip
上采样 x8
ConvTranspose2d 输出 RGB
每个 UpSampleBlock 内部有 4 个 residual block
```

### `models/vector_quantizer.py`

量化器实现。

包含：

- `ProjectedEmbedding`：SimVQ 的冻结 embedding + 可训练 linear projection。
- `VectorQuantizer`：patch-wise SimVQ。
- `ChannelwiseVectorQuantizer`：channel-wise CVQ/SimVQ。
- `VanillaVectorQuantizer`：普通 VQ，embedding 直接训练。

当前 A/B/C 使用：

- A：`VectorQuantizer`
- B/C 第 1 层：`ChannelwiseVectorQuantizer`
- B/C 第 2 层：`VectorQuantizer`

### `models/vector_quantizer_vitvq.py`

ViT-VQ / QBridge 相关量化器，主要给历史 `ViTvqNoCompress` 实验使用。当前 A/B/C 不使用。

### `models/channel.py`

训练时的有限块长信道扰动模块。注意它和 `communications/` 里的真实 LDPC 测试链路不是同一个层次。

### `models/attention.py`

bottleneck self-attention 模块。

当前 A/B/C 中：

```text
USE_BOTTLENECK_ATTENTION=False
```

所以没有启用。

### `models/swinir_enhance.py`

轻量 SwinIR 风格后处理网络。历史 Swin enhance 实验使用，当前 A/B/C 不使用。

### `models/__init__.py`

Python package 标识文件。

## 5. `data/` 目录

### `data/datasets.py`

数据集和 dataloader 逻辑。

支持：

- 普通文件夹图像数据集 `ImageDataset`
- tar/zip 流式读取 `ArchiveImageDataset`
- train/val/test 三种 transform

关键逻辑：

- train 默认：`Resize(256) + RandomCrop(256)`，如果设置 `SIMVQ_TRAIN_RESIZE` 则用固定 resize。
- train 带 `RandomHorizontalFlip()`。
- val 默认：`Resize(256) + CenterCrop(256)`。
- test 默认：`Resize(768,512)`。
- test 如果设置 `SIMVQ_TEST_NO_RESIZE=1`，则不 resize。

## 6. `losses/` 目录

### `losses/deepsc_loss.py`

训练损失。

包含：

- MSE 重建损失
- 可选 MS-SSIM loss
- 可选 VGG perceptual loss
- 多层 VQ loss 加权

当前 A/B/C 实际使用：

```text
MSE_LOSS_WEIGHT = 1.0
MS_SSIM_LOSS_WEIGHT = 0.0
LPIPS_LOSS_WEIGHT = 0.0
```

所以当前主要是：

```text
MSE reconstruction loss + weighted VQ loss
```

## 7. `communications/` 目录

### `communications/channel.py`

真实测试链路里的 AWGN / Rician 信道函数。

当前 SNR=0 测试使用：

```text
awgn_channel()
```

### `communications/modulation.py`

调制和解调相关函数。

包含：

- BPSK modulation / demodulation / LLR
- QPSK modulation / LLR
- 16QAM 相关函数

当前主要使用：

```text
bpsk_modulate()
bpsk_llr()
```

### `communications/ldpc_coding.py`

Sionna 5G NR LDPC 编码/解码封装。

注意：

- 代码里会屏蔽 TensorFlow GPU，让 LDPC 在 CPU 上跑，避免和 PyTorch 抢 GPU。
- 当前测试配置是 `k=128,n=256,R=0.5`。

### `communications/evaluate.py`

通信链路相关历史评估辅助文件。当前主测试更多通过 `evaluation/quality.py` 和 `test_real.py` 完成。

## 8. `evaluation/` 目录

### `evaluation/quality.py`

评测核心函数。

包含：

- `evaluate_no_channel()`：源重建上界。
- `evaluate_ldpc_channel()`：LDPC + modulation + AWGN + decode 后重建。
- `evaluate_uncoded_channel()`：无 LDPC 编码的信道测试。

当前 A/B/C SNR=0 结果来自：

```text
evaluate_ldpc_channel()
```

### `evaluation/__init__.py`

Python package 标识文件。

## 9. `training/` 目录

### `training/schedules.py`

训练调度逻辑。

控制：

- skip dropout 从初始值退火到最终值
- VQ layer loss weights 从初始值退火到最终值
- channel probability 从 0 逐步升到 1

当前默认：

```text
CHANNEL_PROB_START_EPOCH=80
CHANNEL_PROB_END_EPOCH=120
```

也可以用环境变量覆盖。

## 10. `monitoring/` 目录

### `monitoring/codebook.py`

码本使用率监控。

统计：

- active code count
- active ratio
- perplexity
- dead code count
- min L2 distance
- collapse ratio

训练中每 10 epoch 左右会写入 codebook metrics CSV。

## 11. `utils/` 目录

### `utils/bit_utils.py`

index 和 bit 的转换工具。

用于真实链路测试：

```text
indices -> bits -> LDPC/BPSK/AWGN/LDPC decode -> bits -> indices
```

已支持 patch-wise index 和 channel-wise index。

### `utils/checkpoint_utils.py`

checkpoint 加载和模型重建工具。

功能：

- 从 checkpoint 读取 state dict
- 推断码本大小
- 按当前 Config 构建兼容模型
- 加载 checkpoint 权重

对 CVQ 很重要：channel-wise 第一层码本向量维度是 `32*32=1024`，但 decoder 特征通道仍是 `256`，所以加载时不能简单把码本维度当 decoder channel。

### `utils/experiment_io.py`

CSV 写入工具。

用于：

- epoch metrics
- codebook metrics

### `utils/math_utils.py`

数学辅助函数。当前主流程中不是最核心文件。

### `utils/metrics.py`

图像质量指标。

包含：

- MS-SSIM 计算
- PSNR 由 `evaluation/quality.py` 中 MSE 计算

### `utils/reproducibility.py`

随机种子设置。

训练和测试中通常使用 seed 42。

## 12. `tools/` 目录

工具脚本，不是主训练入口。

### `tools/auto_evaluate_completed_experiments.py`

自动检测已完成实验并评估的工具。

### `tools/start_auto_eval_watcher.sh`

启动自动评估 watcher 的 shell 脚本。

### `tools/evaluate_cb128_16_tiled_kodak.py`

针对 `cb128-16` 的 tiled Kodak 测试工具。

### `tools/export_codebook_metrics_from_logs.py`

从日志导出码本指标。

### `tools/generate_codebook_utilization_workbook.py`

生成码本使用率 Excel/汇总表。

### `tools/generate_detailed_experiment_report.py`

生成详细实验报告。

### `tools/rebuild_unified_experiment_report.py`

重建统一实验报告。

### `tools/write_pretty_experiment_report_uno.py`

生成更可读的实验报告。

### `tools/monitor_quality.py`

监控训练质量的辅助脚本。

### `tools/smoke_large_codebook_train_step.py`

大码本训练 step smoke test，用于验证显存/前向/反向是否能跑。

### `tools/__init__.py`

Python package 标识文件。

## 13. `experiments/` 目录

实验结果和文档目录。这里文件非常多，不建议当作源码目录维护。

### 重要文档

- `experiments/abc_rate041_044_detailed_config_analysis_20260609.md`：A/B/C 配置、结构和结果分析。
- `experiments/compression_rate_codebook_selection_record_20260609.md`：压缩率和码本选择推导。
- `experiments/cvq_hybrid_083_training_plan_20260608.md`：0.083 CVQ 方案推导。
- `experiments/rate042_cvq_training_config_20260608.md`：早期 0.042 方案。
- `experiments/rate042_large_to_small_training_config_20260608.md`：大到小码本方案记录。
- `experiments/experiment_log.md`：历史实验日志总表。

### 指标 CSV

命名一般为：

```text
experiments/<experiment_name>_epoch_metrics.csv
experiments/<experiment_name>_codebook_metrics.csv
experiments/<experiment_name>_screening.csv
```

含义：

- `epoch_metrics.csv`：每轮 train/val loss、best 标记等。
- `codebook_metrics.csv`：码本使用率、perplexity 等。
- `screening.csv`：早期实验筛选记录。

### 评测 JSON

当前最重要目录：

```text
experiments/eval_best_20260609_kodak256_snr0/
experiments/eval_best_20260609_rate041_large_to_small/
```

含义：

- `kodak256_snr0`：Kodak-256 no-resize, SNR=0 测试结果。
- `rate041_large_to_small`：768x512 或早期 no-channel/SNR=0 结果。

### `experiments/logs/`

当前训练日志目录。

已清理为主要保留当前正在训练的 A/B/C 日志，其它旧日志在：

```text
experiments/logs/archive/
```

### `experiments/tensorboard/`

TensorBoard 事件文件目录。体积可能很大，主要给可视化用。

### `experiments/archive/`

旧实验产物归档。

### `experiments/reports/`

Excel/CSV 等汇总报告。

### `experiments/snapshots/`

训练中保存的图像快照，如果对应实验启用了 snapshot。

### `experiments/variants/`

历史变体配置。例如：

- `preexisting_k128_config.py`

## 14. `checkpoints/` 目录

模型权重目录。每个子目录对应一个实验名。

常见文件：

```text
best_vq_deepsc.pth
last_checkpoint.pth
```

区别：

- `best_vq_deepsc.pth`：验证集 val recon loss 最佳时的模型 state dict，测试时通常用这个。
- `last_checkpoint.pth`：断点续训用，包含模型、optimizer、scheduler、epoch、rng state 等。

当前 A/B/C 关键 checkpoint：

- A：`checkpoints/quality_v2_B_larger_rate044_A_patch_cb16-2_unet2_ds8x2_k16-2/`
- B：`checkpoints/quality_v2_B_larger_rate041_B_hybridcvq_cb65536-8192_unet2_ds8x2_k65536-8192/`
- C：`checkpoints/quality_v2_B_larger_rate041_C_hybridcvq_nested_cb65536-8192_unet2_ds8x2_k65536-8192/`

## 15. `document/` 目录

### `document/Channel-wise Vector Quantization.pdf`

CVQ 论文 PDF。之前关于 channel-wise quantization、mixed CVQ、nested dropout 的方案讨论基于这篇文章。

## 16. 根目录 Markdown 文档

### `ABLATION_PLAN.md`

历史 ablation 计划。

### `EXPERIMENTS_OVERVIEW.md`

实验总览。

### `PROJECT_STRUCTURE.md`

旧项目结构说明。当前这份 `project_file_map_20260609.md` 是更新后的、更贴近当前 A/B/C 状态的说明。

### `REFACTOR_LOG.md`

重构记录。

## 17. 哪些文件是“核心源码”

如果只看最关键代码，优先看这些：

```text
config.py
train.py
test_real.py
data/datasets.py
models/deepsc.py
models/semantic_encoder.py
models/semantic_decoder.py
models/vector_quantizer.py
losses/deepsc_loss.py
evaluation/quality.py
communications/ldpc_coding.py
communications/modulation.py
communications/channel.py
utils/bit_utils.py
utils/checkpoint_utils.py
training/schedules.py
monitoring/codebook.py
```

## 18. 哪些文件是“实验入口”

当前最重要：

```text
run_exp12_rate044_A_patch_cb16_2.sh
run_exp13_rate041_B_hybridcvq_cb65536_8192.sh
run_exp14_rate041_C_hybridcvq_nested_cb65536_8192.sh
test_real.py
```

原始基准：

```text
run_exp9_larger_cb128_16.sh
```

## 19. 哪些目录不建议手工乱改

```text
checkpoints/
experiments/tensorboard/
experiments/logs/
experiments/archive/
experiments/interim_results/
experiments/auto_results/
```

这些目录主要是训练和测试产物。需要整理时，建议移动到 archive，而不是直接删除。

## 20. 当前项目为什么显得乱

主要原因：

1. 历史实验很多，`run_exp*.sh`、CSV、JSON、checkpoint 都留在同一个项目里。
2. `experiments/` 同时承担了“结果目录”和“文档目录”两个角色。
3. 多个实验阶段共用同一套 `config.py`，大量行为由环境变量决定，新读代码时不容易看出实际配置。
4. checkpoint、日志、TensorBoard、评测 JSON 没有完全按实验阶段分层归档。
5. A/B/C 新方案是在已有项目上增量加的 CVQ，所以和旧 VQ、ViT-VQ、SwinIR、NoQuant 实验共存。

建议后续整理方向：

```text
scripts/
  train/
  eval/
docs/
  experiment_plans/
  analysis/
outputs/
  checkpoints/
  metrics/
  eval_json/
  logs/
src/
  models/
  data/
  training/
  evaluation/
```

但在当前训练还在跑的情况下，不建议立刻大规模移动源码和产物。更稳妥的做法是：先用 md 把现状标清楚，再等当前 A/B/C 实验结束后做一次系统归档。
