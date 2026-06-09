# 项目结构说明

日期：2026-05-28

本项目现在把 **代码**、**模型权重**、**实验输出** 分开放置，避免训练文件、日志文件、模型文件混在一起。

## 顶层文件夹

### `checkpoints/`

只存放模型权重。当前项目里只保留这一个 checkpoint 根目录。

- `checkpoints/quality_v2_A_curriculum_unet2_ds8x2_k16-32/`
  - A 消融实验权重，只验证课程学习。

- `checkpoints/quality_v2_B_backbone_unet2_ds8x2_k16-32/`
  - B 消融实验权重，A + 主干网络升级。

- `checkpoints/quality_v2_C_full_unet2_ds8x2_k16-32/`
  - C 消融实验权重，B + MS-SSIM 混合损失 + bottleneck attention。

- `checkpoints/quality_v2_unet2_ds8x2_k16-32/`
  - full 参考实验权重，配置等价 C，但命名不是严格的 C。

每个实验目录只保留：

```text
best_vq_deepsc.pth
last_checkpoint.pth
```

含义：

```text
best_vq_deepsc.pth：用于测试和结果对比
last_checkpoint.pth：用于中断后恢复训练
```

旧的 `vq_deepsc_epoch_10.pth`、`vq_deepsc_epoch_20.pth` 等周期性权重已经删除，后续训练也不再保存这些文件。

旧说明：

- `checkpoints/quality_v2_unet2_ds8x2_k16-32/`
  - `0.083 BPP` 低码率 full 参考实验的权重目录。
  - `best_vq_deepsc.pth`：验证集表现最好的模型权重。
  - `last_checkpoint.pth`：用于继续训练的断点，包含模型、优化器、调度器、随机数状态等信息。
  - `vq_deepsc_epoch_200.pth`：当前 200 轮训练完成后保留的最终 epoch 权重。

- `checkpoints/observed_001_baseline/`
  - 清理前已经存在的旧 baseline 权重归档。
  - 这里只保留 `best_vq_deepsc.pth` 和 `last_checkpoint.pth`。
  - 旧的中间 epoch 权重已经删除。

- `checkpoints/quality_v1_unet2_ds4x2_k64/`
  - 上一轮二层 `[4,2]`、`K=[64,64]` 实验归档。

### `communications/`

通信物理层相关代码。

- `channel.py`：AWGN 信道函数。
- `ldpc_coding.py`：Sionna LDPC 编码器和解码器封装。
- `modulation.py`：BPSK 调制、解调和 LLR 相关函数。
- `evaluate.py`：兼容旧调用方式的评估入口，内部转到 `evaluation/quality.py`。

### `data/`

数据集和 dataloader 相关代码。

- `datasets.py`：图像数据集类和 `get_dataloader`。

### `evaluation/`

评估指标相关代码，PSNR、MS-SSIM、BPP 等计算逻辑放在这里。

- `quality.py`
  - `evaluate_no_channel`：无信道情况下的重建质量评估，也就是源重建上限。
  - `evaluate_ldpc_channel`：真实 LDPC + BPSK + AWGN 链路下的质量评估。
  - `evaluate_uncoded_channel`：无 LDPC，仅 BPSK + AWGN 的质量评估。

- `bpp.py`
  - `calculate_bpp`：根据量化索引计算 bits-per-pixel。

### `experiments/`

只存放实验输出和实验记录，不再存放模型权重。

- `quality_v2_unet2_ds8x2_k16-32_epoch_metrics.csv`：当前低码率主实验的逐 epoch 指标记录。
- `quality_v1_unet2_ds4x2_k64_epoch_metrics.csv`：上一轮二层 `[4,2]`、`K=[64,64]` 实验记录。
- `observed_001_epoch_metrics.csv`：旧 baseline 训练记录归档。
- `quality_v2_unet2_ds8x2_k16-32_screening.csv`：当前低码率主实验的阶段性 PSNR/MS-SSIM 筛选记录。
- `quality_v1_unet2_ds4x2_k64_screening.csv`：上一轮实验筛选记录。
- `baseline_best_epoch079_nochannel.json`：旧 baseline 最佳 epoch 的无信道评估结果。
- `experiment_log.md`：实验过程、训练次数、修改方案和结果记录。
- `logs/`：训练、评估、监控脚本的 stdout 日志。
- `snapshots/`：筛选过程产生的 JSON 指标快照；重复 `.pth` 权重副本已经删除。
- `tensorboard/`：TensorBoard 事件文件。
- `variants/`：历史配置方案归档。

### `losses/`

训练损失函数。

- `deepsc_loss.py`：重建 MSE 损失和加权 VQ 损失。

### `models/`

网络结构和模型组件，只放模型定义相关代码。

- `deepsc.py`：`DeepSC` 总模型，负责组合编码器、量化器、信道、解码器，并提供训练、验证、测试 forward 路径。
- `semantic_encoder.py`：语义编码器和下采样模块。
- `semantic_decoder.py`：语义解码器、上采样模块、skip-dropout 模块。
- `vector_quantizer.py`：SimVQ 量化器，以及底层码本距离和统计相关函数。
- `channel.py`：训练中使用的可微有限块长信道近似模块。

### `monitoring/`

训练监控和诊断相关工具。

- `codebook.py`
  - 码本利用率统计。
  - 码本利用率命令行打印格式。
  - 码本统计信息写入 TensorBoard。

### `tools/`

项目自动化脚本，不属于核心模型结构。

- `monitor_quality.py`：定期评估当前主实验 checkpoint，并把 PSNR/MS-SSIM 写入 JSON/CSV。

### `training/`

训练策略辅助代码。

- `schedules.py`：dropout 和 VQ-loss 权重调度逻辑。

### `utils/`

通用工具函数。

- `bit_utils.py`：量化索引和 bitstream 之间的转换。
- `checkpoint_utils.py`：加载 checkpoint、推断码本配置、从 checkpoint 构建模型。
- `experiment_io.py`：把训练指标追加写入 CSV。
- `math_utils.py`：数学辅助函数。
- `metrics.py`：PSNR/MS-SSIM 相关辅助函数。
- `reproducibility.py`：随机种子设置。

## 顶层文件

### `config.py`

当前主实验的集中配置文件。

现在 U-Net 层数、每层下采样倍率、每层码本大小、每层 VQ 损失权重、skip dropout、channel curriculum 和实验目录名都集中由这里管理。

常用修改方式：

```python
UNET_DEPTH = 2  # 当前二层 U-Net
DOWNSAMPLE_STRIDES = [8, 2]
NUM_EMBEDDINGS_LIST = [16, 32]
```

以后如果要训练三层或四层，优先只改这个数字：

```python
UNET_DEPTH = 3
UNET_DEPTH = 4
```

默认规则：

- `DOWNSAMPLE_STRIDES` 可以自动生成，也可以手动指定。当前低码率实验手动指定为 `[8, 2]`。
- `EMBEDDING_DIM_LIST` 会根据 `BASE_CHANNELS` 和层数自动生成。
- `NUM_EMBEDDINGS_LIST` 可以每层单独指定。当前低码率实验为 `[16, 32]`，估算 BPP 为 `0.08203125`。
- `LAYER_LOSS_WEIGHTS_INIT`、`LAYER_LOSS_WEIGHTS_FINAL` 会按层数自动生成。
- `SKIP_DROPOUT_P_INIT`、`SKIP_DROPOUT_P_FINAL` 会自动生成长度为 `UNET_DEPTH - 1` 的列表。
- `EXPERIMENT_NAME` 会自动包含结构信息，例如 `quality_v2_unet2_ds8x2_k16-32`。

重要路径：

- `CHECKPOINT_DIR = "./checkpoints/quality_v2_unet2_ds8x2_k16-32"`
- `LOG_DIR = "./experiments/tensorboard/quality_v2_unet2_ds8x2_k16-32"`

### `train.py`

主训练入口。

主要职责：

- 构建模型、损失函数、优化器、学习率调度器。
- 根据 `config.py` 中的 checkpoint 路径继续训练。
- 执行训练和验证循环。
- 调用 `monitoring/codebook.py` 记录码本诊断信息。
- 逐 epoch 追加写入 CSV 指标。
- 保存 `best_vq_deepsc.pth`、`last_checkpoint.pth` 和周期性 checkpoint。

### `test_real.py`

图像质量评估命令行入口。

示例：

```bash
python test_real.py --checkpoint checkpoints/quality_v2_unet2_ds8x2_k16-32/best_vq_deepsc.pth --no-channel
python test_real.py --checkpoint checkpoints/quality_v2_unet2_ds8x2_k16-32/best_vq_deepsc.pth --snrs 0 3 6 9 12
```

### `run_train.sh`

训练启动脚本，日志写入 `experiments/logs/`。

### `REFACTOR_LOG.md`

代码结构重构和文件清理过程记录。

### `PROJECT_STRUCTURE.md`

当前这个项目结构说明文件。

## 本次清理已删除的内容

- 重复 checkpoint 根目录：`experiments/checkpoints/`。
- 旧的顶层 TensorBoard 目录：`logs/`。
- 旧的顶层训练日志：`train_output.log`。
- 不必要的中间 epoch checkpoint，只保留当前需要的 best、last 和最终 epoch-200 权重。
- `experiments/snapshots/` 下重复保存的 `.pth` 筛选权重副本。
- Python 自动生成的 `__pycache__` 目录。

## 当前放置规则

- 代码放在 `models/`、`training/`、`evaluation/`、`monitoring/`、`communications/`、`utils/`、`tools/`。
- 模型权重只放在 `checkpoints/`。
- 实验日志、CSV、JSON、TensorBoard 文件只放在 `experiments/`。
