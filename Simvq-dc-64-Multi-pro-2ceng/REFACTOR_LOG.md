# Refactor Log

Date: 2026-05-28

## Goal

整理 `Simvq-dc-64-Multi-pro-2ceng` 的代码职责边界，让文件名和内容更匹配，同时保留原来的训练、测试和监控入口，避免破坏已有用法。

## Main Changes

### 1. 拆出码本监控逻辑

Before:

- `models/deepsc.py` 同时包含模型结构、前向传播、码本利用率统计和格式化打印。

After:

- `models/deepsc.py` 只保留 `DeepSC` 模型本身。
- 新增 `monitoring/codebook.py`：
  - `compute_codebook_utilization`
  - `print_codebook_utilization`
  - `write_codebook_tensorboard`
- `train.py` 改为从 `monitoring.codebook` 调用这些函数。

Reason:

- 码本监控是训练/诊断工具，不属于模型定义本身。

### 2. 拆出评估指标逻辑

Before:

- `test_real.py` 里混有 CLI、checkpoint 解析、模型构建、无信道评估、LDPC 真实链路评估。
- `communications/evaluate.py` 与 `test_real.py` 中存在重复评估实现。

After:

- 新增 `evaluation/quality.py`：
  - `evaluate_no_channel`
  - `evaluate_ldpc_channel`
  - `evaluate_uncoded_channel`
- `test_real.py` 变成轻量 CLI 入口。
- `communications/evaluate.py` 保留旧函数名，但只作为兼容转发层。

Reason:

- 评估指标应集中放在 `evaluation/` 中，测试脚本只负责解析参数和展示结果。

### 3. 拆出 BPP 计算

Before:

- `test_BPP.py` 文件名大小写不统一，且同时包含模型加载、BPP 计算和输出展示。

After:

- BPP 相关文件已被移除（改用压缩率计算）。

### 4. 拆出 checkpoint 与复现实验工具

新增：

- `utils/checkpoint_utils.py`
  - `extract_state_dict`
  - `load_model_state_dict`
  - `infer_codebook_config`
  - `build_model_from_checkpoint`
- `utils/reproducibility.py`
  - `setup_seed`
- `utils/experiment_io.py`
  - `append_epoch_record`

Reason:

- checkpoint 推断码本大小、构建模型、固定随机种子、写 CSV 记录都是通用工具，不应分散在测试和训练脚本中。

### 5. 拆出训练调度逻辑

Before:

- `compute_schedule` 放在 `train.py` 顶部。

After:

- 新增 `training/schedules.py`：
  - `compute_schedule`
- `train.py` 只保留训练主流程。

Reason:

- dropout/VQ 权重阶段调度属于训练策略，单独放入 `training/` 更清楚。

### 6. 移动自动监控脚本

Before:

- `experiments/monitor_quality.py` 是完整实现，但 `experiments/` 更适合放日志、结果、checkpoint 等实验产物。

After:

- 新增 `tools/monitor_quality.py` 作为实际实现。
- 删除 `experiments/monitor_quality.py`，避免实验产物目录里混入工具脚本。

Reason:

- 自动化工具放 `tools/`；实验产物留在 `experiments/`。

## Compatibility Kept

以下旧命令仍可用：

```bash
python train.py
python test_real.py --checkpoint <path> --no-channel
python test_real.py --checkpoint <path> --snrs 0 3 6 9 12
```
```

## Files Added

- `evaluation/__init__.py`
- `evaluation/quality.py`
- `monitoring/__init__.py`
- `monitoring/codebook.py`
- `training/__init__.py`
- `training/schedules.py`
- `tools/__init__.py`
- `tools/monitor_quality.py`
- `utils/checkpoint_utils.py`
- `utils/experiment_io.py`
- `utils/reproducibility.py`
- `REFACTOR_LOG.md`

## Files Simplified

- `models/deepsc.py`
- `train.py`
- `test_real.py`
- `communications/evaluate.py`

## Validation

Ran Python bytecode compilation for all project `.py` files after the refactor:

```bash
find . -name '*.py' -not -path './__pycache__/*' -not -path '*/__pycache__/*' -print0 | \
  xargs -0 /workspace/yi/.conda/envs/work/bin/python -m py_compile
```

Also checked representative imports:

```bash
python - <<'PY'
import test_real
from communications.evaluate import evaluate_metrics_with_channel
from monitoring.codebook import compute_codebook_utilization
print('imports_ok')
PY
```

Both checks passed.

Checked CLI help for the main entry points:

```bash
python test_real.py --help
python test_BPP.py --help
python tools/monitor_quality.py --help
```

## Notes

- This is a structural refactor. It does not intentionally change model behavior, loss computation, channel simulation, or checkpoint format.
- Existing experimental logs and checkpoints were left in place.
- Generated `__pycache__` directories were removed after validation.

## Cleanup Follow-up

After the structural refactor, checkpoint storage was normalized:

- Active checkpoints now live in `checkpoints/quality_v1_unet2_ds4x2_k64`.
- Archived baseline checkpoints now live in `checkpoints/observed_001_baseline`.
- The duplicate `experiments/checkpoints` directory was removed.
- Intermediate periodic checkpoints were deleted; only `best_vq_deepsc.pth`, `last_checkpoint.pth`, and the final active `vq_deepsc_epoch_200.pth` were retained.
- Duplicated `.pth` screening snapshots under `experiments/snapshots` were deleted; JSON metrics remain.
- `tools/monitor_quality.py` now evaluates the active checkpoint directly instead of copying checkpoint files into `experiments/snapshots`.
- Stale top-level `logs/`, `train_output.log`, and generated `__pycache__` directories were removed.

## Architecture Configuration Follow-up

Date: 2026-05-28

Goal:

- 让当前二层 U-Net 结构可以平滑扩展为三层、四层，而不用改模型主体代码。
- 让 checkpoint、TensorBoard、筛选结果文件名自动带上网络层数、下采样倍率和码本大小，方便并行管理多个实验。

Changes:

- `config.py`
  - 新增 `UNET_DEPTH` 作为主要结构开关。
  - `NUM_DOWNSAMPLE_BLOCKS` 由 `UNET_DEPTH` 派生。
  - `DOWNSAMPLE_STRIDES` 自动生成：二层 `[4, 2]`，三层 `[4, 2, 2]`，四层 `[4, 2, 2, 2]`。
  - `EMBEDDING_DIM_LIST` 根据 `BASE_CHANNELS` 和层数自动生成。
  - `NUM_EMBEDDINGS_LIST` 默认由 `NUM_EMBEDDINGS_PER_LAYER` 扩展到每一层。
  - `LAYER_LOSS_WEIGHTS_INIT`、`LAYER_LOSS_WEIGHTS_FINAL`、`SKIP_DROPOUT_P_INIT`、`SKIP_DROPOUT_P_FINAL` 按层数自动生成。
  - 新增 `ESTIMATED_SOURCE_BPP`，用于在训练启动时查看结构改变带来的源端码率变化。
  - 新增 `validate()`，训练和评估前检查每个列表长度是否和层数匹配。
  - `EXPERIMENT_NAME` 自动生成，例如当前二层为 `quality_v1_unet2_ds4x2_k64`。

- `models/deepsc.py`
  - 移除对全局 `Config` 的直接依赖。
  - skip dropout、信道编码率、block length、SNR 范围改为从构造参数传入。
  - 增加层数、码本列表、embedding 维度、下采样列表的一致性检查。

- `train.py`
  - 训练启动时调用 `cfg.validate()`。
  - run id、metrics 文件、checkpoint 目录都使用自动生成的实验名。
  - 启动日志打印 U-Net 层数、下采样倍率、估算源端 BPP、每层特征维度、每层码本大小。

- `tools/monitor_quality.py`
  - 默认监控路径改为从 `Config` 读取。
  - 筛选日志和筛选 CSV 使用自动实验名。

- Existing active checkpoint directory was renamed:
  - Before: `checkpoints/quality_v1_k64`
  - After: `checkpoints/quality_v1_unet2_ds4x2_k64`

Usage:

```python
# config.py
UNET_DEPTH = 3
```

Then the default experiment name becomes:

```text
quality_v1_unet3_ds4x2x2_k64
```

For four layers:

```python
UNET_DEPTH = 4
```

Default experiment name:

```text
quality_v1_unet4_ds4x2x2x2_k64
```

## quality_v2 Low-BPP Upgrade

Date: 2026-05-28

Goal:

- 在约 `0.083 BPP` 下启动一套更强的 SimVQ 全链路升级实验。
- 先通过纯信源训练稳定码本，再逐步加入信道噪声，最后进行全信道鲁棒微调。
- 提升低码率下采样和重建网络容量，为后续对比 BPG 做准备。

Main configuration:

```text
EXPERIMENT_FAMILY = quality_v2
EXPERIMENT_NAME = quality_v2_unet2_ds8x2_k16-32
UNET_DEPTH = 2
DOWNSAMPLE_STRIDES = [8, 2]
NUM_EMBEDDINGS_LIST = [16, 32]
ESTIMATED_SOURCE_BPP = 0.08203125
```

Code changes:

- `config.py`
  - 切换到 `quality_v2`。
  - 设置 `[8,2]` 下采样和 `[16,32]` 码本。
  - 新增 channel curriculum、GroupNorm、SiLU、残差块数、上采样方式、bottleneck attention、混合损失权重配置。

- `training/schedules.py`
  - `compute_schedule` 现在返回 `channel_prob`。
  - `channel_prob` 调度：epoch 0-79 为 0，80-119 线性升到 1，120 以后为 1。

- `models/deepsc.py`
  - 训练/验证阶段支持按 `channel_prob` 决定是否经过信道噪声。
  - Phase 1 中使用干净量化特征，避免早期码本被信道扰动破坏。
  - 接入 bottleneck attention。

- `models/semantic_encoder.py`
  - 支持 GroupNorm、SiLU、可配置残差块数量。
  - stride=4/8 使用级联 2x 下采样，避免单个大步幅卷积直接丢失过多空间信息。

- `models/semantic_decoder.py`
  - 支持 GroupNorm、SiLU、可配置残差块数量。
  - 支持 `UPSAMPLE_MODE = "bilinear"`，减少大倍率上采样方块伪影。

- `models/attention.py`
  - 新增 bottleneck self-attention 模块。

- `losses/deepsc_loss.py`
  - 新增可微 MS-SSIM 损失。
  - 重建损失变为 `0.8*MSE + 0.2*(1-MS-SSIM)`。

- `train.py`
  - 打印并记录新结构、BPP、channel_prob。
  - epoch metrics 追加 `channel_prob` 字段。

Validation before training:

- `Config.validate()` passed.
- `quality_v2_unet2_ds8x2_k16-32` summary reported `ESTIMATED_SOURCE_BPP = 0.08203125`.
- CPU forward/backward smoke test passed with output shape `(1, 3, 256, 256)`.
