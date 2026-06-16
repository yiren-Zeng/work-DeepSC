# SimVQ 实验统一索引

生成目的：本文件作为项目实验的统一入口文档，不替代已有的 `PROJECT_STRUCTURE.md`、`EXPERIMENTS_OVERVIEW.md`、`ABLATION_PLAN.md` 以及 `experiments/` 下的详细实验记录。

项目路径：

```text
/workspace/yi/work/Simvq-dc-64-Multi-pro-2ceng
```

## 1. 文档定位

已有文档大致分为两类：

- 早期结构与 A/B/C 消融说明：`PROJECT_STRUCTURE.md`、`EXPERIMENTS_OVERVIEW.md`、`ABLATION_PLAN.md`
- 当前实验快照与 rate041/044 分析：`experiments/project_file_map_20260609.md`、`experiments/abc_rate041_044_detailed_config_analysis_20260609.md`、`experiments/compression_rate_codebook_selection_record_20260609.md`

阅读顺序建议：

1. 先看本文件，确认当前主线、历史实验和待确认实验。
2. 看 `experiments/project_file_map_20260609.md`，了解当前项目文件和最新 A/B/C。
3. 看 `experiments/abc_rate041_044_detailed_config_analysis_20260609.md`，了解当前 rate041/044 A/B/C 的配置和结果。
4. 如需追溯早期 A/B/C，再看 `ABLATION_PLAN.md` 和 `EXPERIMENTS_OVERVIEW.md`。

## 2. 当前推荐入口

当前主线训练脚本已经整理到：

```text
scripts/train/current/
```

当前主线测试脚本在：

```text
scripts/eval/
```

推荐优先使用：

| 用途 | 脚本 | 说明 |
|---|---|---|
| 当前 A 训练 | `scripts/train/current/run_exp12_rate044_A_patch_cb16_2.sh` | patch-wise SimVQ baseline |
| 当前 B 训练 | `scripts/train/current/run_exp13_rate041_B_hybridcvq_cb65536_8192.sh` | SimVQ + CVQ |
| 当前 C 训练 | `scripts/train/current/run_exp14_rate041_C_hybridcvq_nested_cb65536_8192.sh` | SimVQ + CVQ + nested channel dropout |
| 当前 A 测试 | `scripts/eval/test_A_patch_rate044.sh` | 默认 SNR=0, BPSK |
| 当前 B 测试 | `scripts/eval/test_B_hybrid_rate041.sh` | 默认 SNR=0, BPSK |
| 当前 C 测试 | `scripts/eval/test_C_hybrid_nested_rate041.sh` | 支持 `DEBUG_PDB=1` 和 `DEBUGPY=1` |

## 3. experiments/ 与 checkpoints/ 边界

### `checkpoints/`

`checkpoints/` 只放模型权重。

常见结构：

```text
checkpoints/<experiment_name>/
├── best_vq_deepsc.pth
└── last_checkpoint.pth
```

含义：

- `best_vq_deepsc.pth`：验证集表现最好的权重，通常用于测试和对比。
- `last_checkpoint.pth`：最近训练断点，用于继续训练，通常包含模型、优化器、调度器、epoch、随机数状态等。

### `experiments/`

`experiments/` 只放实验输出，不放主模型权重。

常见内容：

- `<experiment_name>_epoch_metrics.csv`：逐 epoch 训练/验证指标。
- `<experiment_name>_codebook_metrics.csv`：码本利用率等统计。
- `<experiment_name>_screening.csv`：阶段性筛选记录。
- `logs/`：训练、评估、监控日志。
- `tensorboard/`：TensorBoard 事件文件。
- `auto_results/`：自动评估 JSON。
- `eval_best_20260609_*`：当前 A/B/C 重点测试结果。
- `snr0_results/`：早期 SNR=0 结果汇总。
- `interim_results/`：阶段性和中间测试结果。
- `reports/`：汇总报告。
- `archive/`：历史归档输出。

## 4. 所有实验总表

说明：

- “当前主线”指 2026-06-09 文档中重点分析的 rate041/044 A/B/C。
- “早期 A/B/C”指 `SIMVQ_EXPERIMENT_STAGE=A/B/C` 控制的 curriculum/backbone/full 消融。
- “待确认”表示可以看到目录或结果，但脚本、训练完整性或用途不能从当前文件中完全确认，不能删除。

| 实验名 | 阶段 | 目的 | 关键配置 | 启动方式 | checkpoint 路径 | 输出结果路径 | 当前状态 | 是否推荐继续使用 |
|---|---|---|---|---|---|---|---|---|
| `observed_001_baseline` | 历史 baseline | 最早观察到的 baseline | 旧二层 SimVQ，BPP 较高 | 待确认 | `checkpoints/observed_001_baseline/` | `experiments/observed_001_epoch_metrics.csv`, `experiments/baseline_best_epoch079_nochannel.json` | 历史归档 | 否，仅对照 |
| `quality_v1_unet2_ds4x2_k64` | v1 历史 | 高码率二层模型对照 | `DOWNSAMPLE_STRIDES=4,2`, `K≈64` | 待确认 | `checkpoints/quality_v1_unet2_ds4x2_k64/` | `experiments/quality_v1_unet2_ds4x2_k64_epoch_metrics.csv`, `experiments/snr0_results/quality_v1_k64.json` | 已完成 | 否，仅历史对照 |
| `quality_v2_A_curriculum_unet2_ds8x2_k16-32` | 早期 A | curriculum + 低码率验证 | stage=A, `K=16,32`, BN/PReLU, res=1/1 | 环境变量启动，见 `ABLATION_PLAN.md` | `checkpoints/quality_v2_A_curriculum_unet2_ds8x2_k16-32/` | `experiments/quality_v2_A_curriculum_unet2_ds8x2_k16-32_epoch_metrics.csv`, `*_screening.csv`, `experiments/snr0_results/current_A_k16-32.json` | 已完成 | 否，作为早期消融保留 |
| `quality_v2_B_backbone_unet2_ds8x2_k16-32` | 早期 B | A + backbone 升级 | stage=B, `K=16,32`, GroupNorm/SiLU, res=2/2 | 环境变量启动，见 `ABLATION_PLAN.md` | `checkpoints/quality_v2_B_backbone_unet2_ds8x2_k16-32/` | `experiments/quality_v2_B_backbone_unet2_ds8x2_k16-32_epoch_metrics.csv`, `*_screening.csv`, `experiments/snr0_results/current_B_k16-32.json` | 已完成 | 否，作为早期消融保留 |
| `quality_v2_C_full_unet2_ds8x2_k16-32` | 早期 C | B + attention/full 配置 | stage=C, `K=16,32`, bottleneck attention | 环境变量启动，见 `ABLATION_PLAN.md` | `checkpoints/quality_v2_C_full_unet2_ds8x2_k16-32/` | `experiments/quality_v2_C_full_unet2_ds8x2_k16-32_epoch_metrics.csv`, `*_screening.csv`, `experiments/snr0_results/current_C_k16-32.json` | 已完成 | 否，作为早期消融保留 |
| `quality_v2_A_curriculum_unet2_ds8x2_k64-256` | 早期 A 重训 | A 的 `K=64,256` 对照 | stage=A, `K=64,256` | 环境变量启动 | `checkpoints/quality_v2_A_curriculum_unet2_ds8x2_k64-256/` | `experiments/quality_v2_A_curriculum_unet2_ds8x2_k64-256_epoch_metrics.csv`, `*_screening.csv`, `experiments/snr0_results/current_A_k64-256.json` | 已完成 | 否，仅对照 |
| `quality_v2_B_backbone_unet2_ds8x2_k64-256` | 早期 B 重训 | B 的 `K=64,256` 对照 | stage=B, `K=64,256` | 环境变量启动 | `checkpoints/quality_v2_B_backbone_unet2_ds8x2_k64-256/` | `experiments/quality_v2_B_backbone_unet2_ds8x2_k64-256_epoch_metrics.csv`, `*_screening.csv`, `experiments/snr0_results/current_B_k64-256.json` | 已完成 | 否，仅对照 |
| `quality_v2_C_full_unet2_ds8x2_k64-256` | 早期 C 重训 | C 的 `K=64,256` 对照 | stage=C, `K=64,256` | 环境变量启动 | `checkpoints/quality_v2_C_full_unet2_ds8x2_k64-256/` | `experiments/quality_v2_C_full_unet2_ds8x2_k64-256_epoch_metrics.csv`, `*_screening.csv`, `experiments/snr0_results/current_C_k64-256.json` | 已完成 | 否，仅对照 |
| `quality_v2_B_larger_unet2_ds8x2_k64-256` | 扩容基准 | 扩大 backbone 容量 | stage=B, `K=64,256`, `BASE_CHANNELS=128`, res=4/4 | `scripts/train/baseline/run_exp2_larger.sh` | `checkpoints/quality_v2_B_larger_unet2_ds8x2_k64-256/` | `experiments/quality_v2_B_larger_unet2_ds8x2_k64-256_epoch_metrics.csv`, `experiments/auto_results/*larger*` | 已完成 | 是，作为基准 |
| `quality_v2_B_larger_cb128-16_unet2_ds8x2_k128-16` | 当前基准来源 | rate041/044 A/B/C 的主要参照 | stage=B, larger backbone, `K=128,16`, SimVQ | `scripts/train/baseline/run_exp9_larger_cb128_16.sh` | `checkpoints/quality_v2_B_larger_cb128-16_unet2_ds8x2_k128-16/` | `experiments/quality_v2_B_larger_cb128-16_unet2_ds8x2_k128-16_epoch_metrics.csv`, `*_codebook_metrics.csv`, `experiments/auto_results/*cb128-16*` | 已完成 | 是，重点保留 |
| `quality_v2_B_larger_rate044_A_patch_cb16-2_unet2_ds8x2_k16-2` | 当前 A | patch-wise baseline | stage=B, `K=16,2`, `axis=patch,patch`, nested=0 | `scripts/train/current/run_exp12_rate044_A_patch_cb16_2.sh` | `checkpoints/quality_v2_B_larger_rate044_A_patch_cb16-2_unet2_ds8x2_k16-2/` | `experiments/quality_v2_B_larger_rate044_A_patch_cb16-2_unet2_ds8x2_k16-2_epoch_metrics.csv`, `*_codebook_metrics.csv`, `experiments/eval_best_20260609_*/*A_patch*.json`, `experiments/logs/train_exp12*.log` | 已完成 200 epoch | 是，当前主线 |
| `quality_v2_B_larger_rate041_B_hybridcvq_cb65536-8192_unet2_ds8x2_k65536-8192` | 当前 B | SimVQ + CVQ | stage=B, `K=65536,8192`, `axis=channel,patch`, `CVQ=32x32,patch`, nested=0 | `scripts/train/current/run_exp13_rate041_B_hybridcvq_cb65536_8192.sh` | `checkpoints/quality_v2_B_larger_rate041_B_hybridcvq_cb65536-8192_unet2_ds8x2_k65536-8192/` | `experiments/quality_v2_B_larger_rate041_B_hybridcvq_cb65536-8192_unet2_ds8x2_k65536-8192_epoch_metrics.csv`, `*_codebook_metrics.csv`, `experiments/eval_best_20260609_*/*B_hybrid*.json`, `experiments/logs/train_exp13*.log` | 已完成 200 epoch | 是，当前主线 |
| `quality_v2_B_larger_rate041_C_hybridcvq_nested_cb65536-8192_unet2_ds8x2_k65536-8192` | 当前 C | SimVQ + CVQ + nested channel dropout | stage=B, `K=65536,8192`, `axis=channel,patch`, `CVQ=32x32,patch`, nested=0.25 | `scripts/train/current/run_exp14_rate041_C_hybridcvq_nested_cb65536_8192.sh` | `checkpoints/quality_v2_B_larger_rate041_C_hybridcvq_nested_cb65536-8192_unet2_ds8x2_k65536-8192/` | `experiments/quality_v2_B_larger_rate041_C_hybridcvq_nested_cb65536-8192_unet2_ds8x2_k65536-8192_epoch_metrics.csv`, `*_codebook_metrics.csv`, `experiments/eval_best_20260609_*/*C_hybrid_nested*.json`, `experiments/logs/train_exp14*.log` | 已完成 200 epoch | 是，当前主线 |
| `quality_v2_B_larger_cb4096-65536_unet2_ds8x2_k4096-65536` | 大码本历史 | 扩容基准上改码本 | stage=B, larger backbone, `K=4096,65536` | `scripts/train/archive/run_exp5_larger_cb4096_65536.sh` | `checkpoints/quality_v2_B_larger_cb4096-65536_unet2_ds8x2_k4096-65536/` | `experiments/quality_v2_B_larger_cb4096-65536_unet2_ds8x2_k4096-65536_epoch_metrics.csv`, `*_codebook_metrics.csv`, `experiments/interim_results/*cb4096*` | 约 51 epoch 后停止，可恢复 | 否，历史对照 |
| `quality_v2_B_larger_cb16384-256_unet2_ds8x2_k16384-256` | 大码本历史 | 扩容基准上改码本 | stage=B, larger backbone, `K=16384,256` | `scripts/train/archive/run_exp6_larger_cb16384_256.sh` | `checkpoints/quality_v2_B_larger_cb16384-256_unet2_ds8x2_k16384-256/` | `experiments/quality_v2_B_larger_cb16384-256_unet2_ds8x2_k16384-256_epoch_metrics.csv`, `*_codebook_metrics.csv`, `experiments/interim_results/*cb16384*` | 已完成 | 否，历史对照 |
| `quality_v2_B_larger_ViTvqNoCompress_unet2_ds8x2_k64-256` | 量化器对照 | ViTvq/QBridge NoCompress | `QUANTIZER_TYPE=vitvq_nocompress`, `K=64,256` | `scripts/train/archive/run_exp7_larger_vitvq_nocompress_k64_256.sh` | `checkpoints/quality_v2_B_larger_ViTvqNoCompress_unet2_ds8x2_k64-256/` | `experiments/quality_v2_B_larger_ViTvqNoCompress_unet2_ds8x2_k64-256_epoch_metrics.csv`, `*_codebook_metrics.csv`, `experiments/auto_results/*ViTvq*` | 已完成 | 否，历史对照 |
| `quality_v2_B_larger_NoQuant_unet2_ds8x2_k64-256` | no-channel 上限 | 无量化直通，测自编码器上限 | `QUANTIZER_TYPE=none`, larger backbone | `scripts/train/archive/run_exp8_larger_noquant.sh` | `checkpoints/quality_v2_B_larger_NoQuant_unet2_ds8x2_k64-256/` | `experiments/quality_v2_B_larger_NoQuant_unet2_ds8x2_k64-256_epoch_metrics.csv`, `experiments/auto_results/*NoQuant*_no_channel.json` | 已完成 | 否，但作为上限对照保留 |
| `quality_v2_B_larger_cb128-16_VQ_unet2_ds8x2_k128-16` | 量化器对照 | 原始 trainable VQ | `QUANTIZER_TYPE=vq`, `K=128,16` | `scripts/train/archive/run_exp10_larger_cb128_16_vq.sh` | `checkpoints/quality_v2_B_larger_cb128-16_VQ_unet2_ds8x2_k128-16/` | `experiments/quality_v2_B_larger_cb128-16_VQ_unet2_ds8x2_k128-16_epoch_metrics.csv`, `*_codebook_metrics.csv`, `experiments/auto_results/*VQ*` | 已完成 | 否，历史对照 |
| `quality_v2_B_larger_cb128-16_DIV2K-Flickr2K_768x512_unet2_ds8x2_k128-16` | 数据集/高分辨率历史 | DIV2K + Flickr2K，768x512 resize | archive dataset, model parallel, `K=128,16` | `scripts/train/archive/run_exp11_larger_cb128_16_div2k_flickr2k_768x512.sh` | `checkpoints/quality_v2_B_larger_cb128-16_DIV2K-Flickr2K_768x512_unet2_ds8x2_k128-16/` | `experiments/quality_v2_B_larger_cb128-16_DIV2K-Flickr2K_768x512_unet2_ds8x2_k128-16_epoch_metrics.csv`, `*_codebook_metrics.csv`, `experiments/auto_results/*DIV2K*` | 已完成 | 否，历史对照 |
| `quality_v2_B_LPIPS_unet2_ds8x2_k64-256` | 历史实验 | 感知损失实验 | `SIMVQ_LPIPS_WEIGHT=0.1`, stage=B | `scripts/train/archive/run_exp1_lpips.sh` | `checkpoints/quality_v2_B_LPIPS_unet2_ds8x2_k64-256/` | `experiments/quality_v2_B_LPIPS_unet2_ds8x2_k64-256_epoch_metrics.csv`, `experiments/quality_v2_B_LPIPS_unet2_ds8x2_k64-256_*.json` | 已完成 | 否，历史对照 |
| `quality_v2_B_SwinEnhance_unet2_ds8x2_k64-256` | 历史实验 | SwinIR 后处理增强 | `SIMVQ_USE_SWINIR_ENHANCE=1`, blocks=4 | `scripts/train/archive/run_exp3_swin.sh` | `checkpoints/quality_v2_B_SwinEnhance_unet2_ds8x2_k64-256/` | `experiments/quality_v2_B_SwinEnhance_unet2_ds8x2_k64-256_epoch_metrics.csv`, `experiments/interim_results/*SwinEnhance*` | 已完成/历史 | 否，历史对照 |
| `quality_v2_B_DynSwinEnhance_unet2_ds8x2_k64-256` | 历史实验 | 更重 SwinIR + base=96 | `SIMVQ_USE_SWINIR_ENHANCE=1`, blocks=6, `BASE_CHANNELS=96` | `scripts/train/archive/run_exp4_dynamic_swin.sh` | `checkpoints/quality_v2_B_DynSwinEnhance_unet2_ds8x2_k64-256/` | `experiments/quality_v2_B_DynSwinEnhance_unet2_ds8x2_k64-256_epoch_metrics.csv`, `experiments/interim_results/*DynSwin*` | 约 103 epoch 快照 | 否，历史对照 |
| `quality_v2_B_larger_rate042_A_patch_cb8-16_unet2_ds8x2_k8-16` | rate042 待确认 | patch 低码率尝试 | `K=8,16` | 当前未找到对应顶层脚本 | `checkpoints/quality_v2_B_larger_rate042_A_patch_cb8-16_unet2_ds8x2_k8-16/` | `experiments/quality_v2_B_larger_rate042_A_patch_cb8-16_unet2_ds8x2_k8-16_epoch_metrics.csv` | 仅见 1 epoch | 待确认 |
| `quality_v2_B_larger_rate042_B_hybridcvq_cb8192-16384_unet2_ds8x2_k8192-16384` | rate042 待确认 | hybrid CVQ 尝试 | `K=8192,16384`，细节待确认 | 当前未找到对应脚本 | `checkpoints/quality_v2_B_larger_rate042_B_hybridcvq_cb8192-16384_unet2_ds8x2_k8192-16384/` | `experiments/tensorboard/quality_v2_B_larger_rate042_B_hybridcvq_cb8192-16384_unet2_ds8x2_k8192-16384/` | checkpoint 目录无 `.pth` | 待确认 |
| `quality_v2_B_larger_rate042_C_hybridcvq_nested_cb8192-16384_unet2_ds8x2_k8192-16384` | rate042 待确认 | nested hybrid CVQ 尝试 | `K=8192,16384`，细节待确认 | 当前未找到对应脚本 | `checkpoints/quality_v2_B_larger_rate042_C_hybridcvq_nested_cb8192-16384_unet2_ds8x2_k8192-16384/` | `experiments/tensorboard/quality_v2_B_larger_rate042_C_hybridcvq_nested_cb8192-16384_unet2_ds8x2_k8192-16384/` | checkpoint 目录无 `.pth` | 待确认 |
| `debug_B_hybridcvq_unet2_ds8x2_k8192-16384` | debug 待确认 | hybrid CVQ 调试 | debug 名称，细节待确认 | 当前未找到对应脚本 | `checkpoints/debug_B_hybridcvq_unet2_ds8x2_k8192-16384/` | `experiments/tensorboard/debug_B_hybridcvq_unet2_ds8x2_k8192-16384/` | checkpoint 目录无 `.pth` | 待确认 |

## 5. A/B/C 消融实验区别

项目中存在两套容易混淆的 A/B/C 语境。

### 5.1 早期严格 A/B/C：curriculum -> backbone -> full

这套由 `config.py` 中 `_stage_settings(stage)` 和 `SIMVQ_EXPERIMENT_STAGE` 控制。

```bash
SIMVQ_EXPERIMENT_STAGE=A
SIMVQ_EXPERIMENT_STAGE=B
SIMVQ_EXPERIMENT_STAGE=C
```

区别：

| 阶段 | 含义 | 关键差异 |
|---|---|---|
| A = curriculum | 只验证课程学习和低码率方案 | BatchNorm, PReLU, encoder/decoder res blocks=1/1, nearest upsample, no attention |
| B = A + backbone | 在 A 基础上升级主干网络 | GroupNorm, SiLU, encoder/decoder res blocks=2/2, bilinear upsample |
| C = B + MS-SSIM / attention | 在 B 基础上加入 full 方案因素 | bottleneck attention 开启；文档中称 MS-SSIM/attention，但当前 `config.py` 中 MS-SSIM 权重为 0.0，实际应以代码为准 |

对应 checkpoint：

```text
checkpoints/quality_v2_A_curriculum_unet2_ds8x2_k16-32/
checkpoints/quality_v2_B_backbone_unet2_ds8x2_k16-32/
checkpoints/quality_v2_C_full_unet2_ds8x2_k16-32/
checkpoints/quality_v2_A_curriculum_unet2_ds8x2_k64-256/
checkpoints/quality_v2_B_backbone_unet2_ds8x2_k64-256/
checkpoints/quality_v2_C_full_unet2_ds8x2_k64-256/
```

### 5.2 当前 rate041/044 A/B/C：patch -> hybrid CVQ -> nested

这套当前更重要，来自 `scripts/train/current/`。

| 当前方案 | 含义 | 关键环境变量 |
|---|---|---|
| A | patch-wise SimVQ baseline | `SIMVQ_EXP_FAMILY=quality_v2_B_larger_rate044_A_patch_cb16-2`, `SIMVQ_NUM_EMBEDDINGS_LIST=16,2`, `SIMVQ_QUANTIZER_AXIS_LIST=patch,patch` |
| B | A/B 对比中的 hybrid CVQ | `SIMVQ_EXP_FAMILY=quality_v2_B_larger_rate041_B_hybridcvq_cb65536-8192`, `SIMVQ_NUM_EMBEDDINGS_LIST=65536,8192`, `SIMVQ_QUANTIZER_AXIS_LIST=channel,patch`, `SIMVQ_CVQ_CODEWORD_SHAPES=32x32,patch` |
| C | B + nested channel dropout | B 的配置 + `SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA=0.25` |

注意：

- 当前 rate041/044 A/B/C 的 `SIMVQ_EXPERIMENT_STAGE` 都设置为 `B`，因为它们共同使用 B/larger 风格主干。
- 当前 A/B/C 的核心差异不是 `_stage_settings(A/B/C)`，而是量化轴、码本大小、CVQ codeword shape 和 nested channel dropout。

## 6. test_real.py 如何测试

`test_real.py` 是当前主测试入口，支持 no-channel 和真实链路测试。

### 6.1 no-channel 测试

no-channel 测源重建上限，不经过 LDPC/BPSK/AWGN。

示例：

```bash
SIMVQ_EXPERIMENT_STAGE=B \
SIMVQ_EXP_FAMILY=quality_v2_B_larger_rate041_C_hybridcvq_nested_cb65536-8192 \
SIMVQ_NUM_EMBEDDINGS_LIST=65536,8192 \
SIMVQ_DOWNSAMPLE_STRIDES=8,2 \
SIMVQ_UNET_DEPTH=2 \
SIMVQ_BASE_CHANNELS=128 \
SIMVQ_ENCODER_RES_BLOCKS=4 \
SIMVQ_DECODER_RES_BLOCKS=4 \
SIMVQ_QUANTIZER_TYPE=simvq \
SIMVQ_QUANTIZER_AXIS_LIST=channel,patch \
SIMVQ_CVQ_CODEWORD_SHAPES=32x32,patch \
SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA=0.25 \
python -u test_real.py \
  --checkpoint checkpoints/quality_v2_B_larger_rate041_C_hybridcvq_nested_cb65536-8192_unet2_ds8x2_k65536-8192/best_vq_deepsc.pth \
  --no-channel \
  --json-output experiments/eval_manual_C_no_channel.json
```

### 6.2 LDPC + BPSK + AWGN 测试

真实链路测试流程：

```text
image
-> encoder / quantizer
-> quantized indices
-> bits
-> LDPC encode
-> BPSK or QPSK modulation
-> AWGN channel
-> LLR
-> LDPC decode
-> bits back to indices
-> decoder reconstruction
-> MS-SSIM / PSNR
```

示例：

```bash
scripts/eval/test_C_hybrid_nested_rate041.sh \
  --checkpoint checkpoints/quality_v2_B_larger_rate041_C_hybridcvq_nested_cb65536-8192_unet2_ds8x2_k65536-8192/best_vq_deepsc.pth \
  --snrs 0 \
  --modulation bpsk \
  --json-output experiments/eval_manual_C_snr0_bpsk.json
```

也可以直接调用：

```bash
python -u test_real.py \
  --checkpoint <checkpoint_path> \
  --snrs 0 3 6 9 12 \
  --modulation bpsk \
  --json-output <result.json>
```

### 6.3 权重与环境变量必须匹配

`test_real.py` 会从 checkpoint 推断码本大小，但模型结构仍依赖当前环境变量中的配置，例如：

- `SIMVQ_EXPERIMENT_STAGE`
- `SIMVQ_BASE_CHANNELS`
- `SIMVQ_ENCODER_RES_BLOCKS`
- `SIMVQ_DECODER_RES_BLOCKS`
- `SIMVQ_DOWNSAMPLE_STRIDES`
- `SIMVQ_QUANTIZER_AXIS_LIST`
- `SIMVQ_CVQ_CODEWORD_SHAPES`
- `SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA`

因此测试某个 checkpoint 时，必须使用与该 checkpoint 对应的 shell 脚本或手动设置完全匹配的环境变量。

当前最安全的测试入口：

```bash
scripts/eval/test_A_patch_rate044.sh
scripts/eval/test_B_hybrid_rate041.sh
scripts/eval/test_C_hybrid_nested_rate041.sh
```

## 7. 当前关键结果位置

当前 A/B/C 重点结果主要在：

```text
experiments/eval_best_20260609_rate041_large_to_small/
experiments/eval_best_20260609_kodak256_snr0/
experiments/eval_best_20260609_kodak768_snr0/
```

其中：

- `rate041_large_to_small`：当前 A/B/C 的 no-channel 与链路复测结果。
- `kodak256_snr0`：Kodak 256x256 no-resize，LDPC 1/2 + BPSK + SNR=0。
- `kodak768_snr0`：Kodak 768x512 resize，LDPC 1/2 + BPSK + SNR=0。

## 8. 脚本目录说明

当前脚本整理为：

```text
scripts/
├── eval/
│   ├── test_A_patch_rate044.sh
│   ├── test_B_hybrid_rate041.sh
│   └── test_C_hybrid_nested_rate041.sh
└── train/
    ├── run_train.sh
    ├── current/
    │   ├── run_exp12_rate044_A_patch_cb16_2.sh
    │   ├── run_exp13_rate041_B_hybridcvq_cb65536_8192.sh
    │   └── run_exp14_rate041_C_hybridcvq_nested_cb65536_8192.sh
    ├── baseline/
    │   ├── run_exp2_larger.sh
    │   └── run_exp9_larger_cb128_16.sh
    └── archive/
        ├── run_exp1_lpips.sh
        ├── run_exp3_swin.sh
        ├── run_exp4_dynamic_swin.sh
        ├── run_exp5_larger_cb4096_65536.sh
        ├── run_exp6_larger_cb16384_256.sh
        ├── run_exp7_larger_vitvq_nocompress_k64_256.sh
        ├── run_exp8_larger_noquant.sh
        ├── run_exp10_larger_cb128_16_vq.sh
        └── run_exp11_larger_cb128_16_div2k_flickr2k_768x512.sh
```

说明：

- `current/`：当前主线训练脚本。
- `baseline/`：当前主线依赖或引用的重要基准脚本。
- `archive/`：历史实验和对照实验脚本。归档不代表废弃，仍可用于复现。
- `eval/`：当前主线测试脚本。

## 9. 待确认项

以下目录或实验不能直接删除，也不能擅自归类为废弃：

```text
checkpoints/debug_B_hybridcvq_unet2_ds8x2_k8192-16384/
checkpoints/quality_v2_B_larger_rate042_B_hybridcvq_cb8192-16384_unet2_ds8x2_k8192-16384/
checkpoints/quality_v2_B_larger_rate042_C_hybridcvq_nested_cb8192-16384_unet2_ds8x2_k8192-16384/
```

原因：

- 这些目录当前未发现 `.pth` 权重文件。
- 但 `experiments/tensorboard/` 中存在相关目录。
- 不能仅凭目录为空或文件名判断其无效。

另一个待确认实验：

```text
quality_v2_B_larger_rate042_A_patch_cb8-16_unet2_ds8x2_k8-16
```

该实验有 checkpoint 和 epoch CSV，但当前扫描未找到对应训练 shell，且 CSV 显示记录很少，应继续标记为待确认。

## 10. 新增实验规范

新增实验时建议遵守：

1. 不复制 `train.py`、`test_real.py` 或模型源码。
2. 通过新增 shell 脚本设置环境变量表达实验差异。
3. `SIMVQ_EXP_FAMILY` 必须唯一、可读、能表达实验目的。
4. checkpoint、CSV、log、JSON 文件名必须能反推出实验名。
5. 如果改变测试集、resize、SNR、modulation，必须写进结果 JSON 文件名或旁边文档。
6. 新实验完成后更新本索引。
7. 不确定用途的产物必须标记为“待确认”，不能删除。

