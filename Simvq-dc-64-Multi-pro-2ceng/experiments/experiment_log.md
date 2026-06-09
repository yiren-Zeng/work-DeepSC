# Experiment Log

## Target

- Dataset for final quality: Kodak (24 images).
- Transmission test: LDPC rate 0.5, BPSK, SNR values `0, 3, 6, 9, 12` dB.
- BPG reference currently supplied by `../BPG/bpg_test_output.log`:
  `PSNR=32.30 dB`, `MS-SSIM=0.9777` for
  `kodak-decompress-0.5-bpsk-ratio-0.28`.
- A fair claim of outperforming BPG additionally requires confirming which
  SNR produced that BPG folder and matching effective BPP/channel usage.

## Runs

| Run | Started | Source | Configuration or change | Status | Result |
| --- | --- | --- | --- | --- | --- |
| observed-001 | 2026-05-26 | already running when tracking began | Two-layer SimVQ checkpoint with `K=[64,64]`; active process reported 500 planned epochs, while on-disk config had changed to `K=[128,128]` and 400 epochs | stopped after exporting 90 completed epochs because its no-channel upper bound is below BPG; stdout recovered from deleted descriptor; best epoch is 79 (`Val Recon Loss=0.0097922534`) | BPP `0.4688`; no-channel PSNR/MS-SSIM `26.9998 / 0.9417`; real-chain at SNR 0 dB `26.9464 / 0.9404` |
| quality_v1_unet2_ds4x2_k64 | 2026-05-26 | new quality improvement trial | `K=[64,64]`, lighter VQ weights `[0.25,0.5]` to `[0.25,0.25]`, skip dropout `0.1` to `0.0` by 40% of training, LR `5e-5`, micro-batch `24`, checkpoint directory `checkpoints/quality_v1_unet2_ds4x2_k64` | completed 200 epochs | best checkpoint, last checkpoint, and final epoch-200 checkpoint retained; intermediate periodic checkpoints and duplicated screening checkpoint snapshots removed during cleanup |

## Changes

| Change | Date | Description | Reason |
| --- | --- | --- | --- |
| tracking-001 | 2026-05-26 | Preserve timestamped training logs and resumed checkpoints in `run_train.sh`; append per-epoch training metrics in `train.py`; allow checkpoint/JSON and fast no-channel upper-bound selection in evaluation scripts; save resume state after updating the best validation threshold. | Make repeated trials measurable without destroying previous evidence, screen poor candidates cheaply, and avoid stale best-model state after resume. |
| model-001 | 2026-05-26 | Add `quality_v1_unet2_ds4x2_k64`: preserve baseline `K=[64,64]`/BPP while reducing early skip dropout and VQ-loss dominance, increasing generator LR, and separating checkpoints. | Baseline cannot reach BPG even before channel corruption, so its reconstruction optimization must improve before real-chain evaluation is worthwhile. |
| structure-002 | 2026-05-28 | Add configurable `UNET_DEPTH`; auto-generate downsample strides, embedding dimensions, per-layer codebook sizes, VQ loss weights, skip-dropout lists, source-BPP estimate, and experiment/checkpoint names. Current active checkpoint directory renamed to `quality_v1_unet2_ds4x2_k64`. | Make 2/3/4-layer U-Net experiments switchable by changing the depth number, while keeping checkpoints and logs easy to distinguish. |
| model-002 | 2026-05-28 | 启动 `quality_v2_unet2_ds8x2_k16-32` 低码率终极实验：`DOWNSAMPLE_STRIDES=[8,2]`，`NUM_EMBEDDINGS_LIST=[16,32]`，估算 BPP 为 `0.08203125`；加入 channel_prob 课程学习、GroupNorm、SiLU、级联下采样、bilinear 上采样、瓶颈 Self-Attention、MSE+MS-SSIM 混合损失。 | 在约 0.083 BPP 下对标 BPG，同时尽量缓解码本早期坍缩和大步幅下采样信息损失。 |

## 当前正式实验：quality_v2_unet2_ds8x2_k16-32

### 实验目的

本轮实验尝试在约 `0.083 BPP` 的极低码率下提升 SimVQ 重建质量，并为后续真实链路 `LDPC + BPSK + AWGN` 对比 BPG 做准备。

### 码率计算

```text
DOWNSAMPLE_STRIDES = [8, 2]
NUM_EMBEDDINGS_LIST = [16, 32]
第 1 层：log2(16) / 8^2  = 4 / 64  = 0.0625
第 2 层：log2(32) / 16^2 = 5 / 256 = 0.01953125
总 BPP = 0.08203125
```

### 三阶段训练调度

```text
Epoch 0-79:
  channel_prob = 0.0
  纯信源训练，不经过信道噪声，让 Encoder、SimVQ、Decoder 先学习稳定离散表征。

Epoch 80-119:
  channel_prob 从 0.0 线性增加到 1.0
  每个 batch 按概率决定是否经过信道扰动，让模型逐步适应噪声。

Epoch 120-200:
  channel_prob = 1.0
  全部 batch 开启随机 SNR 信道扰动，进行鲁棒微调。
```

### 网络结构升级

```text
归一化：BatchNorm2d -> GroupNorm
激活函数：PReLU -> SiLU
下采样：stride=8 改为 2x -> 2x -> 2x 级联平滑下采样
上采样：nearest -> bilinear
残差块：编码器/解码器每个采样块中各使用 2 个残差块
瓶颈：最深层特征加入 1 层 Bottleneck Self-Attention
```

### 损失函数

```text
Recon Loss = 0.8 * MSE + 0.2 * (1 - MS-SSIM)
Total Loss = Recon Loss + 分层加权 VQ Loss
```

### 本轮关键产物路径

```text
checkpoints/quality_v2_unet2_ds8x2_k16-32/
experiments/quality_v2_unet2_ds8x2_k16-32_epoch_metrics.csv
experiments/quality_v2_unet2_ds8x2_k16-32_screening.csv
experiments/snapshots/quality_v2_unet2_ds8x2_k16-32/
experiments/logs/train_quality_v2_unet2_ds8x2_k16-32-*.log
```

## Execution Attempts

| Invocation | Scheme | Completed epochs attributable to invocation | Outcome |
| --- | --- | --- | --- |
| observed-001 | baseline configuration found running | 90 | stopped and archived after poor no-channel ceiling |
| quality_v1_unet2_ds4x2_k64-001 | model-001 | 1 | produced the initial checkpoint; interrupted during epoch 2 to detach it cleanly |
| quality_v1_unet2_ds4x2_k64-002-resume | model-001 | 0 | background launch did not remain alive after returning from its parent command |
| quality_v1_unet2_ds4x2_k64-003-resume | model-001 | in progress from epoch 2 | running in an independent session; monitored automatically |
| quality_v2_unet2_ds8x2_k16-32-debug | model-002 | 0 | 前台排错启动；代码进入训练但 GPU 3 显存不足，报 CUDA OOM 后停止；未形成有效 epoch 记录。 |
| quality_v2_unet2_ds8x2_k16-32-timeout | model-002 | 0 | 使用 GPU 0 前台限时 20 秒冒烟测试；成功进入 epoch 1，确认 Phase 1 为 clean channel，随后由 timeout 主动停止。 |
| quality_v2_unet2_ds8x2_k16-32-001 | model-002 | 1 个 epoch 已完成训练和验证，但 CSV 写入失败 | 正式后台训练曾使用 GPU 0；第 1 个 epoch 完成并保存 `best_vq_deepsc.pth`，随后因为 epoch CSV 新增 `channel_prob` 字段而旧表头不兼容，进程退出。已修复 `utils/experiment_io.py` 和 CSV 表头。 |
| quality_v2_unet2_ds8x2_k16-32-002-gpu3 | model-002 | running | 按用户要求切换到 GPU 3 重新启动后续训练；训练 PID `3952922`，监控 PID `3952923`，训练和监控均绑定 GPU 3。训练日志：`experiments/logs/train_quality_v2_unet2_ds8x2_k16-32-002-gpu3.log`；监控日志：`experiments/logs/monitor_quality_v2_unet2_ds8x2_k16-32-002-gpu3.log`。 |
| quality_v2_A_curriculum_unet2_ds8x2_k16-32-001-gpu1 | model-002-A | running | 消融 A：只验证 channel curriculum 和 0.082 BPP 码率方案；保留原始 BatchNorm/PReLU、单残差块、nearest 上采样、无 attention、纯 MSE。训练 PID `3550416`，监控 PID `3550417`，绑定 GPU 1。 |
| quality_v2_B_backbone_unet2_ds8x2_k16-32-001-gpu2 | model-002-B | running | 消融 B：在 A 基础上加入 GroupNorm、SiLU、级联下采样、bilinear 上采样、加深残差块；仍无 attention，仍纯 MSE。训练 PID `3550418`，监控 PID `3550419`，绑定 GPU 2。 |
| quality_v2_C_full_unet2_ds8x2_k16-32-001-gpu3 | model-002-C | running | 严格命名的 C 消融已启动：在 B 基础上加入 MS-SSIM 混合损失和 bottleneck attention。训练 PID `1842320`，监控 PID `1842321`，绑定 GPU 3。 |
| quality_v2_C_full_unet2_ds8x2_k16-32-002-resume-gpu3 | model-002-C | running | 根据用户要求清理周期性 epoch 权重，并修改训练代码后从 `last_checkpoint.pth` 恢复 C 实验；训练 PID `1578484`，监控 PID `1578485`。后续只保留 `best_vq_deepsc.pth` 和 `last_checkpoint.pth`。 |
| quality_v2_A_curriculum_unet2_ds8x2_k16-32-one_step-001-gpu0 | model-002-A-one-step | running | 按用户要求重跑 A：A 关闭级联下采样，第一层使用一步 `stride=8` 卷积；退火时间恢复为 `PHASE1_END=0.1`、`PHASE2_END=0.4`；损失保持纯 MSE + 分层加权 VQ。旧 A checkpoint/CSV/log/snapshot/tensorboard 已归档到 `experiments/archive/quality_v2_A_curriculum_unet2_ds8x2_k16-32_before_one_step_20260529_184342/`。训练 PID `2533143`，监控 PID `2533144`，绑定 GPU 0。训练日志：`experiments/logs/train_quality_v2_A_curriculum_unet2_ds8x2_k16-32-one_step-001-gpu0.log`；监控日志：`experiments/logs/monitor_quality_v2_A_curriculum_unet2_ds8x2_k16-32-one_step-001-gpu0.log`。 |
| quality_v2_B_backbone_unet2_ds8x2_k16-32-one_step-001-gpu2 | model-002-B-one-step | running | 按用户要求重跑 B：B 关闭级联下采样，第一层使用一步 `stride=8` 卷积；退火时间恢复为 `PHASE1_END=0.1`、`PHASE2_END=0.4`；损失保持纯 MSE + 分层加权 VQ。旧 B checkpoint/CSV/log/snapshot/tensorboard 已归档到 `experiments/archive/quality_v2_B_backbone_unet2_ds8x2_k16-32_before_one_step_20260529_192824/`。训练 PID `3175241`，监控 PID `3175242`，绑定 GPU 2。训练日志：`experiments/logs/train_quality_v2_B_backbone_unet2_ds8x2_k16-32-one_step-001-gpu2.log`；监控日志：`experiments/logs/monitor_quality_v2_B_backbone_unet2_ds8x2_k16-32-one_step-001-gpu2.log`。 |
| quality_v2_C_full_unet2_ds8x2_k16-32-one_step_mse-001-gpu3 | model-002-C-one-step-mse | running | 按用户要求重跑 C：C 关闭级联下采样，第一层使用一步 `stride=8` 卷积；退火时间恢复为 `PHASE1_END=0.1`、`PHASE2_END=0.4`；损失改为与 B 一致的纯 MSE + 分层加权 VQ；保留 C 相对 B 的唯一结构差异：1 层 bottleneck attention。旧 C 训练进程已停止，旧 checkpoint/CSV/log/snapshot/tensorboard 已归档到 `experiments/archive/quality_v2_C_full_unet2_ds8x2_k16-32_before_one_step_mse_20260529_193756/`。训练 PID `3181607`，监控 PID `3181608`，绑定 GPU 3。训练日志：`experiments/logs/train_quality_v2_C_full_unet2_ds8x2_k16-32-one_step_mse-001-gpu3.log`；监控日志：`experiments/logs/monitor_quality_v2_C_full_unet2_ds8x2_k16-32-one_step_mse-001-gpu3.log`。 |
| quality_v2_A/B/C_unet2_ds8x2_k64-256 | model-003-k64-256 | running | 用户将码本改为 `[64,256]` 后并行重训 A/B/C。源端 BPP 为 `6/64 + 8/256 = 0.125`，若按 LDPC `R=0.5` 计算 coded bpp 为 `0.25`。A/B/C 分别绑定 GPU 0/1/2，监控筛选使用 GPU 3。训练 run 分别为 `quality_v2_A_curriculum_unet2_ds8x2_k64-256-001-gpu0`、`quality_v2_B_backbone_unet2_ds8x2_k64-256-001-gpu1`、`quality_v2_C_full_unet2_ds8x2_k64-256-001-gpu2`；训练 PID 分别为 `3676989`、`3676991`、`3676993`，监控 PID 分别为 `3676990`、`3676992`、`3676994`。输出 CSV/日志/checkpoint 均使用自动实验名 `*_k64-256`。 |
| quality_v2_B_larger_cb4096-65536_unet2_ds8x2_k4096-65536 | model-004-larger-codebook | stopped, resumable | 基于已完成扩容方案 `quality_v2_B_larger_unet2_ds8x2_k64-256`，仅将码本改为 `[4096,65536]`。保持 `BASE_CHANNELS=128`、编码器/解码器残差块数 `4/4`、`DOWNSAMPLE_STRIDES=[8,2]`、总 batch `24`、micro-batch `24` 和其余 Stage B 条件不变。源端 BPP 为 `12/64 + 16/256 = 0.25`，LDPC `R=0.5` 后发送 BPP 为 `0.5`。原训练 PID `1586181` 已按要求停止；已保留 Epoch 51 后的 `best_vq_deepsc.pth` 与 `last_checkpoint.pth`，可恢复训练。为支持大码本，量化器使用等价的分块最近邻搜索，并移除显式 one-hot 矩阵。 |
| quality_v2_B_larger_cb16384-256_unet2_ds8x2_k16384-256 | model-004-larger-codebook | running | 基于已完成扩容方案 `quality_v2_B_larger_unet2_ds8x2_k64-256`，仅将码本改为 `[16384,256]`。保持 `BASE_CHANNELS=128`、编码器/解码器残差块数 `4/4`、`DOWNSAMPLE_STRIDES=[8,2]`、总 batch `24`、micro-batch `24` 和其余 Stage B 条件不变。源端 BPP 为 `14/64 + 8/256 = 0.25`，LDPC `R=0.5` 后发送 BPP 为 `0.5`。绑定 GPU 1，训练 PID `1586185`。为支持大码本，量化器使用等价的分块最近邻搜索，并移除显式 one-hot 矩阵。 |
| quality_v2_B_larger_ViTvqNoCompress_unet2_ds8x2_k64-256 | model-005-vitvq-nocompress | running | 保留原扩容方案 `quality_v2_B_larger_unet2_ds8x2_k64-256`，新增可选量化器类型 `vitvq_nocompress`。两层码本仍为 `[64,256]`，两层 SimVQ 同时替换为 `QBridgeNoCompress-S`；U-Net 主干、采样倍率、损失、batch 和课程训练条件不变。该实验使用独立 checkpoint、CSV 和日志目录，绑定 GPU 0。训练 PID `623612`，会话 `simvq_exp7_vitvq_gpu0`，日志 `experiments/logs/train_exp7_larger_vitvq_nocompress_k64-256-20260602-011729.log`。 |
| quality_v2_B_larger_NoQuant_unet2_ds8x2_k64-256 | model-006-noquant-upper-bound | running | 基于已达到约 27 dB 的扩容方案 `quality_v2_B_larger_unet2_ds8x2_k64-256`，移除两层量化模块，Encoder 特征直接输入 Decoder。保持扩宽主干、采样倍率、损失、batch、数据集和训练轮数不变。由于不再存在离散索引，该方案关闭离散信道扰动与码本监控，仅用于测量纯自编码器重建上限。绑定物理 GPU 2，训练 PID `2589131`，会话 `simvq_exp8_noquant_gpu2`，日志 `experiments/logs/train_exp8_larger_noquant-20260602-161423.log`。 |
| quality_v2_B_larger_cb128-16_unet2_ds8x2_k128-16 | model-007-simvq-rate-redistribution | running | 基于约 27 dB 的扩容 SimVQ 方案，仅将码本从 `[64,256]` 改为 `[128,16]`。保持扩宽主干、采样倍率、SimVQ 类型、损失、batch、数据集和训练轮数不变。总 Source BPP 仍为 `7/64 + 4/256 = 0.125`，用于验证将码率预算向第一层倾斜的效果。绑定物理 GPU 1，训练 PID `2986254`，会话 `simvq_exp9_cb128-16_gpu1`，日志 `experiments/logs/train_exp9_larger_cb128-16-20260602-223140.log`；恢复权重保存在独立目录 `checkpoints/quality_v2_B_larger_cb128-16_unet2_ds8x2_k128-16/`。 |

Counts as of this record: `12` training process invocations observed/started,
`5` model configurations measured or running, and `2` performance-oriented
model/configuration modifications introduced in this project.

## Ablation Schedule

为了修正此前 A/B/C 一次性融合导致的归因问题，后续消融实验按如下定义分析：

| Stage | Experiment | GPU | Purpose |
| --- | --- | --- | --- |
| Full reference | `quality_v2_unet2_ds8x2_k16-32` | GPU 3 | 当前已经在跑的融合版本，作为 full 参考，不作为逐阶段归因的唯一证据。 |
| A | `quality_v2_A_curriculum_unet2_ds8x2_k16-32` | GPU 1 | 只看课程学习 + 0.082 BPP 码率方案的效果。 |
| B | `quality_v2_B_backbone_unet2_ds8x2_k16-32` | GPU 2 | 在 A 基础上加入基础网络算子和主干结构升级，评估 backbone 改造的增益。 |
| C | `quality_v2_C_full_unet2_ds8x2_k16-32` | pending | 在 B 基础上加入 MS-SSIM 混合损失和 bottleneck attention；可在 A/B/full 任一完成释放 GPU 后启动。 |

说明：

- A/B/C 的定义仍然是递进关系，最终分析按 A -> B -> C 比较。
- 训练可以并行启动，因为每个实验从零训练、使用独立 checkpoint/CSV/log，不存在权重依赖。
- 当前 full 实验先跑完；它和后续严格命名的 C 实验是否都保留，等 A/B 结果出来后再决定。

## Artifacts

- `../checkpoints/quality_v1_unet2_ds4x2_k64`: active `quality_v1_unet2_ds4x2_k64` checkpoints.
- `../checkpoints/observed_001_baseline`: archived baseline checkpoints.
- `quality_v1_unet2_ds4x2_k64_epoch_metrics.csv`: appended automatically on epochs run after
  `tracking-001` is loaded by a new/resumed training process.
- `logs/train_<run_id>.log`: stdout/stderr from future training launches.
- `logs/baseline_epoch088_*.log`: evaluation begun from the best checkpoint
  observed while the active run had reached epoch 88.
- `variants/preexisting_k128_config.py`: preserved copy of the incompatible
  on-disk `K=[128,128]` configuration found before starting `quality_v1_unet2_ds4x2_k64`.
- `../tools/monitor_quality.py`: automatically evaluates the best `quality_v1_unet2_ds4x2_k64`
  checkpoint at epochs 5, 10, and every 10 epochs thereafter; only runs the
  expensive real-chain test once the no-channel quality ceiling meets BPG.
  It writes JSON/CSV metrics only and does not duplicate checkpoint files.
- `quality_v1_unet2_ds4x2_k64_screening.csv`: monitor output table once the first
  screening milestone completes.
