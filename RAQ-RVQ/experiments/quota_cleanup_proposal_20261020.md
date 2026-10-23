# /workspace 配额清理与归档记录

日期：2026-10-20（Asia/Shanghai）。用户已确认将下列 27 个文件移到 `/var/tmp` 临时归档，保留文件和迁回记录，随后在原保存目录、GPU 3 上从头重跑训练。

## 原因与恢复计划

- 账号 yi 在 /workspace 的个人硬配额为 300 GiB；盘点时仅剩约 99 MiB。
- uniform_channel1 训练完成第 1/200 个 epoch 后，保存 last_checkpoint.pth 写盘失败而退出。
- 第 1 轮 best_vq_deepsc.pth 通过 ZIP 校验并可用 torch.load 加载，浮点权重均为有限值；last_checkpoint.pth 不完整。
- 续训检查点缺少完整优化器状态；清理确认并释放配额后，在物理 GPU 3 从头重跑 200 epochs，权重仍保存到原实验目录。
- 现有模型结构、K 采样、信道设置、数据和训练超参数不因存储处理而改变。

## 候选范围

以下路径均相对于 `/workspace/yi/work/RAQ-RVQ/checkpoints/`。共 27 个文件，按文件实际占用合计约 11.810 GiB。

每组只包含下面逐项列出的文件。保留各方案的 best_vq_deepsc.pth、epoch_0200_vq_deepsc.pth 和有效的 last_checkpoint.pth。中间 epoch 权重用于回看训练过程，移出或删除后无法再从原路径读取这些中间版本。

### 1. 单尺度 src64：9 个中间权重，约 0.985 GiB

目录：`shiyan_independent_raq_rvq_single_scale_src64_trg2-64_d2_curriculum_rate094_A_patch_ch256_res6-6_mixture_per_k_unet1_ds8_k64/`

- epoch_0020_vq_deepsc.pth
- epoch_0040_vq_deepsc.pth
- epoch_0060_vq_deepsc.pth
- epoch_0080_vq_deepsc.pth
- epoch_0100_vq_deepsc.pth
- epoch_0120_vq_deepsc.pth
- epoch_0140_vq_deepsc.pth
- epoch_0160_vq_deepsc.pth
- epoch_0180_vq_deepsc.pth

### 2. src128：9 个中间权重，约 5.593 GiB

目录：`shiyan_independent_raq_rvq_src128-128_trg2-128_d2_curriculum_rate094_A_patch_ch256-512_res6-6_mixture_per_k_unet2_ds8x2_k128/`

- epoch_0020_vq_deepsc.pth
- epoch_0040_vq_deepsc.pth
- epoch_0060_vq_deepsc.pth
- epoch_0080_vq_deepsc.pth
- epoch_0100_vq_deepsc.pth
- epoch_0120_vq_deepsc.pth
- epoch_0140_vq_deepsc.pth
- epoch_0160_vq_deepsc.pth
- epoch_0180_vq_deepsc.pth

### 3. src16：2 个中间权重，约 1.240 GiB

目录：`shiyan_independent_raq_rvq_src16-16_trg2-16_d2_curriculum_rate094_A_patch_ch256-512_res6-6_mixture_per_k_unet2_ds8x2_k16/`

- epoch_0160_vq_deepsc.pth
- epoch_0180_vq_deepsc.pth

### 4. src32：2 个中间权重，约 1.240 GiB

目录：`shiyan_independent_raq_rvq_src32-32_trg2-32_d2_curriculum_rate094_A_patch_ch256-512_res6-6_mixture_per_k_unet2_ds8x2_k32/`

- epoch_0160_vq_deepsc.pth
- epoch_0180_vq_deepsc.pth

### 5. src64：4 个中间权重，约 2.482 GiB

目录：`shiyan_independent_raq_rvq_src64-64_trg2-64_d2_curriculum_rate094_A_patch_ch256-512_res6-6_mixture_per_k_unet2_ds8x2_k64/`

- epoch_0120_vq_deepsc.pth
- epoch_0140_vq_deepsc.pth
- epoch_0160_vq_deepsc.pth
- epoch_0180_vq_deepsc.pth

### 6. uniform_channel1：1 个损坏的续训检查点，约 0.270 GiB

目录：`shiyan_independent_raq_rvq_src64-64_trg2-64_d2_uniform_channel1_rate094_A_patch_ch256-512_res6-6_unet2_ds8x2_k64/`

- last_checkpoint.pth（ZIP 结构不完整，不能正常续训）

## 处理选项（用户已确认并执行第 1 项）

1. 将上述 27 个文件移至 `/var/tmp/yi/raq-rvq-quota-archive-20261020/`，保持相对目录结构，保存逐文件大小、哈希及原路径映射。释放配额约 11.810 GiB，文件可迁回；归档位于临时存储。
2. 永久删除上述 27 个文件，并保存删除清单、逐文件大小及哈希。释放配额约 11.810 GiB。
3. 保留这些候选文件，继续盘点其他对象。

执行第 1 项后已核对个人配额和 GPU 3 占用，并重新启动训练。本记录与归档脚本通过 apply_patch 留下可审阅变更。

## 已授权的执行方式

归档脚本：`scripts/maintenance/archive_quota_checkpoints_20261020.py`。范围固定为上文 27 个文件；跨磁盘复制后验证 SHA-256，校验通过才移除原文件，并逐文件更新迁移记录。

执行命令（项目根目录）：

```bash
python -u scripts/maintenance/archive_quota_checkpoints_20261020.py --archive
```

机器可读记录同时保存到：

- `experiments/quota_archive_manifest_20261020.json`
- `/var/tmp/yi/raq-rvq-quota-archive-20261020/manifest.json`

需要迁回时，先确认 `/workspace` 的个人配额足够。可按记录中的相对路径逐个恢复，恢复命令也会验证哈希；已有原路径文件时拒绝覆盖：

```bash
python scripts/maintenance/archive_quota_checkpoints_20261020.py --restore --relative-path '清单中的相对路径'
```

省略 `--relative-path` 会迁回尚未恢复的中间 epoch 权重。按用户随后明确的要求，已知损坏旧断点继续留在归档位置，不迁回；任何已有目标文件仍拒绝覆盖。逐文件记录包含最终迁回路径和时间。

## 执行与验证结果

- 27 个批准文件均已完成跨磁盘归档，每个文件都在复制后通过 SHA-256 校验才移除原文件。
- 归档实际占用合计约 11.810 GiB，原磁盘个人配额使用量从约 299.903 GiB 降至约 288.093 GiB，剩余约 11.907 GiB（训练重启前读数）。
- 其余 22 个检查点文件的 inode、大小和修改时间与归档前一致，包括各方案保留的最佳、最终和有效续训权重。
- 归档日志：`experiments/logs/quota_archive_20261020.log`。逐文件原路径、归档路径、大小、SHA-256 和迁移时间见上述两份 manifest。
- 在原 uniform_channel1 训练目录中，使用一个已有的完整本地 Adam 续训检查点进行 CPU 加载、重新保存、fsync 和 ZIP 全量校验。成功保存 1,996,845,899 字节、校验 2,514 个 ZIP 条目；验证通过后移除该临时测试文件。报告：`experiments/checkpoint_storage_probe_20261020.json`，完成于 2026-10-20 10:22:15（Asia/Shanghai）。

## GPU 3 重启记录

- 启动时间：2026-10-20 10:22:47（Asia/Shanghai）。
- Run ID：`independent-raq-rvq-uniform-channel1_gpu3-20261020-102247-quota-restart`。
- 设置：物理 GPU 3，`SIMVQ_RESUME=0`，200 epochs，总 batch 和 micro batch 均为 24，随机种子 42。
- 训练算法与训练、测试脚本保持原样；使用现有训练脚本启动。四个 K 全程独立均匀采样，ChannelProb=1，SNR 范围为 0～15 dB。
- 权重目录仍为 `checkpoints/shiyan_independent_raq_rvq_src64-64_trg2-64_d2_uniform_channel1_rate094_A_patch_ch256-512_res6-6_unet2_ds8x2_k64/`。
- 训练日志：`experiments/logs/train_independent-raq-rvq-uniform-channel1_gpu3-20261020-102247-quota-restart.log`。
- 启动参数和原失败运行关联记录：`experiments/logs/train_independent-raq-rvq-uniform-channel1_gpu3-20261020-102247-quota-restart.json`。
- 启动器 PID：2765132。训练仍在进行，未声明 200 epochs 已完成。

## 用户授权迁回（2026-10-20）

释放 TensorBoard 日志空间后，用户要求将归档文件迁回原位置，随后明确“旧的损坏断点就不需要管它了”。最终执行范围为 26 个中间 epoch 权重，损坏旧断点保留在归档位置。

- 26 个中间 epoch 权重迁回原目录和原文件名。
- 旧损坏断点不迁回、不改名，继续保留在原 `/var/tmp` 归档位置。当前训练正在使用的有效 `last_checkpoint.pth` 保持原有安排。
- 每个文件复制后验证 SHA-256，校验通过后才移除 `/var/tmp` 中的对应副本。
- 两份迁移 manifest 继续保留原始归档信息，并增加迁回状态、时间和最终路径。
- 本次只更新归档与恢复工具和迁移记录，训练参数和训练进程保持原有安排。

执行命令（项目根目录）：

```bash
python -u scripts/maintenance/archive_quota_checkpoints_20261020.py --restore
```

迁回执行日志：`experiments/logs/quota_restore_20261020.log`。

### 迁回完成与验证

- 26 个中间 epoch 权重已全部迁回原目录和原文件名，实际占用合计约 11.54042 GiB。每个文件均经过 SHA-256 校验，原修改时间保留。
- 旧损坏断点仍保留在原 `/var/tmp` 归档位置，状态维持 `archived_verified`，没有迁回或改名。
- 其余 21 个固定检查点文件的 inode、大小和修改时间与迁回前一致。
- 重跑训练使用的有效 `last_checkpoint.pth` 保留，文件大小为 1,996,894,875 字节，ZIP 结构可正常读取。GPU 3 训练进程 PID 2765144 继续运行，核对时已推进至第 3/200 轮、Step 461/540。
- 为落实用户补充的排除范围，首批迁回在完成 21 个文件后主动中断，再处理剩余 5 个。日志中的 `KeyboardInterrupt` 对应这次主动范围调整，不是再次出现写盘错误；由该操作产生的临时复制文件已校验并清理，相关记录保存在 manifest 的 `restore_scope_updates` 中。
- 当前配额读数及完整迁回汇总保存在两份 manifest 的 `restore_summary` 中。
- 已验证恢复工具会跳过已恢复文件和旧损坏断点，同时保留当前有效检查点。训练与测试代码未因迁回操作而调整。
