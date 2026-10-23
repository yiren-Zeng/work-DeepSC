# 三个 TensorBoard 日志文件删除记录

日期：2026-10-20（Asia/Shanghai）。

## 用户授权与操作范围

用户明确授权：删除前面确认的三个 `events.out.tfevents.…` 文件，保留 checkpoint，释放约 82.47 GiB。

本次只对以下三个精确文件路径执行逐文件 unlink，不执行目录递归删除：

| 文件（相对 `/workspace/yi/work/RAQ-VAE-main/runs/`） | 实际占用 |
| --- | ---: |
| `seq2seq_src64_20260909_173605_993462/tensorboard/events.out.tfevents.1788946572.kaisa.167986.0` | 28.63559 GiB |
| `seq2seq_ds8_dualk_src64_20260910_113736_956705/tensorboard/events.out.tfevents.1789011463.kaisa.1521259.0` | 27.06387 GiB |
| `seq2seq_ds8_src64_20260910_005420_629332/tensorboard/events.out.tfevents.1788972866.kaisa.762241.0` | 26.77005 GiB |

预检确认三个文件均为当前用户拥有的普通文件，且没有进程打开这些文件。其 TensorBoard 历史记录将随删除而移除。

## checkpoint 保护与验证

- 三个实验各有 6 个 `.ckpt` 文件，共 18 个，存放在各自独立的 `checkpoints/` 子目录。
- 删除前记录每个 checkpoint 的完整路径、大小、inode、修改时间和 SHA-256。
- 删除后逐项复核上述信息及 checkpoint 文件列表，确保 18 个文件均完整保留。
- 同时核对三个实验目录中其余文件的列表、大小、inode 和修改时间。
- 删除前后读取 `/workspace` 个人配额，核对实际释放量和剩余可用空间。

动态生成的逐文件执行结果与验证数据保存到：`experiments/tensorboard_log_deletion_20261020.json`。执行状态和最终配额以该机器可读记录及下方结果为准。

## 执行结果

- 已完成，仅删除上表三个精确路径的 TensorBoard 日志文件。
- 按文件实际占用共释放 82.46952 GiB；个人配额使用量实测下降约 82.46950 GiB，差异来自同期小型记录写入。
- 三个实验的 18 个 checkpoint 全部保留；删除前后 SHA-256、大小、inode、修改时间均一致。
- 三个实验目录的其余文件列表、大小、inode 和修改时间也均保持一致。
- 删除后 `/workspace` 个人配额已用约 203.07 GiB，剩余约 96.93 GiB，总配额 300 GiB。
- 本次只新增此说明文档及动态生成的同名 JSON 审计记录，不修改训练或测试代码。
