# 单支路 K=32：GPU 4，200 轮正式训练

用户于 2026-10-23 授权在 GPU 4 上运行已验证的单支路方案。
本次只启动训练并记录运行信息，不修改模型、训练参数或已有启动脚本。

## 启动参数

- 工作目录：`/workspace/yi/work/VQ-DeepSC`
- 脚本：`scripts/train/train_k32_first_branch.sh`
- 实验：`k32_first_branch`，尺度 0，码本 `[32]`，单级解码器
- 总轮数：200；batch size 24；其余参数沿用已交付配置
- 物理 GPU：4，UUID `GPU-5f174e61-459c-e68c-b9fd-afb7d3d3b9ad`
- 脚本设置 `CUDA_VISIBLE_DEVICES=4`，进程内设备为 `cuda:0`
- 启动前 GPU 4 显存占用为 0 MiB，无同方案训练进程，正式权重目录不存在
- 默认启用断点恢复；本次没有历史正式权重，因此从头训练，不使用小样本检查权重

等价启动命令如下；实际通过独立进程会话启动，并保存 PID，避免对话结束影响运行：

```bash
cd /workspace/yi/work/VQ-DeepSC
GPU_ID=4 VQ_DEEPSC_NUM_EPOCHS=200 VQ_DEEPSC_RESUME=1 \
VQ_DEEPSC_CHECKPOINT_DIR=/workspace/yi/work/VQ-DeepSC/checkpoints/k32_first_branch \
VQ_DEEPSC_LOG_DIR=/workspace/yi/work/VQ-DeepSC/logs/k32_first_branch \
PYTHONDONTWRITEBYTECODE=1 \
nohup bash scripts/train/train_k32_first_branch.sh \
  > experiments/train/k32_first_branch_200ep_gpu4_20261023/launcher.log 2>&1 < /dev/null &
```

## 输出与审阅记录

- [launch_manifest.json](launch_manifest.json)：实际命令、环境参数、启动 PID、时间和输出位置
- [launcher.log](launcher.log)：后台启动及训练终端日志
- 主训练日志：`experiments/logs/train_k32_first_branch.log`
- 最优权重：`checkpoints/k32_first_branch/best_vq_deepsc.pth`
- 恢复权重：`checkpoints/k32_first_branch/last_checkpoint.pth`
- TensorBoard：`logs/k32_first_branch/`

日志、权重和 TensorBoard 按原有 Git 忽略规则保留；训练中的文件会持续更新。
训练状态以日志和实际进程为准，启动记录不代表 200 轮已完成。

## 启动验证

已确认后台启动进程 PID 为 `878176`，主训练 Python PID 为 `878223`；
会话已独立于启动终端。日志已进入 `Epoch [1/200]` 并连续完成多个训练步骤。
实际进程环境为 `CUDA_VISIBLE_DEVICES=4`、`VQ_DEEPSC_DEVICE=cuda:0`、
`VQ_DEEPSC_NUM_EPOCHS=200`、`VQ_DEEPSC_NUM_EMBEDDINGS_LIST=32`。

`nvidia-smi` 确认 PID `878223` 使用物理 GPU 4 对应 UUID，启动检查时占用
10,864 MiB 显存，未发现 Python 异常。实时步骤和最后一次核对时间见 manifest。

本次文件变更为新增本记录、自动生成 `launch_manifest.json`，以及在
`experiments/first_branch_k32/README.md` 中链接正式训练记录；代码和启动脚本均未改变。
