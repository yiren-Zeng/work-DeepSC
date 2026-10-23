# 新增两组 last-two copy-cascade 训练

沿用 `scripts/train/train_k64_16_last2_cascade_200ep.sh` 的训练条件，分别新增：

| 实验 | 活跃码本 | 物理 GPU | Epoch |
|---|---|---|---|
| k64_64_last2_copy_200ep | [64,64] | 3 | 200 |
| k512_16_last2_copy_200ep | [512,16] | 4 | 200 |

Cars196 训练及验证集、seed=42、batch=24、micro batch=24、Adam（lr=1.75e-5，betas=(0.5,0.999)）、StepLR（100 epoch 后乘 0.5）、commitment cost=0.25、模型结构、数据预处理和损失函数均沿用原方案。首次启动从头训练，自动恢复只读取各自独立目录的 `last_checkpoint.pth`。

## 新增脚本

- `scripts/train/train_k64_64_last2_cascade_200ep.sh`
- `scripts/train/train_k512_16_last2_cascade_200ep.sh`
- `scripts/test/test_k64_64_last2_cascade_200ep.sh`
- `scripts/test/test_k512_16_last2_cascade_200ep.sh`
- `scripts/train/supervise_new_last2_cascade_200ep.sh`

测试条件沿用原测试脚本，只替换实验名、码本、对应权重及 GPU。现有方案和公共代码均未修改；启动前后核对了 77 个已有源码文件的 SHA256。哈希和训练条件保存在 `launch_manifest.json`。

## 后台运行与恢复

两个守护进程通过 `nohup` 和独立会话启动，脱离启动终端，忽略 SIGHUP。每个实验有独立文件锁，重复启动会退出，避免同一方案启动两份。训练进程异常退出后，守护进程等待 60 秒并自动从该实验最近保存的完整 epoch 恢复；原训练入口每个 epoch 保存完整训练检查点（模型、优化器、调度器和随机状态）。完成 200 epoch 后不再重启。

终端关闭或 SSH 断开不影响训练。机器关机或重启会停止进程，之后可重新运行守护命令并从检查点恢复。`runtime/<experiment>/STOP` 文件会禁用退出后的自动重启，但不会立即停止正在进行的训练。

输出根目录：`/home/yi/vq_deepsc_runs_20260919`。

- 训练日志：`console/train_<experiment>.log`
- 守护日志：`console/supervisor_<experiment>.log`
- 权重：`checkpoints/<experiment>/best_vq_deepsc.pth`
- 续训检查点：`checkpoints/<experiment>/last_checkpoint.pth`
- TensorBoard：`logs/<experiment>/`
- 运行状态与 PID：`runtime/<experiment>/`

查看日志：

```bash
tail -f /home/yi/vq_deepsc_runs_20260919/console/train_k64_64_last2_copy_200ep.log
tail -f /home/yi/vq_deepsc_runs_20260919/console/train_k512_16_last2_copy_200ep.log
```

需要在机器重启后恢复时：

```bash
cd /workspace/yi/work/VQ-DeepSC
nohup bash scripts/train/supervise_new_last2_cascade_200ep.sh k64_64_last2_copy_200ep >> /home/yi/vq_deepsc_runs_20260919/console/supervisor_k64_64_last2_copy_200ep.log 2>&1 < /dev/null &
nohup bash scripts/train/supervise_new_last2_cascade_200ep.sh k512_16_last2_copy_200ep >> /home/yi/vq_deepsc_runs_20260919/console/supervisor_k512_16_last2_copy_200ep.log 2>&1 < /dev/null &
```
