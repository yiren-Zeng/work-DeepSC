# 五阶段独立测试脚本

这些脚本逐一测试单教师可变码率 RAQ pipeline 的五个阶段。它们不会在某个
checkpoint 缺失时回退到上一阶段，并会校验 checkpoint 中的 `stage` metadata。

## 脚本与默认测试内容

| 脚本 | Checkpoint stage | 默认测试 |
|---|---|---|
| `test_stage1_src_teacher_gpu2.sh` | `src_teacher` | SRC `2048x2048` |
| `test_stage2_identity_warmup_gpu2.sh` | `identity_warmup` | 四个 near-max Profile |
| `test_stage3_variable_rate_gpu2.sh` | `variable_rate` | 固定六锚点 |
| `test_stage4_joint_lite_gpu2.sh` | `joint_lite` | 固定六锚点 |
| `test_stage5_channel_finetune_gpu2.sh` | `channel_finetune` | clean 六锚点 + 固定 SNR 曲线 |

Stage 1 使用普通 `DeepSC` SRC 路径；Stage 2–5 使用
`VariableRateDeepSC`。Stage 5 的 clean 测试保留 teacher guard，固定 SNR
测试关闭 clean teacher guard，并显式启用索引信道。

## 运行

```bash
cd /workspace/yi/work/shiyan-2
GPU_ID=2 bash scripts/eval/variable_rate/test_stage1_src_teacher_gpu2.sh
GPU_ID=2 bash scripts/eval/variable_rate/test_stage2_identity_warmup_gpu2.sh
GPU_ID=2 bash scripts/eval/variable_rate/test_stage3_variable_rate_gpu2.sh
GPU_ID=2 bash scripts/eval/variable_rate/test_stage4_joint_lite_gpu2.sh
GPU_ID=2 bash scripts/eval/variable_rate/test_stage5_channel_finetune_gpu2.sh
```

Stage 3–5 测试全部 121 个 Profile：

```bash
GPU_ID=2 ALL_PROFILES=1 \
bash scripts/eval/variable_rate/test_stage3_variable_rate_gpu2.sh
```

快速单 batch 检查：

```bash
GPU_ID=2 MAX_BATCHES=1 NUM_WORKERS=0 \
bash scripts/eval/variable_rate/test_stage3_variable_rate_gpu2.sh
```

只打印最终命令、不加载 checkpoint 或数据：

```bash
DRY_RUN=1 bash scripts/eval/variable_rate/test_stage4_joint_lite_gpu2.sh
```

Stage 5 可配置：

```bash
CHANNEL_SNRS="0 6 12 15" \
CHANNEL_CODING_RATE=0.5 \
CHANNEL_MOD_BITS=1 \
bash scripts/eval/variable_rate/test_stage5_channel_finetune_gpu2.sh
```

- `RUN_CLEAN=0`：跳过 clean 测试。
- `RUN_CHANNEL=0`：跳过固定 SNR 测试。
- `PROFILES="2048x2048;2048x16"`：覆盖默认 Profile。
- `CHECKPOINT=...`、`TEACHER_CHECKPOINT=...`：覆盖默认 checkpoint。
- `DATASET=...`、`TEST_RESIZE=...`：覆盖测试数据及尺寸。

所有结果写入 `experiments/eval/` 下的独立 Stage 目录。评分显式使用
`0.8 × weighted mean PSNR + 0.2 × worst PSNR`，与训练 checkpoint 选择口径一致。
