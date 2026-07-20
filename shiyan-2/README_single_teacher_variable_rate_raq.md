# 单教师可变码本 RAQ（独立方案）

本目录是从原始 `shiyan/` 隔离出来的新方案。新代码、checkpoint、日志和评估结果都写入
`/workspace/yi/work/shiyan-2`；训练/评估脚本不会把输出写回原始 `shiyan/`。

## 1. 方案边界

- 完整教师只有一个：SRC `[2048,2048]`。
- RAQ 是一个统一的可变码率生成器系统，不是 121 个彼此独立的生成器。
- 两层支持 `K_l ∈ {2,4,8,16,32,64,128,256,512,1024,2048}`，所以完整 profile 空间为 121 种 `(K0,K1)`。
- Encoder 和 Decoder 各只有一个共享实例；完整 profile 的连续码率嵌入通过轻量 FiLM/conditional affine 调节特征。
- 旧的 dynamic RAQ-RVQ、routed source codebook 和伪教师路径不参与本方案。
- RAQ 图像重建采用 `dual` 梯度路径，使真实图像重建损失能够直接更新生成码本及 Transformer。

“一个统一生成器”指一个 `VariableRateRAQGenerator` 和一个共享的完整 profile rate conditioner。
由于两个尺度的嵌入维度分别为 `D0` 和 `D1`，系统内部保留两个尺度专用 projection/head；它们不是
按 profile 复制的 121 个完整生成器。

## 2. 逐层码本规则

对第 `l` 层，源码本记为 `W_src,l ∈ R^(2048×D_l)`。

当 `K_l = 2048` 时采用结构硬旁路：

```text
W_raq,l = W_src,l
```

该层不会调用 Transformer。这样 `[2048,2048]`、`[2048,K1]` 和 `[K0,2048]` 都会逐层执行正确的
继承/生成组合。

当 `K_l < 2048` 时：

```text
W_raq,l(K_l) = S_l,K(W_src,l) + DeltaW_theta,l(W_src,l, K0, K1)
```

`S_l,K` 由目标 query 对 2048 个源码字做 cross-attention pooling 得到；Transformer 只预测残差
`DeltaW`。生成器使用完整 `(K0,K1)` 的 rate embedding，因此同一层也能感知另一层的目标码率。

一个必须明确的梯度事实是：`[2048,2048]` 的两层都被硬旁路时，RAQ Generator 不参与前向图，
因此它在该 profile 下数学上不可能得到梯度。Stage 2 虽然必须包含 `[2048,2048]` 做恒等保护和 FiLM
校准，但也必须包含 `[2048,1024]`、`[1024,2048]`、`[1024,1024]` 等 near-max profile，不能只训练
全最大 profile。

## 3. Profile 条件与采样

连续码率输入为：

```text
r = [log2(K0), log2(K1)]
```

小型 MLP 生成共享 rate embedding，并供两层码本生成器、Encoder FiLM 和 Decoder FiLM 使用。

Stage 3～5 的目标集合为 `all`，即自动展开全部 121 个 profile。每个 sandwich 训练窗口包含：

1. 最大 profile `[2048,2048]`；
2. 配置的最小 profile（正式配置为 `[2,2]`）；
3. 可配置数量的随机中间 profile；
4. 随机项优先选择历史出现次数较少的 profile。

采样器状态和每个 profile 的实际出现次数会随 checkpoint 保存。多个 profile 依次前向/反向，loss
按实际 profile 数和实际梯度累积窗口长度归一化，尾部不足一个完整累积窗口时也不会放大梯度。

## 4. 单一冻结教师

Stage 1 训练一个不带 RAQ 的 `[2048,2048]` SRC 模型。Stage 2～5 加载该 checkpoint 为完全独立的
`teacher_model`：

- `teacher_model.eval()`；
- 所有教师参数 `requires_grad=False`；
- 教师前向位于 `no_grad` 区域；
- 教师与 student 不共享参数对象；
- 每个 micro-batch 只执行一次教师前向，再供该窗口内各 profile 使用。

教师提供最大码率重建、Encoder 特征、SRC 量化特征和两个最大源码本。当前 student 的 SRC 分支
`.detach()` 不被当作教师。

## 5. 损失

对一个完整 profile `K=(K0,K1)`：

```text
L_K = L_rec
    + lambda_vq L_vq
    + lambda_out(K) L_output_distill
    + lambda_feat(K) L_feature_distill
    + lambda_identity L_identity
    + lambda_hier L_hierarchy
    + lambda_div L_diversity
```

- `L_rec`：student 与真实图像之间的 MSE/MS-SSIM/真实 LPIPS 损失，是主监督。
- `L_vq`：两层 RAQ VQ loss；逐层权重可从初值线性调度到终值。
- `L_output_distill`：student 重建与冻结教师重建之间的图像损失。
- `L_feature_distill`：两层 student RAQ 量化特征与冻结教师特征的加权 MSE。
- `L_identity`：对 `K_l=2048` 的层检查/保护 `W_raq,l=W_src,l`；结构硬旁路仍是主要保证。
- `L_hierarchy`：令 `W_K` 接近 `Merge(W_2K)`，Merge 对相邻父码字做聚合，不要求“小码本等于大码本前 K 个码字”。
- `L_diversity`：只抽样 `P` 个不同码字对，并计算 `relu(margin-distance)^2`，内存复杂度为 `O(PD)`，不会构造 `K×K` 距离矩阵。

码率归一化分数为：

```text
rho(K) = ((log2(K0)-1) + (log2(K1)-1)) / 20
```

输出和特征蒸馏权重都采用：

```text
lambda(K) = low + (high-low) * rho(K)^gamma
```

默认输出蒸馏为 `low=0.02, high=0.20, gamma=2`，默认特征蒸馏为
`low=0.01, high=0.10, gamma=2`。因此低码率主要由真实图像监督，高码率获得更强的教师约束。

## 6. 五阶段正式训练

所有脚本默认使用物理 GPU 2：先设置 `CUDA_VISIBLE_DEVICES=2`，Python 内始终使用逻辑
`cuda:0`。默认 Python 为 `/home/yi/.conda/envs/work/bin/python`。

| 阶段 | 脚本 | 默认 epoch | 训练重点 | 必需输入 |
|---|---|---:|---|---|
| 1 | `run_stage1_src_teacher_gpu2.sh` | 200 | 只训练 SRC `[2048,2048]` | 无 |
| 2 | `run_stage2_identity_warmup_gpu2.sh` | 20 | max 恒等保护 + near-max 训练 Generator/Rate/FiLM | Stage 1 teacher |
| 3 | `run_stage3_variable_rate_gpu2.sh` | 120 | 无信道、全部 121 profile、sandwich | teacher + Stage 2 student |
| 4 | `run_stage4_joint_lite_gpu2.sh` | 40 | Generator + 小学习率 Decoder 后部；Encoder 默认冻结 | teacher + Stage 3 student |
| 5 | `run_stage5_channel_finetune_gpu2.sh` | 40 | 全 profile 信道微调，前 10 epoch 逐步提高信道概率 | teacher + Stage 4 student |

完整串联命令：

```bash
cd /workspace/yi/work/shiyan-2
GPU_ID=2 bash scripts/train/variable_rate/run_pipeline_gpu2.sh
```

各阶段也可单独执行。例如已有 Stage 1 teacher 后执行 Stage 2：

```bash
cd /workspace/yi/work/shiyan-2
GPU_ID=2 \
SIMVQ_SRC_TEACHER_CHECKPOINT=/workspace/yi/work/shiyan-2/checkpoints/single_teacher_variable_rate_raq_stage1_src_teacher/best_src_teacher.pth \
bash scripts/train/variable_rate/run_stage2_identity_warmup_gpu2.sh
```

脚本会在启动 Python 前检查数据目录、teacher 和前一阶段 student checkpoint。为避免污染原项目，
输入 checkpoint 也必须位于 `shiyan-2/checkpoints/`。缺失时立即退出，不会静默从头训练下一阶段。

严格 checkpoint 链为：

```text
stage1_src_teacher/best_src_teacher.pth
  -> stage2_identity_warmup/best_variable_rate_raq.pth
  -> stage3_variable_rate/best_variable_rate_raq.pth
  -> stage4_joint_lite/best_variable_rate_raq.pth
  -> stage5_channel_finetune/best_variable_rate_raq.pth
```

## 7. 固定 profile 评估

默认独立评估六个固定 profile：

```text
[2048,2048], [2048,16], [16,2], [1024,256], [512,64], [64,16]
```

运行最终 Stage 5 checkpoint：

```bash
cd /workspace/yi/work/shiyan-2
GPU_ID=2 bash scripts/eval/test_single_teacher_variable_rate_raq_gpu2.sh
```

评估全部 121 种 profile：

```bash
ALL_PROFILES=1 GPU_ID=2 \
bash scripts/eval/test_single_teacher_variable_rate_raq_gpu2.sh
```

评估其他阶段可显式覆盖：

```bash
CHECKPOINT=/workspace/yi/work/shiyan-2/checkpoints/<experiment>/best_variable_rate_raq.pth \
TEACHER_CHECKPOINT=/workspace/yi/work/shiyan-2/checkpoints/<teacher-experiment>/best_src_teacher.pth \
EVAL_RUN_NAME=stage3_fixed_profiles \
bash scripts/eval/test_single_teacher_variable_rate_raq_gpu2.sh
```

结果写到 `experiments/eval/<run>/`，包括汇总 JSON、合并 CSV 和每个 profile 的独立 CSV。每个
profile 记录 PSNR、MS-SSIM、真实 LPIPS、reconstruction loss、active ratio、perplexity、dead code、
collapse ratio、最小 L2 距离，以及可选 SRC 专家参考差距。

checkpoint 综合分数为加权平均 PSNR 与最差 profile PSNR 的凸组合；只有 `[2048,2048]` 相对冻结
教师的 PSNR 下降不超过阈值（默认 `0.30 dB`）时，checkpoint 才有资格成为 best。

## 8. 第一组正式实验默认值

脚本采用以下可覆盖默认值：

| 变量 | 默认值 | 含义 |
|---|---:|---|
| `GPU_ID` | `2` | 物理 GPU |
| `SIMVQ_BASE_CHANNELS` | `32` | 主干基础通道数 |
| `SIMVQ_EMBEDDING_DIM_LIST` | `64,128` | 两层量化维度 |
| `SIMVQ_DOWNSAMPLE_STRIDES` | `8,2` | 两层下采样步幅 |
| `SIMVQ_MICRO_BATCH_SIZE` | `4` | 单次 micro-batch |
| `SIMVQ_TOTAL_BATCH_SIZE` | `16` | 梯度累积后的目标 batch |
| `SIMVQ_RAQ_SANDWICH_NUM_RANDOM` | `1` | sandwich 中随机中间 profile 数 |
| `SIMVQ_RAQ_TARGET_PROFILES` | `all` | 原子 profile 集合；也支持分号列表 |
| `SIMVQ_RAQ_MIN_PROFILE` | `2x2` | sandwich 最小 profile（Stage 2 脚本覆盖为 `1024x1024`） |
| `SIMVQ_RAQ_RATE_EMBED_DIM` | `64` | 完整 profile rate embedding 维度 |
| `SIMVQ_RAQ_RATE_HIDDEN_DIM` | `128` | rate MLP 隐藏维度 |
| `SIMVQ_RAQ_GENERATOR_MODEL_DIMS` | `256,256` | 两个尺度内部 head 的 Transformer 维度 |
| `SIMVQ_RAQ_GENERATOR_ATTENTION_DIM` | `64` | 源码本 cross-attention key/query 维度 |
| `SIMVQ_RAQ_TRANSFORMER_HEADS` | `8` | 每个尺度 Transformer heads |
| `SIMVQ_RAQ_TRANSFORMER_LAYERS` | `2` | 每个尺度 Transformer 层数 |
| `SIMVQ_RAQ_LAYER_VQ_WEIGHTS` | `0.25,0.5` | 两层 VQ 初始权重 |
| `SIMVQ_RAQ_LAYER_VQ_WEIGHTS_FINAL` | 同初值 | 两层 VQ 终值，训练期间线性插值 |
| `SIMVQ_RAQ_OUTPUT_DISTILL_WEIGHT_LOW/HIGH` | `0.02/0.20` | 输出蒸馏低/高码率端点 |
| `SIMVQ_RAQ_FEATURE_DISTILL_WEIGHT_LOW/HIGH` | `0.01/0.10` | 特征蒸馏低/高码率端点 |
| `SIMVQ_RAQ_IDENTITY_WEIGHT` | `1.0` | 最大码本恒等检查项 |
| `SIMVQ_RAQ_HIERARCHY_WEIGHT` | `0.05` | 2K 层级一致性权重 |
| `SIMVQ_RAQ_DIVERSITY_WEIGHT` | `0.01` | sampled diversity 权重 |
| `SIMVQ_RAQ_DIVERSITY_NUM_PAIRS` | `4096` | 每个生成层抽样码字对数 |
| `SIMVQ_RAQ_AMP` | `1` | CUDA 混合精度 |
| `SIMVQ_RAQ_GRAD_CLIP_NORM` | `1.0` | 梯度裁剪阈值 |
| `SIMVQ_RAQ_VAL_MAX_BATCHES` | `32` | 每轮固定 profile 验证批次数 |
| `SIMVQ_RAQ_VAL_AVERAGE_WEIGHT/WORST_WEIGHT` | `0.8/0.2` | 平均/最差 profile 评分权重 |
| `SIMVQ_RAQ_VAL_MAX_PSNR_DROP_DB` | `0.30` | 最大 profile 教师保护阈值 |
| `SIMVQ_SNR_RANGE_DB` | `0,15` | Stage 5 AWGN 训练 SNR 范围 |
| `SIMVQ_RAQ_CHANNEL_RAMP_EPOCHS` | `10` | Stage 5 信道概率线性升高时长 |

epoch 可用 `STAGE1_EPOCHS` 到 `STAGE5_EPOCHS` 分别覆盖；学习率可用配置中的
`SIMVQ_RAQ_STAGE*_LR`、`SIMVQ_RAQ_STAGE4_DECODER_LR`、`SIMVQ_RAQ_STAGE5_DECODER_LR` 等覆盖。
Stage 4/5 的 Encoder 默认不训练，只有显式设置 `SIMVQ_RAQ_STAGE4_TRAIN_ENCODER=1` 或
`SIMVQ_RAQ_STAGE5_TRAIN_ENCODER=1` 才以远低于 Generator 的学习率解冻后部。

## 9. 结果解释与风险

- 121 profile 支持是模型和训练目标，不代表尚未完成的长期训练已经达到每个 profile 与独立 SRC
  专家相差 `0.1～0.3 dB`；必须以固定 profile 评估结果为准，不能伪造结论。
- `[2048,2048]` 的 Generator 梯度为零是硬旁路的必然结果，不是实现错误；near-max 样本负责训练
  Generator，最大 profile 负责保护共享主干、FiLM 和输出恒等性。
- 全 121 profile + LPIPS 的完整验证耗时显著高于六 profile 验证。正式训练可每轮验证六个锚点，
  定期或训练结束后再执行 `ALL_PROFILES=1`。
- Stage 5 以前必须先确认无信道性能稳定；若最大 profile 保护门未通过，checkpoint 不应继续作为
  下一阶段的正式起点。
- 每个 checkpoint 都需要保留模型构造配置、训练 stage、teacher 路径、profile 计数、优化器/
  scheduler 和验证摘要；独立评估不会猜测缺失的模型结构。

无需 pytest 的必做检查入口为：

```bash
cd /workspace/yi/work/shiyan-2
/home/yi/.conda/envs/work/bin/python tests/run_variable_rate_checks.py
```
