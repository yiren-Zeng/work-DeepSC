# Dynamic RAQ-RVQ2 与普通 RAQ Curriculum 测试方案的区别

对比脚本：

- `scripts/eval/test_src2048_2048_dynamic_raq_rvq2_ch256_512.sh`
- `scripts/eval/test_src2048_2048_raq2_2048_curriculum_ch256_512.sh`

## 1. 核心区别概览

| 对比项 | Dynamic RAQ-RVQ2 方案 | 普通 RAQ Curriculum 方案 |
|---|---|---|
| 测试量化结构 | 两级残差量化 | 单级量化 |
| 测试分支 | `TEST_USE_RAQ_RVQ=1`，进入 RAQ-RVQ 测试分支 | 未启用 `TEST_USE_RAQ_RVQ`，进入普通 RAQ 测试分支 |
| 每个尺度的输出 | `q = q1 + q2` | `q = q1` |
| 第二级输入 | 第一级量化后的残差 `r1 = feature - q1` | 不存在第二级 |
| 第二级 RAQ generator | 独立、训练过、带 allocation 条件的 generator | 不存在 |
| 默认目标 K | `[2048, 16]` | `[2048, 2048]` |
| 默认分级 K | `[[64,32],[4,4]]` | 不分级，直接 `[2048,2048]` |
| 默认每尺度索引 bit | 第一尺度 `6+5=11 bit`；第二尺度 `2+2=4 bit` | 第一尺度 `11 bit`；第二尺度 `11 bit` |
| 第二级零码字 | 启用，第 0 个 codeword 强制为零向量 | 不适用 |
| 是否支持测试时指定有序分配 | 支持，例如 `32,64;8,2` | 不支持 RVQ 分配 |
| 默认 checkpoint | Dynamic RAQ-RVQ2 专用 checkpoint | 普通 RAQ Curriculum checkpoint |
| 默认 GPU | GPU 2 | GPU 0 |

## 2. 实际量化流程的区别

### 2.1 普通 RAQ Curriculum：单级 RAQ

普通方案对每个尺度的 encoder feature 只进行一次动态码本生成和一次量化。

对尺度 `i`：

```text
feature_i
   │
   ├─ 普通 RAQ generator 根据 K_total 生成目标码本 C_i
   │
   └─ 使用 C_i 量化一次，得到 q_i

Decoder 输入：q_i
```

其等价形式为：

```text
q_i = Q(feature_i, C_i)
```

脚本默认：

```bash
SIMVQ_RAQ_TARGET_LIST="2048,2048"
```

因此两个尺度分别使用一个大小为 2048 的目标码本：

```text
尺度 0：K_total = 2048，单个索引需要 log2(2048) = 11 bit
尺度 1：K_total = 2048，单个索引需要 log2(2048) = 11 bit
```

这里没有把 2048 拆成多个 stage，也没有 residual 的第二次量化。

### 2.2 Dynamic RAQ-RVQ2：两级残差 RAQ-RVQ

Dynamic 方案会把每个尺度的总 bit 预算拆给两个有先后顺序的 RVQ stage。

对尺度 `i`：

```text
feature_i
   │
   ├─ Stage 1 RAQ generator 生成 C1_i
   │       └─ q1_i = Q(feature_i, C1_i)
   │
   ├─ 计算第一级残差
   │       └─ r1_i = feature_i - q1_i
   │
   ├─ 独立的 Stage 2 RAQ generator 生成 C2_i
   │       └─ q2_i = Q(r1_i, C2_i)
   │
   └─ 两级结果相加
           └─ q_i = q1_i + q2_i

Decoder 输入：q1_i + q2_i
```

其等价形式为：

```text
q1_i = Q1(feature_i; K1_i)
r1_i = feature_i - q1_i
q2_i = Q2(r1_i; K2_i, allocation_i)
q_i  = q1_i + q2_i
```

第二级不是再次量化原始 feature，而是量化第一级没有表示好的 residual。

## 3. 默认 K 配置的区别

### 3.1 普通方案的默认 K

```bash
SIMVQ_RAQ_TARGET_LIST="2048,2048"
```

对应：

| 尺度 | K_total | 实际量化级数 | bit/token |
|---|---:|---:|---:|
| Scale 0 | 2048 | 1 | 11 |
| Scale 1 | 2048 | 1 | 11 |

### 3.2 Dynamic 方案的默认 K

```bash
SIMVQ_RAQ_TARGET_LIST="2048,16"
SIMVQ_TEST_RAQ_RVQ_DEPTH="2"
```

在没有设置 `SIMVQ_TEST_RAQ_RVQ_K_LISTS` 时，代码采用自动均衡 bit 拆分：

```text
K_total = 2048
总预算 = log2(2048) = 11 bit
自动分配 = 6 bit + 5 bit
Stage K = [2^6, 2^5] = [64, 32]

K_total = 16
总预算 = log2(16) = 4 bit
自动分配 = 2 bit + 2 bit
Stage K = [2^2, 2^2] = [4, 4]
```

所以默认得到：

```text
RVQ Stage K Lists: [[64, 32], [4, 4]]
```

对应：

| 尺度 | K_total | Stage 1 K | Stage 2 K | 两级总 bit/token |
|---|---:|---:|---:|---:|
| Scale 0 | 2048 | 64 | 32 | `6 + 5 = 11` |
| Scale 1 | 16 | 4 | 4 | `2 + 2 = 4` |

Dynamic 方案中的 `K_total` 表示总 bit 预算的等效单级码本大小，并不是两个 stage 都使用大小为 `K_total` 的码本。

## 4. 两个默认测试并非相同码率

两个脚本默认值的第一尺度都是 11 bit/token，但第二尺度不同：

```text
普通方案：
Scale 0 = 11 bit/token
Scale 1 = 11 bit/token

Dynamic 方案：
Scale 0 = 11 bit/token，由 6+5 两级共同使用
Scale 1 =  4 bit/token，由 2+2 两级共同使用
```

因此，两个脚本直接按默认参数运行时，不只是“单级与两级”的区别，还同时包含第二尺度 `2048 → 16` 的码率变化。

Dynamic 方案在同一个尺度内部满足：

```text
log2(K1) + log2(K2) = log2(K_total)
```

例如：

```text
log2(64) + log2(32) = 6 + 5 = 11 = log2(2048)
log2(4)  + log2(4)  = 2 + 2 = 4  = log2(16)
```

但这只说明 Dynamic 方案的两级拆分没有超过它自己的 `K_total` bit 预算，并不表示它的默认 `[2048,16]` 与普通方案默认 `[2048,2048]` 具有相同总码率。

## 5. Stage 2 generator 的区别

普通方案只有原来的 RAQ generator：

```text
raqs[i]
```

Dynamic 方案除了保留 Stage 1 使用的原 RAQ generator，还增加：

```text
raqs_rvq_stage2[i]
```

这个 Stage 2 generator 具有两个关键区别：

1. 它有独立参数，不与 Stage 1 共用 generator 权重。
2. 它接收当前 allocation 条件：

```text
(K_total, K1, K2)
```

因此，在相同 `K_total` 下，`[64,32]` 与 `[32,64]` 会作为不同的有序 allocation 交给第二级 generator。

普通方案没有 allocation 条件，因为它根本不做两级拆分。

## 6. 有序 K 分配能力的区别

Dynamic 测试脚本允许通过以下变量指定每个尺度的有序分配：

```bash
SIMVQ_TEST_RAQ_RVQ_K_LISTS="32,64;8,2"
```

若同时指定：

```bash
SIMVQ_RAQ_TARGET_LIST="2048,16"
```

则实际分配为：

```text
Scale 0：K_total=2048，stage_K=[32,64]
Scale 1：K_total=16，stage_K=[8,2]
```

bit 预算仍然分别为：

```text
Scale 0：5 + 6 = 11 bit
Scale 1：3 + 1 = 4 bit
```

分配是有序的：

```text
[32,64]：第一级 K=32，第二级 K=64
[64,32]：第一级 K=64，第二级 K=32
```

这两种分配总 bit 相同，但量化顺序和第二级 residual 的难度不同，所以不是同一个量化过程。

普通 RAQ 方案没有 `K1/K2` 的概念，也不会读取或使用这个分配。

## 7. 第二级零码字的区别

Dynamic 脚本设置：

```bash
SIMVQ_DYNAMIC_RAQ_RVQ_ZERO_CODEWORD="1"
```

因此 Stage 2 生成码本后，第 0 个 codeword 会被替换成全零向量：

```text
C2[0] = 0
```

这使第二级能够选择“不对第一级结果进行修正”。当 residual 已经足够小时，Stage 2 可以输出零增量。

普通方案只有一次量化，没有第二级修正，因此不存在 Stage 2 零码字。

## 8. 测试代码分支的区别

Dynamic 脚本设置：

```bash
SIMVQ_TEST_USE_RAQ_RVQ="1"
```

所以 `test_real.py` 最终调用的模型逻辑是：

```text
forward_test()
  └─ _forward_test_raq_rvq()
```

该分支产生嵌套的两级索引和两级码本：

```text
indices[scale][stage]
codebooks[scale][stage]
```

普通脚本没有设置该变量，默认值为 0，所以调用普通 RAQ 测试逻辑：

```text
forward_test()
  └─ 每个尺度生成一个 RAQ 码本并量化一次
```

其索引和码本结构是单级的：

```text
indices[scale]
codebooks[scale]
```

## 9. checkpoint 的区别

两个脚本加载的不是同一个模型文件。

Dynamic 方案默认加载：

```text
checkpoints/shiyan_dynamic_raq_rvq_src2048-2048_raq2-2048_curriculum_rate044_A_patch_ch256-512_unet2_ds8x2_k2048/best_vq_deepsc.pth
```

普通方案默认加载：

```text
checkpoints/shiyan_raq_src2048-2048_raq2-2048_curriculum_rate044_A_patch_ch256-512_unet2_ds8x2_k2048/best_vq_deepsc.pth
```

两者对应的训练目标不同：

| checkpoint | 训练时 decoder 接收 | Stage 2 是否参与训练 |
|---|---|---|
| Dynamic RAQ-RVQ2 | `q1 + q2` | 是 |
| 普通 RAQ Curriculum | 单级 `q` | 否 |

因此，Dynamic 测试不是在普通 RAQ checkpoint 上临时重复量化两次，而是加载含独立 Stage 2 generator、并按照 residual 流程训练过的专用 checkpoint。

## 10. 其他脚本级差异

| 环境变量或参数 | Dynamic RAQ-RVQ2 | 普通 RAQ Curriculum | 实际区别 |
|---|---|---|---|
| `SIMVQ_EXP_FAMILY` | `shiyan_dynamic_raq_rvq_...` | `shiyan_raq_...` | 选择不同实验目录与模型家族 |
| `SIMVQ_USE_DYNAMIC_RAQ_RVQ` | `1` | 未设置，默认 `0` | Dynamic 模型构建独立 Stage 2 模块 |
| `SIMVQ_TEST_USE_RAQ_RVQ` | `1` | 未设置，默认 `0` | 两级 residual 测试与单级测试的分支开关 |
| `SIMVQ_TEST_RAQ_RVQ_DEPTH` | `2` | 未设置 | Dynamic 固定为两级 RVQ |
| `SIMVQ_TEST_RAQ_RVQ_K_LISTS` | 可选；缺省时自动均衡拆分 | 不使用 | Dynamic 可改变有序 bit allocation |
| `SIMVQ_RAQ_RECON_GRAD_MODE` | `dual` | 未设置，默认 `ste` | 表示两种模型的训练重建梯度配置不同；测试处于 `no_grad`，它不是测试时反向传播差异 |
| `SIMVQ_DYNAMIC_RAQ_RVQ_ZERO_CODEWORD` | `1` | 未设置 | Dynamic 的 Stage 2 第 0 个码字为零向量 |
| `SIMVQ_RAQ_TARGET_LIST` 默认值 | `2048,16` | `2048,2048` | 两个默认测试的第二尺度码率不同 |
| `GPU_ID` 默认值 | `2` | `0` | 默认运行设备不同，不改变算法本身 |

## 11. 最终归纳

这两个测试方案的本质区别是：

```text
普通 RAQ Curriculum：
每个尺度使用一个动态生成的 RAQ 码本，对 feature 做一次量化，decoder 接收 q。

Dynamic RAQ-RVQ2：
每个尺度把 K_total 的 bit 预算有序分配给 K1、K2；Stage 1 量化 feature，
独立且经过训练的 Stage 2 generator 量化 Stage 1 residual，decoder 接收 q1+q2。
```

同时，两份脚本的默认目标 K 也不同：普通方案是 `[2048,2048]`，Dynamic 方案是 `[2048,16]`。所以默认运行结果同时受到“量化结构不同”和“第二尺度码率不同”这两个因素影响。

## 12. 损失函数的区别

### 12.1 测试脚本本身没有训练损失

这两个 eval 脚本运行 `test_real.py` 时都处于 `torch.no_grad()`，不会反向传播，也不会使用训练 loss 更新参数。测试阶段实际统计的是 PSNR 和 MS-SSIM。

这里所说的“损失函数区别”，指两个测试脚本分别加载的 checkpoint 在训练阶段所使用的损失。

### 12.2 两者共同的总损失框架

两份训练脚本都没有显式设置 `SIMVQ_TRAIN_BRANCH`，因此都使用默认的 `joint` 分支。其总损失框架相同：

```text
L_total = L_recon + L_aux
```

在当前配置中：

```text
MSE weight          = 1.0
MS-SSIM loss weight = 0.0
LPIPS loss weight   = 0.0
RAQ repulsion       = 0.0
latent distillation = 0.0
SRC repulsion       = 0.0
```

所以有效损失可以写成：

```text
L_total
  = MSE(x, x_hat_src)
  + MSE(x, x_hat_raq)
  + Σ_i w_i L_vq_src,i
  + Σ_i w_i L_vq_raq,i
```

其中：

- `x_hat_src` 是固定 source codebook `[2048,2048]` 分支的重建结果；
- `x_hat_raq` 是目标 RAQ 分支的重建结果；
- `i` 表示两个 U-Net 尺度；
- `w_i` 是随训练阶段调度的尺度权重，初始为 `[0.25,0.5]`，最终为 `[0.25,0.25]`。

因此，两者没有新增 MS-SSIM、LPIPS 或专门的码率损失；核心区别发生在 `x_hat_raq` 和 `L_vq_raq` 的定义上。

### 12.3 普通 RAQ 的损失

普通 RAQ 在每个尺度只量化一次：

```text
q_i = Q(feature_i, C_i)
x_hat_raq = Decoder(q_0, q_1)
```

所以普通方案的 RAQ 重建损失为：

```text
L_recon_raq_single = MSE(x, Decoder(q_0, q_1))
```

每个尺度只有一个 RAQ VQ loss：

```text
L_vq_raq,i = L_vq(feature_i, q_i)
```

单次 VQ loss 的内部形式为：

```text
L_vq(feature, q)
  = MSE(q, stopgrad(feature))
  + 0.25 × MSE(stopgrad(q), feature)
```

第一项训练生成码本及其可训练参数，第二项是 commitment loss，约束 encoder feature 靠近所选码字。

因此普通方案中与 RAQ 相关的有效部分为：

```text
L_raq_single
  = MSE(x, x_hat_raq_single)
  + Σ_i w_i L_vq(feature_i, q_i)
```

### 12.4 Dynamic RAQ-RVQ2 的损失

Dynamic 方案每个尺度包含两个量化 stage：

```text
q1_i = Q1(feature_i, C1_i)
r1_i = feature_i - stopgrad(q1_i)
q2_i = Q2(stopgrad(r1_i), C2_i)
qsum_i = q1_i + q2_i
x_hat_raq = Decoder(qsum_0, qsum_1)
```

其 RAQ 重建损失变为：

```text
L_recon_raq_rvq
  = MSE(x, Decoder(q1_0 + q2_0, q1_1 + q2_1))
```

每个尺度的 RAQ VQ loss 不再是一个单级 loss，而是两个 stage loss 相加：

```text
L_vq_raq,i
  = L_vq_stage1,i + L_vq_stage2,i

L_vq_stage1,i = L_vq(feature_i, q1_i)
L_vq_stage2,i = L_vq(stopgrad(r1_i), q2_i)
```

所以 Dynamic 方案中与 RAQ 相关的有效部分为：

```text
L_raq_dynamic
  = MSE(x, x_hat_raq_rvq)
  + Σ_i w_i [L_vq_stage1,i + L_vq_stage2,i]
```

这意味着 Dynamic 方案在数值上直接多计算并加入了 Stage 2 的 VQ loss。

### 12.5 最关键的损失差异

| 损失组成 | 普通 RAQ Curriculum | Dynamic RAQ-RVQ2 |
|---|---|---|
| Source 重建 loss | `MSE(x,x_hat_src)` | 相同 |
| Source VQ loss | 每尺度一个 | 相同 |
| RAQ 重建对象 | 单级 `q` 的解码结果 | `q1+q2` 的解码结果 |
| RAQ VQ loss | 每尺度一个单级 VQ loss | 每尺度 Stage 1 与 Stage 2 VQ loss 之和 |
| Stage 2 residual VQ loss | 无 | 有 |
| 显式 residual MSE loss | 无 | 无 |
| 显式 allocation loss | 无 | 无 |
| 显式 rate loss | 无 | 无 |

需要特别注意：代码会记录每一级量化后的 `residual_mse`，但它只是诊断数据：

```text
rvq_residual_mse_list
```

它没有被加进 `L_total`。因此当前 Dynamic 方案没有一个单独的：

```text
λ_res × MSE(residual_after_stage2, 0)
```

第二级主要通过以下两部分学习：

1. Stage 2 自己的 VQ loss；
2. 最终 `q1+q2` 图像重建 loss 的梯度。

### 12.6 重建梯度路径也不同

普通脚本没有设置 `SIMVQ_RAQ_RECON_GRAD_MODE`，因此默认使用：

```text
ste
```

普通单级 RAQ 的重建梯度通过 STE 主要回到 encoder feature；RAQ 生成码本主要由 VQ loss 训练。

Dynamic 脚本显式设置：

```bash
SIMVQ_RAQ_RECON_GRAD_MODE="dual"
```

Dynamic 分支先得到最终量化和：

```text
qsum = q1 + q2
```

然后只对最终 `qsum` 建立一次 gradient bridge。这样做的结果是：

- encoder 只接收一次 STE identity gradient，避免两个 stage 分别做 STE 导致 encoder 梯度重复；
- Stage 1 和独立的 Stage 2 generator 都可以从最终图像重建 loss 接收梯度；
- Stage 2 同时还从自己的 residual VQ loss 接收梯度。

因此，虽然两者都使用 MSE 作为图像重建损失，Dynamic 方案的重建 loss 作用于 `q1+q2`，而且梯度会训练两级生成码本；普通方案的重建 loss 只作用于单级 `q`。

### 12.7 损失函数区别的最终表达

忽略当前权重为 0 的正则项后，普通方案可以概括为：

```text
L_single
  = MSE(x, x_hat_src)
  + MSE(x, Decoder(q))
  + Σ_i w_i L_vq_src,i
  + Σ_i w_i L_vq(feature_i, q_i)
```

Dynamic 方案可以概括为：

```text
L_dynamic
  = MSE(x, x_hat_src)
  + MSE(x, Decoder(q1 + q2))
  + Σ_i w_i L_vq_src,i
  + Σ_i w_i [L_vq(feature_i, q1_i)
             + L_vq(stopgrad(feature_i - q1_i), q2_i)]
```

所以两者最本质的损失差异不是换了一种图像 loss，而是普通方案优化单级量化结果，Dynamic 方案同时优化第一级量化、第二级 residual 量化，以及两级相加后的最终重建结果。
