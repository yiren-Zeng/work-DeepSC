# 单教师、统一 Transformer RAQ Generator 可变码率方案

## 设计说明、配置参考、训练流程与结果解读手册

> 项目目录：`/workspace/yi/work/shiyan-2`  
> 对应实现版本：`single_teacher_variable_rate_raq_v1`  
> 文档依据：当前 `shiyan-2` 中的模型、损失、训练、验证和 shell 脚本逐项核对  
> 本文只描述 `shiyan-2` 新方案；原 `shiyan` 不属于该训练链

---

## 0. 先给出最重要的结论

这个方案只训练一个完整的 SRC 教师 `[2048,2048]`，并用一个统一的可变码率 RAQ 生成系统覆盖 121 个两层 profile：

```text
K0, K1 ∈ {2,4,8,16,32,64,128,256,512,1024,2048}
profile K = (K0, K1)
profile 总数 = 11 × 11 = 121
```

“一个统一的 Generator”不表示两个不同特征维度强行共用同一个输出头，而是：

- 全部 121 个 profile 共用一个 `VariableRateRAQGenerator` 实例；
- 共用一个读取完整 `(K0,K1)` 的 rate conditioner；
- 内部有两个尺度专用 Transformer 码本生成头，因为两层码字维度分别是 `D0` 和 `D1`；
- 不存在按 profile 复制的 121 套 Generator、121 套 Encoder 或 121 套 Decoder。

第二个必须牢记的事实是硬旁路梯度边界：

- 当某层 `Kl=2048` 时，该层直接令 `W_raq,l = W_src,l`，不会调用该层 Transformer 码本生成头；
- 因此 `[2048,2048]` 会让两个 `layer_generators` 都严格得到 `grad=None`；
- Stage 2 不能只训练 `[2048,2048]`，必须同时训练 `[2048,1024]`、`[1024,2048]`、`[1024,1024]` 等 near-max profile。

这里有一个需要按代码精确定义的细节：`raq_generator.rate_conditioner` 仍会在 `[2048,2048]` 前向中计算 rate embedding，并供 Encoder/Decoder FiLM 使用。FiLM 的仿射层初始为零，所以初始时 rate conditioner 不会从 FiLM 得到梯度；FiLM 学出非零权重后，rate conditioner 可能经 FiLM 收到梯度。因此：

> “`[2048,2048]` 下 RAQ Generator 无梯度”的严格含义，是两个 Transformer 码本生成头无梯度；如果把共享 rate conditioner 也算进整个 Python `raq_generator` 模块，则不能声称它在整个训练期间永远无梯度。

这是概念口径与实际 autograd 边界之间最重要的区分。

---

## 1. 这个方案到底解决什么问题

普通固定码率方案通常为一个码本大小训练一套模型。若两层都允许 11 种 `K`，直接为 121 个 profile 各训练一套模型，会带来：

- 121 套训练、存储和部署成本；
- profile 之间没有共享知识；
- 每切换一次码率都要切换完整模型；
- 很难保证所有 profile 的行为一致；
- 最大码率性能容易在低码率联合训练时被破坏。

本方案把问题拆成三个稳定部分：

1. 一个高容量知识锚点：SRC `[2048,2048]` 教师；
2. 一个共享通信骨干：一套 Encoder、一套 Decoder、两层冻结源码本；
3. 一个按完整 profile 编译目标码本的统一 RAQ 系统。

可以把它理解成：

```text
一个最大码率专家（知识库）
          │
          ├── 冻结教师输出：图像、两层量化特征
          │
          └── 冻结学生源码本：W_src,0、W_src,1
                         │
完整 profile (K0,K1) ──► 统一 RAQ “码本编译器”
                         │
                         ├── W_raq,0(K0 | K0,K1)
                         └── W_raq,1(K1 | K0,K1)
                                      │
图像 ─► 共享 Encoder ─► 两层量化 ─► 共享 Decoder ─► 重建图像
          ▲                               ▲
          └──────── profile FiLM ─────────┘
```

profile 不是两个互不相关的随机整数，而是一条原子命令。`(2048,16)` 和 `(16,2048)` 是两个不同的完整工作点，生成、训练、计数、验证和 checkpoint 指标都以完整二元组为单位。

---

## 2. 方案边界与不变量

当前实现有以下硬边界：

- 只有两层量化尺度；
- 两层源码本固定为 `[2048,2048]`；
- 支持的目标 `K` 只能取 2 到 2048 的 11 个二次幂；
- 量化器固定为 SimVQ patch quantizer；
- 只有一个完整教师；
- 旧 dynamic RAQ-RVQ、routed source codebook 和“用学生 SRC 分支冒充教师”的路径不参与；
- Stage 2～5 的学生源码本始终冻结；
- Stage 5 当前只支持 AWGN 有限块长索引噪声；
- 固定验证始终是 clean/no-channel，以便不同阶段 checkpoint 可比较。

用户真正部署时只需提供：

```text
输入图像 x
目标 profile (K0,K1)
可选信道参数
```

无需选择 121 个模型中的某一个。

---

## 3. 符号与两层数据流

对尺度 `l∈{0,1}`：

- `W_src,l ∈ R^(2048×Dl)`：冻结的最大源码本；
- `Kl`：该层目标码本大小；
- `W_raq,l ∈ R^(Kl×Dl)`：生成或旁路得到的目标码本；
- `zl`：Encoder 输出并经过 profile FiLM 的连续特征；
- `ql`：按最近邻从 `W_raq,l` 选出的量化特征；
- `sg(·)`：stop-gradient；
- `x`：真实图像；
- `x_hat_K`：profile `K=(K0,K1)` 下的学生重建；
- `x_hat_T`：冻结教师的最大码率重建。

正式脚本使用：

```text
D0 = 64
D1 = 128
W_src,0 shape = [2048, 64]
W_src,1 shape = [2048, 128]

输入 shape   = [B,3,256,256]
layer 0 特征 = [B,64,32,32]
layer 1 特征 = [B,128,16,16]
```

Python 配置类本身的架构默认值更大，但 `_common.sh` 在正式实验中覆盖为上述轻量首轮配置。后文会分开列出两者。

在这个固定分辨率和 stride 下，若暂不考虑熵编码、信道编码与调制冗余，两个索引流的原始 bpp 近似为：

```text
bpp_raw(K) =
  (32×32 / 256×256) · log2(K0)
  + (16×16 / 256×256) · log2(K1)

= log2(K0)/64 + log2(K1)/256
```

因此损失里的 `rho(K)` 对两层 log-rate 等权，只是稳定、单调的蒸馏调权启发式，不等于真实 bpp。浅层 token 数是深层的 4 倍，同样增加 1 bit 对实际 bpp 的贡献不同。

---

## 4. 完整 profile 的 rate conditioning

### 4.1 输入

模型先构造：

```text
r_raw(K) = [log2(K0), log2(K1)]
```

由于两层源码本最大值都是 2048，`log2(2048)=11`，送入 rate MLP 的实际归一化输入是：

```text
r_model(K) = [log2(K0)/11, log2(K1)/11]
```

随后经过：

```text
Linear(2 → 128) → SiLU → Linear(128 → 64) → LayerNorm
```

得到一个共享的 64 维 rate embedding。正式配置中：

- `RATE_EMBED_DIM=64`；
- `RATE_HIDDEN_DIM=128`。

这个 embedding 同时供：

- 第 0 层码本生成头；
- 第 1 层码本生成头；
- 两个 Encoder 输出尺度的 FiLM；
- 两个 Decoder 输入尺度的 FiLM。

因此第 0 层生成 `K0` 个码字时仍能感知 `K1`，第 1 层也能感知 `K0`。这正是“完整 profile 条件”，而不是逐层独立采样。

### 4.2 不要混淆两种码率归一化

代码中存在两种有不同用途的归一化：

1. 模型条件输入：

   ```text
   [log2(K0)/11, log2(K1)/11]
   ```

   最小 `[2,2]` 对应 `[1/11,1/11]`，并不映射为 0。

2. 损失调度分数：

   ```text
   rho(K) = ((log2(K0)-1) + (log2(K1)-1)) / 20
   ```

   它把 `[2,2]` 映射为 0，把 `[2048,2048]` 映射为 1。

前者服务于神经网络条件输入，后者服务于蒸馏权重调度，不能互换。

---

## 5. 统一 Transformer RAQ 码本生成器

### 5.1 为什么内部仍有两个尺度头

第 0 层码字维度为 64，第 1 层为 128。一个输出投影不可能同时直接产生两种不同维度，因此统一系统内部有两个 `LayerResidualCodebookGenerator`。

它们是“按尺度分头”，不是“按 profile 分模型”：

```text
统一系统
├── 共享 rate conditioner
├── layer generator 0：服务全部 121 个 profile 的第 0 层
└── layer generator 1：服务全部 121 个 profile 的第 1 层
```

每个尺度头正式默认：

- Transformer model dim：256；
- cross-attention query/key dim：64；
- Transformer depth：2；
- heads：8；
- FFN multiplier：4；
- dropout：0。

### 5.2 `Kl<2048` 时的生成公式

对一个目标层，先取前 `Kl` 个可学习 target query：

```text
Q_l,K = PoolQuery_l[0:Kl] + Linear_rate→query(e_K)
```

源码本只做内容投影，不加源码字位置编码：

```text
Key_l = LayerNorm(Linear_key(W_src,l))
```

cross-attention 权重为：

```text
A_l,K = softmax(Q_l,K Key_l^T / sqrt(d_attn))
```

其中：

```text
A_l,K shape = [Kl, 2048]
每一行对 2048 个源码字求和为 1
```

所以基线聚合码本是一个显式的凸组合：

```text
S_l,K = A_l,K W_src,l
```

然后把以下三项相加形成目标 token：

```text
T_l,K =
    Linear_baseline(S_l,K)
  + Linear_query(Q_l,K)
  + Linear_rate(e_K)
```

目标侧 Transformer 只在这 `Kl` 个 target token 上建模，再预测残差：

```text
DeltaW_l,K = Linear_delta(Transformer(T_l,K))
W_raq,l(K) = S_l,K + DeltaW_l,K
```

`Linear_delta` 的权重和偏置全部零初始化，因此初始生成码本严格从有意义的聚合 `S_l,K` 起步，不会在训练第一步加入随机输出残差。

零初始化还带来一个首步梯度细节：第一次反向时，残差 Transformer 和残差 token 投影暂时会被零值末层挡住，`Linear_delta` 先获得更新；但 query/key/rate-to-query 仍可通过聚合 `S_l,K` 获得梯度。`Linear_delta` 离开零点后，残差 Transformer 开始正常接收梯度。

`S_l,K` 位于源码本凸包内，但 `DeltaW_l,K` 没有凸包约束，所以训练后的最终 `W_raq,l` 可以离开源码本凸包。

生成计算本身也不是常数复杂度：

```text
cross-attention：O(K·2048)
target self-attention：O(K²)
```

最大 `K=2048` 被硬旁路，因此最大的常规生成档是 1024；开启 hierarchy 时还会额外生成每个活跃层的 `2K` parent，parent 为 2048 时再次走硬旁路。训练 diversity 不创建 `K²` 距离矩阵，并不表示 Transformer self-attention 没有 `K²` 成本。

### 5.3 为什么源码本不使用位置编码

源码本是一组码字，不应因存储行顺序改变语义。当前实现只给 target query 可学习身份，不给 source codeword 添加位置编码，因此 source 被当作集合处理。

target query 的顺序仍有意义，因为输出需要产生 `Kl` 个可区分的目标码字。层级损失中的“相邻父码字合并”也依赖目标侧顺序。

### 5.4 `Kl=2048` 时的硬旁路

当目标大小等于源码本大小时：

```text
W_raq,l = W_src,l
```

实现返回的是同一个源码本张量对象，不是复制、近似映射或额外 Transformer 输出，同时：

- `aggregation=W_src,l`；
- `residual=zeros_like(W_src,l)`；
- `attention=None`；
- 对应 `bypass_flag=True`；
- 对应 layer generator 不被调用。

四类 profile 的行为是：

| profile 类型 | 第 0 层 | 第 1 层 |
|---|---|---|
| `[2048,2048]` | 硬旁路 | 硬旁路 |
| `[2048,K1]` | 硬旁路 | 生成 |
| `[K0,2048]` | 生成 | 硬旁路 |
| `[K0,K1]`，两者均小于 2048 | 生成 | 生成 |

硬旁路保护的是“码本张量恒等”。它不自动保证最终图像永远与 SRC 路径逐像素相同，因为 profile FiLM 在训练后可以改变 Encoder/Decoder 特征。最大档的图像级保护依赖：

- FiLM 零初始化带来的初始恒等；
- 最大档真实图像重建；
- 高权重教师蒸馏；
- 验证期 teacher-drop guard。

---

## 6. Encoder/Decoder 的共享 FiLM

每个尺度的 Encoder 输出和 Decoder 输入各有一个独立 ConditionalAffine：

```text
FiLM(h,e) = h ⊙ (1 + gamma(e)) + beta(e)
```

其中 `gamma`、`beta` 由同一个线性层产生。该线性层的权重和偏置都初始化为 0，所以初始：

```text
gamma(e)=0
beta(e)=0
FiLM(h,e)=h
```

这个设计有两个目的：

1. Stage 2 从教师复制学生骨干后，新增条件模块不会立即破坏原模型；
2. 一套 Encoder/Decoder 可以逐渐学会针对不同 `(K0,K1)` 调整特征分布。

FiLM 是共享骨干适配码率的主要机制。它不是为每个 profile 保存一组独立参数，而是由连续 rate embedding 动态产生仿射量。

---

## 7. SimVQ、最近邻与 dual reconstruction gradient

### 7.1 SimVQ 源码本

每个源码本由：

```text
冻结的底层 Embedding E
+
可训练线性投影 P
```

组成，实际码本权重为：

```text
W_src = P(E)
```

Stage 1 中底层 `E` 按 SimVQ 设计天然冻结，投影 `P` 可训练。Stage 2～5 会把整个学生 source quantizer 冻结，因此 `E` 和 `P` 都不再更新。

### 7.2 精确最近邻

对每个连续 token `z`，索引为：

```text
i* = argmin_i ||z - W_raq[i]||²
q  = W_raq[i*]
```

索引决策在 `no_grad` 中按行分块计算，不会一次物化过大的 token×codebook 距离矩阵。索引本身不可导，但选中码字可以通过 gather 参与后续梯度。

### 7.3 VQ 损失

每层实现为：

```text
L_codebook,l   = MSE(q_l, sg(z_l))
L_commit,l     = MSE(sg(q_l), z_l)
L_vq,l         = L_codebook,l + beta · L_commit,l
beta           = 0.25
```

两层聚合：

```text
L_vq = alpha0 · L_vq,0 + alpha1 · L_vq,1
```

默认：

```text
alpha0 = 0.25
alpha1 = 0.50
lambda_vq = 1.0
```

`alpha0,alpha1` 支持从初值到终值按 epoch 线性插值；正式默认终值等于初值，所以实际保持不变。

### 7.4 为什么普通 STE 不够

普通 STE：

```text
q_ste = z + sg(q-z)
```

前向数值等于 `q`，但重建梯度主要回到 `z`，不会从图像重建项直接更新生成码字。

本方案固定使用 dual bridge：

```text
q_dual = z + sg(q-z) + (q-sg(q))
```

它的前向数值仍严格等于 `q`，但反向同时提供：

```text
∂q_dual/∂z = 1
∂q_dual/∂q = 1
```

因此真实图像重建损失可以同时更新：

- Encoder 侧连续特征路径；
- 被选中的生成码字；
- 进一步回到 RAQ cross-attention、Transformer 和残差头。

训练器会在第一次非最大 profile 反向后检查所有活跃 layer generator 的梯度绝对值和必须大于 0；若没有梯度，训练直接报错。

---

## 8. 单一冻结教师与学生初始化

### 8.1 Stage 1 教师

Stage 1 训练一套无 RAQ、无信道的 DeepSC：

```text
SRC profile = [2048,2048]
```

它是后续唯一完整教师。

### 8.2 Stage 2 如何创建学生

Stage 2 新建一个独立 `VariableRateDeepSC`，然后通过 `state_dict` 把以下模块从教师复制到学生：

- semantic encoder；
- semantic decoder；
- bottleneck attention；
- 两个 vector quantizer。

“复制”不是参数对象别名。训练器会检查 teacher 和 student 的 parameter id 没有交集。

### 8.3 教师冻结契约

Stage 2～5 中：

- teacher 始终 `eval()`；
- channel probability 始终 0；
- 所有 teacher parameter 都 `requires_grad=False`；
- teacher forward 使用 `torch.no_grad()`；
- 每个 micro-batch 只做一次独立教师前向；
- 同一 micro-batch 的所有 sandwich profile 复用这份冻结输出；
- 每轮反向后都检查教师没有 grad。

教师提供：

- 最大码率重建图像；
- 两层 Encoder 特征；
- 两层 raw quantized feature；
- 两层 VQ loss 与索引诊断；
- 两个最大源码本。

蒸馏目标来自这个独立教师，不来自学生自己的 detached SRC 分支。

---

## 9. 原子 profile sandwich sampler

### 9.1 每个窗口采什么

一个梯度累积窗口固定使用同一组 profile：

```text
最大 profile
+ 配置的最小 profile
+ N 个历史计数最少的中间 profile
```

默认 `N=1`。中间 profile 先选择当前出现次数最少的桶，再用固定 RNG 在并列者中随机打破平局。

“均衡覆盖”只针对中间候选。max/min 每个窗口都固定出现，所以它们的累计次数必然远高于任一中间 profile，这是 sandwich 的预期行为，不是采样器失衡。

Stage 2：

```text
候选集合：
2048x2048
2048x1024
1024x2048
1024x1024

每个窗口实际执行：
2048x2048
+ 1024x1024
+ 两个 mixed near-max 中较少采样的一个
```

所以 Stage 2 不是每个窗口都执行四个 profile，而是三个；两个 mixed profile 会跨窗口均衡覆盖。

Stage 3～5：

```text
候选集合：全部 121 个
每个窗口：2048x2048 + 2x2 + 1 个中间 profile
```

### 9.2 与梯度累积的关系

正式配置：

```text
micro batch = 4
total batch = 16
accumulation steps = 4
```

每个 accumulation window 只采样一次 profile 列表，四个 micro-batch 复用。计数器记录的是实际 micro-batch/profile forward 次数，而不只是 optimizer window 次数。

损失归一化为：

```text
normalized_loss =
    L(profile, microbatch)
    / (当前窗口实际 microbatch 数 × 当前窗口 profile 数)
```

最后一个不足四个 micro-batch 的尾窗口使用真实窗口长度，不会错误放大或缩小梯度。

各 profile 依次 forward/backward；不会把所有 profile 的 student graph 同时保留在显存中。

### 9.3 sampler 可恢复性

checkpoint 保存：

- 完整 profile 列表；
- min/max profile；
- 随机 profile 数；
- 每个 profile 的计数；
- cycle 数；
- Python sampler RNG state。

resume 时严格检查 profile 空间和采样设置一致，防止静默改变覆盖分布。

---

## 10. 总损失函数

对一个原子 profile `K=(K0,K1)`：

```text
L_K =
    L_rec(x_hat_K, x)
  + lambda_vq · L_vq
  + lambda_out(K) · L_out(x_hat_K, sg(x_hat_T))
  + lambda_feat(K) · L_feat
  + lambda_id · L_id
  + lambda_hier · L_hier
  + lambda_div · L_div
```

真实图像重建 `L_rec` 是主监督。教师、层级和多样性都是辅助约束。

### 10.1 图像重建损失

```text
L_img(a,b) =
    lambda_mse · MSE(a,b)
  + lambda_ms · (1-MS-SSIM(a,b))
  + lambda_lpips · LPIPS_VGG(a,b)
```

正式默认：

```text
lambda_mse    = 1.0
lambda_ms     = 0.0
lambda_lpips  = 0.0
```

因此当前正式训练实际以 native `[-1,1]` 图像范围上的 MSE 为重建项。开启 MS-SSIM 时，内部会先映射到 `[0,1]`。学生阶段开启 LPIPS 时使用 `lpips.LPIPS(net="vgg")`。

实现细节：Stage 1 的历史 `DeepSCLoss` 在 `LPIPS_LOSS_WEIGHT>0` 时实际启用的是 VGG-19 多层特征 L1；Stage 2～5 的新损失则使用 `lpips` 包。正式权重为 0，所以当前首轮实验不受这个命名差异影响。若未来开启感知损失，应先决定是否统一两阶段定义。

### 10.2 输出蒸馏

```text
L_out = L_img(x_hat_K, sg(x_hat_T))
```

它复用与重建项相同的 MSE/MS-SSIM/LPIPS 组合，但 target 是冻结教师重建。

### 10.3 特征蒸馏

学生使用两层 RAQ raw quantized feature，教师使用两层冻结 SRC raw quantized feature：

```text
L_feat =
  (u0·MSE(q_student,0, sg(q_teacher,0))
 + u1·MSE(q_student,1, sg(q_teacher,1)))
  / (u0+u1)
```

默认 `u0=u1=1`。

这里比较的不是 Encoder feature。raw quantized feature 是 bridge 前选中码字 `q`；最近邻 assignment 已 detach，因此该蒸馏项会直接更新选中生成码字和 Generator，但不会经 assignment 反传到 Encoder。Encoder 仍从图像重建、commitment 等路径获得梯度。

### 10.4 码率感知蒸馏权重

```text
rho(K) =
  ((log2(K0)-1) + (log2(K1)-1)) / 20

lambda(K) =
  low + (high-low) · rho(K)^gamma
```

输出蒸馏默认：

```text
low=0.02, high=0.20, gamma=2
```

特征蒸馏默认：

```text
low=0.01, high=0.10, gamma=2
```

例子：

| profile | `rho` | `lambda_out` | `lambda_feat` |
|---|---:|---:|---:|
| `[2,2]` | 0 | 0.0200 | 0.0100 |
| `[2048,2]` | 0.5 | 0.0650 | 0.0325 |
| `[2048,2048]` | 1 | 0.2000 | 0.1000 |

这样低码率不会被要求不现实地完全复制最大码率教师，而高码率会更强地锚定教师。

### 10.5 最大码本 identity loss

对所有 `Kl=2048` 的层：

```text
L_id =
  mean_l MSE(W_raq,l, sg(W_src,l))
```

默认 `lambda_id=1.0`。

当前硬旁路返回同一个源码本张量，所以这项按结构应精确为 0。它更像一致性断言和未来改动的保险丝，不是当前训练中的主要有效梯度。最大档图像质量不能靠这项保护，仍要看蒸馏和 teacher-drop guard。

### 10.6 层级一致性损失

若 `Kl<2048`，系统额外生成该层 `2Kl` 的父码本。完整父 profile 为：

```text
第 0 层父 profile：(2K0, K1)
第 1 层父 profile：(K0, 2K1)
```

相邻两个父码字做均值合并：

```text
Merge(W_2K) = reshape(W_2K, [K,2,D]).mean(axis=1)
```

损失为：

```text
L_hier =
  mean_active_layers MSE(W_K, sg(Merge(W_2K)))
```

默认 `lambda_hier=0.05`。

当前实现会 detach 父码本，因此该项只把子码本拉向父码本，不反向更新父码本分支。这是一种单向教师式层级约束，不是双向耦合。它也不假设“小码本等于大码本前 K 行”。

还有一个顺序假设：主 cross-attention 不给 source 加位置编码，但当 `K=1024` 时，`2K=2048` parent 是硬旁路源码本，`Merge` 会按源码本当前行号相邻成对。因此层级顶端依赖稳定的 source row 顺序；不能把整套 hierarchy 描述成对源码本任意行置换都完全不变。

### 10.7 sampled diversity loss

每个生成层随机采样 `P` 个码字对。每对内部保证两个索引不同，但不同 pair 之间允许重复：

```text
d(i,j) = ||w_i-w_j||_2 / sqrt(D)
L_div,l = mean_pairs relu(margin-d(i,j))²
L_div   = mean_generated_layers L_div,l
```

默认：

```text
lambda_div = 0.01
margin     = 0.5
P          = 4096
```

训练内存复杂度是 `O(PD)`，不会构造 `K×K` pairwise matrix。它只用于 `Kl<2048` 的生成层。

仓库里旧的通用 `CodebookRepulsionLoss` 可构造 dense `K×K` 距离，但它不属于 `VariableRateRAQLoss`；当前五阶段 runner 也没有开启 Stage 1 的 source repulsion 权重。不要把旧 dense repulsion 与本方案的 sampled diversity 混写。

### 10.8 一个 profile 的默认总损失展开

在当前正式 MSE-only 配置下：

```text
L_K =
    MSE(x_hat_K, x)
  + 1.0 · (0.25 L_vq,0 + 0.50 L_vq,1)
  + lambda_out(K) · MSE(x_hat_K, sg(x_hat_T))
  + lambda_feat(K) · L_feat
  + 1.0 · L_id
  + 0.05 · L_hier
  + 0.01 · L_div
```

---

## 11. 五阶段训练设计

### 11.1 总表

| 阶段 | 名称 | epoch | profile | 训练参数 | 信道 |
|---|---|---:|---|---|---|
| 1 | `src_teacher` | 200 | SRC `[2048,2048]` | SRC 模型全部可训练参数 | 关闭 |
| 2 | `identity_warmup` | 20 | max + near-max | 两个生成头、rate conditioner、全部 FiLM | 关闭 |
| 3 | `variable_rate` | 120 | 全 121；max+min+1 | 两个生成头、rate conditioner、全部 FiLM | 关闭 |
| 4 | `joint_lite` | 40 | 全 121；max+min+1 | 上述模块 + Decoder 尾部；Encoder 默认冻结 | 关闭 |
| 5 | `channel_finetune` | 40 | 全 121；max+min+1 | Stage 4 范围，以更小 LR 微调 | AWGN 概率渐增 |

### 11.2 Stage 1：训练唯一 SRC 教师

目标：

- 先得到稳定的最大码率重建；
- 学好共享 Encoder、Decoder 和两个 `2048` SimVQ 源码本；
- 不引入 RAQ、FiLM 条件或信道扰动。

默认：

```text
lr = 5e-5
epochs = 200
optimizer = Adam(beta1=0.5, beta2=0.999)
weight_decay = 0
```

best checkpoint 只按验证 PSNR 最大选择。

### 11.3 Stage 2：identity/near-max warmup

学生骨干从教师复制，源码本随后冻结。训练：

- 两个 layer generator；
- rate conditioner；
- Encoder FiLM；
- Decoder FiLM。

profile 候选：

```text
2048x2048
2048x1024
1024x2048
1024x1024
```

默认每窗口实际执行 max、`1024x1024` 和一个 mixed near-max。

学习率：

```text
layer generators = 2e-4
rate conditioner = min(2e-4,1e-4) = 1e-4
FiLM             = min(2e-4,1e-4) = 1e-4
```

Stage 2 的关键不是让 max profile 训练 Transformer，而是：

- 用 max profile 校准共享 FiLM 和最大档输出；
- 用 near-max profile 真正训练 Transformer 码本头；
- 在容量变化较小的区域先建立稳定映射。

需要知道，Stage 2 的 epoch validation 仍使用全阶段统一的六个固定 profile，其中包含多个 Stage 2 尚未训练的低码率档。它们会进入 Stage 2 的综合 score。这样可以提前暴露低档的初始行为，但也意味着 Stage 2 best 的 score 不只是 near-max 指标；解读 Stage 2 时应同时看 teacher guard、near-max 单档结果和综合 score。

### 11.4 Stage 3：全部 121 profile 的 clean 训练

目标集合强制为 `all`，最小 profile 强制为 `2x2`。

学习率：

```text
layer generators = 1e-4
rate conditioner = 1e-4
FiLM             = 1e-4
```

共享 Encoder/Decoder 主干仍冻结，避免一开始为极低码率破坏教师能力。主要让生成器和条件模块学会完整 rate surface。

### 11.5 Stage 4：joint-lite

在 Stage 3 基础上额外解冻：

- Decoder 最后一个 up block；
- Decoder final 层。

默认：

```text
RAQ/rate/FiLM lr = 5e-5
Decoder tail lr  = 1e-5
Encoder          = frozen
```

可选设置 `SIMVQ_RAQ_STAGE4_TRAIN_ENCODER=1`，此时解冻：

- Encoder 最后一个 block；
- bottleneck attention；
- 学习率 `1e-6`。

joint-lite 的思想是先让码本生成稳定，再用小学习率调整解码端对多码率量化分布的适配，而不是一次性全模型联合训练。

### 11.6 Stage 5：信道微调

训练范围与 Stage 4 相同，但学习率更小：

```text
RAQ/rate/FiLM lr = 2e-5
Decoder tail lr  = 5e-6
Encoder optional = 5e-7
```

信道激活概率按 epoch 从 0 线性升到 1。默认 ramp 为 10 个 epoch；代码以 `ramp_epochs-1` 为分母，所以 epoch index 0 为 0，第 10 个训练 epoch 已达到 1。

每次信道激活时：

- SNR 从 `[0,15] dB` 均匀采样；
- coding rate 默认 0.5；
- coded block length 为 256 bits；
- 调制 bit 数按 SNR 从 `{1,2,4}` 的适用子集中采样；
- 根据有限块长近似计算 BER；
- 对量化索引的二进制位做 Bernoulli 翻转；
- 越界索引 clamp 到 `[0,K-1]`。

信道开关、SNR 和调制阶数在每个 profile forward 中独立采样；同一 micro-batch 的三个 sandwich profile 不要求共享同一信道随机实例。这里的 “AWGN” 也不是直接向 latent 加高斯噪声，更准确地说是“由 AWGN 有限块长公式推导 BER，再对索引比特做翻转”。

有限块长近似核心为：

```text
gamma = 10^(SNR/10)
C = log2(1+gamma)
V = (1-(1+gamma)^(-2)) · (log2(e))²
rho_block = Q(sqrt(L)·(C-R_transport)/sqrt(V))
BER = clamp(rho_block/(R_transport·L), max=0.5)
```

解码器前向使用噪声索引对应的码字，但梯度桥为：

```text
q_decoder =
  q_clean_dual + sg(q_noisy-q_clean_dual)
```

所以前向数值是 `q_noisy`，反向仍沿 clean dual path 更新 Encoder、选中码字和 Generator。

Stage 5 训练有信道，checkpoint 固定验证仍关闭信道。这能让 best 选择与前四阶段保持同一 clean 指标口径，但不代表 clean 指标已经衡量了信道鲁棒性；如需信道曲线，应另做固定 SNR 测试。

### 11.7 所有阶段共有的优化设置

- Adam：`betas=(0.5,0.999)`；
- weight decay：0；
- AMP：CUDA 上默认开启，FP16 autocast；
- gradient clip norm：1.0；
- seed：42，确定性模式；
- sampler seed：3407；
- cosine LR multiplier：从 1.0 逐渐降到 0.05；
- 每 epoch 验证一次；
- 每 10 epoch 保存一个编号 checkpoint；
- 冻结子模块会被强制设为 `eval()`，防止 BatchNorm running stats 漂移。

---

## 12. 配置：Python 默认值与正式脚本值

不要只看 `config_variable_rate.py` 的类默认值就判断正式实验。shell 的 `_common.sh` 会在启动前覆盖一部分参数。

### 12.1 架构与数据

| 配置 | Python 默认 | 正式脚本默认 | 说明 |
|---|---:|---:|---|
| `SIMVQ_UNET_DEPTH` | 2 | 2 | 固定两尺度 |
| `SIMVQ_BASE_CHANNELS` | 256 | 32 | 首轮正式实验使用轻量骨干 |
| `SIMVQ_EMBEDDING_DIM_LIST` | `512,1024` | `64,128` | 必须等于 base×2、base×4 |
| `SIMVQ_DOWNSAMPLE_STRIDES` | `8,2` | `8,2` | 两层步幅 |
| `SIMVQ_ENCODER_RES_BLOCKS` | 4 | 4 | 每级残差块 |
| `SIMVQ_DECODER_RES_BLOCKS` | 4 | 4 | 每级残差块 |
| `SIMVQ_NORM_TYPE` | group | group | 默认 GroupNorm |
| `SIMVQ_GROUP_NORM_GROUPS` | 32 | 32 | group 数 |
| `SIMVQ_ACTIVATION` | silu | silu | 激活 |
| `SIMVQ_USE_CASCADE_DOWNSAMPLE` | 0 | 0 | 关闭级联下采样 |
| `SIMVQ_USE_BOTTLENECK_ATTENTION` | 1 | 1 | 一个 bottleneck attention block |
| `SIMVQ_QUANTIZER_TYPE` | simvq | simvq | 当前方案强制 |
| source codebook | `2048,2048` | `2048,2048` | 不允许修改 |
| train/val resize | `256×256` | `256×256` | |

正式数据路径：

```text
train = /workspace/yi/work/Cars196/train_data
val   = /workspace/yi/work/Cars196/val_data
test  = /workspace/yi/work/Kodak-256-transform-resize
```

### 12.2 Generator 与 profile

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `SIMVQ_RAQ_RATE_EMBED_DIM` | 64 | profile embedding 维度 |
| `SIMVQ_RAQ_RATE_HIDDEN_DIM` | 128 | rate MLP 隐层 |
| `SIMVQ_RAQ_GENERATOR_MODEL_DIMS` | `256,256` | 两尺度 Transformer dim |
| `SIMVQ_RAQ_GENERATOR_ATTENTION_DIM` | 64 | pooling query/key dim |
| `SIMVQ_RAQ_TRANSFORMER_HEADS` | 8 | 每尺度 heads |
| `SIMVQ_RAQ_TRANSFORMER_LAYERS` | 2 | 每尺度层数 |
| `SIMVQ_RAQ_TRANSFORMER_DROPOUT` | 0 | 正式默认 |
| `SIMVQ_RAQ_GENERATOR_FEEDFORWARD_MULTIPLIER` | 4 | FFN multiplier |
| `SIMVQ_RAQ_TARGET_PROFILES` | all | Stage 2 和 Stage 3～5 脚本会强制各自集合 |
| `SIMVQ_RAQ_MIN_PROFILE` | `2x2` | Stage 2 强制 `1024x1024` |
| `SIMVQ_RAQ_SANDWICH_NUM_RANDOM` | 1 | 每窗口中间 profile 数 |
| `SIMVQ_RAQ_PROFILE_SAMPLER_SEED` | 3407 | sampler RNG |

profile 文本格式：

```text
一个 profile：K0xK1
多个 profile：K0xK1;K0xK1;...
全部 profile：all
```

重复 profile、非二次幂、超出支持集合或不是两个元素都会在启动时失败。

Stage 2 以及 Stage 3～5 的 shell 对 target/min 是无条件赋值，外部同名环境变量不会覆盖。若确实需要改变正式采样空间，应先明确这是实验方案变化，而不是以为简单传环境变量就已经生效。

### 12.3 batch、日志与验证

| 配置 | Python 默认 | 正式脚本默认 |
|---|---:|---:|
| `SIMVQ_MICRO_BATCH_SIZE` | 8 | 4 |
| `SIMVQ_TOTAL_BATCH_SIZE` | 24 | 16 |
| accumulation | 3 | 4 |
| `SIMVQ_NUM_WORKERS` | 8 | 8 |
| `SIMVQ_RAQ_LOG_INTERVAL` | 25 optimizer steps | 同左 |
| `SIMVQ_RAQ_VAL_INTERVAL` | 1 epoch | 同左 |
| `SIMVQ_RAQ_VAL_MAX_BATCHES` | 32 | 同左 |
| `SIMVQ_RAQ_TRAIN_MAX_BATCHES` | 0 | 0，表示完整 dataloader |
| `SIMVQ_RAQ_SAVE_EVERY` | 10 | 10 |
| `SIMVQ_RAQ_AMP` | 1 | 1 |
| `SIMVQ_RAQ_GRAD_CLIP_NORM` | 1.0 | 1.0 |

`SIMVQ_TOTAL_BATCH_SIZE` 必须能被 `SIMVQ_MICRO_BATCH_SIZE` 整除。

通过 stage shell 覆盖 epoch 时，优先使用 `STAGE1_EPOCHS`～`STAGE5_EPOCHS`，或统一设置 `NUM_EPOCHS`。stage shell 会自行重写 `SIMVQ_RAQ_STAGE*_EPOCHS`，所以直接在外部只设置后者不一定生效。默认依次为 `200/20/120/40/40`。

### 12.4 损失默认

| 环境变量 | 默认值 |
|---|---:|
| `SIMVQ_RAQ_RECON_MSE_WEIGHT` | 1.0 |
| `SIMVQ_RAQ_RECON_MS_SSIM_WEIGHT` | 0.0 |
| `SIMVQ_RAQ_RECON_LPIPS_WEIGHT` | 0.0 |
| `SIMVQ_RAQ_VQ_WEIGHT` | 1.0 |
| `SIMVQ_RAQ_LAYER_VQ_WEIGHTS` | `0.25,0.5` |
| `SIMVQ_RAQ_LAYER_VQ_WEIGHTS_FINAL` | 同初值 |
| `SIMVQ_RAQ_OUTPUT_DISTILL_WEIGHT_LOW/HIGH` | `0.02/0.20` |
| `SIMVQ_RAQ_OUTPUT_DISTILL_GAMMA` | 2 |
| `SIMVQ_RAQ_FEATURE_DISTILL_WEIGHT_LOW/HIGH` | `0.01/0.10` |
| `SIMVQ_RAQ_FEATURE_DISTILL_GAMMA` | 2 |
| `SIMVQ_RAQ_FEATURE_LAYER_WEIGHTS` | `1,1` |
| `SIMVQ_RAQ_IDENTITY_WEIGHT` | 1.0 |
| `SIMVQ_RAQ_HIERARCHY_WEIGHT` | 0.05 |
| `SIMVQ_RAQ_DIVERSITY_WEIGHT` | 0.01 |
| `SIMVQ_RAQ_DIVERSITY_MARGIN` | 0.5 |
| `SIMVQ_RAQ_DIVERSITY_NUM_PAIRS` | 4096 |

### 12.5 验证 profile 与 checkpoint 评分

训练期固定六个 profile：

```text
2048x2048
2048x16
16x2
1024x256
512x64
64x16
```

默认 profile 权重相同。也可用 `SIMVQ_RAQ_VAL_PROFILE_WEIGHTS` 指定：

```text
2048x2048=2;2048x16=1;16x2=1;...
```

如果只显式写一部分 profile，未写项的原始权重默认为 1.0，随后全部一起归一化；未写项不是自动变成 0。

训练期综合分数：

```text
w_worst =
  VAL_WORST / (VAL_AVERAGE + VAL_WORST)
  = 0.2 / (0.8+0.2)
  = 0.2

score =
  0.8 · weighted_mean_PSNR
  + 0.2 · worst_profile_PSNR
```

同时必须满足：

```text
teacher_PSNR - student_max_profile_PSNR <= 0.30 dB
```

只有 guard 通过且 score 创新高，才保存 `best_variable_rate_raq.pth`。

### 12.6 信道默认

| 环境变量 | 默认值 |
|---|---:|
| `SIMVQ_CHANNEL_TYPE` | AWGN |
| `SIMVQ_SNR_RANGE_DB` | `0,15` |
| `SIMVQ_CHANNEL_CODING_RATE_TRAIN` | 0.5 |
| `SIMVQ_CHANNEL_CODING_RATE_VAL` | 0.5 |
| `SIMVQ_BLOCK_LENGTH` | 256 |
| `SIMVQ_RAQ_CHANNEL_RAMP_EPOCHS` | 10 |
| `SIMVQ_RAQ_CHANNEL_PROB_START` | 0 |
| `SIMVQ_RAQ_CHANNEL_PROB_END` | 1 |

---

## 13. 标准运行流程

### 13.1 运行前检查

```bash
cd /workspace/yi/work/shiyan-2

test -d /workspace/yi/work/Cars196/train_data
test -d /workspace/yi/work/Cars196/val_data
test -x /home/yi/.conda/envs/work/bin/python
nvidia-smi -i 2
```

shell 脚本还会自动检查：

- 当前项目根目录；
- Python 入口；
- 数据集目录；
- 上一阶段 checkpoint；
- checkpoint realpath 必须位于 `shiyan-2/checkpoints`；
- experiment/family 名不能包含 `/` 或 `..`。

### 13.2 完整五阶段前台运行

```bash
cd /workspace/yi/work/shiyan-2
GPU_ID=2 bash scripts/train/variable_rate/run_pipeline_gpu2.sh
```

脚本把物理 GPU 2 映射为进程内逻辑 `cuda:0`：

```text
CUDA_VISIBLE_DEVICES=2
SIMVQ_DEVICE=cuda:0
```

因此 Python 日志显示 `cuda:0` 是正确的，并不表示使用了物理 GPU 0。

### 13.3 后台运行并保存文本日志

pipeline 自身不会后台化，也不会自动重定向 stdout/stderr。需要后台运行时可使用：

```bash
cd /workspace/yi/work/shiyan-2
mkdir -p experiments/logs

setsid -f bash -c 'cd /workspace/yi/work/shiyan-2 && exec env GPU_ID=2 bash scripts/train/variable_rate/run_pipeline_gpu2.sh > experiments/logs/single_teacher_variable_rate_raq_gpu2_pipeline.log 2>&1 < /dev/null'
```

监控：

```bash
tail -f /workspace/yi/work/shiyan-2/experiments/logs/single_teacher_variable_rate_raq_gpu2_pipeline.log
nvidia-smi -i 2
```

### 13.4 严格 checkpoint 链

```text
checkpoints/single_teacher_variable_rate_raq_stage1_src_teacher/
└── best_src_teacher.pth
          │
          ▼
checkpoints/single_teacher_variable_rate_raq_stage2_identity_warmup/
└── best_variable_rate_raq.pth
          │
          ▼
checkpoints/single_teacher_variable_rate_raq_stage3_variable_rate/
└── best_variable_rate_raq.pth
          │
          ▼
checkpoints/single_teacher_variable_rate_raq_stage4_joint_lite/
└── best_variable_rate_raq.pth
          │
          ▼
checkpoints/single_teacher_variable_rate_raq_stage5_channel_finetune/
└── best_variable_rate_raq.pth
```

pipeline 每阶段结束后检查 best 文件。若 guard 导致某阶段始终没有合格 best，pipeline 会停止，不会静默拿 last 继续。

### 13.5 分阶段运行

Stage 1：

```bash
cd /workspace/yi/work/shiyan-2
GPU_ID=2 bash scripts/train/variable_rate/run_stage1_src_teacher_gpu2.sh
```

Stage 2：

```bash
GPU_ID=2 \
SIMVQ_SRC_TEACHER_CHECKPOINT=/workspace/yi/work/shiyan-2/checkpoints/single_teacher_variable_rate_raq_stage1_src_teacher/best_src_teacher.pth \
bash scripts/train/variable_rate/run_stage2_identity_warmup_gpu2.sh
```

Stage 3：

```bash
GPU_ID=2 \
SIMVQ_SRC_TEACHER_CHECKPOINT=/workspace/yi/work/shiyan-2/checkpoints/single_teacher_variable_rate_raq_stage1_src_teacher/best_src_teacher.pth \
SIMVQ_RAQ_STUDENT_CHECKPOINT=/workspace/yi/work/shiyan-2/checkpoints/single_teacher_variable_rate_raq_stage2_identity_warmup/best_variable_rate_raq.pth \
bash scripts/train/variable_rate/run_stage3_variable_rate_gpu2.sh
```

Stage 4/5 同理，student checkpoint 指向紧邻的上一阶段 best。

Stage 3～5 会严格检查：

- 保存的 stage 名必须正好是上一阶段；
- checkpoint 记录的 teacher 路径必须与当前 teacher 一致；
- model state 必须严格匹配。

teacher 一致性当前比较的是解析后的绝对路径，不是 checkpoint 内容 hash。因此即使权重文件内容相同，移动 teacher 文件后也会被视为不同 teacher；正式链中应保持 Stage 1 teacher 路径稳定。

### 13.6 自定义实验 family

```bash
GPU_ID=2 \
SIMVQ_EXP_FAMILY=my_variable_rate_exp \
bash scripts/train/variable_rate/run_pipeline_gpu2.sh
```

五个阶段必须使用同一个 family。family 只允许普通目录名。

### 13.7 resume 同一阶段

以恢复 Stage 3 为例：

```bash
cd /workspace/yi/work/shiyan-2

GPU_ID=2 \
SIMVQ_RESUME=1 \
SIMVQ_RESUME_PATH=/workspace/yi/work/shiyan-2/checkpoints/single_teacher_variable_rate_raq_stage3_variable_rate/last_checkpoint.pth \
bash scripts/train/variable_rate/run_stage3_variable_rate_gpu2.sh
```

resume checkpoint 必须来自同一个 stage。Stage 2 best 作为 Stage 3 初始化是正常阶段迁移，不叫 resume。

Stage 3～5 的配置预检即使在 resume 模式下仍要求 `SIMVQ_RAQ_STUDENT_CHECKPOINT` 指向一个存在的前一阶段 checkpoint；正常 family 路径下 wrapper 已有默认值。真正恢复当前 stage 状态的文件仍是 `SIMVQ_RESUME_PATH`。

恢复内容包括：

- model；
- optimizer；
- cosine scheduler；
- AMP scaler；
- epoch、global step、best score；
- sampler counts、cycles、RNG；
- PyTorch CPU/CUDA RNG。

恢复从保存 epoch 的下一轮开始。建议保持原 `SIMVQ_EXPERIMENT_NAME`，否则新日志和 CSV 会写到另一套目录。

可重复性边界：checkpoint 保存了 Torch CPU/CUDA RNG 和 sampler 自己的 `random.Random`，但没有保存全局 Python `random` 与 NumPy RNG。Stage 5 的信道开关、SNR 和调制阶数使用全局 Python random，因此中断后 resume 不承诺 bitwise 完全复现未中断轨迹。

另外，`NUM_EPOCHS` 表示这个 stage 的目标总 epoch，不是“从 checkpoint 起再训练多少轮”。直接重新运行完整 pipeline 也不会自动识别并 resume 已有阶段；它会从 Stage 1 重新开始，并可能覆盖 checkpoint、向已有 CSV 追加行。

---

## 14. 独立测试与评估

### 14.1 必做结构检查

不依赖 pytest：

```bash
cd /workspace/yi/work/shiyan-2
/home/yi/.conda/envs/work/bin/python tests/run_variable_rate_checks.py
```

它检查：

- 最大档两个码本逐元素硬旁路；
- 新初始化 FiLM 下最大档重建与 SRC 路径一致；
- mixed profile 的 shape 和 bypass flag；
- 非最大档重建梯度进入相应 layer generator；
- 最大档两个 layer generator `grad=None`；
- 学生源码本和教师无梯度；
- 121 profile sampler 覆盖与 state restore；
- 教师蒸馏 target detach；
- diversity 未使用完整 `cdist/pdist`；
- 原 DeepSC 路径仍可运行。

若安装了 pytest，可补充：

```bash
/home/yi/.conda/envs/work/bin/python -m pytest \
  tests/test_profile_sampler.py \
  tests/test_profile_validation.py \
  tests/test_variable_rate_loss.py
```

### 14.2 固定六 profile Kodak 评估

```bash
cd /workspace/yi/work/shiyan-2
GPU_ID=2 TEST_RESIZE=256x256 \
bash scripts/eval/test_single_teacher_variable_rate_raq_gpu2.sh
```

这里建议显式设置 `TEST_RESIZE=256x256`。测试 dataloader 在未设置时默认 resize 到 `768x512`，即使目录名包含 “Kodak-256” 也会再次变换图像。这会改变评估口径并显著增加前向开销。若要保持原始图像尺寸，可单独使用底层的 `SIMVQ_TEST_NO_RESIZE=1`，但所有对比实验必须统一预处理。

### 14.3 全部 121 profile

```bash
GPU_ID=2 ALL_PROFILES=1 TEST_RESIZE=256x256 \
bash scripts/eval/test_single_teacher_variable_rate_raq_gpu2.sh
```

限定测试批数：

```bash
GPU_ID=2 ALL_PROFILES=1 MAX_BATCHES=10 TEST_RESIZE=256x256 \
bash scripts/eval/test_single_teacher_variable_rate_raq_gpu2.sh
```

### 14.4 指定 checkpoint

```bash
GPU_ID=2 \
CHECKPOINT=/workspace/yi/work/shiyan-2/checkpoints/<student-dir>/best_variable_rate_raq.pth \
TEACHER_CHECKPOINT=/workspace/yi/work/shiyan-2/checkpoints/<teacher-dir>/best_src_teacher.pth \
EVAL_RUN_NAME=stage3_fixed_profiles \
TEST_RESIZE=256x256 \
bash scripts/eval/test_single_teacher_variable_rate_raq_gpu2.sh
```

student 和 teacher checkpoint 都必须位于 `shiyan-2/checkpoints`。学生 checkpoint 必须含 `model_config` 元数据；评估入口不会猜测模型结构。

### 14.5 训练评分与独立评估评分的差异

训练期默认：

```text
80% weighted mean + 20% worst
```

但 `evaluate_variable_rate.py` 的命令行默认 `--worst-profile-weight=0.25`，所以独立测试 shell 当前实际使用：

```text
75% weighted mean + 25% worst
```

测试 shell 没有透传该参数。若需要与训练 checkpoint score 完全一致，应直接调用 Python 并显式添加：

```bash
CUDA_VISIBLE_DEVICES=2 /home/yi/.conda/envs/work/bin/python -u evaluate_variable_rate.py \
  --checkpoint /workspace/yi/work/shiyan-2/checkpoints/single_teacher_variable_rate_raq_stage5_channel_finetune/best_variable_rate_raq.pth \
  --teacher-checkpoint /workspace/yi/work/shiyan-2/checkpoints/single_teacher_variable_rate_raq_stage1_src_teacher/best_src_teacher.pth \
  --dataset /workspace/yi/work/Kodak-256-transform-resize \
  --device cuda:0 \
  --batch-size 1 \
  --num-workers 2 \
  --test-resize 256x256 \
  --profiles '2048x2048;2048x16;16x2;1024x256;512x64;64x16' \
  --csv experiments/eval/training_weight_match/profiles.csv \
  --per-profile-csv-dir experiments/eval/training_weight_match/per_profile \
  --json experiments/eval/training_weight_match/results.json \
  --max-teacher-drop-db 0.30 \
  --worst-profile-weight 0.2
```

---

## 15. 验证指标应该怎么读

### 15.1 图像质量

每个 profile 记录：

- PSNR：在图像映射到 `[0,1]` 后逐图计算再平均，越高越好；
- MS-SSIM：越高越好；
- LPIPS：真实 VGG LPIPS，无像素损失 fallback，越低越好；
- reconstruction loss：默认是 native `[-1,1]` 上的 MSE，越低越好。

### 15.2 码本使用率

每层记录：

- `active_count`：验证集中至少被选过一次的码字数；
- `active_ratio=active_count/K`；
- perplexity：使用分布的指数熵；
- dead code count：`K-active_count`。

注意：低 `K` 的 active ratio 容易高，并不自动表示重建好；应与 PSNR、perplexity 和几何指标一起看。

### 15.3 码本几何

每层记录：

- min L2 distance；
- collapse count；
- collapse ratio。

验证代码会按行分块做精确最近邻搜索，不在内存中长期保留完整 `K×K` 矩阵。当前 `K≤2048` 时统计覆盖全部码字，所以是 exact；但计算量仍接近 `O(K²D)`。

这与训练 diversity loss 不同：

- 训练 diversity：抽样 `P` 对，`O(PD)`；
- 验证 geometry：精确最近邻，分块内存，近 `O(K²D)` 计算。

所以 `ALL_PROFILES=1` 会明显较慢。

### 15.4 weighted mean、worst 与 guard

三者回答不同问题：

- weighted mean：总体质量；
- worst PSNR：最弱 profile 是否被平均值掩盖；
- teacher guard：最大档是否因可变码率联合训练而退化。

训练的 best 逻辑：

```text
先检查 teacher guard
    不通过 → 本轮没有 best 资格
    通过   → 再比较综合 score
```

### 15.5 可选 SRC expert gap

`SIMVQ_RAQ_SRC_REFERENCE_PSNR` 或测试的 `SRC_REFERENCE_PSNR` 可以提供 JSON：

```json
{
  "2048x16": 30.10,
  "16x2": 24.80
}
```

此时：

```text
src_psnr_gap_db = reference_SRC_PSNR - variable_rate_student_PSNR
```

这些 reference 只是离线标量，不会加载额外教师，也不会把方案变成 121 教师。当前训练链始终只有一个完整 `[2048,2048]` 教师。

训练配置中的 profile weight 语法是 `K0xK1=weight;...`；独立 evaluator 的 `PROFILE_WEIGHTS` 则要求 JSON 字符串或 JSON 文件。两者语法不同，不能直接复制。

---

## 16. 输出文件与 checkpoint 内容

### 16.1 每阶段输出

```text
checkpoints/<experiment-name>/
├── best_src_teacher.pth                 # 仅 Stage 1
├── best_variable_rate_raq.pth           # Stage 2～5
├── last_checkpoint.pth
└── epoch_NNNN.pth                       # 默认每 10 epoch

experiments/tensorboard/<experiment-name>/
experiments/<experiment-name>_epoch_metrics.csv
experiments/<experiment-name>_profile_metrics.csv
experiments/<experiment-name>_profile_metrics/<profile>.csv
experiments/snapshots/<experiment-name>/profile_sampling_counts.json
```

`profile_sampling_counts.json` 可用于检查 121 个 profile 是否覆盖、最少/最多计数是否失衡。

### 16.2 独立评估输出

```text
experiments/eval/<run-name>/
├── profiles.csv
├── results.json
└── per_profile/
    ├── 2048x2048.csv
    └── ...
```

重复使用同一个 `EVAL_RUN_NAME` 时，CSV 会追加，`results.json` 会覆盖。

### 16.3 checkpoint 保存内容

新 checkpoint 包含：

- format 和 version；
- stage、epoch、global step；
- model state；
- optimizer、scheduler、AMP scaler state；
- profile sampler state；
- best score；
- 唯一 teacher checkpoint 路径；
- 训练 config；
- 可重建模型的 `model_config`；
- validation 摘要；
- channel probability 等 extra state；
- CPU/CUDA RNG state。

保存先写同目录临时文件，再用原子 `os.replace` 替换目标，降低中断时产生半截 checkpoint 的风险。

### 16.4 当前容易误解的未使用配置

- `_common.sh` 会导出 `SIMVQ_CODEBOOK_METRICS_PATH`，但当前 `train_variable_rate.py` 不单独写这个文件；码本指标实际在 profile metrics CSV 中；
- stage shell 会导出 `SIMVQ_BEST_CHECKPOINT_NAME`，但训练入口当前直接使用固定文件名，不读取该变量。
- 通用 `SIMVQ_RAQ_GENERATOR_LR`、`SIMVQ_RAQ_DECODER_LR`、`SIMVQ_RAQ_ENCODER_LR` 和 `SIMVQ_LEARNING_RATE_G` 不控制当前五阶段的实际 param group；应使用各 Stage 专用 LR；
- `SIMVQ_RICIAN_K_FACTOR` 虽存在于配置，但当前验证只允许 AWGN，不会进入正式 Stage 5。

不要因为找不到单独 codebook metrics 文件而判断监控没有运行。

---

## 17. 如何判断每个阶段是否成功

### Stage 1

关注：

- 验证 PSNR 是否稳定上升；
- `best_src_teacher.pth` 是否生成；
- 是否出现 codebook 大量死亡或训练损失异常；
- 后续所有 stage 都必须引用同一个 teacher 文件。

### Stage 2

关注：

- 日志出现 non-bypass profile 的 generator gradient contract 通过；
- `[2048,2048]` teacher drop 在 0.30 dB 内；
- `1024x1024` 和两个 mixed profile 能正常生成；
- 不能只看 identity loss，因为它结构上接近精确 0。

### Stage 3

关注：

- profile coverage 从少到多，最终覆盖 121/121；
- max/min/intermediate 的 PSNR 不发生异常分叉；
- low-rate profile 允许比教师差，但应对真实图像持续改善；
- worst PSNR 不应持续坍塌。

### Stage 4

关注：

- Decoder 尾部小学习率是否带来整体提升；
- 最大档是否因 Decoder 变化触发 guard；
- 若 max 档退化，不应简单继续 Stage 5。

### Stage 5

关注：

- channel probability 是否按 0→1 变化；
- 训练 loss 在高 channel probability 下是否稳定；
- clean validation 是否保住；
- 另做固定 SNR 测试后才能判断真实信道鲁棒性。

---

## 18. 常见问题与调参顺序

### 18.1 OOM

优先顺序：

1. 降低 `SIMVQ_MICRO_BATCH_SIZE`；
2. 保持 total batch，并相应增加 accumulation；
3. 降低 Generator model dim 或 Transformer depth；
4. 降低验证 `MAX_BATCHES`；
5. 全 121 验证放到阶段结束后。

不要为了省显存去掉 sandwich 的 max/min，它们分别承担最大档保护和低码率边界约束。

### 18.2 最大档 teacher guard 不通过

优先检查：

- FiLM 学习率是否过大；
- Stage 4/5 Decoder tail 学习率是否过大；
- teacher checkpoint 是否一致；
- max profile 是否始终出现在 sampler；
- 输出/特征蒸馏是否被意外设为 0；
- checkpoint 与数据预处理是否一致。

可尝试降低 FiLM/Decoder LR，或适度提高高码率蒸馏权重。不要直接关闭 guard 来掩盖退化。

### 18.3 低码率质量差

低码率容量本来就小，不应简单把教师蒸馏提高到与最大档一样强。优先：

- 确认真实图像重建项仍是主项；
- 增加 `SANDWICH_NUM_RANDOM` 或训练 epoch；
- 检查 profile 覆盖；
- 查看低档 active ratio、perplexity；
- 评估层级损失和 diversity 是否过强。

### 18.4 码本坍缩

依次检查：

- diversity pair count 是否实际大于 0；
- margin 与码字尺度是否匹配；
- active ratio/perplexity 是否同样下降；
- 是否只有某一层、某些 profile 坍缩；
- sampler 是否长期遗漏这些 profile。

必要时小幅提高 diversity weight 或 pair 数。当前设计刻意不使用训练期完整 `K²` repulsion。

### 18.5 profile 覆盖慢

默认每窗口只有一个中间 profile。全部 119 个中间 profile 至少需要 119 个采样窗口才能各出现一次；正式 accumulation 为 4，每窗口跨四个 micro-batch。

可提高 `SIMVQ_RAQ_SANDWICH_NUM_RANDOM`，代价是每个 micro-batch 增加更多顺序 forward/backward。

### 18.6 Stage 2 没有 Generator 梯度

如果日志只执行 `[2048,2048]`，配置就是错误的。当前正式脚本强制 near-max 集合且启动检查禁止 min=max；训练器还会在没有执行非最大 profile 时终止。

### 18.7 独立评估分数与训练日志对不上

先检查 worst weight：

- 训练：0.20；
- 独立 CLI 默认：0.25。

其次检查 profile 列表、profile weights、数据集、resize 和 max batches。

---

## 19. 当前实现的已知限制

- 121 profile 是支持空间和训练目标，不代表长期训练结果已经自动达到每个 profile 相对独立专家只差 0.1～0.3 dB；
- 0.30 dB teacher guard 只约束最大 profile，不约束所有中低码率相对假想专家的差距；
- 当前只有一个完整教师；其他 profile 的“专家差距”必须用外部参考 PSNR 评估；
- 正式首轮骨干较轻：base 32、维度 64/128；质量上限可能低于更大骨干；
- 层级父码本在损失中 detach，是单向约束；
- 感知训练损失默认关闭，验证仍要求真实 VGG LPIPS；
- 全 121 profile 的精确几何验证计算昂贵；
- Stage 5 当前只实现 AWGN；
- clean checkpoint validation 不能替代固定 SNR 鲁棒性曲线；
- max 码本硬旁路不等于训练后的 max 图像路径绝对恒等，因为 FiLM 仍参与；
- 两个 Transformer 码本生成头在 max profile 下严格无梯度，共享 rate conditioner 的梯度边界则取决于 FiLM。

---

## 20. 代码调用关系

核心入口：

```text
scripts/train/variable_rate/run_stage*.sh
  └── config_variable_rate.py
  └── train_variable_rate.py
       ├── training/frozen_teacher.py
       │    └── models/deepsc.py
       ├── models/variable_rate_deepsc.py
       │    ├── models/variable_rate_raq.py
       │    ├── models/vector_quantizer.py
       │    ├── models/semantic_encoder.py
       │    ├── models/semantic_decoder.py
       │    └── models/channel.py
       ├── training/profile_sampler.py
       ├── losses/variable_rate_raq_loss.py
       ├── evaluation/profile_validation.py
       └── utils/variable_rate_checkpoint.py

scripts/eval/test_single_teacher_variable_rate_raq_gpu2.sh
  └── evaluate_variable_rate.py
       └── evaluation/profile_validation.py
```

关键职责：

| 文件 | 职责 |
|---|---|
| `config_variable_rate.py` | profile 空间、五阶段、损失、LR、数据和 guard 配置 |
| `models/variable_rate_raq.py` | rate embedding、两尺度聚合+残差 Transformer、硬旁路 |
| `models/variable_rate_deepsc.py` | 共享 Encoder/Decoder、FiLM、dual VQ、信道连接 |
| `models/vector_quantizer.py` | SimVQ、分块最近邻、STE/dual bridge |
| `training/frozen_teacher.py` | 唯一教师构建、冻结、一次前向复用、学生复制 |
| `training/profile_sampler.py` | 原子 sandwich、least-seen coverage、恢复状态 |
| `losses/variable_rate_raq_loss.py` | profile 复合损失与码率调度 |
| `train_variable_rate.py` | optimizer 分组、五阶段循环、梯度契约、验证和保存 |
| `evaluation/profile_validation.py` | 六档/121 档指标、几何、score 与 guard |
| `utils/variable_rate_checkpoint.py` | metadata-rich 原子 checkpoint |

---

## 21. 最短执行清单

1. 确认训练/验证数据目录存在。
2. 在物理 GPU 2 上运行 `run_pipeline_gpu2.sh`。
3. Stage 1 必须产生 `best_src_teacher.pth`。
4. Stage 2 日志必须出现非旁路 Generator 梯度检查通过。
5. Stage 2～5 每阶段必须通过最大档 0.30 dB guard，才能产生正式 best。
6. 检查 `profile_sampling_counts.json`，Stage 3～5 应最终覆盖 121/121。
7. 先做六档 clean 评估，再做全部 121 档。
8. 若比较训练 score，独立评估显式设 `--worst-profile-weight 0.2`。
9. Stage 5 结束后另做固定 SNR 曲线，不能只看 clean 验证。
10. 所有新日志、checkpoint 和结果都留在 `shiyan-2`，不要回写原 `shiyan`。

---

## 22. 一句话理解

这不是“为 121 个 profile 训练 121 个模型”，而是“用一个最大码率教师、一个共享 Encoder/Decoder、一个完整 profile conditioner 和两个尺度专用 Transformer 头，把冻结的 2048 源码本动态编译成任意两层目标码本”；最大档用结构硬旁路守住源码本，near-max 与全码率 sandwich 负责给生成头提供可学习梯度，真实图像重建负责最终任务，码率感知蒸馏、层级和多样性约束负责让整个 rate surface 更稳定。
