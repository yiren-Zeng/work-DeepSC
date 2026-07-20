---
marp: true
theme: default
paginate: true
size: 16:9
math: katex
header: 单教师 · 统一生成器 · 可变码率 RAQ
footer: Single-Teacher Variable-Rate RAQ
style: |
  section {
    font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
    font-size: 27px;
    line-height: 1.25;
  }
  h1 { color: #17365d; }
  h2 { color: #24557a; }
  strong { color: #b54708; }
  table { font-size: 19px; }
  code { font-size: 0.88em; }
  .small { font-size: 21px; }
  .tiny { font-size: 17px; }
  .center { text-align: center; }
  .accent { color: #b54708; font-weight: 700; }
  .ok { color: #16794a; font-weight: 700; }
  .warn { color: #b42318; font-weight: 700; }
---

<!--
使用说明：
1. 本文件可直接交给支持 Marp 的 Markdown 工具导出为 PPT/PDF；“---”表示换页。
2. HTML 注释是讲解提示或视觉建议，不会显示在导出的幻灯片中。
3. 建议主讲 20–25 分钟；时间不足时可略讲 03、20、21、26 页，附录仅用于答疑。
4. 第 24–25 页为 2026-07-17 训练中期快照，正式汇报前应以最终测试结果替换。
-->

<!-- _class: lead -->

# 单教师、统一 Transformer RAQ Generator

## 面向 121 个两层 Profile 的可变码率语义通信方案

**一个最大码率教师 · 一个共享编解码器 · 一个统一码本生成系统**

<br>

汇报人：________  
日期：________

<!--
讲解提示：
本方案的核心不是“为 121 个码率训练 121 个模型”，而是让一个共享系统根据完整 profile 动态生成两层目标码本。

视觉建议：
标题页只保留三个关键词：单教师、统一生成器、121 profiles。
-->

---

# 01 研究背景：固定码率模型的代价

两层码本各支持 11 种大小：

$$K_l \in \{2,4,\ldots,2048\},\quad l\in\{0,1\}$$

直接采用固定码率方案意味着：

- 可能需要 **121 套完整模型**
- 每个码率独立训练、存储和部署
- Profile 之间无法共享知识
- 最大码率能力容易在多码率训练中退化
- 切换码率等同于切换模型

> 目标：用一个共享系统覆盖完整二维码率空间。

<!--
视觉建议：
左侧画 121 个小模型，右侧画 1 个统一模型；中间用存储、训练、部署成本做对比。
-->

---

# 02 研究问题与设计目标

## 核心问题

如何只使用一个最大码率教师，在不复制 121 套模型的前提下，支持任意两层目标 Profile？

## 四个设计目标

1. **统一性**：全部 Profile 共用同一生成系统
2. **最大档保护**：$[2048,2048]$ 不破坏源码本
3. **可训练性**：真实重建梯度可进入动态生成码本
4. **覆盖性**：121 个 Profile 都能被稳定采样、验证和恢复

<!--
讲解提示：
后面所有模块都对应这四个目标：统一 Generator、硬旁路、dual gradient、coverage-aware sampler。
-->

---

# 03 方案边界

| 项目 | 当前方案 |
|---|---|
| 完整教师 | 仅 1 个 SRC $[2048,2048]$ |
| Profile | 原子二元组 $(K_0,K_1)$ |
| Profile 数量 | $11\times11=121$ |
| Encoder / Decoder | 各 1 个共享实例 |
| 源码本 | 两层各 2048，Stage 2–5 冻结 |
| 量化器 | SimVQ patch quantizer |
| 信道 | Stage 5：AWGN-derived 索引比特翻转 |
| 不参与本方案 | 旧 dynamic RAQ-RVQ、routed source、伪教师 |

<div class="small">

当前实现固定为两层；Profile 的生成、采样、计数、验证均以完整二元组为单位。

</div>

---

# 04 方案的主要贡献

1. **统一可变码率码本生成**
   - 一个顶层 Generator 系统覆盖 121 个 Profile

2. **聚合 + 残差的 Transformer RAQ**
   - 先从源码本形成可解释聚合，再预测残差

3. **逐层硬旁路**
   - $K_l=2048$ 时严格返回原源码本

4. **Dual reconstruction gradient**
   - 重建梯度同时进入 Encoder 和生成码本

5. **单教师、多档联合训练**
   - 码率感知蒸馏 + 层级约束 + sampled diversity

6. **五阶段训练与最大档门控**
   - 从 SRC 教师逐步过渡到全 Profile 与信道微调

<!--
视觉建议：
用六个图标或六边形呈现，不要放长段落。
-->

---

# 05 整体系统架构

```text
                         唯一冻结教师 [2048,2048]
                         ├─ 教师重建图像
                         └─ 两层 raw quantized features

完整 Profile (K0,K1)
        │
        ▼
共享 Rate Conditioner ───────────────┐
        │                             │
        ▼                             ▼
统一 RAQ Generator              Encoder / Decoder FiLM
├─ Scale-0 Transformer Head           │
└─ Scale-1 Transformer Head           │
        │                             │
        ├─ W_raq,0 [K0,D0]            │
        └─ W_raq,1 [K1,D1]            │
                  │                   │
图像 ─► 共享 Encoder ─► 两层量化 ─► 共享 Decoder ─► 重建
```

> 共享的是一个系统与一个完整 Profile 条件，不是 121 套参数。

<!--
视觉建议：
正式 PPT 中将上图重绘成横向框图：教师在上、学生主路径在下、Profile 条件从左侧注入。
-->

---

# 06 “一个 Generator”到底是什么意思

```text
1 × VariableRateRAQGenerator
├── 1 × RateProfileEmbedding       ← 全 Profile 共享
├── 1 × Layer Generator 0          ← 服务所有 Profile 的第 0 层
└── 1 × Layer Generator 1          ← 服务所有 Profile 的第 1 层
```

为什么需要两个尺度头？

- 两层码字维度不同：$D_0=64,\ D_1=128$
- 每个尺度头都有自己的 cross-attention 和 Transformer
- **不存在按 Profile 复制的网络**
- 也不能表述为“两层完全共享同一个 Transformer 核心”

<div class="accent center">

按尺度分头，按 Profile 共享

</div>

---

# 07 Profile 空间与实际码率

$$K_0,K_1\in\{2,4,8,16,32,64,128,256,512,1024,2048\}$$

正式 $256\times256$ 配置：

| 层 | Feature | Token 数 | 源码本 |
|---|---:|---:|---:|
| 0 | $64\times32\times32$ | 1024 | $2048\times64$ |
| 1 | $128\times16\times16$ | 256 | $2048\times128$ |

忽略熵编码和信道冗余：

$$\mathrm{bpp}_{raw}=\frac{\log_2K_0}{64}+\frac{\log_2K_1}{256}$$

> 两层的 $\log_2K$ 对真实 bpp 的贡献不同；损失中的等权 rate score 是调权启发式。

---

# 08 完整 Profile 条件

原始输入：

$$r_{raw}=[\log_2K_0,\log_2K_1]$$

模型实际输入：

$$r_{model}=\left[\frac{\log_2K_0}{11},\frac{\log_2K_1}{11}\right]$$

正式 Rate MLP：

```text
Linear(2→128) → SiLU → Linear(128→64) → LayerNorm
```

同一个 64 维 embedding 同时供：

- 两个码本生成头
- 两个 Encoder FiLM
- 两个 Decoder FiLM

> 第 0 层生成 $K_0$ 时仍能感知 $K_1$，反之亦然。

---

# 09 RAQ 码本生成：聚合 + 残差

对任一非最大层 $K_l<2048$：

$$Q_K=Q_{learned}[0:K]+U_re$$

$$A_K=\mathrm{softmax}\left(\frac{Q_K\,\mathrm{LN}(U_kW_{src})^T}{\sqrt{d_a}}\right)$$

$$S_K=A_KW_{src}$$

$$W_K=S_K+\Delta W_K$$

其中：

- $S_K$：对全部 2048 个源码字的凸聚合
- Source 无位置编码，被视为集合
- Target query 保留输出码字身份与顺序
- Transformer 只预测残差 $\Delta W_K$
- 残差输出层零初始化，初始 $W_K=S_K$

<!--
视觉建议：
画三段式：Source codebook → cross-attention pooling → target Transformer residual。
-->

---

# 10 硬旁路：最大档的结构保护

当某层 $K_l=2048$：

$$W_{raq,l}=W_{src,l}$$

| Profile | Layer 0 | Layer 1 |
|---|---|---|
| $(2048,2048)$ | 硬旁路 | 硬旁路 |
| $(2048,K_1)$ | 硬旁路 | 生成 |
| $(K_0,2048)$ | 生成 | 硬旁路 |
| $(K_0,K_1)$ | 生成 | 生成 |

硬旁路返回同一个源码本张量：

- 不调用对应 Transformer head
- residual 为 0
- attention 为 None
- 码本 identity 是结构事实，不是近似损失

---

# 11 必须讲清楚的梯度边界

## 对 $(2048,2048)$：

<div class="ok">

两个 Transformer 码本生成头严格无梯度

</div>

原因：两层都被硬旁路，生成头没有进入计算图。

## 但需要精确区分：

- Shared rate conditioner 仍被调用
- Rate embedding 仍供 Encoder/Decoder FiLM 使用
- FiLM 初始为 identity，首步不会把梯度传给 conditioner
- FiLM 学出非零权重后，conditioner 可经 FiLM 获得梯度

> Stage 2 必须包含 near-max Profile，不能只训练全最大档。

---

# 12 共享 Encoder / Decoder 的 FiLM

每层 Encoder 输出与 Decoder 输入都进行：

$$\mathrm{FiLM}(h,e)=h\odot(1+\gamma(e))+\beta(e)$$

四个独立适配器：

```text
Encoder FiLM 0      Encoder FiLM 1
Decoder FiLM 0      Decoder FiLM 1
```

初始化：

$$\gamma(e)=0,\quad\beta(e)=0$$

因此训练开始时：

$$\mathrm{FiLM}(h,e)=h$$

> 一套共享骨干通过连续条件适配 121 个 Profile。

---

# 13 SimVQ 与 Dual Gradient

最近邻选择：

$$i^*=\arg\min_i\|z-W_K[i]\|^2,\qquad q=W_K[i^*]$$

普通 STE：

$$q_{ste}=z+\mathrm{sg}(q-z)$$

本方案 Dual Bridge：

$$q_{dual}=z+\mathrm{sg}(q-z)+(q-\mathrm{sg}(q))$$

前向数值仍为 $q$，反向同时满足：

$$\frac{\partial q_{dual}}{\partial z}=I,\qquad
\frac{\partial q_{dual}}{\partial q}=I$$

<div class="accent center">

真实图像重建梯度同时更新 Encoder 与选中生成码字

</div>

---

# 14 单一冻结教师

Stage 1 只训练一个 SRC 教师：

$$K_T=(2048,2048)$$

Stage 2–5：

- Teacher 与 Student 是独立模型对象
- 通过 `state_dict` 复制初始化，不共享参数
- Teacher 始终 `eval + no_grad`
- 每个 micro-batch 只前向一次
- 同一窗口的所有 Profile 复用教师输出
- Student 源码本始终冻结

教师提供：

1. 最大码率重建图像
2. 两层冻结 raw quantized features

> 不加载 121 个教师；其他码率专家只可作为离线参考指标。

---

# 15 总损失函数

对 Profile $K=(K_0,K_1)$：

$$
\begin{aligned}
\mathcal L_K={}&
\mathcal L_{rec}
+\lambda_{vq}\mathcal L_{vq}\\
&+\lambda_{out}(K)\mathcal L_{out}
+\lambda_{feat}(K)\mathcal L_{feat}\\
&+\lambda_{id}\mathcal L_{id}
+\lambda_{hier}\mathcal L_{hier}
+\lambda_{div}\mathcal L_{div}
\end{aligned}
$$

| 项 | 作用 |
|---|---|
| $\mathcal L_{rec}$ | 对真实图像的主监督 |
| $\mathcal L_{vq}$ | 码本与 Encoder commitment |
| $\mathcal L_{out},\mathcal L_{feat}$ | 单教师辅助蒸馏 |
| $\mathcal L_{id}$ | 最大码本结构检查 |
| $\mathcal L_{hier}$ | $K$ 与 $2K$ 层级一致性 |
| $\mathcal L_{div}$ | 防止生成码字聚集 |

---

# 16 码率感知蒸馏

损失调度使用：

$$
\rho(K)=
\frac{(\log_2K_0-1)+(\log_2K_1-1)}{20}
$$

$$
\lambda(K)=low+(high-low)\rho(K)^\gamma
$$

| 蒸馏项 | Low | High | $\gamma$ |
|---|---:|---:|---:|
| 输出蒸馏 | 0.02 | 0.20 | 2 |
| 特征蒸馏 | 0.01 | 0.10 | 2 |

- $[2,2]$：真实图像监督占主导
- $[2048,2048]$：更强地锚定教师
- 避免低码率被不现实地要求完全复制最大码率教师

---

# 17 层级与多样性约束

## 层级一致性

$$
\mathcal L_{hier}
=\mathrm{MSE}
\left(W_K,\mathrm{sg}(\mathrm{Merge}(W_{2K}))\right)
$$

$$\mathrm{Merge}(W_{2K})=
\mathrm{reshape}(K,2,D).\mathrm{mean}(1)$$

- Parent 被 detach：单向约束
- 默认权重：0.05

## Sampled Diversity

$$d_{ij}=\frac{\|w_i-w_j\|_2}{\sqrt D}$$

$$\mathcal L_{div}=\mathbb E[\max(0,m-d_{ij})^2]$$

- 每层采样 4096 对，$m=0.5$
- 内存 $O(PD)$，不构造训练期 dense $K^2$ 距离矩阵

---

# 18 Coverage-aware Sandwich Sampling

每个梯度累积窗口固定执行：

```text
最大 Profile
+ 最小 Profile
+ 1 个历史计数最少的中间 Profile
```

Stage 3–5：

```text
(2048,2048) + (2,2) + least-seen intermediate
```

正式 batch：

| 配置 | 值 |
|---|---:|
| Micro batch | 4 |
| Accumulation | 4 |
| Total batch | 16 |
| Profiles / window | 3 |

损失按“实际窗口长度 × Profile 数”归一化；Sampler counts 与 RNG 随 checkpoint 保存。

---

# 19 五阶段训练流程

```text
Stage 1          Stage 2            Stage 3
SRC Teacher  →  Near-max Warmup  →  All 121 Profiles
  200 ep          20 ep               120 ep
                                           │
                                           ▼
Stage 5          Stage 4
Channel FT   ←   Joint-lite
  40 ep           40 ep
```

| Stage | 核心目标 | 信道 |
|---|---|---|
| 1 | 学好最大码率教师 | 关闭 |
| 2 | 最大档校准 + near-max 训练生成头 | 关闭 |
| 3 | 学完整二维 rate surface | 关闭 |
| 4 | 小学习率适配 Decoder 尾部 | 关闭 |
| 5 | 索引噪声下鲁棒微调 | 开启 |

Stage 2 的四个 near-max 训练 Profile：

$$
(2048,2048),\ (2048,1024),\ (1024,2048),\ (1024,1024)
$$

---

# 20 各阶段可训练范围与学习率

| Stage | Generator / Rate / FiLM | Decoder tail | Encoder tail |
|---|---:|---:|---:|
| 1 | — | SRC 全模型 $5e{-5}$ | — |
| 2 | $2e{-4}$ / $1e{-4}$ / $1e{-4}$ | 冻结 | 冻结 |
| 3 | $1e{-4}$ / $1e{-4}$ / $1e{-4}$ | 冻结 | 冻结 |
| 4 | $5e{-5}$ | $1e{-5}$ | 默认冻结；可选 $1e{-6}$ |
| 5 | $2e{-5}$ | $5e{-6}$ | 默认冻结；可选 $5e{-7}$ |

共同设置：

- Adam $(0.5,0.999)$，weight decay = 0
- AMP，gradient clip = 1.0
- Cosine schedule，最低倍率 0.05
- 冻结子模块强制 `eval()`，避免 running stats 漂移

---

# 21 Stage 5：信道微调

当前“AWGN”实现为：

```text
AWGN 有限块长公式
        ↓
计算 BER
        ↓
量化索引 → 二进制位 → Bernoulli 翻转
        ↓
Corrupted index → Noisy quantized feature
```

默认：

- SNR：$[0,15]$ dB
- Coding rate：0.5
- Block length：256 bits
- Channel probability：前 10 epoch 从 0 线性升到 1

前向使用 noisy feature，反向沿 clean dual path。

> 当前 epoch validation 仍是 clean/no-channel，不等于信道鲁棒性曲线。

---

# 22 验证与 Checkpoint 选择

固定六个锚点：

```text
2048×2048   2048×16   16×2
1024×256    512×64    64×16
```

训练期评分：

$$Score=0.8\cdot PSNR_{mean}+0.2\cdot PSNR_{worst}$$

最大档保护门：

$$PSNR_T-PSNR_{student}^{2048\times2048}\le0.30\ \mathrm{dB}$$

只有同时满足：

1. Guard 通过
2. Score 创历史新高

才保存 `best_variable_rate_raq.pth`。

---

# 23 正式实验配置

| 项目 | 正式值 |
|---|---:|
| 输入尺寸 | $256\times256$ |
| Base channels | 32 |
| 两层维度 | 64 / 128 |
| Strides | 8 / 2 |
| 源码本 | 2048 / 2048 |
| Generator dim | 256 / 256 |
| Transformer | 2 layers，8 heads |
| Rate embedding | 64 |
| Train / Val dataset | Cars196 |
| Test dataset | Kodak |
| 物理设备 | GPU 2 |

<div class="small">

注意：直接运行 Python 的默认骨干是 base 256；正式实验必须以 shell 覆盖后的 base 32 为准。

</div>

---

# 24 当前训练进展（快照）

> 截至 2026-07-17 00:55 CST

| Stage | 进度 | 状态 |
|---|---:|---|
| Stage 1 | 200 / 200 | 完成；best PSNR = 26.4862 dB |
| Stage 2 | 20 / 20 | 完成；best score = 20.6348 |
| Stage 3 | 21 / 120 | 第 22 轮进行中 |
| Stage 4 | 0 / 40 | 未开始 |
| Stage 5 | 0 / 40 | 未开始 |

Stage 3 当前 best（epoch 18）：

- Score：**23.2939**
- Weighted mean PSNR：**24.2611 dB**
- Worst：`16×2 = 19.4249 dB`
- 最大档 teacher drop：**0.0082 dB**
- Profile coverage：**121 / 121**

> 当前数据来自 Cars196 validation 前 128 张图的固定六锚点验证，是训练中期结果；不是 Kodak、全 121 Profile 或最终性能结论。

---

# 25 六个锚点的中期结果

> Stage 3 当前 best checkpoint：epoch 18

| Profile | PSNR / dB | MS-SSIM | LPIPS |
|---|---:|---:|---:|
| $2048\times2048$ | **26.4780** | 0.96939 | 0.29637 |
| $2048\times16$ | 25.0511 | 0.95335 | 0.33490 |
| $16\times2$ | 19.4249 | 0.82079 | 0.52592 |
| $1024\times256$ | 25.8516 | 0.96479 | 0.30888 |
| $512\times64$ | 25.1774 | 0.95734 | 0.33051 |
| $64\times16$ | 23.5835 | 0.93246 | 0.39107 |

$$
\overline{PSNR}_{weighted}=24.2611\ \mathrm{dB},
\qquad PSNR_{worst}=19.4249\ \mathrm{dB}
$$

<div class="tiny">

口径：Cars196 validation 前 128 张图；固定六锚点；clean/no-channel。该表用于观察训练趋势，不替代 Kodak 与全 121 Profile 最终评估。

</div>

---

# 26 为什么训练较慢

Stage 1 每个 micro-batch：

```text
1 × SRC forward/backward
```

Stage 3 每个 micro-batch：

```text
1 × Frozen teacher forward
+ 3 × Student profile forward/backward
+ Hierarchy parent generation
```

每个 epoch 还包含：

- 6 个固定 Profile 验证
- 真正 VGG LPIPS
- 分块精确码本最近邻几何统计

实测：

| 阶段 | 约耗时 / epoch |
|---|---:|
| Stage 1 | 3.5 分钟 |
| Stage 2 | 14 分钟 |
| Stage 3 | 15 分钟 |

> 主要瓶颈不是单次 Generator 前向，而是“多 Profile 反向 + 六档完整验证 + LPIPS/码本几何统计”的组合成本。

---

# 27 当前进度与预计耗时

| 阶段 | 状态 | 实测 / 预计耗时 |
|---|---|---:|
| Stage 1 | 完成 200 / 200 | 约 11.5 h |
| Stage 2 | 完成 20 / 20 | 约 4.6 h |
| Stage 3 | 完成 21 / 120；epoch 22 运行中 | 总计约 29.7 h |
| Stage 4 | 待运行 | 约 10 h |
| Stage 5 | 待运行 | 约 10 h |

截至 2026-07-17 00:55 CST：

- 已运行约 **21.3 小时**
- Epoch 计数进度：$241/420\approx57\%$
- 按阶段耗时加权进度：约 **32%**
- 若不中断，预计仍需 **44–46 小时**

<div class="tiny">

预计总耗时约 65–67 小时，完成时间约为 7 月 18 日晚；该 ETA 按 Stage 2/3 实测吞吐外推，Stage 4/5 可能浮动。

</div>

---

# 28 计划中的对比与消融

## 核心消融

1. Aggregation-only vs. Aggregation + Residual
2. Standard STE vs. Dual reconstruction gradient
3. 无 FiLM vs. 完整 Profile FiLM
4. 固定蒸馏权重 vs. rate-aware 蒸馏
5. 无 hierarchy / 无 diversity
6. Uniform sampling vs. coverage-aware sandwich
7. 直接全量联合训练 vs. 五阶段课程

## 对比维度

- PSNR / MS-SSIM / LPIPS
- Worst-profile PSNR
- 最大档 teacher drop
- Active ratio / perplexity / collapse
- 参数量、训练耗时与部署模型数

---

# 29 能够声称什么，不能声称什么

## 当前可以声称

- 架构支持 121 个原子 Profile
- 只使用一个完整教师
- 两个最大层严格码本硬旁路
- 非最大 Profile 的重建梯度进入生成头
- Profile sampler 已覆盖 121 / 121
- 中期最大档 guard 稳定通过

## 当前不能提前声称

- 所有 Profile 已达到最终最优质量
- 相对 121 个独立专家均只差 0.1–0.3 dB
- Stage 5 已证明完整信道鲁棒性
- Clean validation 等价于固定 SNR 测试

<div class="warn center">

最终结论必须等待 Stage 5 与独立测试完成

</div>

---

# 30 总结

## 核心思想

> 用一个最大码率教师和一个统一码本生成系统，学习完整二维码率空间。

## 关键机制

- 完整 Profile 条件
- Cross-attention 聚合 + Transformer 残差
- 最大档逐层硬旁路
- FiLM 共享骨干适配
- Dual reconstruction gradient
- Rate-aware distillation
- Coverage-aware sandwich
- 五阶段训练 + 最大档 guard

## 最终价值

<div class="accent center">

121 个工作点，1 套可部署模型

</div>

---

<!-- _class: lead -->

# 附录

## 公式、配置与答辩备用页

---

# A1 VQ 与图像损失

每层 VQ：

$$
\mathcal L_{vq,l}
=\mathrm{MSE}(q_l,\mathrm{sg}(z_l))
+\beta\,\mathrm{MSE}(\mathrm{sg}(q_l),z_l)
$$

$$\beta=0.25$$

两层聚合：

$$\mathcal L_{vq}=0.25\mathcal L_{vq,0}+0.50\mathcal L_{vq,1}$$

图像损失：

$$
\mathcal L_{img}
=w_{mse}\mathrm{MSE}
+w_{ms}(1-\mathrm{MS\text{-}SSIM})
+w_{lp}\mathrm{LPIPS}
$$

正式训练默认：

$$w_{mse}=1,\quad w_{ms}=0,\quad w_{lp}=0$$

---

# A2 Identity、Hierarchy 与 Diversity

Identity：

$$\mathcal L_{id}=
\mathrm{mean}_{K_l=2048}\mathrm{MSE}(W_l,\mathrm{sg}(W_{src,l}))$$

当前硬旁路下应严格为 0，主要用于结构防回归。

Hierarchy：

$$\mathcal L_{hier}=
\mathrm{mean}_{K_l<2048}
\mathrm{MSE}(W_K,\mathrm{sg}(\mathrm{Merge}(W_{2K})))$$

Diversity：

$$
\mathcal L_{div}=
\mathrm{mean}\left[\max\left(0,m-\frac{\|w_i-w_j\|_2}{\sqrt D}\right)^2\right]
$$

默认：

$$\lambda_{id}=1,\quad\lambda_{hier}=0.05,\quad
\lambda_{div}=0.01$$

---

# A3 Profile 验证指标

每个 Profile：

- PSNR、MS-SSIM、LPIPS
- Reconstruction loss
- Active count / active ratio
- Perplexity
- Dead code count
- Collapse ratio
- Minimum L2 distance

解释原则：

| 指标 | 关注点 |
|---|---|
| Mean PSNR | 整体质量 |
| Worst PSNR | 最弱工作点 |
| Teacher drop | 最大档退化 |
| Active ratio / perplexity | 码本利用 |
| Collapse / min distance | 码本几何 |

---

# A4 Checkpoint 链

```text
Stage 1
best_src_teacher.pth
        │
        ▼
Stage 2
best_variable_rate_raq.pth
        │
        ▼
Stage 3 → Stage 4 → Stage 5
best_variable_rate_raq.pth
```

Checkpoint 包含：

- Model / optimizer / scheduler / AMP scaler
- Stage、epoch、global step、best score
- Teacher path
- Model config
- Sampler counts 与 sampler RNG
- Validation summary
- Torch CPU/CUDA RNG

---

# A5 运行与评估

完整训练：

```bash
cd /workspace/yi/work/shiyan-2
GPU_ID=2 bash scripts/train/variable_rate/run_pipeline_gpu2.sh
```

固定六档：

```bash
GPU_ID=2 TEST_RESIZE=256x256 \
bash scripts/eval/test_single_teacher_variable_rate_raq_gpu2.sh
```

全部 121 档：

```bash
GPU_ID=2 ALL_PROFILES=1 TEST_RESIZE=256x256 \
bash scripts/eval/test_single_teacher_variable_rate_raq_gpu2.sh
```

注意：独立 evaluator 默认 worst 权重为 0.25；复现训练评分需显式设为 0.20。

---

# A6 常见答辩问题

**Q1：为什么不是 121 个 Generator？**  
所有 Profile 共用一个顶层系统和同一组参数；只有两个按尺度区分的 Transformer head。

**Q2：最大档为什么不能训练生成头？**  
两层硬旁路后，两个生成头都不在计算图中。

**Q3：为什么还要放最大档？**  
用于校准 FiLM、保护共享输出，并执行 teacher-drop guard。

**Q4：为什么需要 Dual Gradient？**  
普通 STE 的重建梯度不能直接更新生成码字；dual bridge 同时更新 Encoder 和选中码字。

**Q5：一个教师如何监督低码率？**  
真实图像始终是主监督；教师权重随码率降低，低档不会被强迫完全复制最大档。

---

<!-- _class: lead -->

# Q & A

## 谢谢
