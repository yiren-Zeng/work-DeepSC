# SimVQ 项目架构与设计模式分析

> 分析日期: 2026-05-15 | 项目路径: `D:\Project\Pycharm_project_new\Simvq-dc-64-Multi-pro`

---

## 一、项目概述

SimVQ (Semantic Image Vector Quantization) 是一个基于深度学习的**语义通信系统**，用于在有限带宽信道上实现高效的图像传输。核心思路是用**多尺度向量量化 (Multi-Scale VQ)** 替代传统图像编码的熵编码，结合**有限码长信道模型**和**物理层 LDPC 编码**，构成完整的语义通信链路。

### 1.1 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 深度学习框架 | PyTorch 2.x | 模型训练/推理 |
| 物理层仿真 | TensorFlow + Sionna | LDPC 编解码 |
| 图像处理 | torchvision + PIL + OpenCV | 数据加载/预处理 |
| 指标计算 | NumPy + SciPy | SSIM/MS-SSIM/PSNR |
| 训练监控 | TensorBoard | 损失/码本利用率可视化 |

### 1.2 模块总览

```
Simvq-dc-64-Multi-pro/
├── config.py                    # 全局配置类
├── train.py                     # 训练主循环 (三阶段调度)
├── test_BPP.py                  # BPP (Bits-Per-Pixel) 测试
├── test_real.py                 # 端到端物理层链路测试
├── consolidate_reports.py       # 报告整合工具
├── models/
│   ├── __init__.py
│   ├── deepsc.py                # 核心模型 DeepSC (整合所有子模块)
│   ├── semantic_encoder.py      # 语义编码器 (4层下采样)
│   ├── semantic_decoder.py      # 语义解码器 (4层上采样 + skip)
│   ├── vector_quantizer.py      # SimVQ 量化器 (冻结嵌入+可训练投影)
│   └── channel.py               # 有限码长信道模型
├── losses/
│   ├── __init__.py
│   └── deepsc_loss.py           # MSE 重建损失 + 多尺度 VQ 损失
├── communications/
│   ├── __init__.py
│   ├── channel.py               # AWGN/Rician 物理信道
│   ├── modulation.py            # BPSK/QPSK/16-QAM 调制解调与软解调 LLR
│   ├── ldpc_coding.py           # 5G NR LDPC 编解码 (Sionna)
│   └── evaluate.py              # 包含完整物理链路的评估函数
├── data/
│   ├── __init__.py
│   └── datasets.py              # 图像数据集加载 (Cars196/Kodak)
├── utils/
│   ├── __init__.py
│   ├── bit_utils.py             # 索引↔比特流转换
│   ├── math_utils.py            # 数学工具函数
│   └── metrics.py               # SSIM / MS-SSIM 指标计算
├── generate_report.py           # 自动生成报告
├── generate_deep_report.py      # 自动生成深度报告
├── generate_supplement_report.py# 自动生成补充报告
└── SimVQ*.docx                  # 已有分析报告 (Word 格式)
```

---

## 二、核心架构：DeepSC 模型

### 2.1 模型结构图

```
输入图像 (B, 3, 256, 256)
        │
        ▼
┌───────────────────────────┐
│    SemanticEncoder        │
│                           │
│  Init Conv (3→64)        │
│  DownSampleBlock 0        │──────► F1 (B, 128, 128, 128)
│    (64→128, stride=2)     │
│  DownSampleBlock 1        │──────► F2 (B, 256, 64, 64)
│    (128→256, stride=2)    │
│  DownSampleBlock 2        │──────► F3 (B, 512, 32, 32)
│    (256→512, stride=2)    │
│  DownSampleBlock 3        │──────► F4 (B, 1024, 16, 16)
│    (512→1024, stride=2)   │
└───────────────────────────┘
        │
        ▼  四层并行
┌───────────────────────────┐
│   VectorQuantizer ×4      │
│   (SimVQ: 冻结Embedding   │
│    + 可训练投影层)         │
│                           │
│  Layer0: K=64, D=128     │
│  Layer1: K=64, D=256     │
│  Layer2: K=64, D=512     │
│  Layer3: K=64, D=1024    │
│                           │
│  每层输出: vq_loss,       │
│  quantized, encoding_idx  │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│  FiniteBlocklengthChannel │
│  (训练时注入索引级噪声)    │
│                           │
│  BER 计算 (有限码长公式)  │
│  比特翻转模拟信道错误      │
└───────────────────────────┘
        │
        ▼  量化特征 (4个尺度)
┌───────────────────────────┐
│    SemanticDecoder        │
│                           │
│  Init (1024→1024)        │
│  UpSampleBlock 0 + skip0  │◄── F̂3 (B, 512, 32, 32) [SkipDropout]
│    (1024→512, ×2)         │
│  UpSampleBlock 1 + skip1  │◄── F̂2 (B, 256, 64, 64) [SkipDropout]
│    (1024→256, ×2)         │
│  UpSampleBlock 2 + skip2  │◄── F̂1 (B, 128, 128, 128) [SkipDropout]
│    (512→128, ×2)          │
│  UpSampleBlock 3          │
│    (256→128, ×2)          │
│  Final ConvTranspose2d   │
│    (128→3, stride=1)      │
└───────────────────────────┘
        │
        ▼
重建图像 (B, 3, 256, 256)
```

### 2.2 关键设计决策

#### 2.2.1 SimVQ 量化器设计

传统 VQ-VAE 的码本嵌入是可训练的，但容易发生码本坍缩 (codebook collapse)。SimVQ 采用了**冻结底层嵌入 + 可训练线性投影层**的设计:

```python
class ProjectedEmbedding(nn.Module):
    # embed.weight: 冻结 (requires_grad=False), 高斯初始化
    # proj: 可训练 Linear(dim, dim, bias=False)
    def forward(self, ids):
        return self.proj(self.embed(ids))  # 冻结嵌入 → 投影
```

**设计动机**: 冻结的随机嵌入提供了稳定的初始分布，投影层则学习将冻结嵌入映射到语义上有意义的空间，避免码本坍缩的同时保留了表达能力。

#### 2.2.2 跳跃连接 Dropout 机制

在解码器的 skip connections 上应用**样本级别的 Dropout**:

```python
class SkipConnectionDropout(nn.Module):
    # 以概率 p 丢弃整个样本的 skip 连接
    mask = (torch.rand(B, 1, 1, 1) > self.p).float()
    return x * mask
```

**设计动机**: 深层 (Layer3, D=1024) 在训练早期需要独立学习语义表示，如果浅层的 skip 信号太强，深层编码器的梯度会不足。通过高概率丢弃浅层 skip，强迫深层自己去学习，然后逐步退火恢复全连接。

#### 2.2.3 三阶段训练调度

| 阶段 | Epoch 占比 | Skip Dropout | Loss 权重 [L0,L1,L2,L3] | 策略 |
|------|-----------|-------------|------------------------|------|
| Phase1 拓荒 | 0%-60% | [0.5, 0.45, 0.05] | [1, 1, 5, 10] | 深层强制学习，浅层高 dropout |
| Phase2 退火 | 60%-90% | 线性衰减→0 | 线性衰减→1 | 逐步恢复全连接 |
| Phase3 微调 | 90%-100% | [0, 0, 0] | [1, 1, 1, 1] | 全连接高保真微调 |

#### 2.2.4 有限码长信道模型

训练时不在连续信号上仿真，而是在**离散索引层面**模拟信道噪声:

1. 根据 SNR 计算 BER (使用有限码长公式: Q 函数近似)
2. 将量化索引转换为比特表示
3. 以 BER 为概率翻转比特位
4. 将翻转后的索引用于解码

这样做的优势是**梯度可以流过量化器**（通过 straight-through estimator），同时信道噪声的影响被建模在离散空间。

---

## 三、数据流分析

### 3.1 训练数据流

```
Cars196 数据集 (256×256 图像)
    │
    ├─ Train: RandomCrop(256) + Normalize([-1,1])
    ├─ Val:   CenterCrop(256) + Normalize([-1,1])
    │
    ▼
DeepSC.forward_train(x)
    │
    ├─ SemanticEncoder(x) → [F1, F2, F3, F4]
    ├─ For each Fi:
    │   ├─ VectorQuantizer(Fi) → (vq_loss, quant_clean, idx)
    │   ├─ Channel.apply_noise(idx, SNR) → corrupted_idx
    │   ├─ get_quantized(corrupted_idx) → quant_noisy
    │   └─ quant_final = clean + (noisy - clean).detach()  # STE 梯度直通
    │
    ├─ SemanticDecoder([F̂1, F̂2, F̂3, F̂4]) → reconstructed
    │
    └─ Loss = MSE(reconstructed, x) + Σ w_i * vq_loss_i
```

### 3.2 推理数据流 (test_real.py)

```
Kodak 数据集 (768×512 图像)
    │
    ▼
DeepSC.forward_test(x) → indices_list (无信道噪声)
    │
    ▼
indices_to_bits(indices) → flat_bits
    │
    ▼
LDPC encode → coded_bits
    │
    ▼
BPSK modulate → symbols
    │
    ▼
AWGN channel (真实物理信道) → noisy_symbols
    │
    ▼
BPSK LLR → ldpc_decode → decoded_bits
    │
    ▼
bits_to_indices → recovered_indices
    │
    ▼
DeepSC.reconstruct_from_indices → reconstructed_image
    │
    ▼
MS-SSIM / PSNR 指标计算
```

### 3.3 训练 vs 推理的关键差异

| 维度 | 训练时 (forward_train) | 推理时 (test_real.py) |
|------|----------------------|---------------------|
| 信道仿真 | 离散索引级比特翻转 (BER 公式) | 真实 AWGN + BPSK 调制 |
| 纠错编码 | 通过 channel_coding_rate 间接建模 | 实际 LDPC 编解码 |
| SNR | 随机采样 [0, 15] dB | 固定测试点 [0,3,6,9,12] dB |
| 梯度 | STE 直通估计器 | 无梯度 |
| 量化解码 | clean + (noisy - clean).detach() | 直接 get_quantized_features |

---

## 四、设计模式与代码组织分析

### 4.1 优点

1. **模块化分离清晰**: `models/`、`losses/`、`communications/`、`utils/`、`data/` 逻辑边界明确
2. **参数集中管理**: `config.py` 通过 Config 类集中管理所有超参数
3. **训练调度解耦**: `compute_schedule()` 函数独立计算各阶段的 dropout/weights，与训练循环分离
4. **三种前向模式**: `forward_train` (训练+噪声)、`forward_val` (验证)、`forward_test` (纯推理) 明确分离
5. **码本监控机制**: `compute_codebook_utilization()` 提供了可解释的码本健康度指标

### 4.2 待改进点

1. **代码重复**: `forward_train` 和 `forward_val` 有 ~90% 的代码重复，仅 `quantized_final` 的 STE 处理不同
2. **硬编码路径**: `test_BPP.py` 和 `test_real.py` 中的 checkpoint 路径硬编码为 `/workspace/yi/work/...`
3. **紧耦合**: `deepsc.py` 直接访问 `Config` 类的静态属性，而非通过构造函数注入
4. **缺乏类型注解**: 大部分函数缺少类型提示，降低了可读性和 IDE 支持
5. **缺少接口抽象**: 没有定义 Encoder/Decoder/Quantizer 的抽象基类

---

## 五、依赖关系图

```
config.py ◄────── train.py, test_BPP.py, test_real.py, deepsc.py
                     │
                     ▼
              models/deepsc.py
               /      |      \
              ▼       ▼       ▼
    semantic_encoder  vector_quantizer  semantic_decoder  channel(AWGN模型)
         │                  │                  │
         └──────────────────┴──────────────────┘
                            │
                     losses/deepsc_loss.py
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        data/datasets  utils/metrics  utils/bit_utils
                             │
                    communications/
                    ├── ldpc_coding.py (依赖 TF/Sionna)
                    ├── modulation.py
                    ├── channel.py
                    └── evaluate.py
```

---

## 六、关键参数分析

### 6.1 模型容量

| 组件 | 参数量级 | 说明 |
|------|---------|------|
| SemanticEncoder | ~3.5M | 4层下采样 (64→128→256→512→1024) |
| SemanticDecoder | ~5.0M | 4层上采样 + skip concat |
| VectorQuantizer ×4 | ~1.0M | 冻结嵌入 64×Σdim_i + 投影层 |
| **总计** | **~9.5M** | 轻量级模型 |

### 6.2 压缩比计算

以 256×256×3 输入 (196,608 bytes raw) 为例:

```
各层空间分辨率: 128² → 64² → 32² → 16²
总 token 数: 128² + 64² + 32² + 16² = 21,760
每 token: log₂(64) = 6 bits
总比特数: 21,760 × 6 = 130,560 bits ≈ 16,320 bytes
理论 BPP: 130,560 / (256×256) = 1.99 bits/pixel
```

### 6.3 梯度累积

`TOTAL_BATCH_SIZE=24, MICRO_BATCH_SIZE=24` → 累积步数为 1，实际没有使用梯度累积。BN 动量调整逻辑在这种情况下没有效果。

---

## 七、训练策略分析

### 7.1 学习率设计

- 主网络: `1.75e-5` (极低，适合微调)
- 码本投影层: `1.75e-4` (10倍，让投影层快速适应)
- Scheduler: StepLR step=100, gamma=0.5

### 7.2 优化器参数分组

```python
# 码本投影层独立学习率
proj_params = [p for n, p in model.named_parameters() if "codebook.proj" in n]
other_params = [p for n, p in model.named_parameters() if "codebook.proj" not in n]
```

**评价**: 参数分组策略合理，投影层是唯一需要快速适应的 VQ 参数，给予更高学习率有助于码本快速收敛。

### 7.3 VQ 损失权重策略

Phase1 中 Layer3 (最深层) 权重为 10，Layer0 (最浅层) 为 1，这是**合理的**——深层码本的通道维度最大 (1024)，量化误差的绝对数值也最大，需要更大的权重来平衡。

---

## 八、总结

### 核心创新点
1. **SimVQ**: 冻结嵌入 + 可训练投影，抗码本坍缩
2. **三阶段调度**: 渐进式 skip dropout 退火 + 损失权重退火
3. **索引级信道建模**: 在离散空间模拟信道噪声，保持梯度流

### 架构风险点
1. **训练/推理信道模型不一致**: 训练用索引翻转，推理走真实 AWGN+BPSK+LDPC，gap 可能较大
2. **码本大小固定 64**: 所有层用相同 K=64，可能不是最优 (浅层可能需要更大的码本)
3. **跳过 RAQ (Rate-Adaptive Quantization)**: Config 注释提到 "无 RAQ 参数"，缺乏速率自适应能力
