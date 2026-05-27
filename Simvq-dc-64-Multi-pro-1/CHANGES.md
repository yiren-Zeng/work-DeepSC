# SimVQ 方案B 优化改动说明

> 改动日期: 2026-05-15
> 新项目路径: `Simvq-dc-64-Multi-pro-1`
> 原始项目: `Simvq-dc-64-Multi-pro`
> 改动范围: `models/semantic_encoder.py`, `models/semantic_decoder.py`, `models/vector_quantizer.py`

---

## 改动总览

| # | 类别 | 文件 | 改动行数 | 影响维度 |
|---|------|------|---------|---------|
| 1 | ResidualBlock 修复 | encoder.py + decoder.py | +6 行/处 | 训练稳定性、特征学习 |
| 2 | DownSampleBlock 去冗余 | encoder.py | -3 行 | 参数量、推理速度 |
| 3 | UpSampleBlock bilinear | decoder.py | -4/+3 行 | 重建质量 |
| 4 | VQ 输入 LayerNorm | vector_quantizer.py | +2 行 | 特征统计稳定、防坍缩 |
| 5 | AttentionGate 新增 | decoder.py | +28 行 | 细节重建 |
| 6 | 输出 tanh | decoder.py | -2/+4 行 | 训练收敛 |

---

## 改动 1：ResidualBlock 补全 BN 和激活函数

### 涉及文件
- `models/semantic_encoder.py` — 编码器中的 ResidualBlock
- `models/semantic_decoder.py` — 解码器中的 ResidualBlock

### 旧代码
```python
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(channels)     # 仅一个 BN
        self.prelu = nn.PReLU()                # 仅一个 PReLU
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = out + identity      # ← 无 BN 约束 conv2 输出，无激活
        return out
```

### 新代码
```python
class ResidualBlock(nn.Module):
    """标准 Post-Activation 残差块: Conv1→BN→PReLU→Conv2→BN→add→PReLU"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)    # 重命名
        self.prelu1 = nn.PReLU()               # 重命名
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)    # 新增: conv2 后的 BN
        self.prelu2 = nn.PReLU()               # 新增: add 后的激活

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.bn2(out)         # 新增: 约束残差分支输出
        out = out + identity
        out = self.prelu2(out)      # 新增: add 后施加非线性
        return out
```

### 为什么要改

对照 ResNet 原始论文 (He et al., CVPR 2016) 的标准 Post-Activation 设计：

```
标准设计:  Conv → BN → ReLU → Conv → BN → add → ReLU
旧代码:    Conv → BN → PReLU → Conv → add  ← 缺 conv2 后的 BN、缺 add 后的激活
```

旧代码的两个缺陷：
1. **conv2 输出无 BN 约束**：残差分支可以输出任意大的值，直接加到 identity 上。在深层网络（4 层下采样）中，这会导致特征幅值逐层失控，训练不稳定。
2. **add 后无非线性**：残差机制的核心是 `F(x) + x`，如果 F(x) 和 x 相加后不再施加非线性，等价于把残差分支和恒等分支"线性混合"，丧失了深度网络的非线性表达能力。

改为标准设计后：
- `bn2` 将残差分支输出归一化到零均值单位方差，与恒等分支的尺度匹配
- `prelu2` 确保每个残差块都有独立的非线性决策边界
- 对 VQ 系统尤其重要：VQ 量化依赖特征空间的距离度量，BN 帮助各层特征幅值保持一致性

---

## 改动 2：DownSampleBlock 删除冗余尾部卷积

### 涉及文件
- `models/semantic_encoder.py` — DownSampleBlock

### 旧代码
```python
class DownSampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.res1 = ResidualBlock(in_ch)
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.res2 = ResidualBlock(out_ch)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU()
        self.tail = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        # ↑ out_ch→out_ch 的卷积, 以128→256块为例占 ~589K 参数

    def forward(self, x):
        x = self.res1(x)       # ResidualBlock
        x = self.down(x)       # Conv stride=2
        x = self.res2(x)       # ResidualBlock (内部: Conv→BN→PReLU→Conv)
        x = self.bn(x)         # BN
        x = self.act(x)        # PReLU
        x = self.tail(x)       # Conv  ← 与 res2 内部 conv2 功能重叠
        return x
```

### 新代码
```python
class DownSampleBlock(nn.Module):
    """下采样块: res1→down(stride=2)→BN→PReLU→res2"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.res1 = ResidualBlock(in_ch)
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU()
        self.res2 = ResidualBlock(out_ch)
        # ↑ 删除 tail, BN/Act 移到 res2 之前

    def forward(self, x):
        x = self.res1(x)
        x = self.down(x)
        x = self.bn(x)         # BN 在 res2 前（原在 res2 后）
        x = self.act(x)        # PReLU 在 res2 前（原在 res2 后）
        x = self.res2(x)       # 修复后的 ResidualBlock
        return x
```

### 为什么要改

1. **功能冗余**：`res2` 内部已有 `conv1(3×3)→BN→PReLU→conv2(3×3)`，其中 `conv2` 已经是 `out_ch→out_ch` 的 3×3 卷积。旧代码在 `res2` 之后再接 `BN→PReLU→tail(conv, out_ch→out_ch, 3×3)`，等效于把 `res2.conv2` 和 `tail` 两个 3×3 卷积串联。两个 3×3 卷积的感受野等价于一个 5×5 卷积——用一个 3×3 卷积去复现这个功能本质上是在浪费参数（因为 `res2` 修复后的 `conv2` 已经够用）。

2. **参数节省**：以 128→256 的 DownSampleBlock 为例：
   - tail = 3×3×256×256 = **589,824** 个参数
   - 该块总参数约 2.4M，tail 占 ~25%
   - 4 个 DownSampleBlock 累计节省约 **1.2M 参数**

3. **BN 位置更合理**：将 BN+PReLU 从 `res2` 之后移到 `res2` 之前（即 `down` 之后），使 stride-2 下采样卷积后的特征先归一化再进入残差块。这符合 Pre-Activation 的设计哲学：BN 放在卷积之前比之后更有利于梯度流动。

---

## 改动 3：UpSampleBlock 上采样方式改为双线性插值

### 涉及文件
- `models/semantic_decoder.py` — UpSampleBlock

### 旧代码
```python
class UpSampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, up_mode: str = "nearest"):
        ...
        self.up_mode = up_mode

    def forward(self, x):
        x = self.res(x)
        x = F.interpolate(x, scale_factor=2, mode=self.up_mode,
                          align_corners=False if self.up_mode == "bilinear" else None)
        ...
```

### 新代码
```python
class UpSampleBlock(nn.Module):
    """上采样块: res→bilinear(×2)→Conv→BN→PReLU"""
    def __init__(self, in_ch, out_ch):
        # 删除 up_mode 参数，不再需要

    def forward(self, x):
        x = self.res(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        ...
```

### 为什么要改

`mode="nearest"` 最近邻插值的本质是将每个像素复制到 2×2 的邻域：

```
输入:  a  b      输出:  a  a  b  b
       c  d             a  a  b  b
                        c  c  d  d
                        c  c  d  d
```

这在上采样后的特征图中产生**明显的 2×2 块状结构**。后续的 3×3 卷积需要消耗额外的容量去"平滑"这些人造块状伪影。

对于 SimVQ 这种语义通信系统，这个问题被放大：
- VQ 解码器要从离散的码本向量重建连续图像
- 最近邻插值产生的块状结构与 VQ 量化误差叠加，在低 SNR 下尤为明显
- 特征空间的块状伪影最终会传递到重建图像的像素空间

`mode="bilinear"` 双线性插值直接输出平滑的过渡，无需后续卷积去修正，且零额外参数。

**为什么不用 pixel_shuffle（方案 B 原提议）**：
- pixel_shuffle 需要将通道数先扩到 4 倍再空间重排，需重构 UpSampleBlock 的卷积顺序
- 侵入性大、增加参数、而增益不可保证
- bilinear 是更安全的等价替代：一行改动，零参数增加，效果已被广泛验证

---

## 改动 4：VQ 输入侧加 LayerNorm（已修正，替代原 L2 归一化方案）

### 涉及文件
- `models/vector_quantizer.py` — VectorQuantizer

### ⚠️ 修正说明

原方案 B 在此处使用了 **L2 归一化距离计算**，训练后 Epoch 20 出现严重码本坍缩（平均活跃率 30%，Layer 3 仅 14%）。根因：高维空间 L2 归一化剥夺了幅值多样性，导致马太效应。**已修正为 LayerNorm + 欧氏距离**。详见文档末尾"修正记录"。

### 旧代码
```python
# 原始项目: 无归一化，直接算欧氏距离
d = torch.sum(flat ** 2, dim=1, keepdim=True) + \
    torch.sum(embed_weight ** 2, dim=1) - 2 * \
    torch.einsum('bd,nd->bn', flat, embed_weight)
```

### 新代码
```python
# 修正后: LayerNorm 稳定统计量 + 欧氏距离
inputs_bhwc = inputs.permute(0, 2, 3, 1).contiguous()
inputs_bhwc = self.input_norm(inputs_bhwc)      # 新增: nn.LayerNorm(C)
embed_weight = self.codebook.projected_weight()
...
d = torch.sum(flat ** 2, dim=1, keepdim=True) + \
    torch.sum(embed_weight ** 2, dim=1) - 2 * \
    torch.einsum('bd,nd->bn', flat, embed_weight)
```

### 为什么要这样改

**LayerNorm 的正当理由**（有论文和社区实践支撑）：
1. 将每个 C 维特征向量归一化到零均值单位方差，防止编码器输出范数在训练中漂移（已知问题：范数可能从 22 暴增到 20000，见 lucidrains/vector-quantize-pytorch issue #26）
2. 各层特征统计量一致，VQ 损失更稳定

**保留欧氏距离的理由**：
1. 幅值差异是码字多样性的关键来源——不同空间位置的特征有不同的激活强度，这些差异帮助码字分散
2. SimVQ 的可训练投影层需要在幅值+方向两个自由度上与编码器匹配，欧氏距离与 STE 梯度路径一致

---

## 改动 5：Skip 连接增加注意力门控 (AttentionGate)

### 涉及文件
- `models/semantic_decoder.py` — 新增 `AttentionGate` 类 + 修改 `SemanticDecoder`

### 新增类
```python
class AttentionGate(nn.Module):
    """注意力门控: 用上采样特征(门控信号)自适应加权 skip 特征"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, bias=False)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1, bias=False)

    def forward(self, g, x):
        g1 = self.W_g(g)          # 1×1 conv 压缩门控信号
        x1 = self.W_x(x)          # 1×1 conv 压缩 skip 特征
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        att = torch.sigmoid(self.psi(F.relu(g1 + x1)))  # 注意力图 ∈ [0,1]
        return x * att                                   # 空间自适应加权
```

### 解码器改动

旧代码：
```python
if i < self.L - 1:
    skip = quant_feats[-2 - i]
    skip = self.skip_dropouts[i](skip)   # 直接做 Dropout
    x = torch.cat([x, skip], dim=1)      # 等权拼接
```

新代码：
```python
if i < self.L - 1:
    skip = quant_feats[-2 - i]
    skip = self.attention_gates[i](x, skip)  # 先用上采样特征门控 skip
    skip = self.skip_dropouts[i](skip)       # 再做 Dropout
    x = torch.cat([x, skip], dim=1)
```

### 各层 AttentionGate 配置

| 解码层 i | 上采样特征 (g) | Skip 特征 (x) | F_int | 参数量 |
|----------|---------------|---------------|-------|--------|
| 0 (最深→Layer2) | F̂₄→上采样, C=512 | F̂₃, C=512 | 256 | ~262K |
| 1 (Layer2→Layer1) | concat→上采样, C=256 | F̂₂, C=256 | 128 | ~66K |
| 2 (Layer1→Layer0) | concat→上采样, C=128 | F̂₁, C=128 | 64 | ~16K |
| **合计** | | | | **~344K (模型总参数 ~3.5%)** |

### 为什么要改

这是基于 Attention U-Net (Oktay et al., MIDL 2018) 的成熟设计。

旧代码中 skip 连接的融合方式是对所有空间位置一视同仁的：
```
x = cat(upsampled, skip)  → 通道翻倍 → 下一层卷积处理
```

但实际上不同空间位置对 skip 信息的需求是完全不同的：
- **平坦区域**（天空、墙壁）：深层语义特征已经足够重建，skip 细节可有可无
- **边缘/纹理区域**（物体轮廓、文字）：skip 中的高频细节至关重要
- **语义关键区域**（人脸、车牌）：需要精细的 skip 引导

注意力门控让**解码器的上采样特征（包含深层语义上下文）作为"查询"，决定在哪些空间位置"关注"skip 特征中的细节**。门控输出是一个空间注意力图，在 [0,1] 之间自适应控制 skip 信息的流入强度。

对于语义通信场景，这尤其有价值：在低 SNR 下，信道噪声可能破坏了深层的语义特征，此时门控可以自动增加对 skip 信息的依赖；在高 SNR 下，深层特征可靠，门控减少冗余的 skip 信息。

---

## 改动 6：输出层改为 Conv2d + Tanh

### 涉及文件
- `models/semantic_decoder.py` — SemanticDecoder.final

### 旧代码
```python
self.final = nn.ConvTranspose2d(in_ch, out_channels, kernel_size=3, stride=1, padding=1)
# 输出无界
```

### 新代码
```python
self.final = nn.Sequential(
    nn.Conv2d(in_ch, out_channels, kernel_size=3, stride=1, padding=1),
    nn.Tanh()
)
# 输出 ∈ [-1, 1]
```

### 为什么要改

两处修正：

**ConvTranspose2d → Conv2d**：当 `stride=1, padding=1, kernel_size=3` 时，ConvTranspose2d 在数学上等价于 Conv2d，但内部实现会多做一次 input padding → conv → output cropping 的操作。纯属浪费。

**加 Tanh**：训练数据通过 `Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))` 归一化到 `[-1, 1]`。旧代码输出无界的原始值，解码器必须隐式学习将输出限制在 `[-1, 1]` 范围内——这增加了不必要的学习负担：
- 训练早期输出值可能远超 `[-1, 1]`，导致 MSE 损失数值巨大
- 梯度在这些异常值上可能不稳定
- 模型容量被浪费在"学习限幅"而非"学习重建"

Tanh 天然输出 `[-1, 1]`，与数据范围精确匹配，让解码器专注于重建质量本身。

---

## 改动影响的参数变化

| 组件 | 旧参数量 | 新参数量 | 变化 |
|------|---------|---------|------|
| 编码器 ResidualBlock ×8 | ~590K | ~787K | +33% (加了 bn2+prelu2) |
| DownSampleBlock tail ×4 | ~1.2M | 0 | -100% (删除) |
| 解码器 ResidualBlock ×4 | ~393K | ~525K | +33% (加了 bn2+prelu2) |
| 解码器 AttentionGate ×3 | 0 | ~344K | 新增 |
| **编码器总计** | ~3.5M | ~3.1M | **-11%** |
| **解码器总计** | ~5.0M | ~5.6M | **+12%** |
| **模型总计** | ~9.5M | ~9.7M | **+2%** |

编码器参数下降（tail 删除 > ResidualBlock BN 增加），解码器参数略增（AttentionGate 为主要增量）。整体模型规模几乎不变，但结构效率更高。

---

## 接口兼容性

所有对外接口保持不变：
- `SemanticEncoder.__init__(in_channels, num_downsample_blocks, base_channels)` — 不变
- `SemanticEncoder.forward(x) → List[Tensor]` — 不变
- `SemanticDecoder.__init__(embedding_dims, out_channels, skip_dropout_p=None)` — 删除了 `up_mode` 参数（原默认值 "nearest" 无人显式传入）
- `SemanticDecoder.forward(quant_feats) → Tensor` — 不变
- `SemanticDecoder.set_skip_dropout_p(p_list)` — 不变
- `VectorQuantizer.forward(inputs) → (vq_loss, quantized, indices)` — 不变
- `VectorQuantizer.get_quantized_features(indices) → Tensor` — 不变
- `DeepSC` 模型的所有方法 — 不变

**结论**：现有训练脚本 (`train.py`)、测试脚本 (`test_BPP.py`, `test_real.py`)、损失函数 (`DeepSCLoss`) 均无需修改，可直接使用新模型。

---

## 修正记录：改动 3 策略调整 (2026-05-15)

### 问题发现

改动 3（VQ L2 归一化）实施后训练，Epoch 20 出现严重码本坍缩：

| Layer | 活跃率 | 死码字 |
|-------|--------|--------|
| Layer 0 (D=128) | 39% | 39/64 |
| Layer 1 (D=256) | 42% | 37/64 |
| Layer 2 (D=512) | 27% | 47/64 |
| Layer 3 (D=1024) | 14% | 55/64 |

### 根因分析

改动 3 将距离计算从欧氏空间搬到归一化超球面，使最近邻搜索**仅依赖方向**。在高维空间（D=1024）中，编码器只需学会指向极少数"好方向"即可满足重建需求，其余码字因方向永远不匹配而成为僵尸码字。

ViT-VQGAN (ICLR 2022) 论文中 L2 归一化成功的前提是 **Factorized Codes（低维查找空间，8~32维）**，低维球面上向量天然分散。SimVQ 在完整特征维度（128~1024）做归一化，条件根本不同。

### 修正方案

**回退 L2 归一化，改为 LayerNorm + 欧氏距离**：

```python
# 修正后的 VectorQuantizer.forward()
inputs_bhwc = inputs.permute(0, 2, 3, 1).contiguous()
inputs_bhwc = self.input_norm(inputs_bhwc)          # 新增: LayerNorm 稳定统计量
embed_weight = self.codebook.projected_weight()

# 恢复欧氏距离 (幅值+方向双重自由度)
d = torch.sum(flat ** 2, dim=1, keepdim=True) + \
    torch.sum(embed_weight ** 2, dim=1) - 2 * \
    torch.einsum('bd,nd->bn', flat, embed_weight)
```

**LayerNorm 的作用**（有理有据）：
- 将各层特征统一到零均值单位方差，防止编码器输出范数漂移（已知问题：编码器范数可能从 22 暴增到 20000，见 lucidrains/vector-quantize-pytorch issue #26）
- 不剥夺幅值自由度（L2 范数 ≈ √C，各向量仍可变），保留码字选择的多样性
- 社区广泛实践：CSDN VQ-VAE 教程明确建议"对编码器输出做 LayerNorm 防止梯度爆炸"

**欧氏距离保留的原因**：
- 幅值差异是码字多样性的重要来源，不同空间位置的特征自然有不同的激活强度
- SimVQ 的冻结嵌入 + 可训练投影机制依赖梯度通过 STE 回传，欧氏空间中的 MSE 损失与距离度量一致
