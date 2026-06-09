# 为什么论文里的 CVQ 很强，但我们当前 B/C 在 768x512 下低于 A

生成时间：2026-06-09  
参考论文：`document/Channel-wise Vector Quantization.pdf`  
对比实验：A / B / C 当前 best checkpoint  
重点问题：论文中 CVQ 重建结果很高，但我们的 B/C 在原始 Kodak `768x512`、SNR=0 下明显低于 A，是否矛盾？

## 1. 结论先说

这不是论文结论和我们实验结果直接矛盾，而是我们当前 B/C 实验与论文 CVQ 实验并不是同一个条件。

论文里的 CVQ 高性能建立在以下前提上：

```text
1. 训练的是完整图像 tokenizer，而不是在已有通信 U-Net 里局部替换一层量化器。
2. VQ 与 CVQ 在相同 token budget 下公平比较，例如 c = h*w = 256。
3. 主要重建实验固定在 256x256 分辨率。
4. 使用 VQGAN 风格训练目标：L2 + commitment + LPIPS + PatchGAN adversarial loss。
5. 可变分辨率扩展不是简单 bilinear resize，而是使用 resampling modules / learnable queries，并做 variable-resolution training。
6. 论文重建指标不包含 LDPC/BPSK 在 SNR=0 下的传输误码。
```

我们的 B/C 当前设置是：

```text
1. 原模型是通信场景的两层 U-Net + SimVQ。
2. 只把第 1 层 skip 特征改成 channel-wise CVQ，第 2 层仍是 patch-wise。
3. 训练尺寸 256x256，测试 768x512 时第 1 层 CVQ 需要从 96x64 resize 到 32x32 查码本，再 resize 回 96x64。
4. 损失函数保持原方案：MSE + VQ loss，没有 LPIPS，没有 GAN。
5. 测试包含 LDPC 1/2 + BPSK + AWGN，SNR=0 dB。
6. A/B/C 码率和 token 分配并不等价。
```

因此，当前结果更准确的解释是：

```text
论文证明了：在它的 tokenizer 架构、训练目标、数据规模、公平 token budget 和可变分辨率模块下，CVQ 可以比传统 VQ 更强。

我们的实验说明：把 CVQ 以最简单形式局部移植到通信 U-Net 的第一层 skip 特征上，并在 768x512 下用 bilinear resize 适配，会明显损失高分辨率空间细节；在 SNR=0 传输条件下，这个问题进一步放大。
```

## 2. 我们当前 A/B/C 的关键结果

### 2.1 Kodak-256 no-resize, SNR=0

测试条件：

```text
测试集：/workspace/yi/work/Kodak-256-transform-resize
测试尺寸：256 x 256
resize：不 resize
信道：LDPC R=1/2 + BPSK
SNR：0 dB
```

结果：

| 方案 | best epoch | 测试压缩率 | MS-SSIM | PSNR |
|---|---:|---:|---:|---:|
| A: patch-wise SimVQ | 153 | `0.0442708` | `0.8956` | `24.2625 dB` |
| B: SimVQ + CVQ | 112 | `0.0755208` | `0.9045` | `25.1364 dB` |
| C: SimVQ + CVQ + nested dropout | 101 | `0.0755208` | `0.9029` | `25.1595 dB` |

在 256x256 no-resize 条件下，B/C 是强于 A 的。这一点反而说明 CVQ 并不是完全无效。

但注意：B/C 在 256x256 下实际压缩率是 `0.07552`，明显高于 A 的 `0.04427`。所以这个结果不能直接说明 B/C 在相同压缩率下更强。

### 2.2 Kodak 768x512, SNR=0

测试条件：

```text
测试集：/workspace/yi/work/Kodak
测试尺寸：Resize(768,512)
信道：LDPC R=1/2 + BPSK
SNR：0 dB
```

结果：

| 方案 | best epoch | 测试压缩率 | MS-SSIM | PSNR |
|---|---:|---:|---:|---:|
| A: patch-wise SimVQ | 153 | `0.0442708` | `0.9047` | `25.3016 dB` |
| B: SimVQ + CVQ | 112 | `0.0407986` | `0.8783` | `24.6866 dB` |
| C: SimVQ + CVQ + nested dropout | 101 | `0.0407986` | `0.8774` | `24.6618 dB` |

在 768x512 条件下，A 明显强于 B/C。

## 3. 论文 CVQ 到底比较的是什么

论文的核心思想是：

```text
传统 VQ：每个 1x1xc 的空间位置 feature vector 一个 index。
CVQ：每个 hxwx1 的通道 feature map 一个 index。
```

论文认为：

- patch-wise VQ 的 patch embedding 高度重复，容易导致码本坍缩；
- channel-wise embedding 更可分，能更充分使用码本；
- channel token 形成天然的一维序列，有利于 next-channel prediction；
- nested dropout 可以诱导 coarse-to-fine 的通道顺序。

论文主实验的几个重要设置：

```text
数据集：ImageNet-1K
图像分辨率：256x256
训练 epoch：100
global batch size：256
默认 codebook size：16384
优化器：Adam, lr=1e-4, weight decay=1e-4
训练目标：pixel-wise L2 + commitment + LPIPS + PatchGAN adversarial loss
```

论文特别强调公平比较：

```text
通过设置 c = h*w = 256，让 CVQ 和 VQ 具有相同 token 数、相同 lookup complexity、相同 memory usage 和相同 training overhead。
```

也就是说，论文中 256-token 设置里：

```text
patch-wise VQ token 数 = 16 * 16 = 256
CVQ token 数 = c = 256
```

这个条件和我们的 768x512 A/B/C 对比不同。

## 4. 最关键差异一：我们的 A/B/C token budget 不公平

在我们的 768x512 测试中：

### A: patch-wise

```text
Layer 1 token = 96 * 64 = 6144
Layer 2 token = 48 * 32 = 1536
```

码本：

```text
K1=16 -> b1=4
K2=2  -> b2=1
```

source bits：

```text
Layer 1 = 6144 * 4 = 24576 bits
Layer 2 = 1536 * 1 = 1536 bits
Total = 26112 bits
```

### B/C: Layer1 channel-wise + Layer2 patch-wise

```text
Layer 1 channel token = 256
Layer 2 patch token = 1536
```

码本：

```text
K1=65536 -> b1=16
K2=8192  -> b2=13
```

source bits：

```text
Layer 1 = 256 * 16 = 4096 bits
Layer 2 = 1536 * 13 = 19968 bits
Total = 24064 bits
```

这里最重要的不是 total bits 差了多少，而是高分辨率 Layer 1 的 bit 分配完全不同：

| 方案 | Layer 1 类型 | Layer 1 token 数 | Layer 1 bit/index | Layer 1 总 bits |
|---|---|---:|---:|---:|
| A | patch-wise | 6144 | 4 | 24576 |
| B/C | channel-wise | 256 | 16 | 4096 |

A 在高分辨率第一层传了 `24576 bits`，B/C 只传了 `4096 bits`。B/C 虽然 `K1=65536` 看起来很大，但第 1 层 token 数太少，所以第 1 层总信息量只有 A 的约 `1/6`。

这和论文的公平设定不同。论文是 `c = h*w = 256`，CVQ 和 VQ token 数对齐；我们的 768x512 A/B/C 对比中，A 的第 1 层 patch token 数随分辨率增长到了 `6144`，B/C 的 channel token 仍固定是 `256`。

结论：

```text
论文没有证明“在任意分辨率下，固定 256 个 channel token 一定能超过数千个 patch token”。
论文证明的是“在相同 token budget 下，CVQ tokenizer 可以比传统 VQ tokenizer 更有效”。
```

## 5. 最关键差异二：论文 256x256 主实验没有我们的 768x512 尺度错配

论文主重建实验固定在 `256x256`。在这个条件下，CVQ codeword 的空间大小和训练/测试 latent 尺寸一致。

我们的 B/C 在训练时：

```text
输入：256x256
Layer 1 特征：32x32
CVQ codeword shape：32x32
```

在测试 `768x512` 时：

```text
输入：768x512
Layer 1 特征：96x64
CVQ codeword shape：32x32
```

当前代码做的是：

```text
96x64 feature channel
-> F.interpolate 到 32x32
-> nearest codebook lookup
-> F.interpolate 回 96x64
-> decoder
```

这会带来明显问题：

1. 查码本前已经把高分辨率 Layer 1 特征从 `96x64` 压到 `32x32`，高频细节被平滑。
2. 查完后再插值回 `96x64`，只是把平滑后的码字放大，不可能恢复被压掉的局部结构。
3. Layer 1 本来是 U-Net 高分辨率 skip，最负责边缘、纹理、颜色边界；这里的信息瓶颈会直接伤害 PSNR/MS-SSIM。

这解释了为什么：

```text
在 Kodak-256 no-resize 下，B/C > A；
但在 Kodak 768x512 下，B/C < A。
```

因为 256x256 下 B/C 没有这个尺度错配；768x512 下有。

## 6. 最关键差异三：论文的可变分辨率 CVQ 不是我们这种 bilinear resize

论文 Appendix D 提到可变分辨率扩展时，不是简单把 feature map bilinear resize 到固定尺寸。

论文做法是：

```text
1. 量化前：用一组 fixed learnable queries cross-attend 任意空间尺寸 h*w 的特征，
   将其 resample 到固定 h0*w0。
2. 量化后：动态生成 h*w target queries，
   将量化后的 channel codeword project 回 decoder 需要的空间尺寸。
3. 训练时：progressively scale input resolution from 256 to 512 and 1024。
```

我们的做法是：

```text
F.interpolate(..., mode="bilinear", align_corners=False)
```

这两者不是同一个级别的适配。

论文的 learnable resampling 可以学习如何保留关键空间结构；我们的 bilinear resize 是固定低通滤波式缩放，会天然损失纹理和边缘。

所以，不能用论文的 variable-resolution 结果直接证明我们当前 B/C 的 768x512 测试应该很高。

## 7. 最关键差异四：论文使用 VQGAN 风格损失，我们是 MSE-only

论文 tokenizer training 使用：

```text
pixel-wise L2 loss
commitment loss
LPIPS loss
PatchGAN adversarial loss
```

我们的当前 A/B/C 使用：

```text
MSE reconstruction loss
VQ loss
MS-SSIM loss = 0
LPIPS loss = 0
GAN loss = 无
```

这会造成两个影响：

1. 论文的 decoder 会被 LPIPS/GAN 推着恢复更自然、更高感知质量的纹理。
2. 我们的模型主要优化像素 MSE，更倾向于平均化和平滑化，尤其当 CVQ 第 1 层被 resize 压缩后，MSE-only 很难补回高频细节。

这也是为什么不能把论文 Table 1 / Table 3 的 rFID、SSIM、PSNR 直接当作我们通信模型的预期。

## 8. 最关键差异五：论文比较的是 CVQ vs vanilla VQ，我们的 A 是 SimVQ patch-wise

论文里 CVQ 的优势很大一部分来自解决传统 VQ 的 codebook collapse。

论文报告：

- patch-wise VQ 在大码本时使用率会严重下降；
- CVQ 可以维持接近 100% 的 codebook utilization；
- 例如 codebook size 到 65,536 时，VQ utilization 很低，而 CVQ 仍很高。

但我们的 A 不是论文里的 vanilla VQ 大码本崩塌 baseline。我们的 A 是：

```text
SimVQ patch-wise
K1=16,K2=2
```

而且 A 的码本非常小，使用率天然容易达到 100%。当前 codebook metrics 显示：

```text
A Layer 1 active_ratio = 1.0
A Layer 2 active_ratio = 1.0
```

B/C 的码本使用率也不差：

```text
B Layer 1 active_ratio ≈ 0.79
B Layer 2 active_ratio ≈ 0.99
C Layer 1 active_ratio ≈ 0.80
C Layer 2 active_ratio ≈ 0.99
```

所以我们的性能差距不是简单的“CVQ 没用起来”。相反，B/C 码本使用还可以，但高分辨率层空间信息量和尺度适配存在更大的瓶颈。

结论：

```text
论文证明 CVQ 相对容易坍缩的 vanilla VQ 很强；
我们的 A 是小码本 SimVQ patch-wise，且高分辨率 token 很多，不是论文里那个弱 VQ baseline。
```

## 9. 最关键差异六：我们的测试包含 SNR=0 的真实链路误码

论文的重建实验是 tokenizer reconstruction，不经过：

```text
index -> bits -> LDPC -> BPSK -> AWGN -> LDPC decode -> bits -> index
```

我们的测试包含 LDPC 1/2 + BPSK + AWGN，SNR=0 dB。

这对 CVQ 特别敏感，原因是：

### 9.1 patch-wise index 错误通常是局部错误

A 的 Layer 1 是 patch-wise：

```text
一个 index 对应一个空间位置的 256 维 feature vector
```

如果某个 index 被信道误码影响，主要破坏一个局部空间位置。

### 9.2 channel-wise index 错误可能破坏整张通道图

B/C 的 Layer 1 是 channel-wise：

```text
一个 index 对应一个完整 channel map
```

如果某个 channel index 错了，就可能把整个通道图替换成另一个 codeword。这个错误是全局性的，不是局部一个 patch。

### 9.3 B/C 单个 index bit 数更大

B/C：

```text
Layer 1 index = 16 bits
Layer 2 index = 13 bits
```

A：

```text
Layer 1 index = 4 bits
Layer 2 index = 1 bit
```

在残余 bit error 存在时，大码本 index 的错误跳转空间更大，错到完全不同 codeword 的概率和破坏性都更高。

所以，即使 B/C 的 no-channel 重建接近，到了 SNR=0 链路下也可能比 A 更吃亏。

## 10. 最关键差异七：我们的 CVQ 是局部移植，不是完整 CVQ tokenizer

论文的 CVQ 是完整 tokenizer 设计：

```text
encoder -> channel-wise quantization -> decoder
```

整个 encoder、codebook、decoder 是围绕 channel-wise tokenization 共同训练的。

我们的 B/C 是：

```text
Layer 1: channel-wise CVQ
Layer 2: patch-wise SimVQ
decoder: 原本为多尺度 U-Net skip 特征设计
```

也就是说，我们并没有重写整个 decoder 让它专门适配 channel-wise token；只是让第 1 层 skip 由 patch-wise 特征变成 channel-wise codeword 重建特征。

这会带来结构错配：

- decoder 仍然强依赖 Layer 1 的空间 skip 细节；
- 但 CVQ Layer 1 在 768x512 下提供的是被 `32x32` codeword 限制后的通道图；
- Layer 2 虽然码本很大，但分辨率低，补不回 Layer 1 的高频细节。

论文里的 CVQ decoder 是为这种 channel token 设计并充分训练的；我们的 decoder 只是被迫接受这种替代特征。

## 11. 最关键差异八：nested dropout 的实现不等价

论文中的 nested channel dropout 不是简单“随机置零后半通道”这么简单。它的 Appendix B 写了：

```text
Lnested(ckeep) =
  Lrecon(Z_truncated)
  + Lquant(Z_truncated)
  + Llpips(Z_truncated)
  + lambda_GAN(ckeep) * LGAN(Z_truncated)
```

并且：

- quantization loss 只对 active channels 计算；
- GAN weight 对 `ckeep` 自适应；
- 以概率 alpha 做 nested dropout，剩余概率做 full channel 训练；
- 目标是让不同 channel prefix 都能独立重建。

我们的 C 当前实现是：

```text
训练时以 alpha=0.25 概率：
随机 c_keep
feat[:, c_keep:, :, :] = 0
然后走同一个 MSE + VQ loss
```

差异：

1. 没有 prefix reconstruction auxiliary objective。
2. 没有 LPIPS。
3. 没有 GAN。
4. 没有 adaptive GAN weight。
5. quant loss 没有按 active channels 做特殊处理。
6. 没有显式保证通道顺序真的 coarse-to-fine。

所以 C 没有明显超过 B，并不奇怪。论文中 nested dropout 主要提升 AR generation，同时保持重建质量；我们的 C 是通信重建任务，而且 nested 实现是简化版。

## 12. 为什么 256x256 下 B/C 变好了

Kodak-256 no-resize 结果：

```text
A: 24.2625 dB
B: 25.1364 dB
C: 25.1595 dB
```

这说明只要满足以下条件，B/C 是有潜力的：

```text
1. 测试尺寸和训练尺寸一致；
2. CVQ codeword shape = Layer 1 feature shape = 32x32；
3. 不需要 96x64 -> 32x32 -> 96x64 的插值；
4. B/C 在 256x256 下实际压缩率为 0.07552，比 A 的 0.04427 高很多。
```

这个结果和论文更接近，因为论文主实验也是固定 256x256。

但这也再次说明：我们目前 B/C 在 768x512 下差，不是“CVQ 理论错了”，而是当前实验设置与论文高性能设置不一致。

## 13. 当前 B/C 低于 A 的主因排序

按影响程度，我认为主要原因排序如下：

### 第一位：Layer 1 高分辨率信息量严重不足

A 的 Layer 1 有：

```text
6144 spatial tokens
24576 bits
```

B/C 的 Layer 1 有：

```text
256 channel tokens
4096 bits
```

高分辨率 skip 特征对图像重建极其重要。B/C 把这部分压得太狠。

### 第二位：768x512 下 CVQ codeword shape 错配

B/C 在 768x512 下第一层是：

```text
96x64 -> 32x32 -> 96x64
```

这会直接损伤局部细节。

### 第三位：我们的 variable-resolution 实现太简单

论文使用 learnable query resampling，并做 variable-resolution training。我们只是 bilinear interpolation。

### 第四位：训练目标和论文不一致

论文使用 LPIPS/GAN；我们 MSE-only。CVQ 的感知纹理优势没有被充分训练出来。

### 第五位：通信误码对 channel-wise token 更全局

一个错误 channel token 影响整张 channel map，一个错误 patch token 只影响局部位置。

### 第六位：A 不是论文中弱 vanilla VQ baseline

A 是小码本 SimVQ，码本使用率 100%，并且空间 token 很多。它是很强的通信重建 baseline。

## 14. 这是否说明文章不适合我们的任务

不是。更准确地说：

```text
文章的 CVQ 思路适合继续借鉴，但不能按当前这种最简替换方式直接期待超过 A。
```

CVQ 对我们的任务仍有价值：

- 它能减少 token 数；
- 它可以用 channel 表示全局结构；
- 它对 256x256 固定分辨率已经表现出潜力；
- 它可能适合渐进传输；
- 它可能降低 AR 或序列建模复杂度。

但要在我们的通信重建任务里超过 A，需要更贴近论文条件，尤其要解决高分辨率信息和可变分辨率 resampling。

## 15. 后续改进建议

### 15.1 不要让第 1 层完全 channel-wise

当前最直接建议：

```text
Layer 1 保留 patch-wise
Layer 2 尝试 channel-wise / CVQ
```

原因：

- Layer 1 是高分辨率细节层；
- A 的优势主要来自 Layer 1 大量空间 token；
- CVQ 放在 Layer 2 更不容易破坏局部纹理。

候选：

```text
Variant D:
Layer 1: patch-wise SimVQ
Layer 2: channel-wise CVQ
```

### 15.2 做 block-wise CVQ，而不是整通道 CVQ

当前 B/C 的 Layer 1 是：

```text
每个 channel 整张图一个 index
```

可以改成：

```text
每个 channel 的局部 block 一个 index
```

例如 Layer 1 测试 `96x64` 时，分成多个 `32x32` 或 `16x16` block。

这样可以保留 CVQ 的 channel 思想，同时不完全丢掉空间自由度。

### 15.3 复现论文的 learnable resampling，而不是 bilinear resize

当前：

```text
F.interpolate to codeword_shape
F.interpolate back
```

建议改成：

```text
pre-quant learnable resampler:
  arbitrary HxW -> fixed H0xW0

post-quant learnable resampler:
  fixed H0xW0 -> target HxW
```

这更接近论文 Appendix D。

### 15.4 做 variable-resolution training

如果测试目标是 `768x512`，训练不能只看 `256x256`。

可选训练策略：

```text
训练中混合：
256x256
512x512 或 512x384
768x512
```

或者直接用 DIV2K/Flickr2K 的高分辨率训练。

### 15.5 让 B/C 和 A 的码率更公平

当前 768x512 下：

```text
A = 0.04427
B/C = 0.04080
```

B/C 更低码率，本身吃亏。

如果希望公平，应至少比较：

```text
相同 compression_ratio
相同 high-resolution layer bit allocation
相同 channel condition
```

### 15.6 加入 LPIPS / MS-SSIM / GAN 或感知损失

论文 tokenizer 的强结果依赖 LPIPS 和 PatchGAN。我们当前 MSE-only 对 CVQ 可能不友好。

更稳妥的下一步：

```text
MSE + 适量 MS-SSIM
或 MSE + LPIPS
```

GAN 可以后置，因为训练复杂度和不稳定性更高。

### 15.7 正确实现 nested channel dropout 的辅助目标

如果继续做 C，建议不只是置零，而是显式训练多个 prefix：

```text
full channels reconstruction loss
prefix channels reconstruction loss
prefix quantization loss
possibly consistency loss
```

否则 channel 顺序不一定真的形成 coarse-to-fine。

## 16. 对下一轮实验的建议优先级

我建议按以下顺序做：

### 优先级 1：Patch Layer1 + CVQ Layer2

目标：验证“不要破坏高分辨率 Layer 1”是否能保住 A 的优势，同时引入 CVQ。

```text
Layer 1: patch-wise SimVQ
Layer 2: channel-wise CVQ
```

### 优先级 2：Layer1 block-CVQ

目标：让第 1 层仍然保留空间 token 数增长，不再只有 256 个 channel token。

### 优先级 3：learnable resampling + variable-resolution training

目标：复现论文 Appendix D 的关键条件，解决 768x512 尺度错配。

### 优先级 4：LPIPS/MS-SSIM 辅助损失

目标：让 CVQ 的通道图表达更关注感知结构，而不是只优化 MSE。

## 17. 最终判断

当前 B/C 在 768x512 下低于 A，不说明论文 CVQ 的结果错误，也不说明 CVQ 不适合我们的项目。更准确的原因是：

```text
我们当前 B/C 与论文 CVQ 的实验条件差异太大；
尤其是 768x512 下第 1 层 channel-wise token 数固定、空间细节 bit 严重不足、
并且使用了简单 bilinear resize 处理可变分辨率。
```

论文结果给我们的启发应该是：

```text
CVQ 要想强，必须和 tokenizer 架构、训练目标、token budget、公平比较条件、
可变分辨率 resampling、nested dropout auxiliary objective 一起设计。
```

当前 B/C 是一个最小可行移植版本，适合验证方向，但还不能代表论文里完整 CVQ 的真实性能。

