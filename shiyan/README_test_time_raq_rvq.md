# Test-time 两级残差 RAQ-RVQ 方案配置与实现说明

## 1. 文档目的

本文档说明当前新增测试脚本所执行的完整实验方案：

```text
scripts/eval/test_src2048_2048_raq_rvq2_fair_ch256_512.sh
```

当前方案确实使用了 RVQ（Residual Vector Quantization，残差向量量化）的核心思想：

1. 第一级量化原始 encoder feature；
2. 计算第一级残差 `residual_1 = feat - q_1`；
3. 第二级量化 `residual_1`，而不是再次量化原始 `feat`；
4. Decoder 接收 `q_1 + q_2`。

但是，这不是一个重新训练过的完整 RVQ 模型，而是一个：

> **test-time zero-shot residual RAQ 实验**

两个 stage 复用现有 checkpoint 中已经训练好的单级 RAQ 生成器，没有新增可学习参数，也没有针对 residual 分布重新训练第二级码本生成器。

---

## 2. 当前使用的 checkpoint

新测试脚本默认加载：

```text
checkpoints/
└── shiyan_raq_src2048-2048_raq2-2048_curriculum_rate044_A_patch_ch256-512_unet2_ds8x2_k2048/
    └── best_vq_deepsc.pth
```

checkpoint 中已经训练好的主要结构为：

| 项目 | scale 0 | scale 1 |
|---|---:|---:|
| Source SimVQ codebook K | 2048 | 2048 |
| Feature channel / embedding dim | 256 | 512 |
| RAQ generator | `self.raqs[0]` | `self.raqs[1]` |
| RAQ 可生成目标 K 范围 | 2～2048 | 2～2048 |

严格 checkpoint 加载仍然使用原来的参数结构。新增 RVQ 开关只是普通 Python 属性，不会加入 `state_dict`。

---

## 3. 必须区分的三种 K

这是当前方案最容易混淆的地方。

### 3.1 Source codebook K

脚本配置：

```bash
export SIMVQ_NUM_EMBEDDINGS_LIST="2048,2048"
```

它表示 checkpoint 中两个尺度已经训练好的 source SimVQ codebook 大小：

```text
scale 0 source K = 2048
scale 1 source K = 2048
```

这两个 source codebook 没有被拆分，也没有被重新建立。

### 3.2 每个尺度的 K_total

当前新脚本配置：

```bash
export SIMVQ_RAQ_TARGET_LIST="2048,16"
```

在原单级 RAQ 基线中，它表示实际动态目标码本大小：

```text
scale 0: 单个 W_trg，K=2048
scale 1: 单个 W_trg，K=16
```

在新 RVQ 分支中，它表示每个尺度允许使用的**总索引比特预算**：

```text
scale 0: K_total=2048 → 总预算 log2(2048)=11 bit/token
scale 1: K_total=16   → 总预算 log2(16)=4 bit/token
```

### 3.3 RVQ 实际 stage K：支持显式灵活配置

开启 RVQ 后，代码不会再生成单个 W2048 和 W16。现在可以通过下面的环境变量显式指定每个尺度有序的 stage K：

```bash
export SIMVQ_TEST_RAQ_RVQ_K_LISTS="32,64;8,2"
```

分号分隔尺度、逗号分隔同一尺度内的 stage，因此上面的含义是：

```text
scale 0: [32, 64]
scale 1: [8, 2]
```

也接受等价的 JSON 写法：

```bash
export SIMVQ_TEST_RAQ_RVQ_K_LISTS='[[32,64],[8,2]]'
```

stage 的顺序有意义：`[32,64]` 表示 K=32 的第一级先量化原始 feature，K=64 的第二级再量化残差；它与 `[64,32]` 码率相同，但量化过程并不等价。

如果不提供 `SIMVQ_TEST_RAQ_RVQ_K_LISTS`，代码保留原来的自动均衡拆分规则：

```text
b_total = log2(K_total)
b_1 = ceil(b_total / 2)
b_2 = floor(b_total / 2)
K_1 = 2^b_1
K_2 = 2^b_2
```

自动拆分结果为：

| 尺度 | K_total | 总 bit | stage 1 | stage 2 | stage bit 总和 |
|---|---:|---:|---:|---:|---:|
| scale 0 | 2048 | 11 | K1=64，6 bit | K2=32，5 bit | 6+5=11 |
| scale 1 | 16 | 4 | K1=4，2 bit | K2=4，2 bit | 2+2=4 |

因此，新方案的实际动态 RAQ codebook 结构为：

```text
rvq_k_lists = [
    [64, 32],  # scale 0
    [4, 4],    # scale 1
]
```

当前新测试脚本默认显式设置为上述 `[[64,32],[4,4]]`，但启动时可以直接覆盖。例如：

```bash
SIMVQ_TEST_RAQ_RVQ_K_LISTS="32,64;8,2" \
NO_CHANNEL=1 \
bash scripts/eval/test_src2048_2048_raq_rvq2_fair_ch256_512.sh
```

代码会在测试前严格验证：

1. 尺度数量必须等于 `UNET_DEPTH`；
2. 每个 stage K 必须是大于等于 2 的二次幂；
3. 每个 stage K 必须位于 `RAQ_MIN_TRG` 到 `RAQ_MAX_TRG` 范围内；
4. 每个尺度必须满足 `sum(log2(stage_K)) = log2(K_total)`。

例如 `[[32,64],[8,2]]` 的预算为 `5+6=11` 和 `3+1=4`，与 `[2048,16]` 完全一致；`[[32,32],[8,2]]` 的第一个尺度只有 10 bit，会直接报错。

需要特别强调：

```text
SIMVQ_RAQ_TARGET_LIST="2048,16"
```

在 RVQ 开启时，不代表仍然使用单个 K=2048 和 K=16 目标码本；它代表的是与原方法相同的总索引 bit 预算。

---

## 4. 模型与特征尺度配置

新脚本使用：

```bash
export SIMVQ_UNET_DEPTH="2"
export SIMVQ_DOWNSAMPLE_STRIDES="8,2"
export SIMVQ_BASE_CHANNELS="128"
export SIMVQ_ENCODER_RES_BLOCKS="4"
export SIMVQ_DECODER_RES_BLOCKS="4"
export SIMVQ_QUANTIZER_TYPE="simvq"
export SIMVQ_QUANTIZER_AXIS_LIST="patch,patch"
```

对于 256×256 测试图像：

| 尺度 | 累计下采样 | Feature shape | 每张图 token 数 |
|---|---:|---|---:|
| scale 0 | 8 | `[1,256,32,32]` | 32×32=1024 |
| scale 1 | 16 | `[1,512,16,16]` | 16×16=256 |

两个尺度始终独立完成 RVQ：

```text
scale 0 的 residual 不会送到 scale 1
scale 1 的 residual 也不会送到 scale 0
```

最终 Decoder 接收的仍然是两个尺度组成的 feature list：

```text
[
    q_sum_scale0,
    q_sum_scale1,
]
```

---

## 5. 测试期开关

新脚本设置：

```bash
export SIMVQ_USE_RAQ="1"
export SIMVQ_TEST_USE_RAQ_RVQ="1"
export SIMVQ_TEST_RAQ_RVQ_DEPTH="2"
export SIMVQ_TEST_RAQ_RVQ_K_LISTS="64,32;4,4"
```

含义分别是：

| 环境变量 | 含义 |
|---|---|
| `SIMVQ_USE_RAQ=1` | 使用 RAQ 动态目标码本分支 |
| `SIMVQ_TEST_USE_RAQ_RVQ=1` | 仅在测试阶段启用 residual RAQ-RVQ |
| `SIMVQ_TEST_RAQ_RVQ_DEPTH=2` | 当前实验固定为两级 residual quantization |
| `SIMVQ_TEST_RAQ_RVQ_K_LISTS` | 按尺度显式指定有序 stage K；不设置则自动拆分 |

当：

```bash
SIMVQ_TEST_USE_RAQ_RVQ=0
```

或者该环境变量没有设置时，`forward_test()` 继续执行原单级 RAQ，不进入 RVQ 分支。

---

## 6. 每个尺度内部的真实 RVQ 流程

以下过程会分别对 scale 0 和 scale 1 执行一次。

### 6.1 获取同一尺度已有的 source codebook

```python
source_quantizer = self.vector_quantizers[scale_idx]
source_codebook = source_quantizer.transformed_weight()
```

同一个尺度的两个 RVQ stage 共享这一份 source codebook。

### 6.2 初始化残差和累计量化结果

```python
residual = feat
quantized_sum = torch.zeros_like(feat)
```

### 6.3 第一级量化原始 feature

以 scale 0 为例，stage 1 使用 K1=64：

```python
W_1 = self.raqs[0].generate_codebook_transformer(
    64,
    source_codebook,
)

q_1 = Q(feat, W_1)
residual_1 = feat - q_1
```

对应实际控制流：

```python
_, _, indices_1, quantized_raw_1 = source_quantizer.forward_raq(
    residual,
    W_1,
    return_raw=True,
)

quantized_sum = quantized_sum + quantized_raw_1
residual = residual - quantized_raw_1
```

此时：

```text
quantized_sum = q_1
residual = feat - q_1
```

### 6.4 第二级量化第一级 residual

scale 0 的 stage 2 使用 K2=32：

```python
W_2 = self.raqs[0].generate_codebook_transformer(
    32,
    source_codebook,
)

q_2 = Q(residual_1, W_2)
residual_2 = residual_1 - q_2
```

注意第二次 `forward_raq()` 的输入是已经更新后的 `residual`：

```python
_, _, indices_2, quantized_raw_2 = source_quantizer.forward_raq(
    residual,
    W_2,
    return_raw=True,
)

quantized_sum = quantized_sum + quantized_raw_2
residual = residual - quantized_raw_2
```

所以最终严格满足：

```text
q_sum = q_1 + q_2

residual_1 = feat - q_1
residual_2 = feat - q_1 - q_2
```

这就是 RVQ 残差量化思想在当前代码中的具体实现。

### 6.5 两个 stage 共享什么、不共享什么

对于同一个尺度：

```text
共享：同一个 self.raqs[scale_idx]
共享：同一个 source codebook
共享：同一个 checkpoint 中的已训练参数

不同：stage K
不同：动态目标码本 W_stage
不同：被量化的输入
```

其中：

```text
stage 1 输入 = feat
stage 2 输入 = feat - q_1
```

两个 U-Net 尺度之间则分别使用自己的 RAQ 模块：

```text
scale 0 使用 self.raqs[0]
scale 1 使用 self.raqs[1]
```

---

## 7. 发送端输出结构

RVQ 开启时，`forward_test()` 返回嵌套结构：

```python
out = {
    "indices": [
        [indices_scale0_stage1, indices_scale0_stage2],
        [indices_scale1_stage1, indices_scale1_stage2],
    ],
    "codebooks": [
        [W_scale0_stage1, W_scale0_stage2],
        [W_scale1_stage1, W_scale1_stage2],
    ],
    "rvq_k_lists": [
        [64, 32],
        [4, 4],
    ],
    "feature_shapes": [
        (32, 32),
        (16, 16),
    ],
    "branch": "raq_rvq",
    "test_raq_rvq_enabled": True,
}
```

普通 SRC 和单级 RAQ 的旧返回结构保持不变。

---

## 8. 接收端如何恢复 q1+q2

接收端不会把两个 stage 的索引合并成一个大索引。

以 scale 0 为例：

```python
q_1 = get_quantized_features(
    indices_scale0_stage1,
    codebook_weight=W_scale0_stage1,
)

q_2 = get_quantized_features(
    indices_scale0_stage2,
    codebook_weight=W_scale0_stage2,
)

quantized_scale0 = q_1 + q_2
```

scale 1 同样恢复：

```python
quantized_scale1 = q_1_scale1 + q_2_scale1
```

Decoder 最终接收：

```python
semantic_decoder([
    quantized_scale0,
    quantized_scale1,
])
```

因此发送端和接收端使用完全一致的 RVQ 加法重建方式。

---

## 9. 当前总 bit 预算

### 9.1 scale 0

Feature token 数：

```text
32×32 = 1024
```

原单级 K=2048：

```text
1024 × 11 = 11264 payload bit
```

新 RVQ：

```text
stage 1: 1024 × 6 = 6144 bit
stage 2: 1024 × 5 = 5120 bit
总计: 6144 + 5120 = 11264 bit
```

### 9.2 scale 1

Feature token 数：

```text
16×16 = 256
```

原单级 K=16：

```text
256 × 4 = 1024 payload bit
```

新 RVQ：

```text
stage 1: 256 × 2 = 512 bit
stage 2: 256 × 2 = 512 bit
总计: 512 + 512 = 1024 bit
```

### 9.3 两个尺度总计

```text
11264 + 1024 = 12288 payload bit/image
```

256×256 图像的 source payload bpp：

```text
12288 / (256×256) = 0.1875 bit/pixel
```

因此：

```text
原单级 [2048,16] payload = 12288 bit
两级 RVQ [[64,32],[4,4]] payload = 12288 bit
```

payload bit 预算严格一致。

---

## 10. 真实 LDPC 信道测试流程

真实信道模式中，每个尺度、每个 stage 都独立执行：

```text
stage indices
→ bit packing
→ LDPC encode
→ modulation
→ AWGN
→ LLR
→ LDPC decode
→ recovered stage indices
```

当前默认参数：

```bash
MODULATION="bpsk"
LDPC_N="256"
LDPC_K="0.5"
SNRS="0 3 6 9 12"
```

此处脚本中的 `LDPC_K=0.5` 实际作为 LDPC rate 使用：

```text
information block k = 256 × 0.5 = 128 bit
coded block n = 256 bit
rate = 1/2
```

对于 256×256 测试图像，四个 stage 的 payload 都能被128整除：

| 尺度/stage | payload bit | LDPC coded bit |
|---|---:|---:|
| scale 0 stage 1 | 6144 | 12288 |
| scale 0 stage 2 | 5120 | 10240 |
| scale 1 stage 1 | 512 | 1024 |
| scale 1 stage 2 | 512 | 1024 |
| 总计 | 12288 | 24576 |

当前标准 256×256 数据上没有额外 LDPC padding，因此 coded bit 也与原单流基线一致。

BPSK 下的实际 transmission ratio：

```text
24576 / (256×256×3) = 0.125
```

如果以后测试任意分辨率，逐 stage 独立 LDPC 可能产生不同的 padding。日志会分别报告：

```text
payload_bits
ldpc_padding_bits
coded_bits
modulation_padding_bits
transmitted_bits
channel_symbols
```

---

## 11. no-channel 测试流程

no-channel 模式不会绕过 RVQ。

实际流程为：

```text
image
→ SemanticEncoder
→ scale 0: q1 + q2
→ scale 1: q1 + q2
→ SemanticDecoder
→ reconstructed image
```

它只是跳过：

```text
LDPC、调制、AWGN、LLR、信道译码
```

因此 no-channel 结果可以直接观察量化方法本身是否有效，不受 BER 和信道噪声影响。

---

## 12. 哪些训练路径没有改变

以下路径均未加入 RVQ：

```text
forward_train()
forward_val()
训练 loss
训练 curriculum
SRC 分支
原单级 RAQ 测试分支
```

训练时 RAQ 仍然是：

```text
q = Q(feat, W_K)
```

而不是：

```text
q1 = Q(feat, W_1)
q2 = Q(feat-q1, W_2)
```

这也是当前实验属于 zero-shot RVQ，而不是 trained RVQ 的根本原因。

---

## 13. 当前方案的核心局限

当前第二级动态码本生成器没有针对第一级 residual 训练。

具体来说：

1. `W_2` 仍由原单级 RAQ generator 生成；
2. generator 的输入仍是同一 source codebook；
3. generator 不读取 residual；
4. generator 没有 RVQ stage ID；
5. `W_2` 没有保证包含零码字；
6. Decoder 没有在 `q_1+q_2` 输入分布上训练。

因此，当前代码虽然正确执行：

```text
residual_1 = feat - q_1
q_2 = Q(residual_1, W_2)
```

但不能保证：

```text
||residual_1 - q_2||² <= ||residual_1||²
```

也就是说，第二级可能让 residual 更小，也可能让 residual 更大。

当前完整 Kodak no-channel 实测正是后者：

| 尺度 | stage 1 后 residual MSE | stage 2 后 residual MSE | 变化 |
|---|---:|---:|---:|
| scale 0 | 0.114314 | 0.136162 | 增加约19.1% |
| scale 1 | 0.046256 | 0.077575 | 增加约67.7% |

因此当前性能下降不是因为 payload bit 预算不公平，而是因为 zero-shot stage 2 没有学会 residual correction。

### scale 1 的 `[4,4]` 额外限制

scale 1 两个 stage 都设置 K=4，并且共享同一个 generator 和同一个 source codebook。

eval 模式下，两次生成的 W4 完全相同。最终表示为：

```text
q_a + q_b
```

由于加法满足交换律，4个码字最多只能形成：

```text
4×5/2 = 10
```

种不同的无序和，而原单级 W16 可以提供16个独立码字。这说明相同4 bit payload 并不自动等于相同表示能力。

---

## 14. 当前观测结果

同一 checkpoint、同一 Kodak 24 图、同一 `[2048,16]` 总预算、no-channel：

| 方法 | 实际量化结构 | PSNR |
|---|---|---:|
| 原单级 RAQ | `[2048,16]` | 26.1338 dB |
| zero-shot 两级 RAQ-RVQ | `[[64,32],[4,4]]` | 22.9457 dB |

下降：

```text
26.1338 - 22.9457 = 3.1881 dB
```

约等于图像域 MSE 增加到原来的：

```text
10^(3.1881/10) ≈ 2.08 倍
```

这个结果只能说明：

> 当前共享单级 RAQ generator 的 zero-shot residual 复用没有成功。

它不能说明：

> 针对 residual 训练过的完整 RVQ 模型一定无效。

---

## 15. 如何运行公平单级基线

原测试脚本默认 target 是 `1024,256`，因此比较 `[2048,16]` 时必须显式覆盖，并显式关闭 RVQ：

```bash
SIMVQ_RAQ_TARGET_LIST=2048,16 \
SIMVQ_TEST_USE_RAQ_RVQ=0 \
NO_CHANNEL=1 \
GPU_ID=2 \
bash scripts/eval/test_src2048_2048_raq2_2048_curriculum_ch256_512.sh
```

期望关键输出：

```text
RAQ目标码本大小: [2048, 16]
Transmission K List: [2048, 16]
No channel ... PSNR: 26.1338 dB
```

---

## 16. 如何运行新 RVQ no-channel 测试

```bash
NO_CHANNEL=1 \
GPU_ID=2 \
bash scripts/eval/test_src2048_2048_raq_rvq2_fair_ch256_512.sh
```

测试其他等码率拆分时只需要覆盖 stage K，无需修改脚本：

```bash
SIMVQ_TEST_RAQ_RVQ_K_LISTS="32,64;8,2" \
NO_CHANNEL=1 \
GPU_ID=2 \
bash scripts/eval/test_src2048_2048_raq_rvq2_fair_ch256_512.sh
```

期望首先看到：

```text
[Test RAQ-RVQ] enabled=True, depth=2
[Test RAQ-RVQ] scale 0: K_total=2048 -> stage_K=[64, 32]
[Test RAQ-RVQ] scale 1: K_total=16 -> stage_K=[4, 4]
```

并在测试结束后看到每尺度：

```text
input feature energy
stage 1 residual energy
stage 2 residual energy
index range
codebook size
payload bits
bit budget match
```

---

## 17. 如何运行新 RVQ 真实信道测试

```bash
GPU_ID=2 \
SNRS="0 3 6 9 12" \
MODULATION="bpsk" \
LDPC_N=256 \
LDPC_K=0.5 \
bash scripts/eval/test_src2048_2048_raq_rvq2_fair_ch256_512.sh
```

真实信道日志还会输出：

```text
每 stage BER
每 stage index error rate
LDPC padding
coded/transmitted bits
payload/transmitted bpp
transmission ratio
与原单流 LDPC coded bits 是否一致
```

---

## 18. 相关代码位置

| 功能 | 文件 |
|---|---|
| 新 RVQ 测试脚本 | `scripts/eval/test_src2048_2048_raq_rvq2_fair_ch256_512.sh` |
| RVQ bit 拆分 | `utils/raq_rvq.py` |
| RVQ 发送端量化 | `models/deepsc.py::_forward_test_raq_rvq()` |
| 单级/RVQ 测试分支选择 | `models/deepsc.py::forward_test()` |
| RVQ 接收端逐 stage 求和 | `models/deepsc.py::reconstruct_from_indices()` |
| 外部动态码本量化 | `models/vector_quantizer.py::forward_raq()` |
| 动态 RAQ codebook 生成 | `models/raq.py::generate_codebook_transformer()` |
| 独立 stage bit packing | `utils/bit_utils.py` |
| no-channel/LDPC 测试 | `evaluation/quality.py` |
| 日志与 JSON 结果 | `test_real.py` |
| 测试期开关配置 | `config.py` |

---

## 19. 最终一句话定义

当前新方案可以准确概括为：

> 对两个 U-Net 尺度分别执行两级残差量化；第一级量化原始 feature，第二级量化 `feat-q1`，接收端恢复 `q1+q2`；两个 stage 共享当前尺度原有的 RAQ generator 和 source codebook，不增加参数，并通过拆分索引 bit 保持与单级 `[2048,16]` 相同的总 payload 预算。

同时必须附带以下限制：

> 第二级 generator 没有在 residual 分布上训练，Decoder 也没有在 `q1+q2` 上训练，因此它是 zero-shot RVQ 验证，不是完整训练过的 RVQ。
