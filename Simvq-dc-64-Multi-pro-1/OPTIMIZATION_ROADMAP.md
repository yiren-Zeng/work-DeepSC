# SimVQ 性能优化与改进路线图

> 分析日期: 2026-05-15 | 基于 ARCHITECTURE_ANALYSIS.md 和 CODE_QUALITY_ANALYSIS.md 的综合分析

---

## 路线图总览

```
Phase 1 (立即修复)        Phase 2 (短期优化)         Phase 3 (中期改进)         Phase 4 (长期演进)
┌─────────────────┐    ┌─────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ 修复 Bug &       │    │ 代码重构 &       │      │ 模型架构改进      │      │ 系统能力扩展      │
│ 消除硬编码       │───▶│ 基础设施完善     │─────▶│ 训练策略优化      │─────▶│ 工程化部署        │
│                  │    │                  │      │                   │      │                   │
│ 工期: 1-2天      │    │ 工期: 3-5天      │      │ 工期: 1-2周       │      │ 工期: 2-4周       │
└─────────────────┘    └─────────────────┘      └──────────────────┘      └──────────────────┘
```

---

## Phase 1: 立即修复 (P0 — 1-2 天)

### Fix 1.1: 修复日志目录日期格式 🔴

**文件**: `train.py:80`
**问题**: `strftime("%Y%M%D-%H%M%S")` 格式字符串错误
**修复**:
```python
# 错误
log_dir = os.path.join(cfg.LOG_DIR, datetime.now().strftime("%Y%M%D-%H%M%S"))
# 正确
log_dir = os.path.join(cfg.LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))
```

---

### Fix 1.2: 将硬编码路径移入 Config 🟠

**文件**: `test_BPP.py:28`, `test_real.py:68`
**问题**: checkpoint 路径硬编码为服务器路径
**修复**:
```python
# config.py 新增
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_vq_deepsc.pth")

# test_BPP.py / test_real.py 改为
checkpoint_path = cfg.CHECKPOINT_PATH
```
同时添加命令行参数支持:
```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default=None)
args = parser.parse_args()
checkpoint_path = args.checkpoint or cfg.CHECKPOINT_PATH
```

---

### Fix 1.3: 添加 .gitignore 🔴

**问题**: 项目缺少 .gitignore，可能误提交大文件
**新建文件**: `.gitignore`
```
# Python
__pycache__/
*.py[cod]
*.so
*.egg-info/

# PyTorch
checkpoints/
*.pth
*.pt

# TensorBoard
logs/
runs/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Environment
venv/
.env
```

---

### Fix 1.4: 添加 requirements.txt 🔴

**新建文件**: `requirements.txt`
```
torch>=2.0.0
torchvision>=0.15.0
tensorflow>=2.12.0
sionna>=0.15.0
numpy>=1.24.0
scipy>=1.10.0
opencv-python>=4.7.0
Pillow>=9.5.0
tqdm>=4.65.0
tensorboard>=2.12.0
```

---

## Phase 2: 短期优化 (P1 — 3-5 天)

### Opt 2.1: 消除代码重复 — 合并 forward_train 和 forward_val 🟠

**文件**: `models/deepsc.py`
**当前状态**: 两个方法 ~90% 代码重复
**重构方案**:
```python
def _forward_impl(self, x, mode='train'):
    """统一的前向传播实现"""
    snr_db = random.uniform(Config.SNR_RANGE_DB[0], Config.SNR_RANGE_DB[1])
    snr_tensor = torch.tensor(snr_db, device=self.device)
    current_mod_bits = self._sample_mod_bits(snr_db)
    current_rc = (Config.CHANNEL_CODING_RATE_TRAIN if mode == 'train'
                  else Config.CHANNEL_CODING_RATE_VAL)

    encoder_features = self.semantic_encoder(x)
    quantized_corrupted = []
    vq_losses = []

    for i, feat in enumerate(encoder_features):
        vq_loss, quantized_clean, encoding_idx = self.vector_quantizers[i](feat)
        vq_losses.append(vq_loss)

        corrupted_idx, _ = self.channel.apply_channel_noise(
            encoding_idx, self.num_embeddings_list[i],
            snr_tensor, current_rc, mod_bits=current_mod_bits
        )
        quantized_noisy = self.vector_quantizers[i].get_quantized_features(corrupted_idx)

        if mode == 'train':
            quantized_final = quantized_clean + (quantized_noisy - quantized_clean).detach()
        else:
            quantized_final = quantized_noisy
        quantized_corrupted.append(quantized_final)

    reconstructed_images = self.semantic_decoder(quantized_corrupted)
    return {"reconstructed_images": reconstructed_images,
            "vq_losses": vq_losses, "current_snr": snr_db}

def forward_train(self, x):
    return self._forward_impl(x, mode='train')

def forward_val(self, x):
    return self._forward_impl(x, mode='val')
```

**收益**: 减少 ~40 行重复代码，降低维护成本

---

### Opt 2.2: 提取公共组件 🟠

**新建文件**: `models/common.py`
```python
import torch.nn as nn

class ResidualBlock(nn.Module):
    """公共残差块 — semantic_encoder 和 semantic_decoder 共用"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn(out)
        out = self.prelu(out)
        out = self.conv2(out)
        return out + identity
```

然后从 `semantic_encoder.py` 和 `semantic_decoder.py` 中删除各自的 `ResidualBlock`，改为 `from .common import ResidualBlock`。

**新建文件**: `utils/random.py`
```python
def setup_seed(seed=42):
    """统一的随机种子设置"""
    import random, os, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
```

---

### Opt 2.3: 为 Config 添加命令行覆盖支持 🟡

**新增功能**: 不修改代码即可调整参数
```python
# config.py 新增方法
@classmethod
def from_args(cls, args=None):
    """从命令行参数覆盖配置"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--device', type=str)
    parser.add_argument('--resume', type=str)
    # ... 更多参数
    args = parser.parse_args(args)
    for k, v in vars(args).items():
        if v is not None:
            setattr(cls, k.upper(), v)
```

---

### Opt 2.4: 为 test_real.py 消除重复的函数定义 🟡

**问题**: `evaluate_metrics_with_channel` 在 `communications/evaluate.py` 和 `test_real.py` 中重复定义
**修复**: 在 `test_real.py` 中改为导入:
```python
from communications.evaluate import evaluate_metrics_with_channel
```

---

## Phase 3: 中期改进 (P2 — 1-2 周)

### Improve 3.1: 训练/推理信道一致性对齐

**问题**: 训练时用索引比特翻转，推理时走真实 AWGN+LDPC 链路
**方案 A (推荐)**: 在训练循环中**周期性地使用真实物理层链路**做验证:
```python
if epoch % 5 == 0:
    # 用真实 AWGN+LDPC 链路评估 MS-SSIM/PSNR
    real_metrics = evaluate_metrics_with_channel(
        model, val_dataloader, snr=10, ldpc_code=ldpc_code
    )
    writer.add_scalar("Val/Real_MS_SSIM@10dB", real_metrics[0], epoch)
```

**方案 B**: 在训练中引入**端到端的软仿真**，用可微 AWGN 替代比特翻转:
```python
# 将索引转化为连续嵌入 → AWGN → 软量化
quantized_signal = self.vector_quantizers[i].get_quantized_features(idx)
noisy_signal = awgn_channel(quantized_signal, snr_db)
# 用最近邻搜索恢复索引
```

**方案 B 需要更多改动**，建议先用方案 A 监控 gap，再决定是否需要方案 B。

---

### Improve 3.2: 探索最优码本配置

**当前**: 所有 4 层均为 K=64
**分析**: 浅层特征 (128² 空间, D=128) 包含更多空间细节，可能需要更大的码本; 深层 (16² 空间, D=1024) 语义更抽象，64 可能足够

**建议实验**:
```python
# 方案 1: 浅层大码本
NUM_EMBEDDINGS_LIST = [256, 128, 64, 64]

# 方案 2: 分层不同大小
NUM_EMBEDDINGS_LIST = [128, 64, 64, 32]

# 方案 3: 仅增大最小层
NUM_EMBEDDINGS_LIST = [512, 64, 64, 64]
```

在 BPP 约束下搜索最优配置。

---

### Improve 3.3: 引入速率自适应 (RAQ)

**目标**: 使同一模型能支持多种 BPP 的工作点
**方案**: 在 VQ 层添加**可变大小的码本掩码**:
```python
class RateAdaptiveVQ(VectorQuantizer):
    def forward(self, x, target_bpp=None):
        if target_bpp is not None:
            # 按需只使用前 M 个码字 (M <= K)
            active_size = self._bpp_to_codesize(target_bpp)
            masked_weight = self.codebook.projected_weight()[:active_size]
            # 在受限码本中最近邻搜索
        ...
```

这允许单个模型在不同带宽条件下灵活传输。

---

### Improve 3.4: 改进码本初始化策略

**当前**: 高斯随机初始化 `N(0, 1/sqrt(D))`
**改进**: 对第一层编码器输出做 K-Means 聚类，用聚类中心初始化码本:
```python
def init_codebook_with_kmeans(encoder, dataloader, layer_idx, device):
    """使用编码器输出的 K-Means 聚类中心初始化码本"""
    features = []
    for images in dataloader:
        with torch.no_grad():
            feats = encoder(images.to(device))
            features.append(feats[layer_idx].cpu())
    all_features = torch.cat(features, dim=0)
    # 在 all_features 上运行 K-Means
    ...
```

**收益**: 码本从第 0 个 epoch 就处于数据分布的合理区域，加速收敛。

---

### Improve 3.5: 添加验证集的 SNR 固定评估

**当前**: 验证循环随机采样 SNR，各 epoch 不可比
**方案**:
```python
# config.py
VAL_SNR_LIST = [0, 5, 10, 15]

# train.py 验证循环
for snr in VAL_SNR_LIST:
    deepsc_model.set_eval_snr(snr)  # 固定 SNR
    val_loss = evaluate(deepsc_model, val_dataloader, snr)
    writer.add_scalar(f"Val/Loss@{snr}dB", val_loss, epoch)
```

---

### Improve 3.6: 添加 MS-SSIM 损失

**当前**: 仅使用 MSE 作为重建损失
**改进**: 结合 MS-SSIM 和 L1/MSE:
```python
# losses/deepsc_loss.py
class DeepSCLoss(nn.Module):
    def __init__(self, alpha=0.84):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.alpha = alpha  # MS-SSIM 权重

    def forward(self, x, x_hat, vq_losses):
        # 需要可微的 MS-SSIM 实现
        ms_ssim_loss = 1 - differentiable_ms_ssim(x_hat, x)
        l1_loss = self.l1(x_hat, x)
        recon_loss = self.alpha * ms_ssim_loss + (1 - self.alpha) * l1_loss
        weighted_vq = sum(w * vl for w, vl in zip(self.layer_weights, vq_losses))
        return recon_loss, weighted_vq
```

---

## Phase 4: 长期演进 (P3 — 2-4 周)

### Evolve 4.1: 统一 PyTorch 物理层仿真

**目标**: 消除对 TensorFlow/Sionna 的依赖
**方案**: 使用 PyTorch 原生实现或 PyTorch 兼容的 LDPC 库:
- 选项 A: 用 `torch` 重写 LDPC BP 解码器
- 选项 B: 使用 `pyldpc` (纯 Python) 或 `aff3ct` (C++ 高性能)
- 选项 C: 将 LDPC 部分作为独立服务，通过 ONNX 导出

**收益**: 单一框架依赖，部署简化，GPU 显存无冲突

---

### Evolve 4.2: 支持可变分辨率

**当前**: 固定 256×256 输入 (训练) 和 768×512 (测试)，但语义编码器使用全卷积，理论支持任意尺寸

**改进**: 
- 将模型改为全卷积 (已满足)
- 测试时支持任意分辨率而不改变 BPP 计算
- 添加多分辨率训练增强鲁棒性

```python
# data/datasets.py
if mode == 'train':
    transform = transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        # ...
    ])
```

---

### Evolve 4.3: 渐进式码本大小训练

**目标**: 从大码本逐步缩减到目标大小，类似蒸馏
**方案**:
```python
# Phase1: K=256 (学习丰富的表示)
# Phase2: 逐步剪枝: 256→192→128→64
# Phase3: K=64 (最终部署码本)
```

在剪枝时使用**码字重要性评分** (基于使用频率) 保留最常用的码字。

---

### Evolve 4.4: 添加量化感知的熵编码

**目标**: 在 VQ 索引上应用熵编码进一步压缩
**方案**: 在 VQ 索引序列上训练一个**小型自回归概率模型** (类似 PixelCNN 的简化版)，用于算术编码:
```python
class IndexEntropyModel(nn.Module):
    """预测 VQ 索引的概率分布用于算术编码"""
    def forward(self, indices_list):
        # 使用轻量级 CNN 预测每个空间位置的码字概率
        # 输出: log-probabilities for arithmetic coding
        ...
```

**收益**: 在此项目 1.99 BPP 基础上再降低 10-20% 的比特率

---

### Evolve 4.5: 集成 CI/CD 与实验管理

**目标**: 规范化实验流程
**方案**:
```
├── .github/workflows/
│   └── test.yml            # GitHub Actions: lint + unit test
├── experiments/
│   └── exp001_baseline/
│       ├── config.yaml      # Hydra/OmegaConf 配置
│       └── metrics.json     # 实验结果记录
├── Makefile                 # 常用命令入口
└── docker/
    └── Dockerfile           # 可复现环境
```

---

### Evolve 4.6: TensorRT / ONNX 推理优化

**目标**: 部署时加速推理
**方案**: 将 SimVQ 编解码器导出为 ONNX，用 TensorRT 做推理优化:
```python
# 导出编码器
torch.onnx.export(
    model.semantic_encoder,
    dummy_input,
    "encoder.onnx",
    opset_version=17
)
# 导出解码器
torch.onnx.export(
    model.semantic_decoder,
    dummy_quantized_features,
    "decoder.onnx",
    opset_version=17
)
```

---

## 优先级矩阵

```
影响力
  ▲
  │  Fix 1.2 (路径)      │  Improve 3.2 (码本配置)
  │  Fix 1.1 (日志)      │  Improve 3.1 (信道对齐)
  │                       │
  │  Opt 2.1 (去重)      │  Improve 3.5 (验证SNR)
  │  Opt 2.2 (公共组件)   │  Improve 3.3 (RAQ)
  │                       │
  │  Fix 1.3 (gitignore)  │  Evolve 4.2 (可变分辨率)
  │  Fix 1.4 (reqs.txt)   │  Evolve 4.4 (熵编码)
  └──────────────────────┴──────────────────────────▶ 实现难度
      低                                高
```

---

## 建议的推进顺序

| 序号 | 条目 | Phase | 预计工作量 | 收益 |
|------|------|-------|-----------|------|
| 1 | Fix 1.1: 修复日志日期格式 | P0 | 5 min | 🟢 修复路径 bug |
| 2 | Fix 1.3: 添加 .gitignore | P0 | 10 min | 🟢 工程规范 |
| 3 | Fix 1.4: 添加 requirements.txt | P0 | 10 min | 🟢 复现保障 |
| 4 | Fix 1.2: 去硬编码路径 | P0 | 30 min | 🟡 可移植性 |
| 5 | Opt 2.1: 合并 forward 方法 | P1 | 1 h | 🟡 维护性 |
| 6 | Opt 2.2: 提取公共组件 | P1 | 2 h | 🟡 代码质量 |
| 7 | Opt 2.4: 消除 evaluate 重复 | P1 | 20 min | 🟡 一致性 |
| 8 | Improve 3.5: 固定 SNR 验证 | P2 | 2 h | 🟠 可复现性 |
| 9 | Improve 3.1: 真实链路验证 | P2 | 4 h | 🔴 模型泛化 |
| 10 | Improve 3.2: 码本配置搜索 | P2 | 1 day | 🟠 性能提升 |
| 11 | Improve 3.6: MS-SSIM 损失 | P2 | 1 day | 🟠 视觉质量 |
| 12 | Improve 3.3: RAQ 速率自适应 | P2 | 3 days | 🟠 实用价值 |
| 13 | Evolve 4.1: 统一框架 | P3 | 1 week | 🔴 部署简化 |
| 14 | Evolve 4.4: 熵编码 | P3 | 1 week | 🟡 压缩率 |
| 15 | Evolve 4.5: CI/CD | P3 | 3 days | 🟢 工程质量 |
