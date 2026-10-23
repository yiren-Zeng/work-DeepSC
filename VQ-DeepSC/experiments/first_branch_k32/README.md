# 单支路 K=32 方案

本方案只保留原四尺度模型的尺度 0：训练输入 `256×256×3`，测试保留原始 `H×W×3`；
经过原编码器的初始卷积及第一个下采样块，得到 `H/2×W/2×128` 特征；用一个 `32×128`
码本量化。解码器复用原模块的一尺度配置，保留初始化卷积、一个残差上采样块
（nearest）和输出卷积，直接恢复原来的 `H×W×3`。不创建其他尺度或跳跃连接。

## 文件与兼容性

| 文件 | 用途 |
| --- | --- |
| [config_first_branch.py](../../config_first_branch.py) | 新增：单尺度、128 通道、K=32、200 轮，以及独立的权重和日志目录 |
| [train_first_branch.py](../../train_first_branch.py) | 新增：调用原训练循环并显式传入单支路配置 |
| [test_real_first_branch.py](../../test_real_first_branch.py) | 新增：调用原测试流程，使用原始尺寸 Kodak，默认 BPSK、LDPC R=1/2 |
| [evaluation/first_branch_images.py](../../evaluation/first_branch_images.py) | 单支路专用：复用原评估器，在同一次重建中导出 PNG |
| [scripts/train/train_k32_first_branch.sh](../../scripts/train/train_k32_first_branch.sh) | 新增：训练启动脚本 |
| [scripts/test/test_k32_first_branch.sh](../../scripts/test/test_k32_first_branch.sh) | 新增：测试启动脚本 |
| [tests/test_first_branch.py](../../tests/test_first_branch.py) | 新增：训练梯度、索引重建、权重读写、旧配置和真实 LDPC 计数验证 |
| [train.py](../../train.py) | 修改：`main` 增加可选 `cfg`；不传参时仍使用原 `Config` |
| [test_real.py](../../test_real.py) | 修改：`test_real` 末尾增加可选 `cfg`；原调用方式和默认参数保留 |
| [experiments/first_branch_k32/README.md](README.md) | 新增：方案、运行方式、变更清单及验证记录 |

编码器、解码器、量化器、损失、通信模块及原有 shell 脚本均不修改。
优化器仍为 Adam，学习率 `1.75e-5`，betas `(0.5, 0.999)`，batch size 24，
新方案按用户要求默认 200 轮，StepLR 每 100 轮乘 0.5，commitment 系数 0.25。
训练、验证继续使用 Cars196；本方案单独使用 `/workspace/yi/work/Kodak` 的 24 张
原始分辨率图片测试：18 张 768×512、6 张 512×768。读取后只转 RGB、ToTensor 和归一化，
不裁剪、不 resize、不切块。其他方案仍使用原配置及其 256×256 数据集。

## 运行

```bash
cd /workspace/yi/work/VQ-DeepSC
bash scripts/train/train_k32_first_branch.sh
bash scripts/test/test_k32_first_branch.sh
```

默认物理 GPU 为 2，进程内对应 `cuda:0`，可用 `GPU_ID=其他编号` 选择。
训练从新方案自己的 `last_checkpoint.pth` 恢复；首次训练从头初始化，
`VQ_DEEPSC_RESUME=0` 可禁用恢复。不加载原四尺度或最后两层方案的权重。

- 最优权重：`checkpoints/k32_first_branch/best_vq_deepsc.pth`
- 恢复权重：`checkpoints/k32_first_branch/last_checkpoint.pth`
- TensorBoard：`logs/k32_first_branch/`
- 训练终端日志：`experiments/logs/train_k32_first_branch.log`
- 测试指标：仅输出到终端，不保存 JSON 或运行日志
- 重建图像：每次测试自动保存 PNG，默认目录为 `generated_images/k32_first_branch/bpsk_ldpc12_snr10/`
- 本方案测试数据：`/workspace/yi/work/Kodak`，路径也在 shell 脚本中显式设置

测试继续支持 `CHECKPOINT`、`SNRS`、`NO_CHANNEL=1` 及调制、码率等附加命令行参数。
`IMAGE_OUTPUT_DIR` 或 `--image-output-dir` 可指定本方案重建图像的根目录。
例如 `SNRS="0 3 6 9" bash scripts/test/test_k32_first_branch.sh`。
终端只显示码本、调制、LDPC 码率、CBR，以及各 SNR 的 PSNR 和 MS-SSIM。
`JSON_OUTPUT` 不再使用，也不接受 `--json-output`。框架初始化和逐层诊断输出被隐藏；
运行异常仍保留 traceback 和非零退出码。其他方案的测试入口和输出方式不变。

### 每次测试自动导出重建图（2026-10-23）

用户确认每次测试自动保存本次信道后的重建图，按参数分目录，并覆盖相同配置的同名文件。
默认生成位置为：

```text
/workspace/yi/work/VQ-DeepSC/generated_images/k32_first_branch/bpsk_ldpc12_snr10/
  kodim01.png
  ...
  kodim24.png
```

图像保持原始横/竖尺寸和原文件名，保存为 RGB PNG。不同 SNR、调制和 LDPC 码率
使用不同目录；非默认 LDPC 码长额外包含 `_n码长`，无信道模式使用 `no_channel/`。
不删除其他目录或文件。PNG 将模型输出反归一化、限制数值范围并四舍五入到 8 bit，
不改变图像几何尺寸。质量指标仍使用原来的浮点重建张量。

本次新增 `evaluation/first_branch_images.py`，修改 `test_real_first_branch.py`、
`scripts/test/test_k32_first_branch.sh`、`tests/test_first_branch.py` 和本文。
单支路独立模块复用既有模型加载和通信评估函数，在 `reconstruct_from_indices`
返回本次重建时直接保存；不重新采样信道，也不再次执行重建。
共享 `test_real.py`、`evaluation/quality.py`、模型、数据加载器及其他方案均未修改。

18 项单元和回归测试全部通过，涵盖文件名顺序、RGB 数值、原尺寸、同名覆盖、
不同参数目录隔离、无信道导出和输出恢复。GPU 4 上实际生成 24 张图片的命令：

```bash
TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 GPU_ID=4 \
bash scripts/test/test_k32_first_branch.sh
```

终端输出保持简洁，实际运行结果为：

```text
K=32 | BPSK | LDPC=1/2 | CBR=5/6 (0.833333)
SNR=10 dB | PSNR=26.8636 dB | MS-SSIM=0.9322
```

核对结果：`kodim01.png` 至 `kodim24.png` 全部保存，18 张 768×512、6 张 512×768，
均为有效 RGB PNG；总大小 14,295,447 字节。显示的 PSNR/MS-SSIM 与导出前一致。
运行前后核对项目内文件，评测只新增这 24 张 PNG，没有生成 JSON 或日志文件。
对按文件名排序的 `文件名 + 空格 + 文件 SHA-256 + 换行` 清单计算 SHA-256，结果为
`3c18178f7afed9ed25bd55f99d3ce7690598312b160f00eb6a7b85bfcf1b95e6`。

### 终端输出调整（2026-10-23，历史记录）

本次仅修改 `scripts/test/test_k32_first_branch.sh`、`test_real_first_branch.py`、
`tests/test_first_branch.py` 和本文。脚本取消结果目录创建和 JSON 参数；独立测试入口
只显示摘要，隐藏 Python 及框架原生日志，退出时恢复输出以保留异常信息。
模型、通信计算和其他方案的代码均未修改。

改用原始分辨率之前，GPU 4 上用完整训练后的最优权重跑完 24 张 256×256 Kodak，终端实际只输出：

```text
K=32 | BPSK | LDPC=1/2 | CBR=5/6 (0.833333)
SNR=6 dB | PSNR=27.4192 dB | MS-SSIM=0.9343
```

验证命令为 `GPU_ID=4 bash scripts/test/test_k32_first_branch.sh`，检查时额外设置
`TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`。
运行前后比较项目内所有文件的清单、大小和修改时间，确认没有新增或改写文件，
stderr 为空。另用不存在的 `--checkpoint` 检查错误路径，仍显示 `FileNotFoundError`
并返回非零退出码，也没有生成文件。

`python -B -m unittest discover -s tests -v` 的 14 项测试全部通过，包含新增加的
摘要输出、禁止保存、隐藏原生日志和异常后输出恢复检查；未保存本次测试日志或结果文件。
此前的小样本 JSON、日志和权重是历史审阅产物，未删除或覆盖。

### 原始分辨率测试（2026-10-23，导出图像前的记录）

按用户要求，仅在 `FirstBranchConfig` 中覆盖测试数据路径，并在
`scripts/test/test_k32_first_branch.sh` 中显式设置
`VQ_DEEPSC_TEST_DATASET_PATH=/workspace/yi/work/Kodak`。该环境参数只由本方案读取，
公共 `config.py`、共享 DataLoader、其他方案及其数据路径均保持原样。
训练和验证仍使用 Cars196；已有 200 轮训练权重继续直接用于测试。

本次修改文件只有 `config_first_branch.py`、`scripts/test/test_k32_first_branch.sh`、
`tests/test_first_branch.py` 和本文。保留脚本现有的 10 dB 默认值以及全部其他通信参数。

15 项单元与回归测试全部通过，包括原始横图、竖图不缩放/不裁剪的尺寸检查，
以及旧 `Config.TEST_DATASET_PATH` 仍为 `Kodak-256-transform-resize` 的隔离检查。
GPU 4 上使用单支路测试入口完整评测原始目录中的 24 张图片，实际输出如下：

```text
K=32 | BPSK | LDPC=1/2 | CBR=5/6 (0.833333)
SNR=10 dB | PSNR=26.8636 dB | MS-SSIM=0.9322
```

验证入口调用参数为 `main(["--snrs", "10"])`，环境为
`CUDA_VISIBLE_DEVICES=4 VQ_DEEPSC_DEVICE=cuda:0 VQ_DEEPSC_TEST_DATASET_PATH=/workspace/yi/work/Kodak`；
另设置 `TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2`
和 `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1`。
结果只在内存中用于断言，不生成 JSON、图片或日志文件。

实际通信诊断核对：原始像素总数为 9,437,184，索引总数为 2,359,296，
payload 为 11,796,480 bit，传输 BPSK 符号为 23,592,960，RGB 标量为 28,311,552；
CBR 精确为 5/6，LDPC 和调制补零均为 0。评测前后文件清单、大小和修改时间完全一致。
最终按修改前的 SHA-256 快照确认，其他已存在文件未被本次操作改动；Git 未跟踪文件清单也未增加。

## CBR

按项目现有口径，CBR 是传输符号数除以输入 RGB 标量个数 `3HW`。

| 项目 | 每张图 |
| --- | ---: |
| 输入原始尺寸 | 768×512 或 512×768 |
| 索引个数 | 384×256 = 98,304 |
| 每索引位数 | log2(32) = 5 |
| payload | 491,520 bit，即 1.25 bpp |
| LDPC | k=128，n=256，3,840 个码块 |
| LDPC 补零 / 调制补零 | 0 / 0 bit |
| 编码后 bit / BPSK 符号 | 983,040 / 983,040 |
| RGB 标量个数 | 768×512×3 = 1,179,648 |
| CBR | **983,040 / 1,179,648 = 5/6 = 0.8333333333333334** |

这里统计的是现有仿真链路，不包含额外的帧头或导频。

## 验证记录

2026-10-23 完成以下检查：

- Python 语法解析、两个新 shell 脚本的 `bash -n`、`git diff --check` 均通过。
- `unittest` 共 12 项全部通过：原有 8 项通信回归测试，以及新增 4 项单支路测试。
- 原训练循环用一张 Cars196 训练图和一张验证图完成 1 轮，执行反向传播并保存最优、恢复权重。
  数据处理、优化器、学习率、损失与 batch size 配置均沿用原值；仅检查阶段使用 1 轮和 0 个 DataLoader worker。
- 新 shell 测试脚本加载这份检查权重，在 6 dB、LDPC `(128,256)`、BPSK 下跑完 24 张 Kodak。
  实际 payload 为 1,966,080 bit，传输符号为 3,932,160，输入 RGB 标量为 4,718,592，
  CBR 为 **5/6**，LDPC 与调制补零均为 0，接收 payload 的 BER 为 0。
- 新配置和训练 shell 脚本的默认轮数均为 200。实现阶段只进行上述小样本检查；
  检查权重只训练过一张图，生成的 PSNR/MS-SSIM 仅用于验证流程，不代表训练完成后的性能。
  后续正式训练见 [GPU 4、200 轮启动记录](../train/k32_first_branch_200ep_gpu4_20261023/README.md)。
- 与修改前的 SHA-256 快照比较，原有文件中仅 `train.py` 和 `test_real.py` 有本次修改。
  原有 7 个已修改文件、123 个未跟踪文件不归入本次变更，也未暂存或提交。

复现检查的命令如下（均在项目根目录执行）：

```bash
set -o pipefail
TF_CPP_MIN_LOG_LEVEL=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=2 \
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests -v \
  2>&1 | tee experiments/first_branch_k32/verification.log
```

生成小样本检查权重的完整命令：

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
PYTHONDONTWRITEBYTECODE=1 python -B - <<'PY' 2>&1 | tee experiments/first_branch_k32/training_smoke.log
from pathlib import Path
from tempfile import TemporaryDirectory

from config_first_branch import FirstBranchConfig
from data.datasets import ImageDataset
from train_first_branch import main

cfg = FirstBranchConfig()
cfg.DEVICE = 'cuda:0'
cfg.NUM_EPOCHS = 1
cfg.NUM_WORKERS = 0
cfg.RESUME = False
cfg.CHECKPOINT_DIR = 'experiments/first_branch_k32/smoke_checkpoints'
cfg.RESUME_PATH = str(Path(cfg.CHECKPOINT_DIR) / 'last_checkpoint.pth')
cfg.LOG_DIR = 'experiments/first_branch_k32/smoke_tensorboard'
with TemporaryDirectory(prefix='vq_first_branch_data_') as temporary:
    for field, split in [('TRAIN_DATASET_PATH', 'train'), ('VAL_DATASET_PATH', 'val')]:
        source = Path(getattr(cfg, field))
        filename = sorted(ImageDataset(str(source)).image_files)[0]
        target = Path(temporary) / split
        target.mkdir()
        (target / filename).symlink_to(source / filename)
        setattr(cfg, field, str(target))
        print(f'Smoke {split} source: {source / filename}', flush=True)
    main(cfg=cfg)
PY
```

下列命令是改为纯终端输出前的历史检查命令，对应已留存的小样本 JSON 和日志；
当前脚本不再保存这些文件：

```bash
set -o pipefail
TF_CPP_MIN_LOG_LEVEL=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 GPU_ID=2 \
CHECKPOINT=experiments/first_branch_k32/smoke_checkpoints/best_vq_deepsc.pth \
JSON_OUTPUT=experiments/first_branch_k32/ldpc_bpsk_smoke.json \
bash scripts/test/test_k32_first_branch.sh \
  2>&1 | tee experiments/first_branch_k32/evaluation_smoke.log
```

生成产物清单：

| 文件 | 内容 |
| --- | --- |
| [verification.log](verification.log) | 12 项测试及真实 LDPC 符号计数的运行日志 |
| [training_smoke.log](training_smoke.log) | 小样本训练、验证及保存权重的日志 |
| [evaluation_smoke.log](evaluation_smoke.log) | 新测试脚本在 24 张 Kodak 上的日志 |
| [ldpc_bpsk_smoke.json](ldpc_bpsk_smoke.json) | 自动生成：测试结果及完整通信诊断 |
| [verification_summary.json](verification_summary.json) | 自动生成：断言检查、CBR、原文件变更清单及产物大小/SHA-256 |
| [smoke_checkpoints/best_vq_deepsc.pth](smoke_checkpoints/best_vq_deepsc.pth) | 单图单轮检查的最优模型 |
| [smoke_checkpoints/last_checkpoint.pth](smoke_checkpoints/last_checkpoint.pth) | 单图单轮检查的恢复权重 |
| `smoke_tensorboard/` | 单轮训练产生的 TensorBoard 事件；具体文件及哈希见 summary |

日志、权重和 TensorBoard 文件按已有 Git 忽略规则保留在磁盘，未修改忽略规则。
两个 JSON 及本文列出的新增源码都计入新增未跟踪文件，交付时通过文件链接审阅。
