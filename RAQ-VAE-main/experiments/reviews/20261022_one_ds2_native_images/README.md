# 单尺度 ds2 原始 Kodak 评测修改记录

## 已确认的行为

- `scripts/test_seq2seq_one_ds2_src256_k32_2048_channel.sh` 默认读取 `/workspace/yi/work/Kodak`，不缩放、不裁剪。
- 输入包括 18 张 768×512 和 6 张 512×768 图片，共 24 张。
- 每次脚本运行创建独立目录，多个目标 K 共用该次运行目录，分别按 K 和测试条件保存最终重建图。
- 保存位置为 `generated_images/<北京时间与唯一标识>/k<K>/<条件>/<原图名>.png`。
- 脚本运行仅写最终重建 PNG，不复制原图，不保存无关重建版本、权重快照、目标码本、JSON、CSV、进度或日志文件。
- 每个 K 仅输出三行：配置、指标、图片目录。依赖库启动信息在内存中捕获；失败时恢复并显示诊断与异常。
- 指标为 PSNR、四位小数 MS-SSIM 及据该显示值计算的 dB、实际整数计数约分得到的 CBR 及小数。无信道时 CBR 为 N/A。

## 使用

从任何工作目录调用：

```bash
bash /workspace/yi/work/RAQ-VAE-main/scripts/test_seq2seq_one_ds2_src256_k32_2048_channel.sh
```

原有默认目标 K=32、无信道测试不变。需要真实信道时：

```bash
NO_CHANNEL=0 CHANNEL=awgn SNR=10 \
  bash /workspace/yi/work/RAQ-VAE-main/scripts/test_seq2seq_one_ds2_src256_k32_2048_channel.sh
```

`TARGETS="32 64"` 可在同次运行测试多个 K；`DATASET` 可指定其他 24 张、像素面积相同且高宽为偶数的图片目录；`OUTPUT_ROOT` 可在项目内指定其他图片根目录。

脚本会传入 `--images-only --concise`；直接调用 Python 并省略 `--images-only` 时，保留原有完整报告模式。未修改模型结构、权重、量化方法、调制、LLR 或 LDPC 算法。

## 本次文件清单

| 文件 | 本次操作 | 用途 |
| --- | --- | --- |
| `scripts/test_seq2seq_one_ds2_src256_k32_2048_channel.sh` | 修改已有文件 | 原图目录、图片输出根目录、每次运行独立路径、仅图片模式。 |
| `eval_seq2seq_one_ds2_src256_k32_2048_channel.py` | 修改已有文件 | 原始尺寸校验与码率规划、动态像素分母、仅写最终 PNG、三行汇总。 |
| `tests/test_one_ds2_native_images.py` | 新增 | 原始尺寸、舍入、CBR、仅图片写入及最终信道结果的回归测试。 |
| `experiments/reviews/20261022_one_ds2_native_images/README.md` | 新增 | 此次使用和审阅记录。 |

前两个文件开始时已含用户既有改动。`before.json` 保存本次操作前的完整内容、SHA-256 和项目 Git 状态。`changes.patch` 基于该快照生成，仅包含本次修改及上述两个新增文本文件；不把已有的共享评测器、训练文档等改动归入本次变更。未暂存、提交或推送。

## 验证

工作目录为 `/workspace/yi/work/RAQ-VAE-main`。

```bash
bash -n scripts/test_seq2seq_one_ds2_src256_k32_2048_channel.sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3 \
  python -m unittest discover -s tests -p 'test_one_ds2*.py' -v
```

9 项相关检查全部通过，包括原有 4 项模型检查；本次新增检查验证了横竖图的码率规划、四位 MS-SSIM 与 dB 一致、实际 CBR、两种信道模式仅写最终重建 PNG、直接读取原权重且不保存权重或 JSON。

完整真实模型验证命令：

```bash
TARGETS="32 64" NO_CHANNEL=1 GPU_ID=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
  bash scripts/test_seq2seq_one_ds2_src256_k32_2048_channel.sh
```

实际使用自动选出的完整训练权重：

```text
runs/seq2seq_one_ds2_src256_k32_2048_original_20261005_101308_427404/checkpoints/epoch=163-step=88560.ckpt
```

实际图片目录：

```text
generated_images/20261022_203744_494182_1749d603/k32/no_channel/
generated_images/20261022_203744_494182_1749d603/k64/no_channel/
```

| K | 图片数 | PSNR (dB) | MS-SSIM | MS-SSIM (dB) |
| --- | --- | --- | --- | --- |
| 32 | 24 | 26.8361 | 0.9570 | 13.6653 |
| 64 | 24 | 29.1392 | 0.9697 | 15.1856 |

对脚本运行前后整个项目的文件集合进行比较，新增文件恰为以上 48 张 PNG，无其他新增项目文件；终端共 6 行。全部 PNG 通过完整性检查、RGB 模式检查及对应原图尺寸检查。

另执行真实信道单包检查：一张原始尺寸图片对应 98,304 个索引，K=32 时载荷为 491,520 bit；LDPC 1/2、BPSK、AWGN 10 dB 实际发送 983,040 个符号，CBR=5/6，解码误比特为 0。此检查验证真实信道数据包，没有重新运行有信道模式的全部 24 张图像指标。

检查命令为 `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3 python -u -`，标准输入程序如下；输出单独保存在 `native_channel_payload.log`。

```python
import sys
from fractions import Fraction
from pathlib import Path
import eval_seq2seq_one_ds2_src256_k32_2048_channel as entry

with entry.quiet_evaluation():
    import numpy as np
    import torch
    reference = Path('/workspace/yi/work/shiyan')
    _, setup_seed = entry.load_shiyan_loader(reference)
    sys.path.append(str(reference))
    from communications.ldpc_coding import get_ldpc_code, ldpc_encode, ldpc_decode
    from communications.modulation import bpsk_modulate, bpsk_llr
    setup_seed(42)
    geometry = entry.evaluation_geometry([(512, 768), (768, 512)])
    payload = np.random.default_rng(42).integers(
        0, 2, geometry['source_indices'] * 5, dtype=np.uint8
    )
    code = get_ldpc_code(128, rate=0.5)
    def llr(received, snr, device, **ignored):
        return bpsk_llr(received, snr, device)
    decoded, stats = entry.shared.transmit_ldpc_stream_exact(
        payload, 10, code, torch.device('cuda:0'), bpsk_modulate, llr,
        1, ldpc_encode, ldpc_decode, 'awgn', 1, use_actual_noise_variance=False,
    )
assert len(decoded) == len(payload) == 491520
assert stats['channel_symbols'] == 983040
assert Fraction(stats['channel_symbols'], geometry['source_values']) == Fraction(5, 6)
assert np.array_equal(decoded, payload)
print('Native channel payload check passed.')
```

## 审阅与验证产物

本目录中的 `before.json`、`changes.patch`、`unit_tests.log`、`native_no_channel.log`、`native_no_channel_audit.json`、`native_channel_payload.log` 是本次开发验证与审阅记录，由代理单独保存，不是测试脚本自动生成的输出。

完整模型运行由外部验证包装器捕获终端输出，并在运行结束后写入本目录；审计 JSON 保存生成命令、环境参数、退出码、终端行数、脚本新增文件清单、所有 PNG 的尺寸和 SHA-256。图片根目录中只保存 PNG。

本次沿用现有 `eval_seq2seq_ds8_dualk_console.py` 的内存日志捕获函数，未修改该文件。补丁包括新增测试和本文档的完整内容，并使用 `git apply --reverse --check` 验证与当前文件一致。
