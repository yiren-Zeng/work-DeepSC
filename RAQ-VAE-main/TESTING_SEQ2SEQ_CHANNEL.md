# Seq2Seq 两层模型：可配置链路测试

入口：`bash scripts/test_seq2seq_channel.sh`。脚本调用 `eval_seq2seq_channel.py`，不会启动、停止或修改训练。
所有输出和缓存都留在本项目；只读复用 `shiyan` 的数据加载、信道编码、调制、信道和指标实现。

## 指定码本、LDPC 码率、调制和 SNR

推荐直接编辑 `scripts/test_seq2seq_channel.sh` 末尾这四行：

```bash
  --target-k 4 \
  --ldpc-rate 1/2 \
  --modulation 16QAM \
  --snr 10 \
```

保存后运行即可，无需再输入参数：

```bash
cd /workspace/yi/work/RAQ-VAE-main
bash scripts/test_seq2seq_channel.sh
```

Shell 入口默认启用精简输出：运行期间不打印逐图指标，结束时只显示本次有信道测试的配置、
PSNR、MS-SSIM、实际传输比和结果路径，不显示无信道对照指标或整段 JSON。
`results.json` 中仍保留无信道对照和逐图信息，供复核使用，不影响终端显示的本次测试结果。

当前内置配置为 K=4、1/2 LDPC、16QAM、10 dB。也仍可用命令行临时覆盖，例如：

```bash
cd /workspace/yi/work/RAQ-VAE-main
bash scripts/test_seq2seq_channel.sh \
  --target-k 4 --ldpc-rate 1/2 --modulation 16QAM --snr 10
```

默认读取训练目录 `runs/seq2seq_src64_20260909_173605_993462/status.json` 当时记录的最佳权重，
复制到本次测试目录后再加载。训练继续更新最佳权重不会影响已经开始的测试。
输出默认是 `runs/eval_seq2seq_k..._时间戳/`，终端打印实际路径。也可以指定 `--output runs/自定义名称`；
目录必须尚不存在，避免覆盖历史结果。相对路径按本项目根目录解释。

| 参数 | 可用值 / 说明 |
|---|---|
| `--target-k` | 2、4、8、16、32、64；固定目标码本大小，上下两层共用 |
| `--ldpc-rate` | `1/2` 或 `0.5`；`3/4` 或 `0.75`；LDPC 总码长固定 256 |
| `--modulation` | `QPSK`、`16QAM`，大小写均可 |
| `--snr` | AWGN SNR，单位 dB，可输入有限小数，沿用原信道的 SNR 定义 |
| `--checkpoint` | 本项目内的可信 `.ckpt`；省略则读取当前记录的最佳权重 |
| `--run-dir` | 模型训练配置目录，默认当前 Cars196 训练目录 |
| `--seed` | 默认 42 |
| `--output` | 本项目内尚不存在的输出目录，省略则自动生成 |
| `--dry-run` | 检查配置并打印传输比计划，不执行推理、不创建测试输出目录 |

基础码本仍为训练时的 64，不是通过 `--target-k` 修改基础码本。目标码本由原 Seq2Seq 生成。
固定 K 模式没有人为预算补位，只有 LDPC/调制所需填充；实际传输比由 K 和链路参数决定，
不会强制等于 1/48。非 2 次幂 K 被拒绝，因为参考定长打包器使用 `int(log2(K))`，直接使用会截断索引。

## 重现第 114 轮的三组测试

以下冻结副本不受训练 top-5 检查点轮换影响：

```bash
cd /workspace/yi/work/RAQ-VAE-main
checkpoint_114="runs/eval_seq2seq_epoch114_snr3_6_10_20260909_222326/epoch114_frozen.ckpt"
bash scripts/test_seq2seq_channel.sh --checkpoint "$checkpoint_114" \
  --target-k 2 --ldpc-rate 1/2 --modulation QPSK --snr 3
bash scripts/test_seq2seq_channel.sh --checkpoint "$checkpoint_114" \
  --target-k 2 --ldpc-rate 3/4 --modulation QPSK --snr 6
bash scripts/test_seq2seq_channel.sh --checkpoint "$checkpoint_114" \
  --target-k 4 --ldpc-rate 1/2 --modulation 16QAM --snr 10
```

三组实际传输比分别为 1/38.4、1/56.8889、1/38.4。

## 自动选择最接近 1/48 的 K

自动模式请直接调用 Python 入口，不使用内置固定 K 的 Shell 脚本：

```bash
CUDA_VISIBLE_DEVICES=1 /home/yi/.conda/envs/work/bin/python -B eval_seq2seq_channel.py \
  --run-dir runs/seq2seq_src64_20260909_173605_993462 \
  --rate-policy nearest --ratio-denominator 48 \
  --ldpc-rate 3/4 --modulation QPSK --snr 6
```

自动模式和固定 K 不可同时使用。`nearest` 按实际传输比与目标比的绝对差选 K，允许高于或低于预算。
传输比定义为复数信道符号数 / (3×H×W)，包括 LDPC 和调制必需填充。
为兼容旧入口，既不传 `--target-k` 也不传 `--rate-policy` 时仍为旧的 `padded` 模式：
选择预算内最大可用 K，并明确补零凑目标预算。常规使用建议显式传 K 或 `nearest`。

## 数据、指标、GPU 和产物

- 数据集仍为 `/workspace/yi/work/Kodak-256-transform-resize` 的 24 张 256×256 RGB 图片。
- 直接调用 `shiyan.data.datasets.get_dataloader`：test、batch=1、shuffle=False、workers=8、pin_memory=True。
- `SIMVQ_TEST_NO_RESIZE=1`，不重新缩放；归一化到 [-1,1]，上下两层索引合并传输。
- PSNR、MS-SSIM 直接复用原 `_image_quality`，逐图计算后算术平均；PNG 可视化裁剪不影响指标。
- 脚本默认物理 GPU1。可用 `GPU_ID=2 bash scripts/test_seq2seq_channel.sh ...` 更改。
  同一权重跨不同型号 GPU 可能出现微小浮点差异；严格比较时建议使用同一张 GPU。
- `results.json`：完整指标、逐图结果、有效载荷 BER、实际传输比。
- `per_image.csv`：逐图结果；`metadata.json`：配置、权重及数据哈希。
- `checkpoint_snapshot.ckpt`、`target_codebook_k*.pt`：本次冻结权重及生成码本。
- `reference/`、`no_channel/`、`channel_*db/`：原图、无信道重建和信道重建。

测试脚本仍使用现有环境 `/home/yi/.conda/envs/work/bin/python` 和本项目 `.runtime/deps`，
没有安装或修改外部项目。可通过 `RAQ_PYTHON` 覆盖解释器路径，但依赖需自行具备。

## 参数和码率回归检查

```bash
/home/yi/.conda/envs/work/bin/python -B tests/test_eval_rate_plan.py
bash scripts/test_seq2seq_channel.sh --target-k 2 --ldpc-rate 1/2 --modulation QPSK --snr 3 --dry-run
```
