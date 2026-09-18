# RAQVAE_TWO_DS8_DUALK：独立双 K 训练与测试

本方案在 1/8、1/16 空间尺度的 DS8 模型上，让 bottom 和 top 分别选择目标码本大小。
基础 EMA 码本仍只有一个且 K=64；Seq2Seq 参数仍完全共享，但每次 forward 始终调用两次，
分别生成 bottom/top 目标码本并分别量化，即使两个 K 相等也不复用生成结果。

## 当前正式训练

- 目录：`runs/seq2seq_ds8_dualk_src64_20260910_113736_956705`
- PID：1521259
- 物理 GPU：4（NVIDIA GeForce RTX 4090 D）
- 随机初始化，从头训练 200 轮；没有加载任何旧权重或优化器状态。
- Cars196：训练 12948 张、验证 3237 张，原数据加载和增强保持不变。
- batch=24，验证 batch=64，workers=8，seed=42，FP32。
- AdamW，学习率 5e-4，默认 weight decay。
- 每两轮验证，按 `val_recon_loss=MSE_src+MSE_trg` 保存 top-5 和 `last.ckpt`。
- 原 source/target 重建 MSE、两分支量化损失、STE、EMA 更新顺序均不改变。

## K 采样

训练和验证时，bottom 与 top 每批各执行一次独立均匀采样：

```python
K_CHOICES = (2, 4, 8, 16, 32, 64)
k_bottom = random.choice(K_CHOICES)
k_top = random.choice(K_CHOICES)
```

没有指定组合、强化组合或混合比例。两次采样共用 seed=42 的 Python RNG，因此可复现。

## 空间尺寸与载荷

256×256 输入产生 bottom 32×32=1024 个索引、top 16×16=256 个索引。固定长度打包载荷为：

```text
1024 × log2(K_bottom) + 256 × log2(K_top) bit
```

以下组合在指定链路中精确产生 3072 个复数信道符号，即传输比 1/64：

| LDPC | 调制 | K_bottom | K_top |
|---|---|---:|---:|
| 1/2 | QPSK | 4 | 16 |
| 3/4 | QPSK | 16 | 4 |
| 3/4 | QPSK | 8 | 64 |
| 1/2 | 16QAM | 32 | 16 |

## 查看训练

```bash
cd /workspace/yi/work/RAQ-VAE-main
sed -n '1,100p' runs/seq2seq_ds8_dualk_src64_20260910_113736_956705/status.json
tail -30 runs/seq2seq_ds8_dualk_src64_20260910_113736_956705/train.log
```

## 测试

编辑 `scripts/test_seq2seq_ds8_dualk_channel.sh` 中的五项配置：

```bash
  --bottom-k 4 \
  --top-k 16 \
  --ldpc-rate 1/2 \
  --modulation QPSK \
  --snr 3
```

运行：

```bash
bash scripts/test_seq2seq_ds8_dualk_channel.sh
```

脚本默认读取本次训练当时记录的最佳权重，并默认使用物理 GPU3，以免占用训练 GPU4。
测试集、加载方式、LDPC、调制、AWGN、PSNR 和 MS-SSIM 与此前方案保持一致。
终端只显示当前有信道测试结果；完整配置、哈希、逐图结果和重建图片保存在 `runs/`。

## 隔离与验证

新模型、训练和测试均使用独立文件及运行目录，不覆盖单 K DS8 或最初的 1/4、1/8 方案。
单元测试覆盖独立随机采样、2 的幂约束、相同 K 时两次生成、形状、严格权重重载和 1/64 码率。
GPU4 smoke test已使用真实 Cars196 数据完成训练、验证和检查点保存。
