# RAQVAE_TWO_DS8 训练与测试

本方案将原 `RAQVAE_TWO` 的 bottom/top 空间尺度从 1/4、1/8 改为 1/8、1/16。
原 `nets/model.py`、`nets/enc_dec.py` 及原训练入口不变；新模型在 `nets/model_ds8.py` 中继承
`RAQVAE_TWO`，只替换 bottom 下采样器和最终上采样器。

## 当前正式训练

- 目录：`runs/seq2seq_ds8_src64_20260910_005420_629332`
- 物理 GPU：4（NVIDIA GeForce RTX 4090 D）
- PID：762241
- 从头训练 200 轮，seed=42，FP32。
- Cars196：训练 12948 张、验证 3237 张；数据加载及增强与原方案一致。
- batch=24，验证 batch=64，workers=8。
- AdamW，学习率 5e-4，默认 weight decay。
- 基础 K=64；目标 K 每训练批均匀采样整数 2..64；验证 K 为 2、4、8、16、32、64。
- top/bottom 共用基础量化器和同一个 Seq2Seq 生成目标码本。
- 继承原重建损失、量化损失、STE、EMA 更新和优化器行为，不加入信道训练或额外损失。
- 每两轮验证，按 `val_recon_loss = MSE_src + MSE_trg` 保留 top-5 和 `last.ckpt`。

## 空间结构

对于 256×256 输入：

| 分支 | 原方案 | DS8 新方案 |
|---|---:|---:|
| bottom | 64×64（1/4） | 32×32（1/8） |
| top | 32×32（1/8） | 16×16（1/16） |
| 总索引数 | 5120 | 1280 |

bottom 使用 `Conv2d(kernel=16,stride=8,padding=4)`；top 的 2 倍下采样和 top-to-bottom 的
2 倍上采样不变；最终使用 `ConvTranspose2d(kernel=16,stride=8,padding=4)` 恢复 256×256。

## 查看状态

```bash
cd /workspace/yi/work/RAQ-VAE-main
sed -n '1,100p' runs/seq2seq_ds8_src64_20260910_005420_629332/status.json
tail -30 runs/seq2seq_ds8_src64_20260910_005420_629332/train.log
```

## 测试

编辑 `scripts/test_seq2seq_ds8_channel.sh` 末尾的 K、LDPC、调制和 SNR，随后运行：

```bash
bash scripts/test_seq2seq_ds8_channel.sh
```

测试器默认读取该训练目录当时记录的最佳权重，使用 Kodak 24 张图片和现有参考信道/指标实现。
新测试器按 1280 个索引计算载荷和实际传输比。例如 K=4、1/2 LDPC、16QAM 为 1/153.6，
不是旧尺度下的 1/38.4。终端仅显示当前有信道测试结果，完整可核验信息保存在结果目录。

## 已完成检查

- CPU 单元测试验证继承关系、32×32 / 16×16 形状、256×256 解码及严格 state dict 重载。
- GPU4 smoke test 使用真实 Cars196 数据完成 4 个训练批与 2 个验证批。
- encoder_b、encoder_t、cbk_enc、cbk_dec、decoder 的首步梯度均有限且非零。
- smoke 检查点对 K=2、4、64 均可严格加载并推理。
- 原方案记录的模型和数据加载源文件哈希保持不变。
