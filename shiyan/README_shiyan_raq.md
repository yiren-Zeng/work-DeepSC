# shiyan: quality_v2_B_larger_rate044_A_patch_cb16-2_ch512-1024 + RAQ

这个目录是一个新的独立实验工程，位置是：

```bash
/workspace/yi/work/shiyan
```

它以 `/workspace/yi/work/Simvq-dc-64-Multi-pro-2ceng` 中的
`quality_v2_B_larger_rate044_A_patch_cb16-2_ch512-1024` 实验为主骨架，并加入
`/workspace/yi/work/vq-dc-raqmae-64-transformer-LR-xiugai-3` 中的 RAQ 动态目标码本思想。

## 一句话说明

原 quality 实验是一个两层 U-Net + patch-wise SimVQ 图像语义通信模型。现在的 `shiyan`
版本保留原来的 Stage B 主干、下采样、特征通道、码率和信道训练课程，同时新增一条 RAQ 支路：

1. source 支路继续使用原 SimVQ 源码本量化。
2. RAQ 支路用 Transformer 根据源码本动态生成目标码本 `W_trg`。
3. 训练时 source/RAQ 两路都重建图像并共同优化。
4. 测试时默认发送 RAQ 目标码本索引。
5. 默认不加载任何预训练权重，从零开始训练。

## 实验脚本配置

RAQ 相关参数不再在 `config.py` 里隐藏默认值。直接导入配置且不传
`SIMVQ_USE_RAQ` 时，RAQ 默认为关闭；如果设置 `SIMVQ_USE_RAQ=1`，必须显式传入
`SIMVQ_RAQ_TARGET_LIST`、`SIMVQ_RAQ_MIN_TRG`、`SIMVQ_RAQ_MAX_TRG`、
`SIMVQ_RAQ_REPULSION_WEIGHT`。当前固定实验入口会在 shell 脚本中写明这些值：

| 项目 | 脚本显式值 |
| --- | --- |
| 实验名 | `shiyan_raq_quality_v2_B_larger_rate044_A_patch_cb16-2_ch512-1024_unet2_ds8x2_k16-2` |
| Stage | `B` |
| U-Net 层数 | `2` |
| 下采样步幅 | `[8, 2]` |
| 总下采样倍率 | `16x` |
| base channels | `256` |
| 特征通道 | `[512, 1024]` |
| 源码本大小 | `[16, 2]` |
| 量化方式 | patch-wise SimVQ |
| RAQ | `SIMVQ_USE_RAQ=1` |
| RAQ 训练目标 K | 从 `[2, 4, 8, 16]` 中按层随机采样 |
| RAQ 测试目标 K | `[16, 2]` |
| RAQ codebook repulsion | `0.05` |
| 估计源端 bpp | `0.06640625` |
| LDPC1/2+BPSK 传输比 | `0.04427083` |
| 预训练 | 默认禁用 |
| 断点续训 | 默认禁用，设置 `SIMVQ_RESUME=1` 才启用 |

## 代码结构

```text
shiyan/
  config.py                         实验配置；RAQ 参数必须由脚本/环境变量显式传入
  train.py                          训练入口，从零训练，source/RAQ 双支路 loss
  test_real.py                      真实链路/无信道评估入口
  models/
    deepsc.py                       主模型，quality 主干 + RAQ 支路
    vector_quantizer.py             SimVQ/VQ 量化器，新增 forward_raq 外部码本量化
    raq.py                          RAQ 目标码本生成器
    transformer.py                  TransformerCodebookGen
    semantic_encoder.py             原 quality 语义编码器
    semantic_decoder.py             原 quality 语义解码器
    channel.py                      有限码长信道误码模拟
  losses/
    deepsc_loss.py                  source/RAQ 兼容损失
  evaluation/
    quality.py                      no-channel、LDPC+BPSK/QPSK 评估
  monitoring/
    codebook.py                     source 和 RAQ 码本利用率监控
  scripts/
    train/current/run_exp12_rate044_A_patch_cb16_2_ch512_1024.sh
                                      从零训练脚本
    eval/test_A_patch_rate044_ch512_1024.sh
                                      默认评估脚本
  experiments/                      训练日志、metrics、tensorboard 输出目录
  checkpoints/                      checkpoint 输出目录
```

## RAQ 是如何接入的

### 1. 动态目标码本生成

`models/raq.py` 中的 `RAQ` 使用 `TransformerCodebookGen`：

```text
源码本权重 W_src -> Transformer encoder
目标码字 id embeddings -> Transformer decoder
输出动态目标码本 W_trg
```

每一层都有一个独立 RAQ 生成器。默认两层分别对应特征维度 `512` 和 `1024`。

### 2. 外部码本量化

`models/vector_quantizer.py` 新增：

```python
VectorQuantizer.forward_raq(inputs, embed_weight)
```

它不使用自身源码本查表，而是使用 RAQ 生成的 `W_trg` 做最近邻量化。原来的
`VectorQuantizer.forward()` 没有改，source 支路仍然按原 SimVQ 工作。

### 3. DeepSC 双支路

`models/deepsc.py` 中：

```text
输入图像
  -> SemanticEncoder
  -> source SimVQ 量化 + 可选信道误码
  -> source reconstruction

同一组 encoder features
  -> RAQ 生成 W_trg
  -> RAQ 量化 + 可选信道误码
  -> RAQ reconstruction
```

训练时 `forward_train()` 返回：

```python
reconstructed_images_src
vq_losses_src
reconstructed_images_raq
vq_losses_raq
W_trg_list
raq_target_list
```

测试时 `forward_test()` 默认返回 RAQ 索引、RAQ 目标码本和实际传输用的 K 列表。

### 4. 信道模拟

训练和验证仍保留原 quality 工程的有限码长误码模拟：

```text
随机 SNR in [0, 15] dB
按 SNR 采样调制阶数 bits/symbol
由有限码长公式估计 BER
在离散索引比特上翻转
```

训练前期仍使用原来的信道课程：

```text
epoch < 80: channel_prob = 0
epoch 80-120: channel_prob 线性升到 1
epoch >= 120: channel_prob = 1
```

### 5. 损失函数

`losses/deepsc_loss.py` 现在支持单支路和 RAQ 双支路。RAQ 开启时：

```text
loss = source_recon
     + RAQ_recon
     + weighted_source_vq
     + weighted_RAQ_vq
     + normalized_RAQ_codebook_repulsion
```

RAQ 排斥项用于鼓励目标码本码字分散。因为本实验特征维度较宽 `[512,1024]`，
排斥项按 embedding 维度做了归一化，避免它压过重建和 VQ 主损失。

## 如何训练

推荐直接运行：

```bash
cd /workspace/yi/work/shiyan
bash scripts/train/current/run_exp12_rate044_A_patch_cb16_2_ch512_1024.sh
```

常用环境变量：

```bash
GPU_ID=0 bash scripts/train/current/run_exp12_rate044_A_patch_cb16_2_ch512_1024.sh
SIMVQ_NUM_EPOCHS=400 bash scripts/train/current/run_exp12_rate044_A_patch_cb16_2_ch512_1024.sh
SIMVQ_TOTAL_BATCH_SIZE=24 SIMVQ_MICRO_BATCH_SIZE=12 bash scripts/train/current/run_exp12_rate044_A_patch_cb16_2_ch512_1024.sh
```

默认从零开始训练：

```bash
SIMVQ_RESUME=0
SIMVQ_PRETRAINED_CHECKPOINT 未设置
```

如果中断后确实想从 `last_checkpoint.pth` 续训，再显式设置：

```bash
SIMVQ_RESUME=1 bash scripts/train/current/run_exp12_rate044_A_patch_cb16_2_ch512_1024.sh
```

训练输出：

```text
checkpoints/shiyan_raq_quality_v2_B_larger_rate044_A_patch_cb16-2_ch512-1024_unet2_ds8x2_k16-2/
  best_vq_deepsc.pth
  last_checkpoint.pth

experiments/
  *_epoch_metrics.csv
  *_codebook_metrics.csv
  logs/
  tensorboard/
```

## 如何评估

默认评估 best checkpoint，SNR=0，BPSK：

```bash
cd /workspace/yi/work/shiyan
bash scripts/eval/test_A_patch_rate044_ch512_1024.sh
```

也可以直接调用：

```bash
python -u test_real.py \
  --checkpoint checkpoints/shiyan_raq_quality_v2_B_larger_rate044_A_patch_cb16-2_ch512-1024_unet2_ds8x2_k16-2/best_vq_deepsc.pth \
  --snrs 0 3 6 9 12 \
  --modulation bpsk
```

无信道上界：

```bash
python -u test_real.py \
  --checkpoint checkpoints/shiyan_raq_quality_v2_B_larger_rate044_A_patch_cb16-2_ch512-1024_unet2_ds8x2_k16-2/best_vq_deepsc.pth \
  --no-channel
```

## 和原 quality 方案相比保留了什么

保留：

1. Stage B 的 GroupNorm + SiLU 主干。
2. 两层 U-Net。
3. 下采样 `[8,2]`。
4. base channels `256`，所以特征通道 `[512,1024]`。
5. patch-wise SimVQ。
6. 源码本大小 `[16,2]` 和 rate-0.044 设定。
7. 原有限码长信道课程。
8. 原 TensorBoard、CSV metrics、checkpoint 输出方式。

改变：

1. 新增 RAQ 动态目标码本支路。
2. 训练 loss 同时优化 source 和 RAQ 两路。
3. 测试默认发送 RAQ 目标码本索引。
4. 默认从零训练，不加载旧 checkpoint。
5. 默认不自动 resume，避免误用历史状态。

## 注意事项

1. 当前 RAQ 接入只支持 `SIMVQ_QUANTIZER_TYPE=simvq` 且 `SIMVQ_QUANTIZER_AXIS_LIST=patch,patch`。
2. 固定训练脚本显式设置 RAQ 评估目标 K 为 `[16,2]`，所以传输比仍对应 rate-0.044 方案。
3. 如果把 `SIMVQ_RAQ_TARGET_LIST` 改大，真实传输比也会改变。
4. `SIMVQ_ALLOW_PRETRAINED=1` 才会允许 `SIMVQ_PRETRAINED_CHECKPOINT` 生效；默认忽略预训练。
5. 若需要完全关闭 RAQ 做对照，可设置 `SIMVQ_USE_RAQ=0`，模型会退回原单支路 SimVQ 训练/测试。
