# A/B/C 消融实验说明

日期：2026-05-28

## 为什么没有单独的 A/B 代码文件

A、B、C 不是通过复制三份训练代码实现的，而是使用 **同一套模型代码 + 环境变量配置开关** 实现。

这样做的原因是：

- 避免 `train_A.py`、`train_B.py`、`train_C.py` 三份代码互相漂移。
- 保证训练循环、评估方式、checkpoint 保存逻辑完全一致。
- 每个方案只改变它应该改变的模块，方便做公平对比。

真正区分 A/B/C 的地方在 `config.py` 的 `_stage_settings()`。

## 配置入口

文件：

```text
config.py
```

关键位置：

```python
def _stage_settings(stage):
    ...
```

通过环境变量选择实验阶段：

```bash
SIMVQ_EXPERIMENT_STAGE=A
SIMVQ_EXPERIMENT_STAGE=B
SIMVQ_EXPERIMENT_STAGE=C
```

如果不设置，默认是：

```bash
SIMVQ_EXPERIMENT_STAGE=C
```

## A 方案：只验证课程学习

实验名：

```text
quality_v2_A_curriculum_unet2_ds8x2_k16-32
```

目的：

```text
只验证 0.082 BPP + channel_prob 课程学习是否有效。
```

配置：

```text
channel curriculum: 开启
BatchNorm: 使用
PReLU: 使用
下采样: stride=8 一步卷积下采样，不使用级联下采样
残差块数量: encoder=1, decoder=1
上采样: nearest
Bottleneck Attention: 关闭
MS-SSIM Loss: 关闭
重建损失: 纯 MSE
退火阶段: PHASE1_END=0.1, PHASE2_END=0.4
```

启动方式：

```bash
SIMVQ_EXPERIMENT_STAGE=A CUDA_VISIBLE_DEVICES=1 \
EXPERIMENT_RUN_ID=quality_v2_A_curriculum_unet2_ds8x2_k16-32-001-gpu1 \
/workspace/yi/.conda/envs/work/bin/python -u train.py
```

输出位置：

```text
checkpoints/quality_v2_A_curriculum_unet2_ds8x2_k16-32/
experiments/quality_v2_A_curriculum_unet2_ds8x2_k16-32_epoch_metrics.csv
experiments/quality_v2_A_curriculum_unet2_ds8x2_k16-32_screening.csv
experiments/logs/train_quality_v2_A_curriculum_unet2_ds8x2_k16-32-001-gpu1.log
```

## B 方案：A + 主干网络升级

实验名：

```text
quality_v2_B_backbone_unet2_ds8x2_k16-32
```

目的：

```text
在 A 的基础上验证主干网络升级带来的收益。
```

配置：

```text
channel curriculum: 开启
GroupNorm: 开启
SiLU: 开启
下采样: stride=8 一步卷积下采样，不使用级联下采样
残差块数量: encoder=2, decoder=2
上采样: bilinear
Bottleneck Attention: 关闭
MS-SSIM Loss: 关闭
重建损失: 纯 MSE
退火阶段: PHASE1_END=0.1, PHASE2_END=0.4
```

启动方式：

```bash
SIMVQ_EXPERIMENT_STAGE=B CUDA_VISIBLE_DEVICES=2 \
EXPERIMENT_RUN_ID=quality_v2_B_backbone_unet2_ds8x2_k16-32-001-gpu2 \
/workspace/yi/.conda/envs/work/bin/python -u train.py
```

输出位置：

```text
checkpoints/quality_v2_B_backbone_unet2_ds8x2_k16-32/
experiments/quality_v2_B_backbone_unet2_ds8x2_k16-32_epoch_metrics.csv
experiments/quality_v2_B_backbone_unet2_ds8x2_k16-32_screening.csv
experiments/logs/train_quality_v2_B_backbone_unet2_ds8x2_k16-32-001-gpu2.log
```

## C 方案：B + 混合损失和注意力

实验名：

```text
quality_v2_C_full_unet2_ds8x2_k16-32
```

目的：

```text
在 B 的基础上验证 MS-SSIM 混合损失和 Bottleneck Attention 的收益。
```

配置：

```text
channel curriculum: 开启
GroupNorm: 开启
SiLU: 开启
下采样: stride=8 一步卷积下采样，不使用级联下采样
残差块数量: encoder=2, decoder=2
上采样: bilinear
Bottleneck Attention: 开启
MS-SSIM Loss: 关闭
重建损失: 纯 MSE
退火阶段: PHASE1_END=0.1, PHASE2_END=0.4
```

启动方式：

```bash
SIMVQ_EXPERIMENT_STAGE=C CUDA_VISIBLE_DEVICES=<空闲GPU> \
EXPERIMENT_RUN_ID=quality_v2_C_full_unet2_ds8x2_k16-32-001 \
/workspace/yi/.conda/envs/work/bin/python -u train.py
```

## 三个方案的对比逻辑

最终比较顺序：

```text
A -> B -> C
```

含义：

```text
A 的结果：课程学习本身的效果
B - A：主干网络升级带来的收益
C - B：MS-SSIM 混合损失 + Bottleneck Attention 带来的收益
```

## 共同低码率配置

A/B/C 都使用同一个低码率配置：

```text
UNET_DEPTH = 2
DOWNSAMPLE_STRIDES = [8, 2]
NUM_EMBEDDINGS_LIST = [16, 32]
ESTIMATED_SOURCE_BPP = 0.08203125
```

码率计算：

```text
第 1 层：log2(16) / 8^2  = 4 / 64  = 0.0625
第 2 层：log2(32) / 16^2 = 5 / 256 = 0.01953125
总 BPP = 0.08203125
```
