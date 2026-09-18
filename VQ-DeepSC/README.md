# VQ-DeepSC

这是从 `Vq-dc-64-Multi` 独立出的无信道多尺度 VQ 图像重建项目。原项目中的有限码长信道、BER 模拟、LDPC、调制、AWGN/Rician 信道及真实传输测试均未包含在本目录中。

## 模型流程

```text
RGB image -> four-scale encoder -> four VQ codebooks
          -> quantized multi-scale features -> decoder -> reconstructed image
```

默认配置使用四层特征 `[128, 256, 512, 1024]`，每层码本大小均为 64。训练损失为重建 MSE 与四层 VQ 损失之和。

## 数据

默认继续使用工作区中的现有数据集：

- 训练：`/workspace/yi/work/Cars196/train_data`
- 验证：`/workspace/yi/work/Cars196/val_data`
- 测试：`/workspace/yi/work/Kodak-256-transform-resize`

训练集采用 `Resize(256) + RandomCrop(256)`，验证集采用 `Resize(256) + CenterCrop(256)`。测试集已经预处理为 256×256，因此测试 DataLoader 不再执行 resize。

数据路径、GPU、batch size、学习率和码本配置都可以在 `config.py` 中修改。

## 训练

```bash
cd /workspace/yi/work/VQ-DeepSC
python train.py
```

项目不会读取原目录中的权重，也不保存每 N 个 epoch 的周期权重。训练期间会在验证重建损失创造新低时更新 `checkpoints/best_vq_deepsc.pth`，并在每个 epoch 结束后更新 `checkpoints/last_checkpoint.pth`，用于恢复模型、优化器、学习率调度器和随机数状态。TensorBoard 日志写入当前项目的 `logs/`，两类输出均由 `.gitignore` 排除。

查看训练曲线：

```bash
tensorboard --logdir /workspace/yi/work/VQ-DeepSC/logs
```

## 未包含的组件

- `communications/`
- `models/channel.py`
- `test_real.py`
- TensorFlow 和 Sionna 依赖
- 原项目的 checkpoints、日志和 Python 缓存

`forward_test()` 仍可输出四层 VQ 索引，`reconstruct_from_indices()` 可以从索引重建图像，便于以后接入新的通信方案。

## Baseline experiments

`scripts/train/` 包含六个最后两层复制级联训练脚本。脚本名称保留原四层历史标签 `[64,64,K2,K3]`，但实际模型只创建尺度 2、3 的两个码本 `[K2,K3]`。输入仍经过完整四层编码器；解码端保留尺度 2 跳跃连接，在缺失的尺度 1、0 跳跃位置分别复制当前解码特征。新实验统一使用 `_last2_copy` 输出目录，不会读取或覆盖旧四码本权重。

`scripts/test/` 包含对应的六个测试脚本。测试链路与指定的 RAQ-RVQ baseline 对齐：256×256 Kodak、单一 combined payload、5G-NR LDPC `(k=128,n=256,R=1/2)`、QPSK、AWGN、默认 6 dB，并输出 PSNR、MS-SSIM、BER、索引错误率和传输开销到独立 JSON 文件。通信模块只被测试程序使用，不参与训练。

同目录下带有 `_last2_cascade` 后缀的六个脚本加载对应的新训练权重，只量化并传输尺度 2、3，并采用与训练一致的深层特征复制解码。终端只汇总码本、调制方式、分数形式压缩率、SNR、PSNR 和 MS-SSIM；完整通信诊断保留在独立 JSON 中。原四层测试脚本及其结果文件不会被覆盖。

两码本测试脚本现命名为 `test_k2_16_last2_cascade.sh`、`test_k4_16_last2_cascade.sh`、`test_k8_16_last2_cascade.sh`、`test_k16_4_last2_cascade.sh`、`test_k32_16_last2_cascade.sh` 和 `test_k128_16_last2_cascade.sh`。脚本、终端输出和 JSON 中的码本配置均只包含两个活动码本。测试使用独立的 `config_last2_test.py`，仍加载现有 `checkpoints/k64_64_*_last2_copy/` 目录中的新方案权重，不移动权重、不改变训练配置。
