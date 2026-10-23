# 三种方法的同条件耗时测试（SNR=4.5 dB）

**本次测试中，RAQ-VAE 的图像编解码速度最快。RAQ-RVQ 在当前配置下没有取得运行时间优势。**

三种方法在同一张 NVIDIA GeForce RTX 4090 D（GPU 2）上运行，输入为 Kodak 的 24 张 256×256 RGB 图像，batch=1，FP32。RAQ-RVQ 使用 `[8,2;8,2]`，VQ-DeepSC 和 RAQ-VAE 使用 `16,16`。三者均为 5120 bit/图像、0.078125 bpp；采用 LDPC 5/8、QPSK 后，每张图像对应 32 个 LDPC 块和 4096 个信道符号。

**TABLE I. 目标索引量化和图像编解码的实测平均耗时（ms/图像）↓。**

| 方法 | 目标索引量化 | 编解码：缓存码本 | 编解码：逐图生成码本 |
|---|---:|---:|---:|
| VQ-DeepSC | 0.381 | 8.449 | 8.450 |
| RAQ-VAE | 0.339 | 1.624 | 7.975 |
| RAQ-RVQ（本文方法） | 0.695 | 13.559 | 26.544 |

注：目标索引量化在已经提取的特征和已经生成的码本上计时，包含距离计算、索引选择以及残差量化所需的中间查表和残差运算。编解码包含一次图像编码、目标量化、接收端查表恢复特征和一次图像解码，不含信道处理。缓存模式在计时前生成目标码本；逐图生成模式每张图像生成一遍所需码本，接收端复用同一组码本。VQ-DeepSC 直接使用固定码本，因此两种模式基本一致。

**TABLE II. 包含 LDPC 和信道处理的端到端仿真实测平均耗时（ms/图像）↓。**

| 方法 | AWGN：缓存码本 | Rician：缓存码本 | AWGN：逐图生成码本 |
|---|---:|---:|---:|
| VQ-DeepSC | 321.146 | 321.440 | 321.380 |
| RAQ-VAE | 314.125 | 314.471 | 320.959 |
| RAQ-RVQ（本文方法） | 326.642 | 326.796 | 339.939 |

注：SNR 均为 4.5 dB，Rician 的线性 K 因子为 10，接收端使用理想 CSI。端到端计时包括图像编码、目标量化、索引打包、LDPC 编码、QPSK 调制、信道仿真、软解调、LDPC 解码、索引恢复和图像解码。三种方法共用同一套通信代码；LDPC 在 CPU 上运行，块长为 (160,256)，解码迭代数为 20。表中的时间是软件仿真处理耗时，不包含真实无线空口的传输等待。

**TABLE III. 单独执行一次目标码本生成的实测平均耗时（ms）↓。**

| 方法 | 所需生成器调用数 | 目标码本生成耗时 |
|---|---:|---:|
| VQ-DeepSC | 0 | —（直接使用固定码本） |
| RAQ-VAE | 2 | 6.380 |
| RAQ-RVQ（本文方法） | 4 | 15.525 |

注：保留各方案规定的独立生成器调用。RAQ-VAE 即使两个尺度均为 K=16，仍执行两次 Seq2Seq 调用。单独模块的耗时与完整编解码中的额外耗时不能直接相加；完整流程中存在 CPU 提交与 GPU 执行的重叠，均以各自实际计时为准。

本文方法的小码本分解减少了需要比较的候选距离项，但当前实现要在两个尺度上依次完成两级量化，增加了逐级依赖和残差处理。候选数量的减少没有在本次 GPU 测试中转化为更短的量化时间。整网速度还受到图像编码器、解码器和码本生成器的影响，因此，局部量化复杂度不能直接代表整网运行速度。

固定码率部署时，本文方法可以预先生成并缓存目标码本。采用这一方式，编解码耗时由 26.544 ms 降至 13.559 ms，减少约 **48.9%**。这是本文方法自身的部署优化收益。与两种基线比较时，论文中的优势表述仍应放在已有证据支持的候选搜索规模、距离矩阵存储及相对 VQ-DeepSC 的码本存储等指标上。

本次统一保留实际传输所需的目标推理分支，省去训练损失、源码本参考重建和质量指标计算。计时前，三种方法在全部 24 张图像上的目标索引均与各自原始量化器完全一致，索引打包与解包校验全部通过。图像提前载入 GPU；checkpoint 加载、磁盘读写和结果保存均不计时。

每个方法、每项计时预热 10 次，随后遍历 24 张图像重复 3 轮，共 72 个样本；每张图像轮换方法执行顺序。使用墙钟计时，并在计时前后同步 CUDA。PyTorch 为 2.7.1+cu118，CUDA 为 11.8，cuDNN 为 9.1；TensorFlow 为 2.21.0，Sionna 为 1.2.2。关闭 AMP 和 TF32，未使用编译；PyTorch 和 TensorFlow 的 CPU 线程数均固定为 1。CPU 为 Intel Xeon Platinum 8352V。

复现命令：

```bash
cd /workspace/yi/work
PYTHONDONTWRITEBYTECODE=1 /home/yi/.conda/envs/work/bin/python -B \
  RAQ-RVQ/experiments/benchmark_equal_conditions.py \
  --gpu 2 --warmup 10 --repeats 3
```

[测速脚本](benchmark_equal_conditions.py) · [原始计时数据（含均值、中位数、P95 和逐样本耗时）](latency_equal_conditions_snr4p5.json) · [运行日志](latency_equal_conditions_snr4p5.log)

使用的 checkpoint：

- [RAQ-RVQ](../checkpoints/shiyan_independent_raq_rvq_src64-64_trg2-64_d2_curriculum_rate094_A_patch_ch256-512_res6-6_mixture_per_k_unet2_ds8x2_k64/best_vq_deepsc.pth)
- [VQ-DeepSC](/home/yi/vq_deepsc_runs_20260919/checkpoints/k16_16_last2_copy_200ep/best_vq_deepsc.pth)
- [RAQ-VAE](../../RAQ-VAE-main/runs/seq2seq_ds8_dualk_src64_20260910_113736_956705/checkpoints/epoch=145-step=78840.ckpt)

