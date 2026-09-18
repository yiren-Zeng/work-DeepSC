# 普通 RAQ 与独立 RAQ-RVQ 实测比较

## 条件

- 数据集：Kodak-256-transform-resize，24 张 RGB 图片，256×256，保持原尺寸。
- GPU：物理 GPU 1，NVIDIA L40S；两模型依次运行。
- 信道：AWGN，SNR=3 dB，LDPC n=256/k=128（码率 1/2），QPSK；随机种子 42。
- 普通 RAQ：目标 K=[8,16]；独立 RVQ：目标 K=[[4,2],[8,2]]。
- 两者均为 4096 bit/图、0.0625 payload bpp、4096 QPSK 符号/图，传输比 1/48。无 LDPC 填充。
- 四次质量测试均调用原始脚本，不传入额外命令行参数；通过 JSON_OUTPUT 保存到本目录。
- 环境中原有的 SIMVQ_* 与评测覆盖变量已清除，GPU_ID=1；脚本本身未修改。

## 重建质量

| 条件 | 普通 RAQ PSNR | 独立 RVQ PSNR | RAQ−RVQ | 普通 RAQ MS-SSIM | 独立 RVQ MS-SSIM |
|---|---:|---:|---:|---:|---:|
| 3 dB | 23.888898 dB | 23.236794 dB | +0.652104 dB | 0.879695 | 0.872867 |
| 无信道 | 23.910743 dB | 23.241668 dB | +0.669075 dB | 0.880532 | 0.873360 |

无信道逐图比较：普通 RAQ 在 24/24 张图片上 PSNR 更高，在 20/24 张图片上 MS-SSIM 更高。明细见 per_image_no_channel.csv。

信道相对于无信道的平均 PSNR 损失：普通 RAQ 0.021845 dB；独立 RAQ-RVQ 0.004874 dB。

当前差距主要来自源重建质量；仅凭这两个不同 checkpoint 的结果，不能认定所有单级 RAQ 都优于 RVQ。

## RVQ 残差诊断

| 尺度 | 第一级后的平均残差 MSE | 第二级后的平均残差 MSE | 第二级带来的变化 |
|---|---:|---:|---:|
| 0 | 0.194545 | 0.151801 | -21.97% |
| 1 | 0.029128 | 0.039946 | +37.14% |

第二尺度的第二级量化增加了平均特征残差 MSE，说明当前 K=[8,2] 下第二级没有降低该尺度的特征误差。这是进一步排查的线索，但单独的特征 MSE 不能证明图像 PSNR 差距的具体原因。
数据取自原脚本无信道结果 diagnostics.rvq_quantization；两模型 encoder 权重不同，不跨模型比较该特征 MSE。

## 模型计算与内存

batch=1、FP32、torch.inference_mode()；预热 5 次，对 24 张图片重复 3 次（共 72 次）。每次计时前后同步 GPU。
计时覆盖 encoder、每图 RAQ 码本生成、量化、索引重建和 decoder；不含数据读取、指标计算、LDPC、调制、AWGN 或启动加载。

| 指标 | 普通 RAQ | 独立 RVQ |
|---|---:|---:|
| 模型总参数 | 104,788,227 | 132,766,979 |
| 推理平均延迟 | 17.438 ms/图 | 26.197 ms/图 |
| 推理中位数延迟 | 17.375 ms/图 | 26.157 ms/图 |
| 推理 P95 延迟 | 17.551 ms/图 | 26.358 ms/图 |
| 平均延迟对应吞吐 | 57.34 图/s | 38.17 图/s |
| PyTorch 峰值已分配显存 | 755.22 MiB | 863.64 MiB |
| checkpoint 文件大小 | 414.53 MiB | 507.07 MiB |

独立 RVQ 参数量增加 26.70%，平均延迟增加 50.23%。
显存值是 PyTorch 已分配内存，包括模型、预加载的 24 张图片及重建临时张量；不等同于 nvidia-smi 的整个进程显存。

## 原脚本整体耗时

| 条件 | 普通 RAQ | 独立 RVQ |
|---|---:|---:|
| 3 dB | 31.132 s | 32.970 s |
| 无信道 | 20.042 s | 19.795 s |

整体耗时各测一次，包括 conda、Python、checkpoint 加载、数据读取、指标，以及信道测试中的 TensorFlow/LDPC CPU 计算；不能据此单独判断模型推理速度。
GPU 1 在开始测试时为空闲，其他 GPU 有训练任务，共享 CPU 的负载可能影响耗时。速度数据用于本机近似比较。

## 文件与复现

- 四次评测原始结果和日志：raq_channel_snr3、independent_rvq_channel_snr3、raq_no_channel、independent_rvq_no_channel 对应的 .json/.log。
- run_manifest.json：原始调用、条件、返回码与耗时。
- *_profile.json：参数数目、72 次延迟、显存与逐图无信道质量。
- summary.json 与 per_image_no_channel.csv：汇总和逐图对比。
- run_comparison.py：依次运行原始脚本。
- run_profiles.py：使用原始脚本设置环境，替换其最终 Python 入口为 profile_reconstruction.py；不改原始脚本。

复现时从仓库目录运行本目录的 run_comparison.py、run_profiles.py，最后运行 summarize.py。复现会覆盖本目录结果。

独立 RVQ 脚本的注释和默认输出文件名仍写着 4×64/LDPC 3/4，实际测试以当前参数 4×2/LDPC 1/2 为准。
num_embeddings_list=[16,4] 是独立 RVQ 的名义元数据，实际布局以 rvq_k_lists=[[4,2],[8,2]] 为准。
结果范围：当前两个 checkpoint、该 Kodak 数据集、当前码率以及一次 seed=42 的 3 dB 信道测试。
