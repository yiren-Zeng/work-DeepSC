# BPG 重建质量评估

在 BPG 目录运行 `python test_bpg.py`，输出 PSNR、MS-SSIM、rFID 和平均 LPIPS。

rFID 使用 [标准 FID Inception](https://github.com/mseitzer/pytorch-fid) 的 2048 维 pool3 特征；LPIPS 使用 [官方 AlexNet v0.1](https://github.com/richzhang/PerceptualSimilarity) 的预训练感知校准权重。

脚本直接导入相邻 `Best_VQ/evaluation/perceptual.py` 的 `PerceptualMetrics`，与 Best_VQ 使用同一个实现、权重缓存、预处理和统计公式。BPG 的 RGB PNG 张量从 `[0,1]` 映射到共享实现要求的 `[-1,1]`；rFID 按整组原图/重建图计算，LPIPS 逐图计算后取平均。

安装依赖：`python -m pip install -r requirements-eval.txt`。需要保留相邻的 `Best_VQ` 目录。

缺失的解码图片继续按 PSNR/MS-SSIM 为 0 计分，并以黑图作为新指标的重建输入，计入全部 24 张图片的统计。
