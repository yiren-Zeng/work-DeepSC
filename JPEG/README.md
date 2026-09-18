`test_jpeg.py` 对比原图 `kodim01.png` 至 `kodim24.png` 与重建图
`val_000001.jp2` 至 `val_000024.jp2`，输出 PSNR、MS-SSIM、rFID 和 LPIPS。

rFID 和 LPIPS 直接复用相邻目录 `Best_VQ/evaluation/perceptual.py`，与 Best_VQ
和 BPG 使用同一份代码、权重、预处理和统计方式：

- rFID 使用 [pytorch-fid 标准 FID Inception](https://github.com/mseitzer/pytorch-fid)
  的 pool3 2048 维特征，对整组原图与重建图计算一次 FID。
- LPIPS 使用 [官方 AlexNet v0.1](https://github.com/richzhang/PerceptualSimilarity)，
  对逐张距离取平均。
- RGB 图片从 `[0,1]` 转为共享实现要求的 `[-1,1]`。
- 缺失的重建文件与 BPG 一样按黑图计入 rFID/LPIPS，并输出失败数量；
  原有 PSNR、MS-SSIM 的缺失文件得分仍为 0。

在 JPEG 目录安装依赖并运行：

```bash
python -m pip install -r requirements-eval.txt
python test_jpeg.py
```

首次运行需要下载标准模型权重，后续复用 PyTorch 缓存。
请保持 JPEG 与 Best_VQ 位于同一父目录。
