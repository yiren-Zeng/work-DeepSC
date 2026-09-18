# 测试指标

`scripts/eval/` 中所有脚本默认输出 MS-SSIM、PSNR、rFID、LPIPS，以及分数形式的传输压缩率 CBR。每个 SNR 独立统计；`--no-channel` 也计算重建质量指标。使用 `--json-output` 时，新指标保存为每个条件下的 `rfid` 和 `lpips` 字段。

安装额外依赖：

```bash
python -m pip install -r requirements-eval.txt
```

- rFID：原图集合与对应重建图集合之间的 FID，使用 [pytorch-fid](https://github.com/mseitzer/pytorch-fid) 的标准 FID Inception 权重和 2048 维 pool3 特征。在输入图像上计算，不写入临时 PNG；输入裁剪到有效范围后映射到 `[0,1]`，Inception 内部缩放到 `299×299`。样本少于 2 张时显示 `N/A`，JSON 保存 `null`。Kodak 等小数据集使用与完整协方差公式等价的 SVD 计算。
- LPIPS：使用 [官方 LPIPS](https://github.com/richzhang/PerceptualSimilarity) 的 AlexNet v0.1 和已训练的感知校准权重；RGB 输入范围为 `[-1,1]`，逐图计算后取平均。

两项指标都是越低越好。首次运行会下载 Inception 和 AlexNet 的预训练权重，之后复用 PyTorch 缓存。
