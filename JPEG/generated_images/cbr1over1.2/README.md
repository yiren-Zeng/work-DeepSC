本目录保存现有 JPEG-2000 重建结果的 24 张 RGB PNG，逐张保持输入 JP2 的原始宽高。

输入为 `JPEG/output_jpeg/val_000001.jp2` 至 `val_000024.jp2`；按照原 `test_jpeg.py` 的对应关系，输出命名为 `kodim01.png` 至 `kodim24.png`。CBR=1/32 使用用户给出的现有实验配置，本次只导出这些已有文件。

PNG 与原评测读取 JP2 后得到的 RGB 像素完全一致，已逐张验证；没有重新压缩、重新模拟信道、调整尺寸或增强图像，原 JP2 和测试脚本均未修改。

前面选择的两个同编号样本为 [kodim02.png](kodim02.png) 和 [kodim03.png](kodim03.png)。完整映射和校验结果见 `image_export.json`。

重新导出：

```bash
/home/yi/.conda/envs/work/bin/python /workspace/yi/work/JPEG/export_reconstruction_png.py
```
