# Kodak 图像重建对比（2026-10-22）

本次从用户指定的三个重建目录中各选取 `kodim05.png`、`kodim21.png`，共六张重建图，并与 Kodak 同编号原图排版。生成两张四列对比图，列名按用户确认使用 Original、JPEG、RAQ-VAE、MDRQ。每张上排为完整图和红色取样框，下排为相同位置、相同倍率的放大细节。

## 成品

| 场景 | PNG | PDF | 细节重点 |
| --- | --- | --- | --- |
| 摩托车 `kodim05` | [PNG](../kodim05_comparison_20261022.png) | [PDF](../kodim05_comparison_20261022.pdf) | 前叉颜色、弹簧、轮辐、轮胎和车身边缘 |
| 灯塔海岸 `kodim21` | [PNG](../kodim21_comparison_20261022.png) | [PDF](../kodim21_comparison_20261022.pdf) | 房屋窗框、屋顶轮廓、烟囱及天空颜色 |

PNG 为 3272 × 1580 像素，记录 300 dpi；PDF 使用相同画面，图像以无损方式嵌入。

## 输入与选择依据

- Original：`/workspace/yi/work/Kodak`
- JPEG：`/workspace/yi/work/JPEG/generated_images/cbr1over1.2`
- RAQ-VAE：`/workspace/yi/work/RAQ-VAE-main/generated_images/20261022_205436_261293_88b35026/k32/awgn_snr10`
- MDRQ：`/workspace/yi/work/Ours-RAQ/experiments/eval_images/20261022_140841_k32_wwydw7h7/snr_10`

四组均为 24 张同编号 RGB PNG，逐张尺寸匹配。先计算全部 24 张的整图 RGB PSNR，再结合场景差异与真实纹理观察选择摩托车和灯塔海岸。它们的 MDRQ 整图 PSNR 分别比另外两种方法中较高者高 4.1451 dB、3.7934 dB。全部排名见 [all_image_psnr.csv](all_image_psnr.csv)。这些是按需求挑选的优势案例，不代表全部 24 张的平均表现，也不表示 MDRQ 在所有图片上都领先。

候选细节通过 144 × 144 窗口、8 像素步长搜索；图像外围保留 16 像素余量，原图梯度 RMS 至少为 12，避免选择缺少细节的纯色区域。按 MDRQ 相对两种基线中较优者的局部 PSNR 优势排序，再人工查看相同位置的四组细节。候选坐标与评分见 [roi_candidates.json](roi_candidates.json)，查看用拼图见 [detail_candidates.png](detail_candidates.png)。

最终红框坐标采用原图左上角为原点的 `[x1,y1,x2,y2)`：

- `kodim05.png`：`[320,224,464,368]`，MDRQ 局部 PSNR 优势至少 6.1857 dB。
- `kodim21.png`：`[112,136,256,280]`，MDRQ 局部 PSNR 优势至少 5.3309 dB。

## 实测指标

图下方显示整图指标。PSNR 以全部 RGB 像素的 MSE 计算，峰值为 255。MS-SSIM 复用现有 `Ours-RAQ/utils/metrics.py::ms_ssim`，输入为原始 RGB PNG 像素除以 255，采用该函数原有五尺度计算；图中沿用参考图的 dB 表示：`-10*log10(1-MS-SSIM)`。原始 0–1 分数也完整保存。

| 图像 | 方法 | PSNR (dB) | MS-SSIM (0–1) | MS-SSIM (dB) | 红框 PSNR (dB) |
| --- | --- | ---: | ---: | ---: | ---: |
| kodim05 | JPEG | 23.0248 | 0.91777787 | 10.8501 | 22.6699 |
| kodim05 | RAQ-VAE | 25.6385 | 0.96347244 | 14.3738 | 23.0906 |
| kodim05 | MDRQ | 29.7836 | 0.98492779 | 18.2182 | 29.2763 |
| kodim21 | JPEG | 27.2902 | 0.95911638 | 13.8845 | 27.4692 |
| kodim21 | RAQ-VAE | 25.5094 | 0.96276410 | 14.2904 | 27.3576 |
| kodim21 | MDRQ | 31.0836 | 0.98554034 | 18.3984 | 32.8001 |

完整精度、指标定义、脚本来源哈希与最终坐标见 [selection_and_metrics.json](selection_and_metrics.json)。图中仅保留两位小数。

## 生成与验证

实际执行环境：`/home/yi/.conda/envs/work/bin/python`。实际命令（工作目录 `/workspace/yi/work`）：

```bash
python -u /workspace/yi/work/comparison_review_20261022/build_comparisons.py --stage analyze > /workspace/yi/work/comparison_review_20261022/analysis.log 2>&1
python -u /workspace/yi/work/comparison_review_20261022/build_comparisons.py --stage render > /workspace/yi/work/comparison_review_20261022/generation.log 2>&1
python -u /workspace/yi/work/comparison_review_20261022/build_comparisons.py --stage review
```

完整图以原生 768 × 512 像素直接贴入画布；四列使用完全相同的红框位置，144 × 144 原始裁剪统一以最近邻放大为 672 × 672。没有修改源文件、调色、锐化、去噪、模糊、超分辨率或生成新细节。

自动验证了全部 96 张输入图像的前后 SHA-256 一致；验证了 8 个整图面板除边框/红框标注外与输入像素一致，以及 8 个放大面板与指定的源图裁剪一致。所选两张的 MDRQ 整图 PSNR、MS-SSIM 与红框区域 PSNR 均严格领先两种基线。两张成品与候选细节均已人工查看。

## 本次文件清单与审阅记录

全部为新增文件，不覆盖已有文件，不修改 Git 暂存区或提交历史。`analysis.log`、`generation.log` 被现有 Git 忽略规则排除，已通过完整文件链接及产物清单留存；其余本次文件均在 Git 状态中核对为新增未跟踪文件。工作区原本存在大量其他修改和未跟踪文件，见本次修改前的 [git_status_before.txt](git_status_before.txt)；这些既有变化不计入本次产物。

- 根目录四个成品：上方表格中的两个 PNG、两个 PDF。
- [build_comparisons.py](build_comparisons.py)：本次新增的可复现筛选、指标计算、排版与验证脚本。
- `README.md`：本次输入路径、选图依据、指标、命令与文件说明。
- [all_image_psnr.csv](all_image_psnr.csv)：全部 24 张的整图 PSNR 对照。
- [roi_candidates.json](roi_candidates.json)：候选局部窗口坐标、纹理与 PSNR 评分。
- [source_manifest.json](source_manifest.json)：96 个输入路径、尺寸和 SHA-256。
- [selection_and_metrics.json](selection_and_metrics.json)：最终两张的完整指标与坐标。
- [verification.json](verification.json)：像素一致性、输入不变和成品校验结果。
- [detail_candidates.png](detail_candidates.png)：人工核查时生成的辅助细节拼图。
- [analysis.log](analysis.log)、[generation.log](generation.log)：两阶段实际运行日志。
- [git_status_before.txt](git_status_before.txt)、[git_status_after.txt](git_status_after.txt)：工作区前后状态快照。
- [artifacts_manifest.json](artifacts_manifest.json)：本次新增文件的路径、大小与 SHA-256；清单本身不做自引用哈希。
- [changes.patch](changes.patch)：本次新增脚本、说明、CSV 与 JSON（除文件清单本身）的真实完整新增补丁，可在界面未展示未跟踪文件时直接审阅；文件清单完整内容单独通过上方链接提供。
