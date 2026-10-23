# Kodak 对比图第二版：手臂与橡皮艇文字

用户确认第一张放大其标出的骑手手臂，包含蓝白袖子字样、肩部到手套；第二张在查看文字候选后选择橡皮艇上的粗体英文，重点比较文字边缘清晰度。继续使用 Original、JPEG、RAQ-VAE、MDRQ 四列，上排完整图和红框，下排对应位置的放大图，保留整图 PSNR、MS-SSIM 指标。

## 成品和修改内容

| 场景 | PNG | PDF | 本次关键变化 |
| --- | --- | --- | --- |
| 摩托车 `kodim05` | [PNG](../kodim05_comparison_20261022_v2.png) | [PDF](../kodim05_comparison_20261022_v2.pdf) | 将车轮取样区域改为用户标出的手臂和袖子字样 |
| 橡皮艇 `kodim14` | [PNG](../kodim14_comparison_20261022_v2.png) | [PDF](../kodim14_comparison_20261022_v2.pdf) | 替换上一版灯塔海岸场景，突出船身 SNOWMASS / WHITEWATER 英文 |

文件直接位于 `/workspace/yi/work`，均为 3272 × 1580 像素、300 dpi PNG，并另存对应的无损嵌图 PDF。文件名添加 `_v2`；上一版的两个 PNG、两个 PDF 和审阅记录保留原样。

## 输入和裁剪坐标

- Original：`/workspace/yi/work/Kodak`
- JPEG：`/workspace/yi/work/JPEG/generated_images/cbr1over1.2`
- RAQ-VAE：`/workspace/yi/work/RAQ-VAE-main/generated_images/20261022_205436_261293_88b35026/k32/awgn_snr10`
- MDRQ：`/workspace/yi/work/Ours-RAQ/experiments/eval_images/20261022_140841_k32_wwydw7h7/snr_10`

源图左上角为原点，坐标格式为 `[x1,y1,x2,y2)`：

- `kodim05.png`：`[352,64,480,192]`，覆盖用户在截图上新增红框标出的手臂。
- `kodim14.png`：`[508,352,636,480]`，覆盖船身两行粗体英文及相邻边缘。

两处均为 128 × 128 原始像素，每列统一最近邻放大至 672 × 672。完整图以原生 768 × 512 像素贴入画布，仅添加边框和红框。没有对源图调色、锐化、去噪、模糊或生成新细节。

筛选时对比了建筑招牌、帽子字样、飞机字样、船名和橡皮艇文字。用户最终选择橡皮艇英文，比较重点是笔画边缘、对比度及与原图的一致性；其他方法仍能辨认部分文字，未据此声称 OCR 识别率。候选的原始坐标、局部 PSNR 和同倍率拼图保留在下方文件清单中。

## 实际指标

图中显示整图指标。PSNR 为原始 RGB 像素上的 `10*log10(255^2/MSE)`；MS-SSIM 使用现有 `Ours-RAQ/utils/metrics.py::ms_ssim` 的五尺度算法，输入为 RGB 像素除以 255。沿用参考图的 MS-SSIM dB 格式：`-10*log10(1-MS-SSIM)`，原始 0–1 分数也保存于 JSON。

| 图像 | 方法 | 整图 PSNR (dB) | MS-SSIM (0–1) | MS-SSIM (dB) | 红框 PSNR (dB) |
| --- | --- | ---: | ---: | ---: | ---: |
| kodim05 手臂 | JPEG | 23.0248 | 0.91777787 | 10.8501 | 21.3856 |
| kodim05 手臂 | RAQ-VAE | 25.6385 | 0.96347244 | 14.3738 | 24.6287 |
| kodim05 手臂 | MDRQ | 29.7836 | 0.98492779 | 18.2182 | 27.9795 |
| kodim14 英文 | JPEG | 25.8108 | 0.93139512 | 11.6364 | 23.7058 |
| kodim14 英文 | RAQ-VAE | 26.5759 | 0.95766379 | 13.7329 | 24.2605 |
| kodim14 英文 | MDRQ | 30.9982 | 0.98401272 | 17.9623 | 27.2291 |

MDRQ 的手臂区域 PSNR 比两种基线中较高者高 3.3507 dB；英文区域高 2.9686 dB。整图 PSNR、MS-SSIM 也在这两个样本中领先。完整精度见 [selection_and_metrics.json](selection_and_metrics.json)。

## 生成命令与验证

执行环境为 `/home/yi/.conda/envs/work/bin/python`，工作目录为 `/workspace/yi/work`。

```bash
python -u /workspace/yi/work/comparison_revision_20261022/revise_comparisons.py --stage inspect > /workspace/yi/work/comparison_revision_20261022/inspection.log 2>&1
python -u /workspace/yi/work/comparison_revision_20261022/revise_comparisons.py --stage render > /workspace/yi/work/comparison_revision_20261022/generation.log 2>&1
python -u /workspace/yi/work/comparison_revision_20261022/revise_comparisons.py --stage review
```

脚本只读复用上一版 `comparison_review_20261022/build_comparisons.py` 中的输入读取、指标和字体辅助函数。生成前后核对全部 96 张源图和上一版 18 个产物的 SHA-256，均未改变。逐像素验证 8 个完整图面板在标注外与输入一致，8 个放大面板与指定源图裁剪一致。两张成品及候选拼图均已人工查看。

## 本次新增文件与审阅记录

本次所有产物均为新增文件，既有工作区改动、上一版文件和 Git 暂存区均未修改。`inspection.log`、`generation.log` 被现有 Git 忽略规则排除，仍通过文件链接与清单保留。其余本次新增文件在 `git status --short --untracked-files=all` 中逐项核对。

- 根目录四个成品：上表两个 `_v2.png` 和两个 `_v2.pdf`。
- [revise_comparisons.py](revise_comparisons.py)：本次新增的候选检查、排版、指标、像素验证与审阅记录脚本。
- `README.md`：本次需求、坐标、指标、生成命令与完整文件说明。
- [source_manifest.json](source_manifest.json)：96 张输入图像的路径与 SHA-256。
- [previous_artifacts_manifest.json](previous_artifacts_manifest.json)：上一版 18 个文件的路径与 SHA-256。
- [candidate_regions.json](candidate_regions.json)：11 个候选区域的坐标、局部 PSNR 和优势评分。
- [candidate_details_1.png](candidate_details_1.png)：手臂、袖子和建筑招牌候选。
- [candidate_details_2.png](candidate_details_2.png)：橡皮艇、飞机和帽子文字候选。
- [candidate_details_3.png](candidate_details_3.png)：船名候选及最终橡皮艇文字取样。
- [selection_and_metrics.json](selection_and_metrics.json)：最终坐标、完整指标与指标/辅助脚本来源哈希。
- [verification.json](verification.json)：输入和旧产物不变、面板像素一致及输出哈希。
- [inspection.log](inspection.log)、[generation.log](generation.log)：实际检查及生成日志。
- [git_status_before.txt](git_status_before.txt)、[git_status_after.txt](git_status_after.txt)：本次操作前后的工作区状态。
- [changes.patch](changes.patch)：所有新增脚本、说明和 JSON（除产物清单本身）的实际完整新增补丁。
- [artifacts_manifest.json](artifacts_manifest.json)：本次新增文件清单、大小和 SHA-256；清单本身通过此链接提供完整内容。
