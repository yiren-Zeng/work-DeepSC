# 将帆船数字放大区域改为船身文字

按用户截图新增红框的位置，将 `kodim10` 四列的红框和下排放大图移至前景帆船深色船身上的白色文字。列标题继续为 Original、JPEG-2000、RAQJSCC、MDRQ，整图指标和版式保持一致。

## 更新成品

- [PNG](../kodim10_comparison_20261022.png)：原路径更新，2248 × 1612 像素、300 dpi。
- [PDF](../kodim10_comparison_20261022.pdf)：原路径更新，同一画面无损嵌图。

仅实际修改这两个既有成品文件；其他四张对比图及其 PDF 的 SHA-256 均确认不变。修改前已保存本图 PNG、PDF 的完整副本，见 [before_manifest.json](before_manifest.json)。

## 裁剪与指标

原图为 `/workspace/yi/work/Kodak/kodim10.png`；三种重建继续使用此前指定目录中的同编号图像，完整路径和输入哈希见 [source_manifest.json](source_manifest.json)。

坐标以原图左上角为原点，格式为 `[x1,y1,x2,y2)`：

- 原船帆数字区域：`[152,28,312,188)`。
- 新船身文字区域：`[224,624,304,704)`，80 × 80 原始像素。

新裁剪比截图中手工标记的范围更紧，完整覆盖目标文字，让字形在下排更突出。四列均采用同一坐标，统一以最近邻放大为 448 × 448，完整图按原生 512 × 768 像素贴入；没有增强、调色或生成新笔画。

图下方保留此前的整图 PSNR、MS-SSIM，因为原始输入完全相同。重新核对整图 RGB PSNR 与缓存值一致，并为新船身区域计算真实局部 PSNR：

| 方法 | 新船身区域 PSNR (dB) |
| --- | ---: |
| JPEG-2000 | 29.8774 |
| RAQJSCC | 29.9686 |
| MDRQ | 34.4357 |

完整指标和验证记录见 [verification_and_metrics.json](verification_and_metrics.json)。

## 实际生成命令与验证

工作目录为 `/workspace/yi/work`，执行环境为 `/home/yi/.conda/envs/work/bin/python`。

```bash
python -u /workspace/yi/work/comparison_hull_text_20261022/replace_crop.py --stage prepare > /workspace/yi/work/comparison_hull_text_20261022/prepare.log 2>&1
python -u /workspace/yi/work/comparison_hull_text_20261022/replace_crop.py --stage render > /workspace/yi/work/comparison_hull_text_20261022/generation.log 2>&1
python -u /workspace/yi/work/comparison_hull_text_20261022/replace_crop.py --stage review
```

`prepare` 保存两个修改前完整副本，记录全部 96 张输入图像以及其他 8 个成品文件的哈希，并生成三种裁剪范围的检查拼图。`render` 恢复旧船帆红框处的原始像素、绘制新红框并替换四个放大面板。

逐像素验证了旧/新红框区域和四个放大面板之外的画面完全一致，包含四列标题及整图指标；四个新放大面板与指定源图裁剪完全一致。全部 96 张输入图像、其他 8 个成品文件的哈希均不变。候选拼图和更新后的完整成品均已人工查看。

## 本次文件变更与审阅

本次实际修改根目录中的 `kodim10_comparison_20261022.png`、`kodim10_comparison_20261022.pdf`，新增本目录中的审阅文件。两个旧成品在操作前已是未跟踪文件，普通 `git diff` 无法反映其本次变化，故通过完整修改前副本、前后 SHA-256 和像素验证保存真实记录。未更改暂存区、提交或推送。

- [replace_crop.py](replace_crop.py)：本次新增的快照、裁剪替换、指标核对及审阅脚本。
- `README.md`：本次变更、坐标、命令、验证和文件清单。
- [before/kodim10_comparison_20261022.png](before/kodim10_comparison_20261022.png)、[before/kodim10_comparison_20261022.pdf](before/kodim10_comparison_20261022.pdf)：修改前完整成品。
- [before_manifest.json](before_manifest.json)：修改前文件路径、副本位置和 SHA-256。
- [source_manifest.json](source_manifest.json)：96 张输入图像的路径、哈希。
- [other_figures_before.json](other_figures_before.json)：其他四张对比图及其 PDF 的哈希。
- [hull_crop_candidates.png](hull_crop_candidates.png)：三个同位置、同倍率的四列裁剪检查拼图。
- [verification_and_metrics.json](verification_and_metrics.json)：最终坐标、指标、像素验证和前后成品哈希。
- [prepare.log](prepare.log)、[generation.log](generation.log)：实际运行日志；被现有 Git 规则忽略，已通过完整链接保留。
- [git_status_before.txt](git_status_before.txt)、[git_status_after.txt](git_status_after.txt)：操作前后工作区状态。
- [changes.patch](changes.patch)：新增脚本、说明、JSON（除产物清单本身）的完整实际新增补丁。
- [artifacts_manifest.json](artifacts_manifest.json)：本次修改与新增文件的完整路径、类型、大小、SHA-256，并区分新增未跟踪文件和被忽略日志。

此前审阅目录保留历史交付记录，本图本次最新裁剪和输出哈希以此目录为准。
