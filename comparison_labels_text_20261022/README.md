# 对比图统一命名并新增帆船数字样本

用户确认将此前四张对比图的列标题全部改为 `JPEG-2000`、`RAQJSCC`；`Original`、`MDRQ` 保持原名称。新增样本在比较建筑招牌、警示牌、墙面字样与帆船数字后，按用户选择采用 `kodim10.png` 的帆船数字 `63455`，重点展示数字边缘和轮廓清晰度。

## 五张最新成品

| 场景 | PNG | PDF | 本次操作 |
| --- | --- | --- | --- |
| 帆船数字 `63455` | [PNG](../kodim10_comparison_20261022.png) | [PDF](../kodim10_comparison_20261022.pdf) | 新增样本，统一列名 |
| 手臂与袖子字样 | [PNG](../kodim05_comparison_20261022_v2.png) | [PDF](../kodim05_comparison_20261022_v2.pdf) | 原路径更新列名 |
| 橡皮艇英文 | [PNG](../kodim14_comparison_20261022_v2.png) | [PDF](../kodim14_comparison_20261022_v2.pdf) | 原路径更新列名 |
| 车轮与前叉 | [PNG](../kodim05_comparison_20261022.png) | [PDF](../kodim05_comparison_20261022.pdf) | 原路径更新列名 |
| 灯塔海岸 | [PNG](../kodim21_comparison_20261022.png) | [PDF](../kodim21_comparison_20261022.pdf) | 原路径更新列名 |

所有成品仍位于 `/workspace/yi/work`。四张旧图为 3272 × 1580 像素；新增原图是竖幅，按原生 512 × 768 像素保留完整画面，新拼图为 2248 × 1612 像素。各 PNG 记录 300 dpi；PDF 使用同一画面无损嵌图。

## 新增样本与真实指标

原图左上角为原点，帆船数字红框坐标为 `[152,28,312,188)`，大小 160 × 160 原始像素。四列均取完全相同的位置，用最近邻统一放大至 448 × 448，没有生成新笔画或改变源图颜色。

| 方法 | 整图 PSNR (dB) | MS-SSIM (0–1) | MS-SSIM (dB) | 数字区域 PSNR (dB) |
| --- | ---: | ---: | ---: | ---: |
| JPEG-2000 | 31.5424 | 0.96906123 | 15.0950 | 32.3522 |
| RAQJSCC | 28.7578 | 0.96902228 | 15.0895 | 32.2822 |
| MDRQ | 34.1274 | 0.98760162 | 19.0664 | 37.1842 |

MDRQ 的数字区域 PSNR 分别高 4.8320 dB、4.9019 dB，整图 PSNR、MS-SSIM 也领先。用户选用的是数字边缘与轮廓清晰度这一比较目标；其他方法仍可辨认部分数字，未据此声称只有 MDRQ 能读清或进行 OCR 识别率测量。

整图 PSNR 为原始 RGB 像素上的 `10*log10(255^2/MSE)`。MS-SSIM 复用 `Ours-RAQ/utils/metrics.py::ms_ssim` 的原有五尺度算法，输入为 RGB 像素除以 255，显示值沿用参考图的 `-10*log10(1-MS-SSIM)` dB 格式。完整精度见 [new_figure_metrics.json](new_figure_metrics.json)。

## 输入路径及后续生成入口

- Original：`/workspace/yi/work/Kodak`
- JPEG-2000：`/workspace/yi/work/JPEG/generated_images/cbr1over1.2`
- RAQJSCC：`/workspace/yi/work/RAQ-VAE-main/generated_images/20261022_205436_261293_88b35026/k32/awgn_snr10`
- MDRQ：`/workspace/yi/work/Ours-RAQ/experiments/eval_images/20261022_140841_k32_wwydw7h7/snr_10`

本次及后续样本使用 [update_labels_and_text.py](update_labels_and_text.py) 的 `new_figure` 入口，标题由 `DISPLAY` 映射统一为新名称。历史脚本和历史审阅目录保留原始记录，未改动。JSON 的源目录/方法键中出现 `JPEG`、`RAQ-VAE` 是来源标识，图中展示名称统一使用 JPEG-2000、RAQJSCC。

## 生成命令和验证

执行环境为 `/home/yi/.conda/envs/work/bin/python`，工作目录为 `/workspace/yi/work`。实际命令如下：

```bash
python -u /workspace/yi/work/comparison_labels_text_20261022/update_labels_and_text.py --stage prepare > /workspace/yi/work/comparison_labels_text_20261022/prepare.log 2>&1
python -u /workspace/yi/work/comparison_labels_text_20261022/update_labels_and_text.py --stage inspect > /workspace/yi/work/comparison_labels_text_20261022/inspection.log 2>&1
python -u /workspace/yi/work/comparison_labels_text_20261022/update_labels_and_text.py --stage labels > /workspace/yi/work/comparison_labels_text_20261022/labels.log 2>&1
python -u /workspace/yi/work/comparison_labels_text_20261022/update_labels_and_text.py --stage new --image kodim10.png --roi 152 28 312 188 > /workspace/yi/work/comparison_labels_text_20261022/new_figure.log 2>&1
python -u /workspace/yi/work/comparison_labels_text_20261022/update_labels_and_text.py --stage review
```

`prepare` 在修改前复制四个旧 PNG、四个旧 PDF 和两个历史脚本，保存完整原文件及 SHA-256；该阶段不覆盖已有快照。`labels` 对每个旧 PNG 仅清除并重绘两个方法标题，随后重新导出对应 PDF。逐像素确认旧图其他部分一致，包括原图、重建图、红框、放大图、指标和其余标题；每张实际修改 6458 个标题像素。

新增图的 4 个完整图面板在边框/红框标注外与输入像素完全一致，4 个放大面板与指定裁剪完全一致。全部 96 张输入图像在操作前后 SHA-256 均不变。所有成品使用真实重建文件，没有调色、锐化、去噪、模糊或超分辨率。候选拼图、新增成品及改名后的代表图均已人工查看。

## 本次文件变更清单与审阅

本次实际修改旧文件 8 个（上表后四行各自的 PNG、PDF），新增首行两个成品及本目录中的审阅文件。旧文件操作前已是未跟踪文件，因此普通 `git diff` 不会显示其本次差异；通过修改前完整快照及前后 SHA-256 保留实际变更记录。未操作 Git 暂存区、提交或推送。

- [renamed_existing_figures.json](renamed_existing_figures.json)：8 个实际修改旧文件的路径、前后 SHA-256、标题修改区域和像素验证结果。
- [before_manifest.json](before_manifest.json)：修改前 10 个完整快照的原路径、快照路径、哈希。
- [before/](before/)：上述四个旧 PNG、四个旧 PDF、两个历史脚本的完整修改前副本；逐项链接也在 `before_manifest.json` 中。
- [update_labels_and_text.py](update_labels_and_text.py)：本次新增的快照、候选比较、统一改名、新样本排版与审阅脚本。
- `README.md`：本次需求、输入、成品、命令、指标及文件说明。
- [source_manifest.json](source_manifest.json)：全部 96 张输入图像的路径、SHA-256。
- [candidate_regions.json](candidate_regions.json)：8 个文字/数字候选区域的坐标和真实局部 PSNR。
- [lettering_candidates_1.png](lettering_candidates_1.png)：建筑招牌文字候选。
- [lettering_candidates_2.png](lettering_candidates_2.png)：警示牌、帆船数字和墙面文字候选。
- [new_figure_metrics.json](new_figure_metrics.json)：新增帆船图的完整指标、尺寸、裁剪坐标、像素验证和输出哈希。
- [prepare.log](prepare.log)、[inspection.log](inspection.log)、[labels.log](labels.log)、[new_figure.log](new_figure.log)：实际运行日志；这些日志被现有 Git 忽略规则排除，已通过完整文件链接和清单保留。
- [git_status_before.txt](git_status_before.txt)、[git_status_after.txt](git_status_after.txt)：本次操作前后的工作区状态快照，核对新增未跟踪文件及已有改动。
- [changes.patch](changes.patch)：新增脚本、说明和 JSON（除产物清单本身）及历史脚本副本的真实完整新增补丁。
- [artifacts_manifest.json](artifacts_manifest.json)：本次修改/新增文件的完整路径、操作类型、大小和 SHA-256；提供完整清单内容并区分旧文件修改与新增未跟踪/被忽略文件。

先前两次审阅目录保留各自交付时的历史哈希。本次旧成品改名的前后记录以本目录中的快照、`renamed_existing_figures.json` 和 `artifacts_manifest.json` 为准。
