# ds2 评测输出变更与审阅记录（2026-10-22）

## 已确认的需求

- CBR 沿用“实际信道符号数 / 原图元素数（H×W×3）”，包含 LDPC 和调制填充，显示精确约分的分数和小数。
- 保留原始 MS-SSIM 均值；终端显示四位小数，dB 使用 `-10 * log10(1 - 四位小数均值)`。先平均，再按终端精度舍入，最后转换。
- 每次运行使用独立目录，按 SNR 保存全部重建图，默认沿用原图名，输出 RGB PNG。

## 本次代码与文档文件

| 文件 | 操作 | 用途 |
| --- | --- | --- |
| `scripts/eval/test_src256_raq32_2048_curriculum_ch256_res6_unet1_ds2.sh` | 修改 | 默认开启图片保存，支持 `SAVE_IMAGES` 和 `IMAGE_OUTPUT_DIR`。 |
| `scripts/eval/test_src256_raq32_2048_curriculum_ch256_res6_unet1_sweep.sh` | 修改 | 显式设置 `SAVE_IMAGES=0`，维持原有仅输出到终端的约定。 |
| `test_real.py` | 修改 | 精确 CBR、MS-SSIM dB、终端汇总与可选 JSON 字段、每次运行独立图片目录。 |
| `evaluation/quality.py` | 修改 | 在有信道和无信道评测中提供重建图片回调。 |
| `utils/image_export.py` | 新增 | 保存当前评测的重建张量，处理原图名、8 位 PNG 转换和同名文件冲突。 |
| `tests/test_eval_reporting.py` | 新增 | 验证统计口径、图片内容和命名、重复运行目录隔离。 |
| `docs/eval_ds2_output_changes_20261022.md` | 新增 | 使用说明、操作范围、验证命令及产物记录。 |

修改开始时，以上已跟踪代码文件均无已有差异。工作区原有的指标 CSV 删除、旧 `...unet1.sh` 删除和未跟踪 `...unet1_ds8.sh` 属于已有改动，不计入本次变更。未执行暂存、提交或推送。

## 使用方式

在项目根目录执行：

```bash
bash scripts/eval/test_src256_raq32_2048_curriculum_ch256_res6_unet1_ds2.sh
```

默认保存结构：

```text
experiments/eval_images/<北京时间>_k256_<唯一标识>/snr_0/kodim01.png
```

多个 SNR 在同一次运行目录下分别保存。无信道评测保存到 `no_channel/`。普通图片文件夹的数据集按原图名保存；相同主文件名会加序号，避免覆盖；不提供原图名的数据源使用顺序编号。

自定义输出根目录及 SNR：

```bash
SNRS="0 3 6" IMAGE_OUTPUT_DIR=/path/to/eval_images \
  bash scripts/eval/test_src256_raq32_2048_curriculum_ch256_res6_unet1_ds2.sh
```

临时关闭保存：

```bash
SAVE_IMAGES=0 bash scripts/eval/test_src256_raq32_2048_curriculum_ch256_res6_unet1_ds2.sh
```

直接调用 Python 时，传入 `--image-output-dir` 才会保存图片。设置原有的 `JSON_OUTPUT` 时，结果还包含 `ms_ssim_db`、各条件图片目录及 CBR 分数字符串。

PNG 从当前评测重建结果转换：`[-1, 1] -> [0, 1]`，截断到合法范围，乘 255 后四舍五入。不修改用于指标计算的张量。CBR 在无信道模式显示 `N/A`；MS-SSIM 均值显示为 `1.0000` 时，按该显示值计算的 dB 为正无穷。JSON 中的 `ms_ssim` 保留完整均值，`ms_ssim_db` 按四位小数均值计算。

## 验证命令与产物

工作目录：`/workspace/yi/work/Ours-RAQ`。

回归测试命令：

```bash
set -o pipefail
mkdir -p experiments/eval_images/verification_20261022
PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=2 \
  python -m unittest discover -s tests -p 'test_eval_reporting.py' -v \
  2>&1 | tee experiments/eval_images/verification_20261022/unit_tests.log
```

真实模型验证命令：

```bash
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=2 OMP_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 \
GPU_ID=0 SNRS='0 3' \
IMAGE_OUTPUT_DIR=/workspace/yi/work/Ours-RAQ/experiments/eval_images/verification_20261022/images \
JSON_OUTPUT=/workspace/yi/work/Ours-RAQ/experiments/eval_images/verification_20261022/channel_results.json \
  bash scripts/eval/test_src256_raq32_2048_curriculum_ch256_res6_unet1_ds2.sh \
  2>&1 | tee experiments/eval_images/verification_20261022/channel_run.log
```

关键参数：默认 ds2 的 `best_vq_deepsc.pth`、源码本 256、RAQ 目标码本 256、Kodak-256 全集、GPU 0、BPSK、LDPC n=256/R=0.5、AWGN、SNR 0/3 dB。

产物根目录为 `experiments/eval_images/verification_20261022/`，运行日志和图片属于自动生成产物；图片内容及哈希在 `artifact_manifest.json` 中记录。`changes.patch` 包含上表全部手工代码和文档改动，包含新增文件的完整内容。

## 初次验证结果（调整 dB 舍入口径前）

- 5 项回归测试全部通过；日志为 `unit_tests.log`。
- 两个 Shell 脚本通过 `bash -n`；本次修改的已跟踪文件通过定向 `git diff --check`。
- 完整 Kodak 测试集在 SNR=0、3 dB 下评测完成，进程退出码 0。
- 两个 SNR 目录各有 24 张 RGB PNG，共 48 张，文件名集合与输入一致，尺寸与对应原图一致，全部通过 PNG 完整性检查。
- 从结果 JSON 的整数计数重新约分 CBR，并重新计算 MS-SSIM 均值的 dB，均与输出一致。
- 已查看 `kodim01.png` 的原图和 SNR=0 重建图，色彩与归一化方向正常。

| SNR (dB) | MS-SSIM | MS-SSIM (dB) | PSNR (dB) | CBR |
| --- | --- | --- | --- | --- |
| 0 | 0.9930 | 21.5467 | 35.4720 | 4/3 (1.33333333) |
| 3 | 0.9936 | 21.9155 | 35.7371 | 4/3 (1.33333333) |

真实验证图片目录：

```text
experiments/eval_images/verification_20261022/images/20261022_135259_k256_tiuhip7i/snr_0/
experiments/eval_images/verification_20261022/images/20261022_135259_k256_tiuhip7i/snr_3/
```

额外自动生成文件为 `channel_results.json`、`artifact_manifest.json` 和 `changes.patch`；`channel_run.log`、`unit_tests.log` 为运行日志，可能被工作区通用日志忽略规则隐藏，但文件实际保留。

补丁生成方式：在工作区 Git 根目录对本次 4 个已跟踪文件运行 `git diff --binary -- <文件清单>`，对本次 3 个新增文本文件分别运行 `git diff --no-index -- /dev/null <文件>`，将实际补丁拼接到 `changes.patch`；未纳入用户原有改动和生成的图片。使用 `git apply --reverse --check` 检查补丁与当前文件一致，不实际应用补丁。

## 后续调整：dB 与终端四位小数一致

用户确认以终端显示的四位小数 MS-SSIM 计算 dB。原先 `21.9155 dB` 对应约 `0.99356646`，四位显示为 `0.9936`；调整后直接使用 `0.9936`，因此显示 `21.9382 dB`。

此次仅修改三个已有文件：

| 文件 | 关键改动 |
| --- | --- |
| `test_real.py` | `_ms_ssim_db` 先按 `.4f` 格式舍入均值再计算，有信道和无信道评测均使用此口径。 |
| `tests/test_eval_reporting.py` | 增加截图数值与舍入到 1 的边界检查，并验证完整评测输出使用舍入后的均值。 |
| `docs/eval_ds2_output_changes_20261022.md` | 记录新口径，并将原有验证数据标为调整前的历史结果。 |

此次修改前快照、增量补丁和测试日志位于 `experiments/reviews/20261022_ms_ssim_db_display_rounding/`。`before.json` 保存三份修改前完整文本；`changes.patch` 仅比较该快照与本次修改后的内容，避免混入早先工作。此处测试文件和本文档在 Git 中仍是先前创建的未跟踪文件，本次操作是继续修改它们。

验证命令：

```bash
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=2 \
  python -m unittest discover -s tests -p 'test_eval_reporting.py' -v \
  2>&1 | tee experiments/reviews/20261022_ms_ssim_db_display_rounding/unit_tests.log
```

此次验证结果：7 项回归检查全部通过；Python 语法检查和定向差异空白检查通过。截图中的 `0.9936` 现在对应 `21.9382 dB`；均值舍入后为 `1.0000` 时显示 `inf`，边界情况也通过测试。
