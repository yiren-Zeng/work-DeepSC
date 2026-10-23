本次已直接使用 `/workspace/yi/work/Kodak` 的全部 24 张原始图片，保留 18 张 768×512 和 6 张 512×768 的原始尺寸及方向，没有缩放、裁剪、分块或拼接模型输入。JPEG-2000 按本轮要求排除。

筛选了六组匹配配置、四个实际 CBR，共完成 14 次独立模型评测，保存 336 张原始分辨率的最终信道重建图。当前选择 **CBR=7/192≈0.0364583**、`kodim04.png` 和 `kodim15.png`，展示红帽网格和眼部轮廓的局部优势。完整图清晰度比先前 256×256 展示更适合比较，但细小针织纹理、睫毛和皮肤纹理仍有损失，模型输出中的局部伪影也完整保留。

共同条件为 AWGN、SNR=10 dB、QPSK、LDPC 1/2（n=256、k=128）、模型和信道种子 42。MDRQ 使用原 RAQ-RVQ 模型及固定权重，两级分配为 **`2,32;8,2`**，有效码本逐尺度乘积为 **`[64,16]`**，与 VQ-DeepSC 的原 `test_k64_16_last2_cascade_200ep.sh` 和 RAQ-VAE 的 bottom/top K 一致。没有训练新模型或更换既有模型权重。

每张图片有 393216 个像素，即 1179648 个 RGB 标量；底层 6144 个索引、顶层 1536 个索引。有效载荷为 `6144×6+1536×4=43008 bit`，编码后为 86016 bit，QPSK 发送 43008 个复符号。因此实际 `CBR=43008/1179648=7/192`。三个方案均零误码，LDPC、调制和人为预算填充均为零。

下面每个单元格依次为 **PSNR / MS-SSIM(dB)**，均取完整原始分辨率图片的原生评测记录；`MS-SSIM(dB)=-10*log10(1-MS-SSIM)`。

| 图片 | MDRQ | VQ-DeepSC | RAQ-VAE |
| --- | ---: | ---: | ---: |
| kodim04.png | 26.8988 / 11.0736 | 26.4438 / 10.0237 | 25.3645 / 9.4752 |
| kodim15.png | 26.4775 / 12.2255 | 25.9915 / 11.3084 | 24.5130 / 10.5470 |

- `kodim04`：MDRQ 的帽檐和网格布局更接近原图；VQ-DeepSC 的部分网格变形，RAQ-VAE 抹平了较多网格。单个针脚和头发仍模糊。
- `kodim15`：MDRQ 较好地保留了虹膜反光、眼球形状和眼睑轮廓；两个对照的形状或颜色偏差更明显。睫毛与皮肤纹理仍模糊。

这两张是按局部结构优势选择的展示样本，没有用它们代替全数据集结论。本次按照最新要求，不强制全数据集多种子平均 PSNR 差值达到 0.2 dB。所选配置的 24 图平均 PSNR 为 MDRQ 25.1693 dB、VQ-DeepSC 25.1214 dB、RAQ-VAE 23.5842 dB。

新版对比图保持两行四列，列顺序为 Original、VQ-DeepSC、RAQ-VAE、MDRQ；上方完整图，下方相同位置的局部放大图，保留红框，底部只显示一组完整图指标。方法名以外的实验参数和局部说明文字已移到记录文件。所有方法使用相同 ROI 和显示比例，最近邻显示，没有锐化、修复或替换模型输出。

- [kodim04 PNG](selected/kodim04_comparison.png) / [PDF](selected/kodim04_comparison.pdf)，ROI `(x,y,w,h)=(160,48,160,160)`。
- [kodim15 PNG](selected/kodim15_comparison.png) / [PDF](selected/kodim15_comparison.pdf)，ROI `(400,112,128,128)`。
- [原始数值与局部诊断](selected/comparison_metrics.json)、[CSV](selected/comparison_metrics.csv)、[所选配置与图片来源](selection.json)。局部 PNG PSNR 仅供诊断，图中显示的是原生完整图指标。

所选配置的全部 24 张原尺寸重建图已经持久保存到各项目目录，每张 PNG 的 SHA256 均核对过：

- [MDRQ](/workspace/yi/work/RAQ-RVQ/generated_images/native_kodak_20261019_k64_16_qpsk_r12/seed_42)
- [VQ-DeepSC](/workspace/yi/work/VQ-DeepSC/generated_images/native_kodak_20261019_k64_16_qpsk_r12/seed_42)
- [RAQ-VAE](/workspace/yi/work/RAQ-VAE-main/generated_images/native_kodak_20261019_k64_16_qpsk_r12/seed_42)

原始大小适配独立于原方案源码。MDRQ 和 VQ-DeepSC 使用原模型与原评测，仅指定数据目录、关闭测试缩放，并在原质量函数计算之后保存解码器输出的 detached CPU 副本。RAQ-VAE 的旧评测把像素数、索引数和形状检查写死为 256×256；[native_vae_adapter.py](native_vae_adapter.py) 只推广这些检查及码率分母，保留原模型、码本生成、索引序列化、LDPC、调制、AWGN 和质量函数。[适配差异](adapter_audit/vae_evaluator_393216_pixels.diff) 可以逐行检查。

为验证适配没有改变结果，单独在旧 256×256 数据上进行了控制实验，24 张 PNG、逐图指标及发送比特计数与原评测**完全一致**：[parity_256.json](adapter_audit/parity_256.json)。256 数据只用于这个控制实验，不计入本次原始分辨率比较。

当前 MDRQ 原 Bash 脚本的 CSCR 默认值与 10 月 13 日方案不同，本次正式结果显式设置 **`USE_CSCR=0`**；RAQ-VAE 也显式使用 `--no-cscr`。初始 CSCR 诊断结果被排除并重跑 MDRQ。零误码下重跑 PNG 与初始诊断逐字节相同，但正式记录采用关闭 CSCR 的运行，保持原编码标签顺序。

[verification.json](verification.json) 确认比较条件匹配，989 个原始代码文件、6 个权重和 24 张原始图片 SHA256 均未变化。[manifest.json](manifest.json) 保存全部配置与输入尺寸、原生顺序及校验值；[comparison_all.json](comparison_all.json) 和 [reports/](reports/) 保存六组比较的完整数值、参数和日志。相同物理条件及有效 K 的 MDRQ 分配变体复用同一组基线结果，来源记录在 `baseline_case_id`。运行环境版本见 [environment.json](environment.json)。

工作目录曾遇到配额限制，因此批量中间 PNG 和早期重复快照暂存在 `/tmp/visual_cbr_native_kodak_20261019`，`cases/`、`inspection/` 指向该处。最终选定的全部 72 张 PNG、两张 PNG/PDF 图表、原始样本、所有实验数值及复现脚本已经保存在 `/workspace/yi/work`，不依赖临时目录长期存在。未清理任何原始数据、既有模型或旧实验结果。

重新运行所选配置（`--rerun` 会把本次同配置旧输出归档后重算；去掉该选项会复用完整结果）：

```bash
/home/yi/.conda/envs/work/bin/python /workspace/yi/work/visual_cbr_native_kodak_20261019/native_experiments.py run --case-id k64_16_qpsk_r12 --rerun
/home/yi/.conda/envs/work/bin/python /workspace/yi/work/visual_cbr_native_kodak_20261019/render_native.py
/home/yi/.conda/envs/work/bin/python /workspace/yi/work/visual_cbr_native_kodak_20261019/verify_native.py
```

全部六组重新评测时省略 `--case-id`。单独重新排版可使用下方命令，它直接读取各项目中持久保存的 PNG 和原始逐图指标，临时目录清理后仍可使用：

```bash
/home/yi/.conda/envs/work/bin/python /workspace/yi/work/visual_cbr_native_kodak_20261019/render_native.py --from-published
```
