本轮重新比较 42 组候选，在保留原模型、权重和原始 Kodak 分辨率的条件下，优先提高两张展示图的整图 PSNR，同时检查 MDRQ 的细节相对优势。JPEG-2000 暂不参与。

最终选择仍为 **CBR=7/192≈0.0364583（约 1/27.43）**，展示图片改为 **kodim03（帽子）与 kodim12（海滩）**。这是本轮候选中按“两张图片都高于两个对照，且两张中较低的 MDRQ 整图 PSNR 尽量高”规则选出的组合。更高的 CBR=5/128 也已测试，但没有找到更好的合格两图组合。

| 图片 | MDRQ PSNR / MS-SSIM(dB) | VQ-DeepSC PSNR / MS-SSIM(dB) | RAQ-VAE PSNR / MS-SSIM(dB) |
| --- | ---: | ---: | ---: |
| kodim03.png | 28.6444 / 13.0205 | 28.2172 / 12.3971 | 26.3414 / 10.9846 |
| kodim12.png | 27.9985 / 12.1328 | 27.7060 / 11.8057 | 25.3838 / 10.0933 |

帽子图 MDRQ 分别领先两个对照 0.4272、2.3029 dB；海滩图分别领先 0.2925、2.6147 dB。所有数值均为完整原始分辨率图像的原生评测记录，MS-SSIM(dB)=-10*log10(1-MS-SSIM)。海滩图四舍五入显示 28.00 dB，原始值是 27.9984998703 dB。

较上一轮展示的 kodim04、kodim15（MDRQ 分别为 26.8988、26.4775 dB），本次换成了整图 PSNR 更高的图片；不是同一张图在同一配置下发生质量提升。帽子 ROI 重点比较绿色帽子的外形和接缝；海滩 ROI 比较粉色脚蹼、脚部与水面的轮廓。MDRQ 的部分形状更接近原图，但细字、布料纹理和水面高频细节仍有损失，不能把更高 PSNR 等同于完整恢复所有纹理。

共同条件：`/workspace/yi/work/Kodak` 全部 24 张原始 RGB 图片，18 张 768×512、6 张 512×768；不缩放、不裁剪模型输入。AWGN、SNR=10 dB、QPSK、LDPC 1/2（n=256、k=128），模型与信道种子 42，CSCR 关闭。所选配置 MDRQ 分配为 **`2,32;8,2`**，逐尺度乘积为 **[64,16]**，与 VQ-DeepSC 原 `test_k64_16_last2_cascade_200ep.sh`、RAQ-VAE 的 bottom/top K 相同。模型结构、权重、量化和通信算法没有修改。

每图 393216 像素、1179648 个 RGB 标量；底层 6144 个索引、顶层 1536 个索引。有效载荷 `6144×6+1536×4=43008 bit`，LDPC 后 86016 bit，QPSK 为 43008 个复符号，因此实际 `CBR=43008/1179648=7/192`。三种方案的预算及编码调制填充均为零，所选结果均零误码。

图版保持两行四列：Original、VQ-DeepSC、RAQ-VAE、MDRQ。上排完整图与红框，下排相同坐标的局部放大，只在底部标一组完整图指标。全部方法使用相同显示比例与最近邻显示，未锐化、修复或替换原始模型输出。

- [帽子图 PNG](selected/kodim03_comparison.png) / [PDF](selected/kodim03_comparison.pdf)，ROI=(448,208,160,160)。
- [海滩图 PNG](selected/kodim12_comparison.png) / [PDF](selected/kodim12_comparison.pdf)，ROI=(224,288,160,160)。
- [完整数值与图片校验值](selected/comparison_metrics.json)、[CSV](selected/comparison_metrics.csv)、[所选配置和输出目录](selection.json)。

所选配置的全部 24 张原尺寸信道重建 PNG 已分别保存到各项目：

- [MDRQ](/workspace/yi/work/RAQ-RVQ/generated_images/high_psnr_native_kodak_20261019_k64_16_qpsk_r12/seed_42)
- [VQ-DeepSC](/workspace/yi/work/VQ-DeepSC/generated_images/high_psnr_native_kodak_20261019_k64_16_qpsk_r12/seed_42)
- [RAQ-VAE](/workspace/yi/work/RAQ-VAE-main/generated_images/high_psnr_native_kodak_20261019_k64_16_qpsk_r12/seed_42)

搜索复用了上一轮 6 个 MDRQ 配置、8 个匹配基线结果，另补测 36 种 MDRQ 分配。有效 K=[64,64] 的 25 种两级分解、有效 K=[64,16] 的 15 种两级分解全部比较，另保留 K=[16,16] 和 [32,16] 对照，共 42 个比较配置、50 组不同模型/配置的有效结果。相同物理条件、码本与固定权重的基线通过 baseline_case_id 复用，完整来源可查。这个有限范围不代表所有可能参数的全局最优。

[full_psnr_ranking.json](full_psnr_ranking.json) 保留 42 组配置下全部逐图 PSNR、MS-SSIM、差值和两图排序，包括未入选结果。[comparison_all.json](comparison_all.json) 与持久保存的 [reports.tar.gz](reports.tar.gz) 保留原生结果、完整命令和日志；[reports/](reports/) 为临时浏览副本。[SEARCH_NOTES.md](SEARCH_NOTES.md) 说明筛选规则、局部检查和调度恢复。末尾八组曾因重复调度发生导出竞争，相关文件已经归档，正式结果采用独立重跑后的数据；后续运行以每组文件锁避免重复写入。

选择两张图片用于展示，不把它们当作整个数据集的平均结论。所选配置全部 24 图的平均 PSNR 仍为 MDRQ 25.1693 dB、VQ-DeepSC 25.1214 dB、RAQ-VAE 23.5842 dB；本轮不强制多种子平均差值 0.2 dB。

原始大小适配独立于原项目源码：MDRQ/VQ-DeepSC 使用原评测器及无增强输出观察钩子；RAQ-VAE 的 [native_vae_adapter.py](native_vae_adapter.py) 仅推广原评测中写死的像素/索引计数、码率分母与输入形状检查。[适配差异](adapter_audit/vae_evaluator_393216_pixels.diff) 可以逐行检查。[旧 256×256 控制实验](adapter_audit/parity_256.json) 已确认适配版与原版的 24 张 PNG、逐图指标与比特计数完全一致；256 数据不计入本轮比较。

[verification.json](verification.json) 核查原始分辨率、逐尺度有效 K、实际 CBR、信道、FEC、无填充、PNG 哈希和完整图指标汇总，以及 988 个原始源文件、6 个权重与 24 张原图的 SHA256 不变。唯一出现变化的是 MDRQ 启动脚本（修改时间 19:18:32，本轮未写入该脚本）；其初始与当前 SHA256、当前内容及独立核验结论见 [launcher_change_review.json](adapter_audit/launcher_change_review.json)。初始全文未保存，因此不声称已恢复逐行差异或全部 989 个源文件字节不变。所有原生结果仍显式满足本次 AWGN、10 dB、QPSK、LDPC、码本和预算条件，模型及通信实现未变。[manifest.json](manifest.json) 保存配置、输入顺序与校验值；[environment.json](environment.json) 保存运行环境。

批量候选中间 PNG 暂存于 /tmp/visual_cbr_native_high_psnr_20261019 及复用的上一轮临时目录，最终 72 张重建 PNG、两张 PNG/PDF 图版、原图样本、全部数值和复现工具持久保存于 /workspace/yi/work。本次没有覆盖上一轮展示文件。

重新排版，无需依赖中间目录：

```bash
/home/yi/.conda/envs/work/bin/python /workspace/yi/work/visual_cbr_native_high_psnr_20261019/render_native.py --from-published
```

重新评测所选配置（归档本轮输出后重算；复用的旧实验目录会先分离，保留旧结果）：

```bash
/home/yi/.conda/envs/work/bin/python /workspace/yi/work/visual_cbr_native_high_psnr_20261019/native_experiments.py run --case-id k64_16_qpsk_r12 --rerun
/home/yi/.conda/envs/work/bin/python /workspace/yi/work/visual_cbr_native_high_psnr_20261019/render_native.py
/home/yi/.conda/envs/work/bin/python /workspace/yi/work/visual_cbr_native_high_psnr_20261019/verify_native.py
```

全部 42 个比较配置的完整复现可省略 --case-id。基线只在所属配置运行一次，其他分配复用同一条件的基线。需要 CUDA 与上述环境，不训练新权重。
