"""Validate the measured comparison and write its Chinese report."""
import csv
import json
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
MODELS = ("raq", "independent_rvq")
labels = {"raq": "普通 RAQ", "independent_rvq": "独立 RAQ-RVQ"}
data = {
    model: {
        condition: json.loads((OUT / (model + "_" + condition + ".json")).read_text())
        for condition in ("channel_snr3", "no_channel", "profile")
    } for model in MODELS
}
manifest = json.loads((OUT / "run_manifest.json").read_text())
timings = {(r["model"], r["condition"]): r["elapsed_seconds"] for r in manifest["runs"]}
metrics = {}
profiles = {}
for model in MODELS:
    metrics[model] = {
        "channel": {metric: data[model]["channel_snr3"]["results"]["3"][metric]
                    for metric in ("psnr", "ms_ssim")},
        "no_channel": {metric: data[model]["no_channel"]["results"]["no_channel"][metric]
                       for metric in ("psnr", "ms_ssim")},
    }
    profiles[model] = data[model]["profile"]
    p = profiles[model]
    assert p["num_images"] == 24
    assert p["input_shapes"] == ["(1, 3, 256, 256)"]
    for metric in ("psnr", "ms_ssim"):
        per_image_mean = statistics.mean(row[metric] for row in p["per_image_no_channel"])
        assert abs(per_image_mean - metrics[model]["no_channel"][metric]) < 1e-4
    assert data[model]["channel_snr3"]["modulation"] == "qpsk"
    assert data[model]["channel_snr3"]["ldpc_rate"] == 0.5

assert profiles["raq"]["target_layout"] == [8, 16]
assert profiles["independent_rvq"]["target_layout"] == [[4, 2], [8, 2]]
total = data["independent_rvq"]["channel_snr3"]["results"]["3"]["diagnostics"]["total"]
assert total["num_images"] == 24
assert total["payload_bits"] == 24 * 4096
assert total["channel_symbols"] == 24 * 4096
assert total["ldpc_padding_bits"] == 0

per_image = {
    model: {row["filename"]: row for row in profiles[model]["per_image_no_channel"]}
    for model in MODELS
}
rows = []
for filename in sorted(per_image["raq"]):
    a, b = per_image["raq"][filename], per_image["independent_rvq"][filename]
    rows.append({
        "filename": filename,
        "raq_psnr": a["psnr"], "independent_rvq_psnr": b["psnr"],
        "psnr_difference_raq_minus_rvq": a["psnr"] - b["psnr"],
        "raq_ms_ssim": a["ms_ssim"], "independent_rvq_ms_ssim": b["ms_ssim"],
        "ms_ssim_difference_raq_minus_rvq": a["ms_ssim"] - b["ms_ssim"],
    })
with (OUT / "per_image_no_channel.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
psnr_wins = sum(row["psnr_difference_raq_minus_rvq"] > 0 for row in rows)
ssim_wins = sum(row["ms_ssim_difference_raq_minus_rvq"] > 0 for row in rows)
a, b = profiles["raq"], profiles["independent_rvq"]
rvq_residuals = data["independent_rvq"]["no_channel"]["results"]["no_channel"]["diagnostics"]["rvq_quantization"]["per_scale"]
summary = {
    "metrics": metrics,
    "quality_difference_raq_minus_rvq": {
        condition: {metric: metrics["raq"][condition][metric] - metrics["independent_rvq"][condition][metric]
                    for metric in ("psnr", "ms_ssim")}
        for condition in ("channel", "no_channel")
    },
    "channel_psnr_loss_db": {
        model: metrics[model]["no_channel"]["psnr"] - metrics[model]["channel"]["psnr"]
        for model in MODELS
    },
    "no_channel_raq_psnr_wins": psnr_wins,
    "no_channel_raq_ms_ssim_wins": ssim_wins,
    "rvq_parameter_increase_percent": 100 * (b["parameters"]["total"] / a["parameters"]["total"] - 1),
    "rvq_mean_latency_increase_percent": 100 * (b["reconstruction_mean_ms"] / a["reconstruction_mean_ms"] - 1),
    "rvq_median_latency_increase_percent": 100 * (b["reconstruction_median_ms"] / a["reconstruction_median_ms"] - 1),
    "rvq_residual_mse_by_scale": [
        {"scale": record["scale"], "after_stage1": record["mean_residual_mse_energies"][0],
         "after_stage2": record["mean_residual_mse_energies"][1],
         "stage2_change_percent": 100 * (record["mean_residual_mse_energies"][1] /
                                         record["mean_residual_mse_energies"][0] - 1)}
        for record in rvq_residuals
    ],
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
lines = [
    "# 普通 RAQ 与独立 RAQ-RVQ 实测比较",
    "",
    "## 条件",
    "",
    "- 数据集：Kodak-256-transform-resize，24 张 RGB 图片，256×256，保持原尺寸。",
    "- GPU：物理 GPU 1，NVIDIA L40S；两模型依次运行。",
    "- 信道：AWGN，SNR=3 dB，LDPC n=256/k=128（码率 1/2），QPSK；随机种子 42。",
    "- 普通 RAQ：目标 K=[8,16]；独立 RVQ：目标 K=[[4,2],[8,2]]。",
    "- 两者均为 4096 bit/图、0.0625 payload bpp、4096 QPSK 符号/图，传输比 1/48。无 LDPC 填充。",
    "- 四次质量测试均调用原始脚本，不传入额外命令行参数；通过 JSON_OUTPUT 保存到本目录。",
    "- 环境中原有的 SIMVQ_* 与评测覆盖变量已清除，GPU_ID=1；脚本本身未修改。",
    "",
    "## 重建质量",
    "",
    "| 条件 | 普通 RAQ PSNR | 独立 RVQ PSNR | RAQ−RVQ | 普通 RAQ MS-SSIM | 独立 RVQ MS-SSIM |",
    "|---|---:|---:|---:|---:|---:|",
]
for condition, label in (("channel", "3 dB"), ("no_channel", "无信道")):
    x, y = metrics["raq"][condition], metrics["independent_rvq"][condition]
    lines.append(f"| {label} | {x['psnr']:.6f} dB | {y['psnr']:.6f} dB | {x['psnr']-y['psnr']:+.6f} dB | {x['ms_ssim']:.6f} | {y['ms_ssim']:.6f} |")
lines += [
    "",
    f"无信道逐图比较：普通 RAQ 在 {psnr_wins}/24 张图片上 PSNR 更高，在 {ssim_wins}/24 张图片上 MS-SSIM 更高。明细见 per_image_no_channel.csv。",
    "",
    "信道相对于无信道的平均 PSNR 损失：" + "；".join(
        f"{labels[model]} {summary['channel_psnr_loss_db'][model]:.6f} dB" for model in MODELS) + "。",
    "",
    "当前差距主要来自源重建质量；仅凭这两个不同 checkpoint 的结果，不能认定所有单级 RAQ 都优于 RVQ。",
    "",
    "## RVQ 残差诊断",
    "",
    "| 尺度 | 第一级后的平均残差 MSE | 第二级后的平均残差 MSE | 第二级带来的变化 |",
    "|---|---:|---:|---:|",
]
for record in summary["rvq_residual_mse_by_scale"]:
    lines.append(f"| {record['scale']} | {record['after_stage1']:.6f} | {record['after_stage2']:.6f} | {record['stage2_change_percent']:+.2f}% |")
lines += [
    "",
    "第二尺度的第二级量化增加了平均特征残差 MSE，说明当前 K=[8,2] 下第二级没有降低该尺度的特征误差。这是进一步排查的线索，但单独的特征 MSE 不能证明图像 PSNR 差距的具体原因。",
    "数据取自原脚本无信道结果 diagnostics.rvq_quantization；两模型 encoder 权重不同，不跨模型比较该特征 MSE。",
    "",
    "## 模型计算与内存",
    "",
    "batch=1、FP32、torch.inference_mode()；预热 5 次，对 24 张图片重复 3 次（共 72 次）。每次计时前后同步 GPU。",
    "计时覆盖 encoder、每图 RAQ 码本生成、量化、索引重建和 decoder；不含数据读取、指标计算、LDPC、调制、AWGN 或启动加载。",
    "",
    "| 指标 | 普通 RAQ | 独立 RVQ |",
    "|---|---:|---:|",
    f"| 模型总参数 | {a['parameters']['total']:,} | {b['parameters']['total']:,} |",
    f"| 推理平均延迟 | {a['reconstruction_mean_ms']:.3f} ms/图 | {b['reconstruction_mean_ms']:.3f} ms/图 |",
    f"| 推理中位数延迟 | {a['reconstruction_median_ms']:.3f} ms/图 | {b['reconstruction_median_ms']:.3f} ms/图 |",
    f"| 推理 P95 延迟 | {a['reconstruction_p95_ms']:.3f} ms/图 | {b['reconstruction_p95_ms']:.3f} ms/图 |",
    f"| 平均延迟对应吞吐 | {a['reconstruction_fps_from_mean']:.2f} 图/s | {b['reconstruction_fps_from_mean']:.2f} 图/s |",
    f"| PyTorch 峰值已分配显存 | {a['gpu_peak_allocated_mib']:.2f} MiB | {b['gpu_peak_allocated_mib']:.2f} MiB |",
    f"| checkpoint 文件大小 | {a['checkpoint_bytes']/2**20:.2f} MiB | {b['checkpoint_bytes']/2**20:.2f} MiB |",
    "",
    f"独立 RVQ 参数量增加 {summary['rvq_parameter_increase_percent']:.2f}%，平均延迟增加 {summary['rvq_mean_latency_increase_percent']:.2f}%。",
    "显存值是 PyTorch 已分配内存，包括模型、预加载的 24 张图片及重建临时张量；不等同于 nvidia-smi 的整个进程显存。",
    "",
    "## 原脚本整体耗时",
    "",
    "| 条件 | 普通 RAQ | 独立 RVQ |",
    "|---|---:|---:|",
]
for condition, label in (("channel_snr3", "3 dB"), ("no_channel", "无信道")):
    lines.append(f"| {label} | {timings[('raq',condition)]:.3f} s | {timings[('independent_rvq',condition)]:.3f} s |")
lines += [
    "",
    "整体耗时各测一次，包括 conda、Python、checkpoint 加载、数据读取、指标，以及信道测试中的 TensorFlow/LDPC CPU 计算；不能据此单独判断模型推理速度。",
    "GPU 1 在开始测试时为空闲，其他 GPU 有训练任务，共享 CPU 的负载可能影响耗时。速度数据用于本机近似比较。",
    "",
    "## 文件与复现",
    "",
    "- 四次评测原始结果和日志：raq_channel_snr3、independent_rvq_channel_snr3、raq_no_channel、independent_rvq_no_channel 对应的 .json/.log。",
    "- run_manifest.json：原始调用、条件、返回码与耗时。",
    "- *_profile.json：参数数目、72 次延迟、显存与逐图无信道质量。",
    "- summary.json 与 per_image_no_channel.csv：汇总和逐图对比。",
    "- run_comparison.py：依次运行原始脚本。",
    "- run_profiles.py：使用原始脚本设置环境，替换其最终 Python 入口为 profile_reconstruction.py；不改原始脚本。",
    "",
    "复现时从仓库目录运行本目录的 run_comparison.py、run_profiles.py，最后运行 summarize.py。复现会覆盖本目录结果。",
    "",
    "独立 RVQ 脚本的注释和默认输出文件名仍写着 4×64/LDPC 3/4，实际测试以当前参数 4×2/LDPC 1/2 为准。",
    "num_embeddings_list=[16,4] 是独立 RVQ 的名义元数据，实际布局以 rvq_k_lists=[[4,2],[8,2]] 为准。",
    "结果范围：当前两个 checkpoint、该 Kodak 数据集、当前码率以及一次 seed=42 的 3 dB 信道测试。",
]
(OUT / "comparison.md").write_text("\n".join(lines) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("Report: " + str(OUT / "comparison.md"))
