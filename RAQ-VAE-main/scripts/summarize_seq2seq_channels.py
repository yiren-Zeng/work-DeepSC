"""Validate and summarize the three same-checkpoint, nearest-rate evaluations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Output must be inside RAQ-VAE-main")
    reports = [(Path(path).resolve(), json.loads(Path(path).read_text())) for path in args.results]
    assert len(reports) == 3
    first = reports[0][1]["metadata"]
    rows = []
    for path, report in reports:
        meta = report["metadata"]
        for key in ("checkpoint_sha256", "dataset_sha256", "dataset_order", "seed",
                    "cuda_visible_devices", "reference_source_sha256"):
            assert meta[key] == first[key], key
        assert len(report["per_image"]) == report["num_images"] == 24
        assert report["totals"]["budget_padding_bits"] == 0
        assert meta["rate_plan"]["policy"] == "nearest"
        snapshot = Path(meta["snapshot"])
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == meta["checkpoint_sha256"]
        selected = meta["rate_plan"]["selected"]
        assert selected["absolute_ratio_error"] == min(
            candidate["absolute_ratio_error"] for candidate in meta["rate_plan"]["candidates"]
        )
        assert all(image["transmission_ratio"] == report["transmission_ratio"]
                   for image in report["per_image"])
        assert abs(statistics.mean(image["psnr"] for image in report["per_image"])
                   - report["metrics"]["psnr"]) < 1e-12
        rows.append({
            "ldpc_rate": meta["ldpc"]["rate"], "modulation": meta["modulation"],
            "snr_db": meta["snr_db"],
            "K": meta["target_K_shared"], "target_ratio": meta["target_transmission_ratio"],
            "actual_ratio": report["transmission_ratio"],
            "actual_ratio_reciprocal": 1 / report["transmission_ratio"],
            "relative_budget_difference_percent": 100 * (report["transmission_ratio"] /
                                                          meta["target_transmission_ratio"] - 1),
            "psnr": report["metrics"]["psnr"], "ms_ssim": report["metrics"]["ms_ssim"],
            "no_channel_psnr": report["metrics"]["no_channel_psnr"],
            "no_channel_ms_ssim": report["metrics"]["no_channel_ms_ssim"],
            "source_ber": report["source_ber"],
            "source_payload_bits_per_image": report["totals"]["source_payload_bits"] // 24,
            "ldpc_padding_bits_per_image": report["totals"]["ldpc_padding_bits"] // 24,
            "channel_symbols_per_image": report["totals"]["channel_symbols"] // 24,
            "results_json": str(path),
        })
    for path, expected in first["reference_source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps({
        "checkpoint_sha256": first["checkpoint_sha256"],
        "checkpoint_epoch_number": first["checkpoint_epoch_number"],
        "seed": first["seed"], "images": 24,
        "gpu": first["gpu"], "checks": "checkpoint, image order/content, code hashes, rates and means verified",
        "results": rows,
    }, indent=2) + "\n")
    with (output / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = [
        "# Seq2Seq RAQ：三种链路，最接近1/48的码本\n",
        f"统一使用第{first['checkpoint_epoch_number']}轮冻结权重，Kodak 24张，AWGN，"
        f"各组SNR见下表，seed={first['seed']}，同一张GPU。数据加载和PSNR/MS-SSIM函数直接复用shiyan。\n",
        "| LDPC码率 | 调制 | SNR/dB | 共享K | 实际传输比 | PSNR/dB | MS-SSIM | 有效载荷BER |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        rate = "1/2" if row["ldpc_rate"] == 0.5 else "3/4"
        text.append(f"| {rate} | {row['modulation'].upper()} | {row['snr_db']:g} | {row['K']} | "
                    f"1/{row['actual_ratio_reciprocal']:.4f} | {row['psnr']:.6f} | "
                    f"{row['ms_ssim']:.6f} | {row['source_ber']:.8f} |")
    text.extend([
        "\n本次按实际传输比与1/48的绝对差选K，允许超过预算。1/38.4比目标多用25%的信道符号；"
        "1/56.8889比目标少用15.625%。三组均没有恰好达到1/48。\n",
        "传输比定义：复数信道符号数/(3×H×W)。没有人为补位来凑预算，只有LDPC分块及调制必需填充。"
        "候选K为2、4、8、16、32、64：现有定长打包器仅对2次幂K无损；非2次幂K若改为向上取整位宽，"
        "也只会共享相邻更大2次幂K的码率，不能提供更接近目标的新定长码率。\n",
        "## 无信道对照与逐图文件\n",
    ])
    for row in rows:
        text.append(f"- LDPC={row['ldpc_rate']:g}, {row['modulation'].upper()}, SNR={row['snr_db']:g} dB, K={row['K']}："
                    f"无信道PSNR={row['no_channel_psnr']:.6f} dB，MS-SSIM={row['no_channel_ms_ssim']:.6f}。"
                    f"[完整结果]({row['results_json']})；"
                    f"[逐图CSV]({Path(row['results_json']).parent / 'per_image.csv'})。")
    text.append(f"\n本表三组均在物理GPU{first['cuda_visible_devices']}上运行；仅汇总显式指定的结果文件。"
                "训练进程和模型文件未改变。\n")
    (output / "REPORT.md").write_text("\n".join(text))
    print(json.dumps(rows, indent=2))
    print(f"REPORT {output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
