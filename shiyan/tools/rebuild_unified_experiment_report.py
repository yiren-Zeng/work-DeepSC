#!/usr/bin/env python3
"""Rebuild a single consolidated experiment report workbook.

The report is generated from source artifacts outside experiments/reports:
epoch metrics CSVs, codebook metrics CSVs, evaluation JSONs, checkpoints and
training logs. This makes experiments/reports safe to clean and recreate.
"""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
REPORTS = EXPERIMENTS / "reports"
OUTPUT_XLSX = REPORTS / "SimVQ_all_experiments_unified_report_20260606.xlsx"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def to_int(value, default=""):
    try:
        return int(float(value))
    except Exception:
        return default


def to_float(value, default=""):
    try:
        return float(value)
    except Exception:
        return default


def fmt_float(value, digits=4):
    if value == "" or value is None:
        return ""
    try:
        return round(float(value), digits)
    except Exception:
        return value


def fmt_pct(value):
    if value == "" or value is None:
        return ""
    return f"{float(value) * 100:.2f}%"


def parse_int_list(text: str) -> list[int]:
    if not text:
        return []
    return [int(item) for item in re.findall(r"\d+", text)]


def parse_experiment_name_from_metric(path: Path, suffix: str) -> str:
    return path.name[: -len(suffix)]


def collect_known_experiments() -> set[str]:
    names: set[str] = set()
    for path in EXPERIMENTS.glob("*_epoch_metrics.csv"):
        names.add(parse_experiment_name_from_metric(path, "_epoch_metrics.csv"))
    for path in EXPERIMENTS.glob("*_codebook_metrics.csv"):
        names.add(parse_experiment_name_from_metric(path, "_codebook_metrics.csv"))
    checkpoint_dir = ROOT / "checkpoints"
    if checkpoint_dir.exists():
        names.update(path.name for path in checkpoint_dir.iterdir() if path.is_dir())
    for path in list((EXPERIMENTS / "auto_results").glob("*.json")) + list((EXPERIMENTS / "interim_results").glob("**/*.json")):
        candidate = infer_experiment_from_path(path, names)
        if candidate:
            names.add(candidate)
    names.discard("")
    names.discard("observed_001_baseline")
    return names


def infer_experiment_from_path(path: Path, known_names: set[str]) -> str:
    text = str(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    checkpoint = str(data.get("checkpoint", ""))
    checkpoint_match = re.search(r"checkpoints/([^/]+)/", checkpoint)
    if checkpoint_match:
        return checkpoint_match.group(1)

    haystack = f"{path.name} {path.parent.name} {text}"
    for name in sorted(known_names, key=len, reverse=True):
        if name and name in haystack:
            return name

    stem = path.stem
    while True:
        cleaned = re.sub(r"_(no_channel|nochannel|snr\d+|bpsk|qpsk|ldpc-r\d+|real_chain)$", "", stem)
        if cleaned == stem:
            break
        stem = cleaned
    stem = re.sub(r"_best_epoch\d+.*$", "", stem)
    stem = re.sub(r"_epoch\d+$", "", stem)
    stem = stem.removesuffix("_final")
    aliases = {
        "dynswin_k64-256": "quality_v2_B_DynSwinEnhance_unet2_ds8x2_k64-256",
        "vitvq_nocompress_k64-256": "quality_v2_B_larger_ViTvqNoCompress_unet2_ds8x2_k64-256",
        "vitvq_nocompress_k64-256_final": "quality_v2_B_larger_ViTvqNoCompress_unet2_ds8x2_k64-256",
        "larger_cb16384-256": "quality_v2_B_larger_cb16384-256_unet2_ds8x2_k16384-256",
        "larger_cb4096-65536": "quality_v2_B_larger_cb4096-65536_unet2_ds8x2_k4096-65536",
        "larger_noquant": "quality_v2_B_larger_NoQuant_unet2_ds8x2_k64-256",
    }
    return aliases.get(stem, stem)


def infer_config_from_name(name: str) -> dict[str, object]:
    stage = ""
    if "quality_v1" in name:
        stage = "quality_v1"
    elif "_A_" in name or "quality_v2_A" in name:
        stage = "A"
    elif "_B_" in name or "quality_v2_B" in name:
        stage = "B"
    elif "_C_" in name or "quality_v2_C" in name:
        stage = "C"

    lower_name = name.lower()
    quantizer = "SimVQ"
    if "noquant" in lower_name:
        quantizer = "None / 无量化直通"
    elif "vitvqnocompress" in lower_name or "vitvq_nocompress" in lower_name:
        quantizer = "ViTvq NoCompress"
    elif "_VQ_" in name or name.endswith("_VQ"):
        quantizer = "Original VQ"

    unet = ""
    match = re.search(r"unet(\d+)", name)
    if match:
        unet = int(match.group(1))
    strides = []
    match = re.search(r"ds([0-9x]+)", name)
    if match:
        strides = [int(item) for item in match.group(1).split("x") if item]
    codebooks = []
    match = re.search(r"_k([0-9-]+)", name)
    if match:
        codebooks = [int(item) for item in match.group(1).split("-") if item]
    if unet and len(codebooks) == 1:
        codebooks = codebooks * int(unet)

    source_bpp = ""
    if strides and codebooks:
        cumulative = 1
        value = 0.0
        for stride, k in zip(strides, codebooks):
            cumulative *= stride
            value += math.log2(k) / (cumulative ** 2)
        source_bpp = value

    base_channels = ""
    if "larger" in name or "cb128-16" in name or "cb16384" in name or "cb4096" in name:
        base_channels = 128
    elif "DynSwinEnhance" in name:
        base_channels = 96
    elif name:
        base_channels = 64

    encoder_blocks = decoder_blocks = ""
    if "larger" in name or "cb128-16" in name or "cb16384" in name or "cb4096" in name:
        encoder_blocks = decoder_blocks = 4
    elif stage == "B":
        encoder_blocks = decoder_blocks = 2
    elif stage == "C":
        encoder_blocks = decoder_blocks = 2
    elif stage == "A":
        encoder_blocks = decoder_blocks = 1

    return {
        "阶段": stage,
        "量化器": quantizer,
        "U-Net层数": unet,
        "下采样步幅": "[" + ",".join(map(str, strides)) + "]" if strides else "",
        "总下采样": math.prod(strides) if strides else "",
        "码本配置": "[" + ",".join(map(str, codebooks)) + "]" if codebooks else "",
        "估算源端BPP": fmt_float(source_bpp, 6),
        "Base Channels": base_channels,
        "Encoder ResBlocks": encoder_blocks,
        "Decoder ResBlocks": decoder_blocks,
        "训练信道课程": "epoch<80 clean; 80-120 线性加入信道; >=120 全信道" if "quality_v2" in name else "",
        "测试建议": test_policy(name, quantizer, codebooks),
    }


def test_policy(name: str, quantizer: str, codebooks: list[int]) -> str:
    if "None" in quantizer:
        return "无量化方案：主要看 No-channel PSNR/MS-SSIM。"
    if codebooks == [128, 16] or codebooks == [64, 256]:
        return "LDPC 1/2 + BPSK: SNR=0dB；同时看 No-channel。"
    if codebooks == [16384, 256] or codebooks == [4096, 65536]:
        return "LDPC 1/2 + QPSK: SNR=4dB；同时看 No-channel。"
    return "按实验目的选择 No-channel 与对应 LDPC 链路。"


def load_log_summaries() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for path in sorted((EXPERIMENTS / "logs").glob("train_*.log")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        matches = re.findall(r"\[Info\] Experiment name:\s*(.+)", text)
        if not matches:
            matches = re.findall(r"Experiment:\s*(quality[^\n]+)", text)
        if not matches:
            continue
        name = matches[-1].strip()
        info = result.setdefault(name, {})
        info["训练日志"] = str(path.relative_to(ROOT))
        gpu = re.findall(r"physical GPU\s+([0-9]+)", text)
        if gpu:
            info["最近训练GPU"] = gpu[-1]
        run = re.findall(r"\[Info\] Experiment run ID:\s*(.+)", text)
        if run:
            info["最近Run ID"] = run[-1].strip()
        loaded = re.findall(r"从预训练权重加载:\s*([0-9]+) 个参数匹配,\s*([0-9]+) 个跳过", text)
        if loaded:
            info["预训练加载情况"] = f"匹配 {loaded[-1][0]}，跳过 {loaded[-1][1]}"
    return result


def load_epoch_summaries() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for path in sorted(EXPERIMENTS.glob("*_epoch_metrics.csv")):
        name = parse_experiment_name_from_metric(path, "_epoch_metrics.csv")
        rows = read_csv(path)
        if not rows:
            continue
        epochs = [to_int(row.get("epoch")) for row in rows if to_int(row.get("epoch")) != ""]
        latest_epoch = max(epochs) if epochs else ""
        best_rows = [row for row in rows if str(row.get("is_best", "")).strip() in {"1", "True", "true"}]
        if best_rows:
            best_row = min(best_rows, key=lambda row: to_float(row.get("best_val_recon", row.get("val_recon")), 1e9))
        else:
            best_row = min(rows, key=lambda row: to_float(row.get("val_recon"), 1e9))
        latest_row = max(rows, key=lambda row: to_int(row.get("epoch"), -1))
        result[name] = {
            "Epoch Metrics": str(path.relative_to(ROOT)),
            "已训练Epoch": latest_epoch,
            "最佳Epoch": to_int(best_row.get("epoch")),
            "最佳Val Recon": fmt_float(best_row.get("val_recon"), 8),
            "最佳Best Val Recon": fmt_float(best_row.get("best_val_recon", best_row.get("val_recon")), 8),
            "最新Train Recon": fmt_float(latest_row.get("train_recon"), 8),
            "最新Train VQ": fmt_float(latest_row.get("train_vq"), 8),
            "最新Channel Prob": latest_row.get("channel_prob", ""),
            "最新Phase": latest_row.get("phase", ""),
        }
    return result


def load_codebook_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(EXPERIMENTS.glob("*_codebook_metrics.csv")):
        name = parse_experiment_name_from_metric(path, "_codebook_metrics.csv")
        for row in read_csv(path):
            rows.append({
                "实验名称": name,
                "Run ID": row.get("run_id", ""),
                "Epoch": to_int(row.get("epoch")),
                "层": f"L{row.get('layer', '')}",
                "层编号": to_int(row.get("layer")),
                "码本大小K": to_int(row.get("codebook_size")),
                "活跃码字数": to_int(row.get("active_count")),
                "利用率": to_float(row.get("active_ratio")),
                "死码数": to_int(row.get("dead_count")),
                "困惑度": to_float(row.get("perplexity")),
                "困惑度/K": (
                    to_float(row.get("perplexity")) / to_float(row.get("codebook_size"))
                    if to_float(row.get("perplexity")) != "" and to_float(row.get("codebook_size")) not in {"", 0}
                    else ""
                ),
                "最小L2距离": to_float(row.get("min_l2_dist")),
                "坍缩码字数": to_int(row.get("collapse_count")),
                "坍缩比例": to_float(row.get("collapse_ratio")),
                "L2统计方式": "精确" if str(row.get("distance_stats_exact", "")) == "1" else f"采样{row.get('distance_reference_count', '')}",
                "源文件": str(path.relative_to(ROOT)),
            })
    return rows


def latest_codebook_by_experiment(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["实验名称"])].append(row)
    latest = {}
    for name, items in grouped.items():
        epoch = max(row["Epoch"] for row in items if row["Epoch"] != "")
        latest[name] = sorted([row for row in items if row["Epoch"] == epoch], key=lambda row: row["层编号"])
    return latest


def load_performance_rows(known_names: set[str]) -> list[dict[str, object]]:
    paths = []
    for base in [EXPERIMENTS / "auto_results", EXPERIMENTS / "interim_results", EXPERIMENTS / "snr0_results"]:
        if base.exists():
            paths.extend(base.glob("**/*.json"))
    paths.extend(path for path in EXPERIMENTS.glob("*.json") if path.name != "auto_eval_manifest.json")

    rows: list[dict[str, object]] = []
    for path in sorted(set(paths)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if "results" not in data:
            continue
        name = infer_experiment_from_path(path, known_names)
        checkpoint = str(data.get("checkpoint", ""))
        epoch_match = re.search(r"epoch[_-]?(\d+)|best_epoch(\d+)", checkpoint + " " + path.name)
        tested_epoch = ""
        if epoch_match:
            tested_epoch = next(group for group in epoch_match.groups() if group)
        modulation = data.get("modulation", "")
        ldpc_rate = data.get("ldpc_rate", "")
        num_embeddings = data.get("num_embeddings_list", "")
        if isinstance(num_embeddings, list):
            num_embeddings = "[" + ",".join(map(str, num_embeddings)) + "]"

        for condition, metric in data.get("results", {}).items():
            if not isinstance(metric, dict):
                continue
            if condition in {"no_channel", "nochannel"}:
                test_type = "No-channel / 无噪声"
                snr = ""
            else:
                test_type = "LDPC链路"
                snr = condition
            rows.append({
                "实验名称": name,
                "测试类型": test_type,
                "SNR(dB)": snr,
                "LDPC Rate": ldpc_rate,
                "调制": modulation,
                "测试Epoch": tested_epoch,
                "码本配置": num_embeddings,
                "PSNR(dB)": fmt_float(metric.get("psnr"), 4),
                "MS-SSIM": fmt_float(metric.get("ms_ssim"), 6),
                "Checkpoint": checkpoint,
                "结果源文件": str(path.relative_to(ROOT)),
            })
    rows.sort(key=lambda row: (str(row["实验名称"]), str(row["测试类型"]), str(row["SNR(dB)"]), str(row["结果源文件"])))
    return rows


def best_performance_summary(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["实验名称"])].append(row)

    def preferred_link_score(name: str, row: dict[str, object]) -> tuple[int, float]:
        cfg = infer_config_from_name(name)
        codebooks = parse_int_list(str(cfg["码本配置"]))
        modulation = str(row.get("调制", "")).lower()
        snr = str(row.get("SNR(dB)", ""))
        if codebooks in ([16384, 256], [4096, 65536]):
            preferred = modulation == "qpsk" and snr == "4"
        elif codebooks in ([64, 256], [128, 16], [16, 32]):
            preferred = modulation in {"bpsk", ""} and snr == "0"
        else:
            preferred = snr == "0"
        return (1 if preferred else 0, to_float(row.get("PSNR(dB)"), -1))

    result: dict[str, dict[str, object]] = {}
    for name, experiment_rows in grouped.items():
        summary: dict[str, object] = {}
        no_channel_rows = [
            row for row in experiment_rows
            if str(row["测试类型"]).startswith("No-channel")
        ]
        if no_channel_rows:
            best_no_channel = max(no_channel_rows, key=lambda row: to_float(row["PSNR(dB)"], -1))
            summary["最佳No-channel PSNR"] = best_no_channel["PSNR(dB)"]
            summary["最佳No-channel MS-SSIM"] = best_no_channel["MS-SSIM"]
            summary["No-channel结果源"] = best_no_channel["结果源文件"]

        link_rows = [
            row for row in experiment_rows
            if not str(row["测试类型"]).startswith("No-channel")
        ]
        if link_rows:
            best_link = max(link_rows, key=lambda row: preferred_link_score(name, row))
            summary["最佳链路PSNR"] = best_link["PSNR(dB)"]
            summary["最佳链路MS-SSIM"] = best_link["MS-SSIM"]
            summary["链路测试条件"] = (
                f"LDPC={best_link['LDPC Rate']}, {best_link['调制']}, "
                f"SNR={best_link['SNR(dB)']}dB"
            )
            summary["链路结果源"] = best_link["结果源文件"]
        result[name] = summary
    return result


def codebook_summary(latest_rows: list[dict[str, object]]) -> dict[str, object]:
    if not latest_rows:
        return {
            "码本监控Epoch": "",
            "平均利用率": "",
            "总死码数": "",
            "平均困惑度/K": "",
            "最小L2距离(全层最小)": "",
            "总坍缩码字数": "",
        }
    return {
        "码本监控Epoch": latest_rows[0]["Epoch"],
        "平均利用率": fmt_pct(sum(float(row["利用率"]) for row in latest_rows) / len(latest_rows)),
        "总死码数": sum(int(row["死码数"]) for row in latest_rows),
        "平均困惑度/K": fmt_pct(sum(float(row["困惑度/K"]) for row in latest_rows) / len(latest_rows)),
        "最小L2距离(全层最小)": fmt_float(min(float(row["最小L2距离"]) for row in latest_rows), 4),
        "总坍缩码字数": sum(int(row["坍缩码字数"]) for row in latest_rows),
    }


def build_rows():
    known = collect_known_experiments()
    log_info = load_log_summaries()
    epoch_info = load_epoch_summaries()
    codebook_rows = load_codebook_rows()
    latest_codebook = latest_codebook_by_experiment(codebook_rows)
    performance_rows = load_performance_rows(known)
    performance_summary = best_performance_summary(performance_rows)
    names = sorted(known | set(epoch_info) | set(latest_codebook) | set(performance_summary) | set(log_info))

    overview = []
    configs = []
    for name in names:
        cfg = infer_config_from_name(name)
        epoch = epoch_info.get(name, {})
        perf = performance_summary.get(name, {})
        cb = codebook_summary(latest_codebook.get(name, []))
        logs = log_info.get(name, {})
        checkpoint = ROOT / "checkpoints" / name
        status_parts = []
        if checkpoint.exists():
            status_parts.append("有checkpoint")
        if epoch.get("已训练Epoch", "") != "":
            status_parts.append(f"已训练{epoch['已训练Epoch']}轮")
        if latest_codebook.get(name):
            status_parts.append("有码本统计")
        if perf:
            status_parts.append("有测试结果")
        status = "；".join(status_parts) if status_parts else "仅发现零散记录"

        overview.append({
            "实验名称": name,
            "状态": status,
            "阶段": cfg["阶段"],
            "量化器": cfg["量化器"],
            "码本配置": cfg["码本配置"],
            "估算源端BPP": cfg["估算源端BPP"],
            "已训练Epoch": epoch.get("已训练Epoch", ""),
            "最佳Val Epoch": epoch.get("最佳Epoch", ""),
            "最佳Val Recon": epoch.get("最佳Val Recon", ""),
            "最佳No-channel PSNR": perf.get("最佳No-channel PSNR", ""),
            "最佳No-channel MS-SSIM": perf.get("最佳No-channel MS-SSIM", ""),
            "最佳链路PSNR": perf.get("最佳链路PSNR", ""),
            "最佳链路MS-SSIM": perf.get("最佳链路MS-SSIM", ""),
            "链路测试条件": perf.get("链路测试条件", ""),
            "码本监控Epoch": cb["码本监控Epoch"],
            "平均利用率": cb["平均利用率"],
            "总死码数": cb["总死码数"],
            "平均困惑度/K": cb["平均困惑度/K"],
            "最小L2距离": cb["最小L2距离(全层最小)"],
            "总坍缩码字数": cb["总坍缩码字数"],
            "最近Run ID": logs.get("最近Run ID", ""),
            "训练日志": logs.get("训练日志", ""),
        })

        configs.append({
            "实验名称": name,
            **cfg,
            "损失配置": "训练主重建损失 MSE；VQ loss 使用分层权重，初始随层递增、后期统一。",
            "信道训练": cfg["训练信道课程"],
            "测试协议": cfg["测试建议"],
            "预训练加载": logs.get("预训练加载情况", ""),
            "Epoch Metrics源": epoch.get("Epoch Metrics", ""),
            "Checkpoint目录": f"checkpoints/{name}" if checkpoint.exists() else "",
        })

    latest_codebook_rows = []
    for name, rows in sorted(latest_codebook.items()):
        for row in rows:
            pretty = dict(row)
            pretty["利用率"] = fmt_pct(pretty["利用率"])
            pretty["困惑度/K"] = fmt_pct(pretty["困惑度/K"])
            pretty["坍缩比例"] = fmt_pct(pretty["坍缩比例"])
            pretty["困惑度"] = fmt_float(pretty["困惑度"], 4)
            pretty["最小L2距离"] = fmt_float(pretty["最小L2距离"], 4)
            latest_codebook_rows.append(pretty)

    codebook_history_rows = []
    for row in codebook_rows:
        pretty = dict(row)
        pretty["利用率"] = fmt_pct(pretty["利用率"])
        pretty["困惑度/K"] = fmt_pct(pretty["困惑度/K"])
        pretty["坍缩比例"] = fmt_pct(pretty["坍缩比例"])
        pretty["困惑度"] = fmt_float(pretty["困惑度"], 4)
        pretty["最小L2距离"] = fmt_float(pretty["最小L2距离"], 4)
        codebook_history_rows.append(pretty)

    training_rows = []
    for name in names:
        row = {"实验名称": name}
        row.update(epoch_info.get(name, {}))
        row.update(log_info.get(name, {}))
        training_rows.append(row)

    guide_rows = [
        {"项目": "本文件如何生成", "说明": "先清空 experiments/reports，再从 reports 之外的原始实验产物重建；不会删除 checkpoint、训练日志、metrics CSV、测试 JSON。"},
        {"项目": "No-channel PSNR", "说明": "无信道噪声、直接由模型重建得到的 PSNR。不同测试集或 Resize/切块方式会改变数值，需看结果源文件。"},
        {"项目": "链路 PSNR", "说明": "经过 LDPC + BPSK/QPSK + AWGN 的测试结果，表中保留 SNR、调制和 LDPC rate。"},
        {"项目": "利用率", "说明": "活跃码字数 / 码本大小 K。低利用率说明容量没有充分用起来。"},
        {"项目": "困惑度/K", "说明": "有效使用码字数占 K 的比例，比单看利用率更能反映分布是否均衡。"},
        {"项目": "死码数", "说明": "统计批次内从未被选择的码字数量。"},
        {"项目": "最小L2距离/坍缩", "说明": "码字间最近邻距离过小会被计入坍缩风险；大码本可能使用采样统计。"},
    ]
    return overview, configs, performance_rows, latest_codebook_rows, codebook_history_rows, training_rows, guide_rows


def selected(rows, headers):
    return [[row.get(header, "") for header in headers] for row in rows]


def sheet(name, title, subtitle, rows, widths=None, tab_color=0x1F4E78):
    if not rows:
        rows = [{"说明": "没有可用记录"}]
    headers = list(rows[0])
    return {
        "name": name,
        "title": title,
        "subtitle": subtitle,
        "headers": headers,
        "rows": selected(rows, headers),
        "widths": widths or {},
        "tab_color": tab_color,
    }


def write_workbook(sheets):
    helper = ROOT / "tools" / "write_pretty_experiment_report_uno.py"
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as handle:
        json.dump({"sheets": sheets}, handle, ensure_ascii=False)
        payload = Path(handle.name)
    try:
        subprocess.run(["/usr/bin/python3", str(helper), str(payload), str(OUTPUT_XLSX)], check=True)
    finally:
        payload.unlink(missing_ok=True)


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    overview, configs, performance, cb_latest, cb_history, training, guide = build_rows()
    subtitle = f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}；数据来源为 reports 外部原始实验产物。"
    sheets = [
        sheet("01_总览", "SimVQ 全部实验统一总览", subtitle, overview,
              {"实验名称": 64, "状态": 36, "训练日志": 70, "链路测试条件": 30}, 0x1F4E78),
        sheet("02_实验配置", "所有方案实验配置明细", "配置由实验名、脚本约定和训练日志综合整理，便于后续优化对照。", configs,
              {"实验名称": 64, "损失配置": 64, "信道训练": 48, "测试协议": 54, "Checkpoint目录": 72}, 0x70AD47),
        sheet("03_测试性能", "测试性能明细", "每行对应一个 JSON 测试结果中的一个条件；请结合结果源文件区分数据集与测试脚本。", performance,
              {"实验名称": 64, "Checkpoint": 90, "结果源文件": 90}, 0xED7D31),
        sheet("04_码本最新", "码本利用率最新统计", "每个实验取最新码本监控 Epoch，逐层展示利用率、困惑度、死码和坍缩。", cb_latest,
              {"实验名称": 64, "源文件": 76, "Run ID": 48}, 0x8064A2),
        sheet("05_码本历史", "码本利用率完整历史", "保留所有 codebook_metrics.csv 中的历史记录，可按实验、Epoch、层筛选。", cb_history,
              {"实验名称": 64, "源文件": 76, "Run ID": 48}, 0xA5A5A5),
        sheet("06_训练摘要", "训练曲线摘要", "来自 *_epoch_metrics.csv 和训练日志的最近运行信息。", training,
              {"实验名称": 64, "Epoch Metrics": 76, "训练日志": 76, "预训练加载情况": 24}, 0x5B9BD5),
        sheet("07_说明", "字段与指标说明", "这张表用于解释关键字段，避免后续看表时混淆测试协议。", guide,
              {"项目": 24, "说明": 100}, 0xFFC000),
    ]
    write_workbook(sheets)
    print(OUTPUT_XLSX)
    print(f"overview_rows={len(overview)} performance_rows={len(performance)} codebook_history_rows={len(cb_history)}")


if __name__ == "__main__":
    main()
