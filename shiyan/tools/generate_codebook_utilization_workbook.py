#!/usr/bin/env python3
"""Generate a styled workbook that consolidates all recorded codebook metrics."""

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from export_codebook_metrics_from_logs import parse_log


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
REPORTS = EXPERIMENTS / "reports"
OUTPUT_XLSX = REPORTS / "SimVQ_codebook_utilization_all_experiments_20260602.xlsx"
OUTPUT_SUMMARY_CSV = REPORTS / "SimVQ_codebook_utilization_latest_summary_20260602.csv"
OUTPUT_HISTORY_CSV = REPORTS / "SimVQ_codebook_utilization_history_20260602.csv"


def canonical_experiment(name: str, source: str = "") -> str:
    if name:
        return name
    if "observed_001" in source:
        return "observed_001"
    if "quality-v1-k64" in source:
        return "quality_v1_unet2_ds4x2_k64"
    return "历史日志_实验名缺失"


def number(value, cast=float, default=""):
    if value in ("", None):
        return default
    return cast(value)


def load_log_rows() -> list[dict[str, object]]:
    rows = []
    for path in sorted((EXPERIMENTS / "logs").glob("*.log")):
        for row in parse_log(path):
            row["experiment"] = canonical_experiment(str(row["experiment"]), str(path))
            row["source_type"] = "训练日志"
            rows.append(row)
    return rows


def load_csv_rows() -> list[dict[str, object]]:
    rows = []
    suffix = "_codebook_metrics.csv"
    for path in sorted(EXPERIMENTS.glob(f"*{suffix}")):
        experiment = path.name[:-len(suffix)]
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({
                    "experiment": experiment,
                    "run_id": row["run_id"],
                    "epoch": int(row["epoch"]),
                    "layer": int(row["layer"]),
                    "codebook_size": int(row["codebook_size"]),
                    "active_ratio": float(row["active_ratio"]),
                    "active_count": int(row["active_count"]),
                    "dead_count": int(row["dead_count"]),
                    "perplexity": float(row["perplexity"]),
                    "min_l2_dist": float(row["min_l2_dist"]),
                    "collapse_count": int(row["collapse_count"]),
                    "collapse_ratio": float(row["collapse_ratio"]),
                    "distance_reference_count": int(row["distance_reference_count"]),
                    "distance_stats_exact": number(row["distance_stats_exact"], int),
                    "source_log": str(path.relative_to(ROOT)),
                    "source_type": "结构化 CSV",
                })
    return rows


def deduplicate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = {}
    for row in rows:
        key = (str(row["experiment"]), int(row["epoch"]), int(row["layer"]))
        current = selected.get(key)
        if current is None or row["source_type"] == "结构化 CSV":
            selected[key] = row
    return sorted(selected.values(), key=lambda row: (
        str(row["experiment"]), int(row["epoch"]), int(row["layer"])
    ))


def latest_trained_epoch(experiment: str) -> int | str:
    path = EXPERIMENTS / f"{experiment}_epoch_metrics.csv"
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max((int(row["epoch"]) for row in csv.DictReader(handle) if row.get("epoch")), default=0)


def experiment_names(rows: list[dict[str, object]]) -> list[str]:
    names = {str(row["experiment"]) for row in rows}
    names.update(
        path.name.removesuffix("_epoch_metrics.csv")
        for path in EXPERIMENTS.glob("*_epoch_metrics.csv")
    )
    aliases = {"observed_001_baseline": "observed_001"}
    names.update(aliases.get(path.name, path.name) for path in (ROOT / "checkpoints").iterdir() if path.is_dir())
    return sorted(names)


def fmt_percent(value) -> str:
    return "" if value == "" else f"{float(value):.2%}"


def distance_mode(row) -> str:
    exact = row["distance_stats_exact"]
    if exact == "":
        return "旧日志未标注"
    if int(exact):
        return "精确"
    return f"采样 {row['distance_reference_count']}"


def quantizer_type(experiment: str) -> str:
    if "NoQuant" in experiment:
        return "无量化直通"
    return "ViTvq NoCompress" if "ViTvqNoCompress" in experiment else "SimVQ"


def latest_groups(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["experiment"])].append(row)
    result = {}
    for experiment, experiment_rows in grouped.items():
        latest_epoch = max(int(row["epoch"]) for row in experiment_rows)
        result[experiment] = [
            row for row in experiment_rows if int(row["epoch"]) == latest_epoch
        ]
    return result


def summary_rows(names, latest):
    rows = []
    for experiment in names:
        metrics = sorted(latest.get(experiment, []), key=lambda row: int(row["layer"]))
        trained_epoch = latest_trained_epoch(experiment)
        monitor_epoch = int(metrics[0]["epoch"]) if metrics else ""
        lag = int(trained_epoch) - monitor_epoch if trained_epoch != "" and monitor_epoch != "" else ""
        status = "已有统计" if metrics else "暂无码本统计"
        if lag not in ("", 0):
            status += f"，滞后 {lag} Epoch"
        by_layer = {int(row["layer"]): row for row in metrics}
        l0, l1 = by_layer.get(0), by_layer.get(1)
        avg_active = sum(float(row["active_ratio"]) for row in metrics) / len(metrics) if metrics else ""
        total_dead = sum(int(row["dead_count"]) for row in metrics) if metrics else ""
        total_collapse = sum(int(row["collapse_count"]) for row in metrics) if metrics else ""

        def field(row, key, ratio=False):
            if not row:
                return ""
            value = row[key]
            return fmt_percent(value) if ratio else value

        notes = ""
        if not metrics:
            notes = "历史训练产物存在，但未找到可解析的码本监控报告。"
        elif total_dead:
            notes = "存在死码，请结合逐层明细定位。"
        elif total_collapse:
            notes = "存在坍缩码字，请检查最小 L2 距离。"
        else:
            notes = "最新监控点未发现死码或坍缩码字。"
        rows.append({
            "实验名称": experiment,
            "量化器": quantizer_type(experiment),
            "统计状态": status,
            "已训练 Epoch": trained_epoch,
            "最新码本监控 Epoch": monitor_epoch,
            "码本配置": "" if not metrics else "[" + ", ".join(str(row["codebook_size"]) for row in metrics) + "]",
            "平均利用率": fmt_percent(avg_active),
            "总死码数": total_dead,
            "总坍缩码字数": total_collapse,
            "L0 利用率": field(l0, "active_ratio", True),
            "L0 困惑度": field(l0, "perplexity"),
            "L0 死码数": field(l0, "dead_count"),
            "L0 最小 L2": field(l0, "min_l2_dist"),
            "L1 利用率": field(l1, "active_ratio", True),
            "L1 困惑度": field(l1, "perplexity"),
            "L1 死码数": field(l1, "dead_count"),
            "L1 最小 L2": field(l1, "min_l2_dist"),
            "备注": notes,
        })
    return rows


def detail_rows(latest):
    rows = []
    for experiment, metrics in sorted(latest.items()):
        for row in sorted(metrics, key=lambda item: int(item["layer"])):
            rows.append({
                "实验名称": experiment,
                "量化器": quantizer_type(experiment),
                "监控 Epoch": row["epoch"],
                "层": f"L{row['layer']}",
                "码本大小 K": row["codebook_size"],
                "活跃码字数": row["active_count"],
                "利用率": fmt_percent(row["active_ratio"]),
                "死码数": row["dead_count"],
                "困惑度": row["perplexity"],
                "困惑度 / K": fmt_percent(float(row["perplexity"]) / int(row["codebook_size"])),
                "最小 L2 距离": row["min_l2_dist"],
                "坍缩码字数": row["collapse_count"],
                "坍缩比例": fmt_percent(row["collapse_ratio"]),
                "L2 统计方式": distance_mode(row),
                "数据来源": row["source_type"],
            })
    return rows


def history_rows(rows):
    result = []
    for row in rows:
        result.append({
            "实验名称": row["experiment"],
            "量化器": quantizer_type(str(row["experiment"])),
            "监控 Epoch": row["epoch"],
            "层": f"L{row['layer']}",
            "码本大小 K": row["codebook_size"],
            "活跃码字数": row["active_count"],
            "利用率": fmt_percent(row["active_ratio"]),
            "死码数": row["dead_count"],
            "困惑度": row["perplexity"],
            "困惑度 / K": fmt_percent(float(row["perplexity"]) / int(row["codebook_size"])),
            "最小 L2 距离": row["min_l2_dist"],
            "坍缩码字数": row["collapse_count"],
            "坍缩比例": fmt_percent(row["collapse_ratio"]),
            "L2 统计方式": distance_mode(row),
            "Run ID": row["run_id"],
            "源文件": row["source_log"],
        })
    return result


def coverage_rows(names, latest):
    rows = []
    for experiment in names:
        metrics = latest.get(experiment, [])
        rows.append({
            "实验名称": experiment,
            "Checkpoint 目录": f"checkpoints/{experiment}",
            "已训练 Epoch": latest_trained_epoch(experiment),
            "是否存在码本报告": "是" if metrics else "否",
            "最新监控 Epoch": int(metrics[0]["epoch"]) if metrics else "",
            "监控层数": len(metrics),
            "说明": "可在总览与逐层明细查看。" if metrics else "旧产物未采集或日志中没有可解析的码本报告。",
        })
    return rows


def guide_rows():
    return [
        {"指标": "利用率", "定义": "活跃码字数 / 码本大小 K。活跃码字是在统计批次中至少被选中过一次的码字。", "阅读建议": "越高越好；低利用率意味着码本容量没有被充分使用。"},
        {"指标": "死码数", "定义": "统计批次中从未被选中的码字数量。", "阅读建议": "通常越少越好；大码本需要同时结合困惑度判断。"},
        {"指标": "困惑度", "定义": "基于码字选择概率熵计算的有效码字数。", "阅读建议": "越接近 K 越均衡；利用率为 100% 但困惑度偏低，仍表示使用分布不均。"},
        {"指标": "困惑度 / K", "定义": "困惑度除以码本大小 K。", "阅读建议": "便于跨不同 K 的方案横向比较。"},
        {"指标": "最小 L2 距离", "定义": "变换后码本中码字最近邻距离的最小值。", "阅读建议": "过小可能表示码字彼此过于接近。"},
        {"指标": "坍缩码字数", "定义": "最近邻距离小于阈值 0.1 的码字估计数量。", "阅读建议": "大于 0 时需要重点排查码本坍缩。"},
        {"指标": "L2 统计方式", "定义": "小码本使用精确计算；过大码本最多抽取 4096 个参考码字估计。", "阅读建议": "采样结果适合诊断趋势，不应当当作精确全量计数。"},
        {"指标": "最新码本监控 Epoch", "定义": "训练期间每 10 Epoch 自动采集一次码本指标。", "阅读建议": "训练中方案可能比当前训练进度滞后数轮，这是正常现象。"},
    ]


def selected(rows, headers):
    return [[row.get(header, "") for header in headers] for row in rows]


def sheet(name, title, subtitle, rows, widths, tab_color):
    headers = list(rows[0]) if rows else ["说明"]
    return {
        "name": name, "title": title, "subtitle": subtitle,
        "headers": headers, "rows": selected(rows, headers),
        "widths": widths, "tab_color": tab_color,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
    rows = deduplicate(load_log_rows() + load_csv_rows())
    latest = latest_groups(rows)
    names = experiment_names(rows)
    summary = summary_rows(names, latest)
    details = detail_rows(latest)
    history = history_rows(rows)
    coverage = coverage_rows(names, latest)
    guide = guide_rows()
    subtitle = f"生成时间：{datetime.now():%Y-%m-%d %H:%M} | 最新可用监控点汇总；训练中方案通常每 10 Epoch 更新一次。"
    sheets = [
        sheet("01_最新总览", "全部方案码本利用率总览", subtitle, summary,
              {"实验名称": 62, "统计状态": 24, "备注": 48, "码本配置": 18}, 0x1F4E78),
        sheet("02_最新逐层明细", "最新监控点逐层码本指标", "困惑度 / K 适合跨码本规模对比；L2 统计方式请区分精确与采样。", details,
              {"实验名称": 62, "L2 统计方式": 18, "数据来源": 16}, 0x70AD47),
        sheet("03_完整历史趋势", "全部历史码本监控记录", "每行对应一个实验、一个监控 Epoch 和一层码本；可使用表头筛选器查看趋势。", history,
              {"实验名称": 62, "源文件": 75, "Run ID": 48, "L2 统计方式": 18}, 0xED7D31),
        sheet("04_覆盖情况", "码本监控数据覆盖情况", "没有报告的方案仍然保留在表中，避免误认为实验不存在。", coverage,
              {"实验名称": 62, "Checkpoint 目录": 72, "说明": 58}, 0xA5A5A5),
        sheet("05_指标说明", "码本指标阅读说明", "用于解释利用率、困惑度、死码和坍缩指标，便于后续优化。", guide,
              {"指标": 22, "定义": 82, "阅读建议": 72}, 0x8064A2),
    ]
    write_csv(OUTPUT_SUMMARY_CSV, summary)
    write_csv(OUTPUT_HISTORY_CSV, history)
    write_workbook(sheets)
    print(f"Wrote {OUTPUT_XLSX.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_SUMMARY_CSV.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_HISTORY_CSV.relative_to(ROOT)}")
    print(f"Experiments: {len(names)}; with metrics: {len(latest)}; history rows: {len(history)}")


if __name__ == "__main__":
    main()
