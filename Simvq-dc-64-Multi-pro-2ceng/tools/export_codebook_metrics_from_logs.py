#!/usr/bin/env python3
"""Export printed codebook utilization reports from training logs to CSV."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


EPOCH_RE = re.compile(r"\[Codebook Utilization\] Epoch (\d+)")
EXPERIMENT_RE = re.compile(r"\[Info\] Experiment name: (.+)")
RUN_ID_RE = re.compile(r"\[Info\] Experiment run ID: (.+)")
LAYER_RE = re.compile(r"Layer (\d+) \(K=(\d+)\)")
STATS_RE = re.compile(
    r"活跃率: ([\d.]+)%.*活跃码字: (\d+)/(\d+).*死码字: (\d+).*"
    r"困惑度: ([\d.]+)/(\d+).*最小L2距离(?:\((精确|采样(\d+))\))?: ([\d.]+).*"
    r"坍缩码字: (\d+)/(\d+) \(([\d.]+)%\)"
)

FIELDS = [
    "experiment", "run_id", "epoch", "layer", "codebook_size",
    "active_ratio", "active_count", "dead_count", "perplexity",
    "min_l2_dist", "collapse_count", "collapse_ratio",
    "distance_reference_count", "distance_stats_exact", "source_log",
]


def parse_log(path: Path) -> list[dict[str, object]]:
    experiment = ""
    run_id = ""
    epoch = None
    layer = None
    rows = []

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := EXPERIMENT_RE.search(line):
            experiment = match.group(1).strip()
        elif match := RUN_ID_RE.search(line):
            run_id = match.group(1).strip()
        elif match := EPOCH_RE.search(line):
            epoch = int(match.group(1))
        elif match := LAYER_RE.search(line):
            layer = int(match.group(1))
        elif match := STATS_RE.search(line):
            if epoch is None or layer is None:
                continue
            mode, sampled_count = match.group(7), match.group(8)
            codebook_size = int(match.group(3))
            rows.append({
                "experiment": experiment,
                "run_id": run_id,
                "epoch": epoch,
                "layer": layer,
                "codebook_size": codebook_size,
                "active_ratio": float(match.group(1)) / 100.0,
                "active_count": int(match.group(2)),
                "dead_count": int(match.group(4)),
                "perplexity": float(match.group(5)),
                "min_l2_dist": float(match.group(9)),
                "collapse_count": int(match.group(10)),
                "collapse_ratio": float(match.group(12)) / 100.0,
                "distance_reference_count": int(sampled_count or codebook_size),
                "distance_stats_exact": "" if mode is None else int(mode == "精确"),
                "source_log": str(path),
            })
    return rows


def write_rows(path: Path, rows: list[dict[str, object]], include_experiment=True) -> None:
    fields = FIELDS if include_experiment else FIELDS[1:-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--write-per-experiment",
        action="store_true",
        help="Also write experiments/<name>_codebook_metrics.csv files.",
    )
    args = parser.parse_args()

    rows = []
    for path in args.logs:
        rows.extend(parse_log(path))
    rows.sort(key=lambda row: (str(row["experiment"]), int(row["epoch"]), int(row["layer"])))
    write_rows(args.output, rows)

    if args.write_per_experiment:
        by_experiment: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            by_experiment.setdefault(str(row["experiment"]), []).append(row)
        for experiment, experiment_rows in by_experiment.items():
            write_rows(
                Path("experiments") / f"{experiment}_codebook_metrics.csv",
                experiment_rows,
                include_experiment=False,
            )

    print(f"Exported {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
