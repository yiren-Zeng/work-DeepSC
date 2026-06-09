#!/usr/bin/env python3
"""Evaluate completed experiments and refresh the Excel report automatically."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "experiments" / "auto_eval_manifest.json"
RESULT_DIR = ROOT / "experiments" / "auto_results"
LOG_DIR = ROOT / "experiments" / "logs"
LOCK_PATH = ROOT / "experiments" / ".auto_eval.lock"
REPORT_SCRIPT = ROOT / "tools" / "rebuild_unified_experiment_report.py"
PYTHON = Path(sys.executable)


def load_manifest() -> dict[str, dict[str, object]]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def latest_epoch(experiment_name: str) -> int:
    path = ROOT / "experiments" / f"{experiment_name}_epoch_metrics.csv"
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return max((int(row["epoch"]) for row in rows if row.get("epoch")), default=0)


def link_result_suffix(settings: dict[str, object]) -> str:
    return str(settings.get("result_suffix", "snr0"))


def result_paths(experiment_name: str, settings: dict[str, object]) -> tuple[Path, Path]:
    return (
        RESULT_DIR / f"{experiment_name}_no_channel.json",
        RESULT_DIR / f"{experiment_name}_{link_result_suffix(settings)}.json",
    )


def is_complete(experiment_name: str, max_epoch: int) -> bool:
    return latest_epoch(experiment_name) >= max_epoch


def has_results(
    experiment_name: str,
    settings: dict[str, object],
    no_channel_only: bool = False,
) -> bool:
    no_channel, link_result = result_paths(experiment_name, settings)
    return no_channel.exists() if no_channel_only else no_channel.exists() and link_result.exists()


def run_command(command: list[str], env: dict[str, str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def evaluate(experiment_name: str, settings: dict[str, object], gpu: str, force: bool) -> bool:
    max_epoch = int(settings.get("max_epoch", 200))
    if not is_complete(experiment_name, max_epoch):
        print(f"[wait] {experiment_name}: epoch {latest_epoch(experiment_name)}/{max_epoch}", flush=True)
        return False
    no_channel_only = bool(settings.get("no_channel_only", False))
    if has_results(experiment_name, settings, no_channel_only) and not force:
        print(f"[skip] {experiment_name}: results already exist", flush=True)
        return False

    checkpoint = ROOT / "checkpoints" / experiment_name / "best_vq_deepsc.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    no_channel, link_result = result_paths(experiment_name, settings)
    env = dict(os.environ)
    env.update({str(key): str(value) for key, value in settings.get("env", {}).items()})
    env["CUDA_VISIBLE_DEVICES"] = gpu
    print(f"[eval] {experiment_name}: no-channel on GPU {gpu}", flush=True)
    run_command(
        [
            str(PYTHON), "-u", "test_real.py", "--checkpoint", str(checkpoint),
            "--no-channel", "--json-output", str(no_channel),
        ],
        env,
        LOG_DIR / f"auto_eval_{experiment_name}_no_channel.log",
    )
    if no_channel_only:
        print(f"[done] {experiment_name}: no-channel-only experiment", flush=True)
        return True
    snrs = [str(item) for item in settings.get("snrs", [0])]
    modulation = str(settings.get("modulation", "bpsk"))
    print(
        f"[eval] {experiment_name}: {modulation.upper()} SNR={','.join(snrs)} dB "
        f"real chain on GPU {gpu}",
        flush=True,
    )
    run_command(
        [
            str(PYTHON), "-u", "test_real.py", "--checkpoint", str(checkpoint),
            "--snrs", *snrs, "--modulation", modulation, "--json-output", str(link_result),
        ],
        env,
        LOG_DIR / f"auto_eval_{experiment_name}_{link_result_suffix(settings)}.log",
    )
    print(f"[done] {experiment_name}", flush=True)
    return True


def refresh_report() -> None:
    print("[report] refreshing Excel workbook", flush=True)
    subprocess.run([str(PYTHON), str(REPORT_SCRIPT)], cwd=ROOT, check=True)


def evaluate_due(args) -> bool:
    manifest = load_manifest()
    names = [args.experiment] if args.experiment else list(manifest)
    changed = False
    with LOCK_PATH.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name in names:
            if name not in manifest:
                raise KeyError(f"{name!r} is not registered in {MANIFEST_PATH.relative_to(ROOT)}")
            changed |= evaluate(name, manifest[name], args.gpu, args.force)
        if changed or args.refresh_report:
            refresh_report()
    return changed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", help="Evaluate one registered experiment only.")
    parser.add_argument("--gpu", default=os.environ.get("AUTO_EVAL_GPU_ID", "0"))
    parser.add_argument("--watch", action="store_true", help="Keep polling for completed experiments.")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--force", action="store_true", help="Re-run evaluations even if JSON results exist.")
    parser.add_argument("--refresh-report", action="store_true", help="Refresh report even if no evaluation ran.")
    return parser.parse_args()


def main():
    args = parse_args()
    while True:
        evaluate_due(args)
        if not args.watch:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
