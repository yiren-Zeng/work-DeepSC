"""Periodically screen a training run against the BPG quality target."""

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "experiments" / "epoch_metrics.csv"
CHECKPOINT_PATH = ROOT / "experiments" / "checkpoints" / "quality_v1_k64" / "best_vq_deepsc.pth"
SNAPSHOT_DIR = ROOT / "experiments" / "snapshots" / "quality_v1_k64"
LOG_DIR = ROOT / "experiments" / "logs"
SCREENING_PATH = ROOT / "experiments" / "quality_v1_k64_screening.csv"
TARGET_PSNR = 32.30
TARGET_MS_SSIM = 0.9777
MAX_EPOCH = 200
POLL_SECONDS = 60


def completed_epochs():
    if not METRICS_PATH.exists():
        return []
    with METRICS_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        int(row["epoch"]) for row in rows
        if row["run_id"].startswith("quality-v1-k64")
    ]


def append_screening(epoch, metrics, snapshot):
    write_header = not SCREENING_PATH.exists()
    passed = metrics["psnr"] >= TARGET_PSNR and metrics["ms_ssim"] >= TARGET_MS_SSIM
    with SCREENING_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow([
                "epoch_seen", "snapshot", "no_channel_psnr", "no_channel_ms_ssim",
                "target_psnr", "target_ms_ssim", "upper_bound_passed"
            ])
        writer.writerow([
            epoch, snapshot.relative_to(ROOT), f"{metrics['psnr']:.6f}",
            f"{metrics['ms_ssim']:.6f}", TARGET_PSNR, TARGET_MS_SSIM, int(passed)
        ])
    return passed


def evaluate(epoch):
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = SNAPSHOT_DIR / f"best_seen_epoch_{epoch:03d}.pth"
    shutil.copy2(CHECKPOINT_PATH, snapshot)
    result_path = SNAPSHOT_DIR / f"best_seen_epoch_{epoch:03d}_nochannel.json"
    log_path = LOG_DIR / f"quality_v1_k64_screen_epoch_{epoch:03d}.log"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = env.get("SCREEN_GPU_ID", "3")
    command = [
        sys.executable, "-u", "test_real.py", "--checkpoint", str(snapshot),
        "--no-channel", "--json-output", str(result_path)
    ]
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return append_screening(epoch, payload["results"]["no_channel"], snapshot)


def full_chain_confirmation(epoch):
    snapshot = SNAPSHOT_DIR / f"best_seen_epoch_{epoch:03d}.pth"
    result_path = SNAPSHOT_DIR / f"best_seen_epoch_{epoch:03d}_real_chain.json"
    log_path = LOG_DIR / f"quality_v1_k64_real_chain_epoch_{epoch:03d}.log"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = env.get("SCREEN_GPU_ID", "3")
    command = [
        sys.executable, "-u", "test_real.py", "--checkpoint", str(snapshot),
        "--json-output", str(result_path)
    ]
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    milestones = [5, 10] + list(range(20, MAX_EPOCH + 1, 10))
    completed = set()
    print(f"Monitoring {CHECKPOINT_PATH}")
    print(f"Upper-bound target: PSNR >= {TARGET_PSNR}, MS-SSIM >= {TARGET_MS_SSIM}")
    while milestones:
        epochs = completed_epochs()
        last_epoch = max(epochs) if epochs else 0
        due = [epoch for epoch in milestones if last_epoch >= epoch]
        for epoch in due:
            if epoch not in completed and CHECKPOINT_PATH.exists():
                print(f"Screening best checkpoint after epoch {epoch}", flush=True)
                passed = evaluate(epoch)
                completed.add(epoch)
                if passed:
                    print("Upper-bound target reached; running real-chain confirmation.", flush=True)
                    full_chain_confirmation(epoch)
                    return
            milestones.remove(epoch)
        if last_epoch >= MAX_EPOCH:
            return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
