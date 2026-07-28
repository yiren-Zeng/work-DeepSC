"""Periodically screen a training run against the BPG quality target."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402


Config.validate()
METRICS_PATH = ROOT / Config.METRICS_PATH
CHECKPOINT_PATH = ROOT / Config.CHECKPOINT_DIR / "best_vq_deepsc.pth"
SNAPSHOT_DIR = ROOT / Config.SNAPSHOT_DIR
LOG_DIR = ROOT / "experiments" / "logs"
SCREENING_PATH = ROOT / Config.SCREENING_PATH
TARGET_PSNR = 32.30
TARGET_MS_SSIM = 0.9777
MAX_EPOCH = 200
POLL_SECONDS = 60


def completed_epochs(metrics_path=METRICS_PATH):
    if not metrics_path.exists():
        return []
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        int(row["epoch"]) for row in rows
        if row["run_id"].startswith(Config.EXPERIMENT_NAME)
    ]


def append_screening(epoch, metrics, checkpoint_path, args):
    write_header = not args.screening_path.exists()
    passed = metrics["psnr"] >= args.target_psnr and metrics["ms_ssim"] >= args.target_ms_ssim
    with args.screening_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow([
                "epoch_seen", "snapshot", "no_channel_psnr", "no_channel_ms_ssim",
                "target_psnr", "target_ms_ssim", "upper_bound_passed",
            ])
        writer.writerow([
            epoch, checkpoint_path.relative_to(ROOT), f"{metrics['psnr']:.6f}",
            f"{metrics['ms_ssim']:.6f}", args.target_psnr, args.target_ms_ssim, int(passed),
        ])
    return passed


def evaluate(epoch, args):
    args.snapshot_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.snapshot_dir / f"best_seen_epoch_{epoch:03d}_nochannel.json"
    log_path = args.log_dir / f"{Config.EXPERIMENT_NAME}_screen_epoch_{epoch:03d}.log"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = env.get("SCREEN_GPU_ID", "3")
    command = [
        sys.executable, "-u", "test_real.py", "--checkpoint", str(args.checkpoint_path),
        "--no-channel", "--json-output", str(result_path),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return append_screening(epoch, payload["results"]["no_channel"], args.checkpoint_path, args)


def full_chain_confirmation(epoch, args):
    result_path = args.snapshot_dir / f"best_seen_epoch_{epoch:03d}_real_chain.json"
    log_path = args.log_dir / f"{Config.EXPERIMENT_NAME}_real_chain_epoch_{epoch:03d}.log"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = env.get("SCREEN_GPU_ID", "3")
    command = [
        sys.executable, "-u", "test_real.py", "--checkpoint", str(args.checkpoint_path),
        "--json-output", str(result_path),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description=f"Monitor {Config.EXPERIMENT_NAME} training and screen checkpoints.")
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT_DIR)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--screening-path", type=Path, default=SCREENING_PATH)
    parser.add_argument("--target-psnr", type=float, default=TARGET_PSNR)
    parser.add_argument("--target-ms-ssim", type=float, default=TARGET_MS_SSIM)
    parser.add_argument("--max-epoch", type=int, default=MAX_EPOCH)
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    return parser.parse_args()


def main():
    args = parse_args()
    milestones = [5, 10] + list(range(20, args.max_epoch + 1, 10))
    completed = set()
    initial_epochs = completed_epochs(args.metrics_path)
    initial_last_epoch = max(initial_epochs) if initial_epochs else 0
    if initial_last_epoch and not args.screening_path.exists():
        completed.update(epoch for epoch in milestones if epoch < initial_last_epoch)
    print(f"Monitoring {args.checkpoint_path}")
    print(f"Upper-bound target: PSNR >= {args.target_psnr}, MS-SSIM >= {args.target_ms_ssim}")
    while milestones:
        epochs = completed_epochs(args.metrics_path)
        last_epoch = max(epochs) if epochs else 0
        due = [epoch for epoch in milestones if last_epoch >= epoch]
        for epoch in due:
            if epoch not in completed and args.checkpoint_path.exists():
                print(f"Screening best checkpoint after epoch {epoch}", flush=True)
                passed = evaluate(epoch, args)
                completed.add(epoch)
                if passed:
                    print("Upper-bound target reached; running real-chain confirmation.", flush=True)
                    full_chain_confirmation(epoch, args)
                    return
            milestones.remove(epoch)
        if last_epoch >= args.max_epoch:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
