"""Train GN variants, evaluate saved epochs, and keep an auditable trial record."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
LOG_PATH = EXPERIMENTS / "AUTO_EXPERIMENT_LOG.md"
RESULTS_PATH = EXPERIMENTS / "auto_trial_results.csv"
TARGET_PSNR = 32.30
TARGET_MS_SSIM = 0.9777

TRIALS = [
    {
        "name": "gn_v1_k64",
        "change": "Replace all encoder/decoder BatchNorm2d layers with GroupNorm(32); retain the source quality-v1 training settings.",
        "env": {},
    },
    {
        "name": "gn_v2_recon_focus",
        "change": "Keep GroupNorm(32), remove initial skip dropout, reduce VQ weights to [0.15, 0.15], and use generator LR 1e-4 to prioritize reconstruction quality.",
        "env": {
            "SIMVQ_SKIP_DROPOUT_INIT": "0.0",
            "SIMVQ_SKIP_DROPOUT_FINAL": "0.0",
            "SIMVQ_LOSS_WEIGHTS_INIT": "0.15,0.15",
            "SIMVQ_LOSS_WEIGHTS_FINAL": "0.15,0.15",
            "SIMVQ_LEARNING_RATE": "1e-4",
        },
    },
]


def append_log(text):
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n\n")


def completed_epochs(name):
    path = EXPERIMENTS / f"{name}_epoch_metrics.csv"
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return max((int(row["epoch"]) for row in rows), default=0)


def run_logged(command, env, log_file):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as output:
        subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)


def candidate_checkpoints(name):
    checkpoint_dir = EXPERIMENTS / "checkpoints" / name
    candidates = sorted(
        checkpoint_dir.glob("vq_deepsc_epoch_*.pth"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    best = checkpoint_dir / "best_vq_deepsc.pth"
    if best.exists():
        candidates.append(best)
    return candidates


def evaluate_candidates(name, env):
    best_record = None
    for checkpoint in candidate_checkpoints(name):
        label = checkpoint.stem
        json_path = EXPERIMENTS / "evaluations" / name / f"{label}_nochannel.json"
        log_file = EXPERIMENTS / "logs" / f"{name}_{label}_nochannel.log"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        run_logged(
            [
                sys.executable, "-u", "test_real.py", "--checkpoint", str(checkpoint),
                "--no-channel", "--json-output", str(json_path),
            ],
            env,
            log_file,
        )
        metrics = json.loads(json_path.read_text(encoding="utf-8"))["results"]["no_channel"]
        record = {
            "trial": name,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "psnr": float(metrics["psnr"]),
            "ms_ssim": float(metrics["ms_ssim"]),
        }
        append_result(record)
        if best_record is None or (record["psnr"], record["ms_ssim"]) > (
            best_record["psnr"], best_record["ms_ssim"]
        ):
            best_record = record
    return best_record


def append_result(record):
    write_header = not RESULTS_PATH.exists()
    with RESULTS_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trial", "checkpoint", "psnr", "ms_ssim"])
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def evaluate_real_chain(record, env):
    json_path = EXPERIMENTS / "evaluations" / record["trial"] / "best_real_chain.json"
    log_file = EXPERIMENTS / "logs" / f"{record['trial']}_best_real_chain.log"
    run_logged(
        [
            sys.executable, "-u", "test_real.py", "--checkpoint", str(ROOT / record["checkpoint"]),
            "--json-output", str(json_path),
        ],
        env,
        log_file,
    )
    return json.loads(json_path.read_text(encoding="utf-8"))["results"]


def main():
    parser = argparse.ArgumentParser(description="Run and screen automatic GN optimization trials.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max-trials", type=int, default=len(TRIALS))
    parser.add_argument("--gpu", default="1")
    args = parser.parse_args()

    append_log(
        f"## Automatic session {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- Requested trials: {min(args.max_trials, len(TRIALS))}; epochs per trial: {args.epochs}; GPU: {args.gpu}.\n"
        f"- Pass target: no-channel PSNR >= {TARGET_PSNR:.2f} dB and MS-SSIM >= {TARGET_MS_SSIM:.4f}."
    )
    for index, trial in enumerate(TRIALS[: args.max_trials], start=1):
        env = dict(os.environ)
        env.update(trial["env"])
        env.update({
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "EXPERIMENT_RUN_ID": f"auto-{trial['name']}",
            "SIMVQ_EXPERIMENT_NAME": trial["name"],
            "SIMVQ_NUM_EPOCHS": str(args.epochs),
        })
        append_log(f"### Trial {index}: `{trial['name']}`\n- Modification: {trial['change']}\n- Status: training started.")
        run_logged(
            [sys.executable, "-u", "train.py"],
            env,
            EXPERIMENTS / "logs" / f"auto_train_{trial['name']}.log",
        )
        best = evaluate_candidates(trial["name"], env)
        if best is None:
            append_log(f"### Trial {index} result\n- Completed epochs: {completed_epochs(trial['name'])}; no saved candidate was produced.")
            continue
        passed = best["psnr"] >= TARGET_PSNR and best["ms_ssim"] >= TARGET_MS_SSIM
        append_log(
            f"### Trial {index} result\n"
            f"- Completed epochs: {completed_epochs(trial['name'])}.\n"
            f"- Best tested checkpoint: `{best['checkpoint']}`.\n"
            f"- No-channel metrics: PSNR={best['psnr']:.6f} dB, MS-SSIM={best['ms_ssim']:.6f}.\n"
            f"- Target passed: {'yes' if passed else 'no'}."
        )
        if passed:
            real_results = evaluate_real_chain(best, env)
            append_log(f"### Trial {index} real-chain confirmation\n```json\n{json.dumps(real_results, indent=2)}\n```")
            return
        if index < min(args.max_trials, len(TRIALS)):
            append_log("- Decision: quality is below target; automatically proceed to the next recorded scheme.")


if __name__ == "__main__":
    main()
