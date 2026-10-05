"""Start source-K=16, target-K<=128 dual-K DS8 training from scratch."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = ("smoke_seq2seq_ds8_dualk_src16_kmax128_" if args.smoke
              else "seq2seq_ds8_dualk_src16_kmax128_")
    directory = ROOT / "runs" / (prefix + timestamp)
    directory.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": args.gpu, "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1", "PYTHONHASHSEED": "42",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TMPDIR": str(ROOT / ".runtime/tmp"),
        "XDG_CACHE_HOME": str(ROOT / ".runtime/cache"),
        "MPLCONFIGDIR": str(ROOT / ".runtime/cache/matplotlib"),
        "TORCH_HOME": str(ROOT / ".runtime/cache/torch"),
    })
    command = [
        sys.executable, "-B", "-u",
        str(ROOT / "train_seq2seq_cars_ds8_dualk_src16_kmax128.py"),
        "--output", str(directory),
    ]
    if args.smoke:
        command.append("--smoke")
    with (directory / "train.log").open("xb") as logfile:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=logfile,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    record = {
        "pid": process.pid, "run_dir": str(directory), "gpu": args.gpu,
        "smoke": args.smoke, "command": command,
    }
    (directory / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
