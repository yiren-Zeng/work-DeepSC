"""Start the isolated 1/8 + 1/16 Seq2Seq training run."""
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
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = "smoke_seq2seq_ds8_src64_" if args.smoke else "seq2seq_ds8_src64_"
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
    command = [sys.executable, "-B", "-u", str(ROOT / "train_seq2seq_cars_ds8.py"),
               "--output", str(directory)]
    if args.smoke:
        command.append("--smoke")
    with (directory / "train.log").open("xb") as logfile:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=logfile,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True)
    record = {"pid": process.pid, "run_dir": str(directory), "gpu": args.gpu,
              "smoke": args.smoke, "command": command}
    (directory / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

