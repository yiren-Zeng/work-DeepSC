"""Launch the single-scale ds2/source-K256/K32..2048 comparison training."""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--embedding-dim", type=int, choices=[64], default=64)
    parser.add_argument("--n-hid", type=int, choices=[64], default=64)
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = (
        "smoke_seq2seq_one_ds2_src256_k32_2048_original_"
        if args.smoke
        else "seq2seq_one_ds2_src256_k32_2048_original_"
    )
    directory = ROOT / "runs" / (prefix + timestamp)
    directory.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONHASHSEED": "42",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TMPDIR": str(ROOT / ".runtime/tmp"),
        "XDG_CACHE_HOME": str(ROOT / ".runtime/cache"),
        "MPLCONFIGDIR": str(ROOT / ".runtime/cache/matplotlib"),
        "TORCH_HOME": str(ROOT / ".runtime/cache/torch"),
    })
    python = os.environ.get(
        "RAQ_PYTHON", "/home/yi/.conda/envs/work/bin/python"
    )
    command = [
        python,
        "-B",
        "-u",
        str(ROOT / "train_seq2seq_cars_one_ds2_src256_k32_2048.py"),
        "--output",
        str(directory),
        "--embedding-dim",
        str(args.embedding_dim),
        "--n-hid",
        str(args.n_hid),
    ]
    if args.smoke:
        command.append("--smoke")
    with (directory / "train.log").open("xb") as logfile:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    record = {
        "pid": process.pid,
        "run_dir": str(directory),
        "gpu": args.gpu,
        "smoke": args.smoke,
        "embedding_dim": args.embedding_dim,
        "n_hid": args.n_hid,
        "command": command,
    }
    (directory / "launch.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
