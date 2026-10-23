"""Launch the independent K256 / K4..4096 experiment with automatic recovery."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ZONE = datetime.timezone(datetime.timedelta(hours=8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="4", help="Physical nvidia-smi GPU index")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume-run", help="Restart the supervisor for an existing run")
    args = parser.parse_args()
    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", f"--id={args.gpu}", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if not gpu_uuid.startswith("GPU-") or "\n" in gpu_uuid:
        raise ValueError("Select exactly one physical GPU")
    if args.resume_run:
        directory = Path(args.resume_run).resolve()
        if not directory.is_relative_to(ROOT / "runs"):
            raise ValueError("Run must be under this project's runs/")
        config = json.loads((directory / "config.json").read_text())
        if config.get("variant") != "RAQVAE_TWO_SRC256_K4_4096_POW2" or bool(config["smoke"]) != args.smoke:
            raise ValueError("Run variant/smoke mode differs")
        if (directory / "STOP").exists():
            raise ValueError("Remove this run's STOP marker before explicitly resuming it")
    else:
        stamp = datetime.datetime.now(ZONE).strftime("%Y%m%d_%H%M%S_%f")
        prefix = "smoke_seq2seq_src256_k4_4096_" if args.smoke else "seq2seq_src256_k4_4096_"
        directory = ROOT / "runs" / (prefix + stamp)
        directory.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu_uuid, "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1", "PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TMPDIR": str(ROOT / ".runtime/tmp"), "XDG_CACHE_HOME": str(ROOT / ".runtime/cache"),
        "MPLCONFIGDIR": str(ROOT / ".runtime/cache/matplotlib"), "TORCH_HOME": str(ROOT / ".runtime/cache/torch"),
    })
    python = os.environ.get("RAQ_PYTHON", "/home/yi/.conda/envs/work/bin/python")
    command = [python, "-B", "-u", str(ROOT / "scripts/supervise_seq2seq_src256_k4_4096.py"),
               "--run-dir", str(directory), "--gpu", args.gpu]
    if args.smoke:
        command.append("--smoke")
    with (directory / "supervisor.log").open("ab") as logfile:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=logfile, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    for _ in range(50):
        if process.poll() is not None:
            raise RuntimeError(f"Supervisor exited ({process.returncode}); see {directory / 'supervisor.log'}")
        if (directory / "supervisor.json").is_file():
            break
        time.sleep(0.1)
    record = {
        "supervisor_pid": process.pid, "run_dir": str(directory), "gpu": args.gpu,
        "gpu_uuid": gpu_uuid, "smoke": args.smoke, "command": command,
    }
    (directory / f"launch_{time.time_ns()}.json").write_text(json.dumps(record, indent=2) + "\n")
    (directory / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    if not args.smoke:
        link = ROOT / "runs/latest_seq2seq_src256_k4_4096"
        temporary = link.with_name(f".{link.name}.{os.getpid()}.tmp")
        temporary.symlink_to(directory.name, target_is_directory=True)
        os.replace(temporary, link)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
