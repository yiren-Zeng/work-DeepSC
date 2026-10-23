"""Keep one training worker alive until completion or an explicit STOP request."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ZONE = datetime.timezone(datetime.timedelta(hours=8))


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def write_json(path, data):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def supervise(directory, command, gpu, poll_seconds=5, retry_seconds=10):
    """Exit codes: 0=completed/stopped, 2=another supervisor holds this GPU lock."""
    directory = Path(directory)
    lock_path = directory.parent / f".seq2seq_src256_k4_4096_gpu{gpu}.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"ALREADY_RUNNING GPU {gpu}: {lock_path}", flush=True)
            return 2
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        requested_stop = {"value": False}

        def stop_handler(signum, frame):
            requested_stop["value"] = True

        signal.signal(signal.SIGTERM, stop_handler)
        signal.signal(signal.SIGINT, stop_handler)
        state = {
            "supervisor_pid": os.getpid(), "worker_pid": None,
            "gpu": str(gpu), "run_dir": str(directory), "restarts": 0,
            "command": command,
        }

        def update(label, **extra):
            state.update(state=label, updated_at=datetime.datetime.now(ZONE).isoformat(), **extra)
            write_json(directory / "supervisor.json", state)

        def stopped():
            return requested_stop["value"] or (directory / "STOP").exists()

        while True:
            if stopped():
                update("stopped", worker_pid=None)
                return 0
            if read_json(directory / "status.json").get("state") == "completed":
                update("completed", worker_pid=None)
                return 0
            with (directory / "train.log").open("ab", buffering=0) as logfile:
                logfile.write((f"\nSUPERVISOR_ATTEMPT {state['restarts'] + 1} "
                               f"{datetime.datetime.now(ZONE).isoformat()}\n").encode())
                try:
                    worker = subprocess.Popen(
                        command, cwd=ROOT, stdin=subprocess.DEVNULL,
                        stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True,
                    )
                except OSError as error:
                    update("retrying", worker_pid=None, launch_error=repr(error))
                    worker = None
                if worker is not None:
                    update("running", worker_pid=worker.pid, last_exit_code=None)
                    print(f"WORKER_STARTED pid={worker.pid} attempt={state['restarts'] + 1}", flush=True)
                    terminated = False
                    while worker.poll() is None:
                        if stopped() and not terminated:
                            worker.terminate()
                            terminated = True
                            update("stopping")
                        time.sleep(poll_seconds)
                        update("stopping" if terminated else "running")
                    exit_code = worker.returncode
                    # A crash's traceback remains in train.log; retain its tail too.
                    with (directory / "train.log").open("rb") as logreader:
                        logreader.seek(max(0, logreader.seek(0, 2) - 8000))
                        tail = logreader.read().decode(errors="replace").splitlines()[-30:]
                    update("worker_exited", worker_pid=None, last_exit_code=exit_code,
                           last_exit_log_tail=tail)
                    print(f"WORKER_EXIT code={exit_code}", flush=True)
            if stopped():
                update("stopped", worker_pid=None)
                return 0
            if worker is not None and worker.returncode == 0 and read_json(
                directory / "status.json"
            ).get("state") == "completed":
                update("completed", worker_pid=None)
                return 0
            state["restarts"] += 1
            delay = min(300, retry_seconds * 2 ** min(state["restarts"] - 1, 5))
            update("retrying", next_attempt_seconds=delay)
            print(f"AUTO_RESUME restart={state['restarts']} delay_seconds={delay}", flush=True)
            deadline = time.monotonic() + delay
            while time.monotonic() < deadline and not stopped():
                time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    directory = Path(args.run_dir).resolve()
    if not directory.is_relative_to(ROOT / "runs"):
        parser.error("Run directory must be under this project's runs/")
    command = [sys.executable, "-B", "-u", str(ROOT / "train_seq2seq_cars_src256_k4_4096.py"),
               "--output", str(directory), "--auto-resume"]
    if args.smoke:
        command.append("--smoke")
    sys.exit(supervise(directory, command, args.gpu))


if __name__ == "__main__":
    main()
