"""Run the two original evaluation scripts sequentially on the same GPU."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path("/workspace/yi/work/shiyan")
OUT = Path(__file__).resolve().parent
SCRIPTS = {
    "raq": "scripts/eval/test_src64_64_raq2_64_curriculum_ch256_512.sh",
    "independent_rvq": "scripts/eval/test_independent_raq_rvq_src64_64_d2_combined.sh",
}
base_env = {k: v for k, v in os.environ.items() if not k.startswith("SIMVQ_")}
for key in ("CHECKPOINT", "SNRS", "MODULATION", "JSON_OUTPUT", "NO_CHANNEL", "GPU_ID"):
    base_env.pop(key, None)
base_env.update({
    "GPU_ID": "1",
    "SNRS": "3",
    "MODULATION": "qpsk",
    "SIMVQ_TEST_DATASET_PATH": "/workspace/yi/work/Kodak-256-transform-resize",
    "SIMVQ_TEST_NO_RESIZE": "1",
    "PYTHONHASHSEED": "42",
    "PYTHONUNBUFFERED": "1",
})
manifest = {
    "dataset": base_env["SIMVQ_TEST_DATASET_PATH"],
    "physical_gpu": 1,
    "snr_db": 3,
    "modulation": "qpsk",
    "ldpc_n": 256,
    "ldpc_k": 128,
    "seed": 42,
    "runs": [],
}
for condition, no_channel in (("channel_snr3", "0"), ("no_channel", "1")):
    for model, script in SCRIPTS.items():
        name = model + "_" + condition
        result_file = OUT / (name + ".json")
        log_file = OUT / (name + ".log")
        env = {**base_env, "NO_CHANNEL": no_channel, "JSON_OUTPUT": str(result_file)}
        command = ["bash", script]
        print("Starting " + name, flush=True)
        started = time.perf_counter()
        with log_file.open("w") as log:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        record = {
            "model": model,
            "condition": condition,
            "script": script,
            "command": command,
            "elapsed_seconds": time.perf_counter() - started,
            "returncode": completed.returncode,
            "result_file": str(result_file),
            "log_file": str(log_file),
        }
        if completed.returncode == 0:
            payload = json.loads(result_file.read_text())
            record["metrics"] = payload["results"]
        manifest["runs"].append(record)
        (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps({k: v for k, v in record.items() if k != "metrics"}), flush=True)
        print("\n".join(log_file.read_text().splitlines()[-10:]), flush=True)
        if completed.returncode:
            raise SystemExit(completed.returncode)
print("All four evaluations completed: " + str(OUT), flush=True)

