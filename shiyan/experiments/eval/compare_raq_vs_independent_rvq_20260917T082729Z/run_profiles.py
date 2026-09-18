"""Invoke the original scripts while replacing their final Python entrypoint."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path("/workspace/yi/work/shiyan")
OUT = Path(__file__).resolve().parent
base_env = {k: v for k, v in os.environ.items() if not k.startswith("SIMVQ_")}
for key in ("CHECKPOINT", "SNRS", "MODULATION", "JSON_OUTPUT", "NO_CHANNEL", "GPU_ID"):
    base_env.pop(key, None)
base_env.update({
    "GPU_ID": "1", "SNRS": "3", "MODULATION": "qpsk", "NO_CHANNEL": "1",
    "SIMVQ_TEST_DATASET_PATH": "/workspace/yi/work/Kodak-256-transform-resize",
    "SIMVQ_TEST_NO_RESIZE": "1", "PYTHONHASHSEED": "42",
    "PERF_PROFILE_FILE": str(OUT / "profile_reconstruction.py"),
})
shell = 'python() { command python "$PERF_PROFILE_FILE" "$@"; }\nexport -f python\nbash "$PERF_EVAL_SCRIPT"'
for model, script in (
    ("raq", "scripts/eval/test_src64_64_raq2_64_curriculum_ch256_512.sh"),
    ("independent_rvq", "scripts/eval/test_independent_raq_rvq_src64_64_d2_combined.sh"),
):
    print("Profiling " + model, flush=True)
    env = {**base_env, "JSON_OUTPUT": str(OUT / (model + "_profile.json")),
           "PERF_EVAL_SCRIPT": script}
    with (OUT / (model + "_profile.log")).open("w") as log:
        completed = subprocess.run(["bash", "-c", shell], cwd=ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        print((OUT / (model + "_profile.log")).read_text(), flush=True)
        raise SystemExit(completed.returncode)
    record = json.loads((OUT / (model + "_profile.json")).read_text())
    print(json.dumps({k: record[k] for k in ("parameters", "reconstruction_mean_ms",
                     "reconstruction_median_ms", "gpu_peak_allocated_mib")}), flush=True)

