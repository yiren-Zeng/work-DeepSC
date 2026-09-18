"""Profile reconstruction using configuration from the original eval script."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path("/workspace/yi/work/shiyan")
sys.path.insert(0, str(ROOT))
import torch
from config import Config
from data.datasets import get_dataloader
from utils.checkpoint_utils import build_model_from_checkpoint
from utils.reproducibility import setup_seed
from evaluation.quality import _image_quality

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--json-output", required=True)
args, _ = parser.parse_known_args()
cfg = Config()
cfg.validate()
setup_seed(42)
device = torch.device(cfg.DEVICE)
model, inferred = build_model_from_checkpoint(args.checkpoint, cfg, device)
model.eval()
parameter_counts = {
    "total": sum(p.numel() for p in model.parameters()),
    "encoder": sum(p.numel() for p in model.semantic_encoder.parameters()),
    "decoder": sum(p.numel() for p in model.semantic_decoder.parameters()),
    "raq_stage1": sum(p.numel() for p in model.raqs.parameters()),
    "raq_stage2": sum(p.numel() for p in model.raqs_rvq_stage2.parameters()),
}
loader = get_dataloader(cfg.TEST_DATASET_PATH, batch_size=1, shuffle=False,
                        mode="test", num_workers=0, pin_memory=False)
images = [image.to(device) for image in loader]

def reconstruct(image):
    out = model.forward_test(image)
    return model.reconstruct_from_indices(
        out["indices"], feature_shapes=out.get("feature_shapes"),
        codebooks=out.get("codebooks"))

measurements = []
with torch.inference_mode():
    for index in range(5):
        result = reconstruct(images[index % len(images)])
    torch.cuda.synchronize()
    del result
    baseline_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for repeat in range(3):
        for image in images:
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = reconstruct(image)
            torch.cuda.synchronize()
            measurements.append((time.perf_counter() - started) * 1000)
            del result

reconstruction_peak_allocated = torch.cuda.max_memory_allocated()
per_image = []
with torch.inference_mode():
    for index, image in enumerate(images):
        reconstructed = reconstruct(image)
        ms_ssim, psnr = _image_quality(image, reconstructed)
        per_image.append({
            "filename": loader.dataset.image_files[index],
            "psnr": float(psnr),
            "ms_ssim": float(ms_ssim),
        })

record = {
    "checkpoint": args.checkpoint,
    "gpu_name": torch.cuda.get_device_name(),
    "torch_version": torch.__version__,
    "inferred": inferred,
    "target_layout": (cfg.INDEPENDENT_RAQ_RVQ_K_LISTS
                      if cfg.USE_INDEPENDENT_RAQ_RVQ else cfg.RAQ_TARGET_LIST),
    "num_images": len(images),
    "input_shapes": sorted({str(tuple(image.shape)) for image in images}),
    "batch_size": 1,
    "dtype": "float32",
    "warmup_iterations": 5,
    "measured_repeats": 3,
    "measured_iterations": len(measurements),
    "parameters": parameter_counts,
    "reconstruction_mean_ms": statistics.mean(measurements),
    "reconstruction_median_ms": statistics.median(measurements),
    "reconstruction_p95_ms": sorted(measurements)[int(0.95 * (len(measurements) - 1))],
    "reconstruction_fps_from_mean": 1000 / statistics.mean(measurements),
    "gpu_allocated_baseline_mib": baseline_allocated / 2**20,
    "gpu_peak_allocated_mib": reconstruction_peak_allocated / 2**20,
    "gpu_incremental_peak_mib": (reconstruction_peak_allocated - baseline_allocated) / 2**20,
    "checkpoint_bytes": Path(args.checkpoint).stat().st_size,
    "latency_ms": measurements,
    "per_image_no_channel": per_image,
    "timing_scope": "encoder + per-image RAQ codebook generation + quantization + index reconstruction + decoder; includes GPU synchronization; excludes loading, quality metrics, LDPC, modulation, AWGN, and checkpoint startup",
}
Path(args.json_output).write_text(json.dumps(record, indent=2))
print(json.dumps({k: v for k, v in record.items() if k != "latency_ms"}, indent=2))
