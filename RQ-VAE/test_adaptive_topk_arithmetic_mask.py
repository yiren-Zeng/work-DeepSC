"""Measure actual arithmetic-code lengths of per-image Top-K masks only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from config import Config
from data.datasets import get_dataloader
from evaluation.adaptive import collect_dense_adaptive_samples
from evaluation.adaptive_arithmetic import evaluate_topk_arithmetic_masks
from test_adaptive_topk_ldpc import _sorted_image_names
from utils.checkpoint_utils import build_model_from_checkpoint
from utils.reproducibility import setup_seed


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_NAME = (
    "quality_v2_B_larger_rate047_rq_ema_unet2_ds8x2_k4-2_d2-2"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / EXPERIMENT_NAME
    / "best_vq_deepsc.pth"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "adaptive_eval"
    / EXPERIMENT_NAME
    / "arithmetic_mask_per_image_topk"
)
DEFAULT_TARGET_ACTIVE_RATES = [step / 10 for step in range(11)]


def _confined_output_path(path, default_name):
    candidate = Path(path) if path else DEFAULT_OUTPUT_DIR / default_name
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            f"output path must stay inside {PROJECT_ROOT}: {candidate}"
        ) from error
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows available for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(output):
    rows = []
    for point in output["results"]:
        stats = point["mask_stats"]
        scale0, scale1 = stats["per_scale"]
        rows.append(
            {
                "target_active_rate": point["target_active_rates"][0],
                "scale0_active_count_per_image": scale0[
                    "active_count_per_image"
                ],
                "scale1_active_count_per_image": scale1[
                    "active_count_per_image"
                ],
                "scale0_raw_mask_bits_per_image": scale0[
                    "raw_mask_bits_mean_per_image"
                ],
                "scale0_arithmetic_mask_bits_mean_per_image": scale0[
                    "arithmetic_mask_bits_mean_per_image"
                ],
                "scale1_raw_mask_bits_per_image": scale1[
                    "raw_mask_bits_mean_per_image"
                ],
                "scale1_arithmetic_mask_bits_mean_per_image": scale1[
                    "arithmetic_mask_bits_mean_per_image"
                ],
                "raw_mask_bits_per_image": stats[
                    "raw_mask_bits_mean_per_image"
                ],
                "arithmetic_mask_bits_mean_per_image": stats[
                    "arithmetic_mask_bits_mean_per_image"
                ],
                "mask_bits_saved_mean_per_image": (
                    stats["mask_bits_saved"] / stats["num_images"]
                ),
                "mask_saving_ratio": stats["mask_saving_ratio"],
                "all_roundtrips_exact": stats["all_roundtrips_exact"],
            }
        )
    return rows


def _per_scale_rows(output):
    rows = []
    for point in output["results"]:
        target = point["target_active_rates"][0]
        for scale in point["mask_stats"]["per_scale"]:
            rows.append({"target_active_rate": target, **scale})
    return rows


def _per_image_rows(output):
    rows = []
    for point in output["results"]:
        target = point["target_active_rates"][0]
        for image in point["per_image"]:
            for scale in image["scales"]:
                rows.append(
                    {
                        "target_active_rate": target,
                        "image_index": image["image_index"],
                        "image_number": image["image_number"],
                        "image_name": image["image_name"],
                        **scale,
                    }
                )
    return rows


def run(args):
    setup_seed(args.seed)
    cfg = Config()
    device = torch.device(cfg.DEVICE)
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    print(
        "CUDA_VISIBLE_DEVICES="
        f"{__import__('os').environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    print(f"Logical device={device}")
    if device.type == "cuda":
        print(f"CUDA device name={torch.cuda.get_device_name(device)}")
    print(f"Loading checkpoint: {checkpoint_path}")
    model, inferred = build_model_from_checkpoint(
        str(checkpoint_path), cfg, device
    )
    num_embeddings = [
        int(value) for value in inferred["num_embeddings_list"]
    ]
    rq_depths = [
        int(value) for value in inferred.get("rq_depth_list", [])
    ]
    if (
        inferred["quantizer_type"] != "rq_ema"
        or num_embeddings != [4, 2]
        or rq_depths != [2, 2]
    ):
        raise ValueError(
            "arithmetic-mask evaluation requires rq_ema K=[4,2], "
            f"depth=[2,2], got type={inferred['quantizer_type']}, "
            f"K={num_embeddings}, depth={rq_depths}"
        )

    loader = get_dataloader(
        root_dir=args.dataset,
        batch_size=1,
        shuffle=False,
        mode="test",
        num_workers=args.num_workers,
        pin_memory=cfg.PIN_MEMORY,
    )
    max_images = None if args.max_images <= 0 else args.max_images
    image_names = _sorted_image_names(loader, max_images)
    print(
        "Encoding images once and caching first-stage token errors; "
        "no reconstruction/channel evaluation will run..."
    )
    samples = collect_dense_adaptive_samples(
        model, loader, device, max_images=max_images
    )
    image_names = image_names[: len(samples)]
    if len(image_names) != len(samples):
        raise RuntimeError("image-name and sample counts differ")

    targets = [float(value) for value in args.target_active_rates]
    if any(not 0.0 <= target <= 1.0 for target in targets):
        raise ValueError("target active rates must be in [0,1]")

    results = []
    for point_index, target in enumerate(targets):
        target_rates = [target, target]
        stats, per_image = evaluate_topk_arithmetic_masks(
            samples, target_rates, image_names=image_names
        )
        per_scale_bits = [
            scale["arithmetic_mask_bits_mean_per_image"]
            for scale in stats["per_scale"]
        ]
        print(
            f"[{point_index + 1}/{len(targets)}] target={target:.0%}: "
            f"arithmetic mask bits/image={per_scale_bits}, "
            f"total={stats['arithmetic_mask_bits_mean_per_image']:.3f}, "
            f"saving={stats['mask_saving_ratio']:.3%}, "
            f"roundtrip={stats['all_roundtrips_exact']}"
        )
        results.append(
            {
                "scan_id": point_index,
                "target_active_rates": target_rates,
                "mask_stats": stats,
                "per_image": per_image,
            }
        )

    output = {
        "schema_version": 1,
        "evaluation": "adaptive_per_image_topk_arithmetic_mask_only",
        "checkpoint": str(checkpoint_path),
        "dataset": str(Path(args.dataset).resolve()),
        "image_order": image_names,
        "num_images": len(samples),
        "quantizer_type": inferred["quantizer_type"],
        "num_embeddings_list": num_embeddings,
        "rq_depth_list": rq_depths,
        "selection": {
            "scope": "each image and each scale independently",
            "active_count_rule": (
                "floor(target_active_rate * token_count + 0.5)"
            ),
            "ordering": (
                "first-stage error descending; ties use ascending raster index"
            ),
            "mask_value": "1=second depth active, 0=STOP",
        },
        "mask_coding": {
            "codec": "32-bit integer binary arithmetic coder",
            "probability_model": (
                "adaptive Laplace counts initialized as zero=1, one=1; "
                "counts update after every decoded symbol"
            ),
            "independent_segments": "one segment per image and scale",
            "scan_order": "row-major raster order",
            "lossless_roundtrip_required": True,
            "counted_bits": "arithmetic mask payload only",
            "excluded_bits": (
                "first-depth indices, active second-depth indices, LDPC, "
                "BPSK, probability headers"
            ),
            "framing_metadata": (
                "token count and arithmetic segment payload length are "
                "assumed known out of band"
            ),
            "framing_overhead_counted": False,
        },
        "results": results,
    }

    json_path = _confined_output_path(args.json_output, "results.json")
    summary_path = _confined_output_path(
        args.csv_output, "mask_summary.csv"
    )
    scale_path = _confined_output_path(
        args.per_scale_csv_output, "mask_per_scale.csv"
    )
    image_path = _confined_output_path(
        args.per_image_csv_output, "mask_per_image.csv"
    )
    json_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(summary_path, _summary_rows(output))
    _write_csv(scale_path, _per_scale_rows(output))
    _write_csv(image_path, _per_image_rows(output))
    print(f"JSON:      {json_path}")
    print(f"Summary:   {summary_path}")
    print(f"Per-scale: {scale_path}")
    print(f"Per-image: {image_path}")
    return output


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Actual lossless arithmetic-code statistics for Top-K masks only"
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--dataset", default="/workspace/yi/work/Kodak-256-transform-resize"
    )
    parser.add_argument(
        "--target-active-rates",
        type=float,
        nargs="+",
        default=DEFAULT_TARGET_ACTIVE_RATES,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--per-scale-csv-output", default=None)
    parser.add_argument("--per-image-csv-output", default=None)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
