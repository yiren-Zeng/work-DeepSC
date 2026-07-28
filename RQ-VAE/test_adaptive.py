"""Rate/distortion scan for adaptive second-stage EMA-RQ evaluation.

The legacy test_real.py path is intentionally untouched.  This entry point
encodes each image once, calibrates optional dataset-global error quantiles,
then reconstructs every threshold point with second-stage STOP sentinels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from config import Config
from data.datasets import get_dataloader
from evaluation.adaptive import (
    build_scan_points,
    collect_dense_adaptive_samples,
    evaluate_adaptive_point,
    parse_scale_pairs,
    pooled_first_stage_errors,
)
from utils.checkpoint_utils import build_model_from_checkpoint
from utils.reproducibility import setup_seed


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "quality_v2_B_larger_rate047_rq_ema_unet2_ds8x2_k4-2_d2-2"
    / "best_vq_deepsc.pth"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "adaptive_eval"
    / "quality_v2_B_larger_rate047_rq_ema_unet2_ds8x2_k4-2_d2-2"
)
DEFAULT_TARGET_ACTIVE_RATES = [1.0, 0.75, 0.5, 0.3, 0.2, 0.1, 0.0]


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


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _csv_rows(points):
    rows = []
    for point in points:
        rate = point["rate"]
        row = {
            "scan_id": point["scan_id"],
            "source": point["source"],
            "psnr": point["psnr"],
            "ms_ssim": point["ms_ssim"],
            "bpp": rate["bpp"],
            "ideal_bpp": rate["ideal_bpp"],
            "exact_raw_bpp": rate["exact_raw_bpp"],
            "dense_fixed_bpp": rate["dense_fixed_bpp"],
            "first_stage_fixed_bpp": rate["first_stage_fixed_bpp"],
            "mask_entropy_bpp": rate["mask_entropy_bpp"],
            "active_index_entropy_bpp": rate["active_index_entropy_bpp"],
            "joint_stop_index_entropy_bpp": rate[
                "joint_stop_index_entropy_bpp"
            ],
            "ideal_bits": rate["ideal_bits"],
            "exact_raw_bits": rate["exact_raw_bits"],
            "dense_fixed_bits": rate["dense_fixed_bits"],
            "num_images": rate["num_samples"],
            "total_image_pixels": rate["total_image_pixels"],
        }
        targets = point.get("target_active_rates")
        for scale, scale_rate in enumerate(rate["per_scale"]):
            prefix = f"scale{scale}_"
            row[prefix + "threshold"] = point["thresholds"][scale]
            row[prefix + "target_active_ratio"] = (
                "" if targets is None else targets[scale]
            )
            row[prefix + "active_ratio"] = scale_rate["active_ratio"]
            row[prefix + "active_tokens"] = scale_rate["active_tokens"]
            row[prefix + "stop_tokens"] = scale_rate["stop_tokens"]
            for key in (
                "first_stage_fixed_bits",
                "dense_fixed_bits",
                "raw_mask_bits",
                "raw_active_index_bits",
                "exact_raw_bits",
                "mask_entropy_bits",
                "active_index_entropy_bits",
                "joint_stop_index_entropy_bits",
                "ideal_bits",
            ):
                row[prefix + key] = scale_rate[key]
                row[prefix + key.removesuffix("_bits") + "_bpp"] = (
                    scale_rate[key] / rate["total_image_pixels"]
                )
        rows.append(row)
    return rows


def _write_csv(path, points):
    rows = _csv_rows(points)
    fieldnames = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(path, points):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "PNG output requires matplotlib; rerun with --no-plot to skip it"
        ) from error

    ordered = sorted(points, key=lambda item: item["rate"]["ideal_bpp"])
    bpp = [item["rate"]["ideal_bpp"] for item in ordered]
    psnr = [item["psnr"] for item in ordered]
    ms_ssim = [item["ms_ssim"] for item in ordered]
    labels = [
        "/".join(
            f"{scale['active_ratio']:.2f}" for scale in item["rate"]["per_scale"]
        )
        for item in ordered
    ]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(bpp, psnr, "o-", linewidth=1.5)
    axes[0].set_xlabel("Ideal entropy source rate (bpp)")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(bpp, ms_ssim, "o-", linewidth=1.5)
    axes[1].set_xlabel("Ideal entropy source rate (bpp)")
    axes[1].set_ylabel("MS-SSIM")
    axes[1].grid(True, alpha=0.3)
    for axis, values in zip(axes, (psnr, ms_ssim)):
        for x_value, y_value, label in zip(bpp, values, labels):
            axis.annotate(
                label,
                (x_value, y_value),
                xytext=(3, 4),
                textcoords="offset points",
                fontsize=7,
            )
    figure.suptitle("Adaptive EMA-RQ: scale0/scale1 second-stage active ratios")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(args):
    setup_seed(42)
    cfg = Config()
    device = torch.device(cfg.DEVICE)
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    model, inferred = build_model_from_checkpoint(
        str(checkpoint_path), cfg, device
    )
    if inferred["quantizer_type"] != "rq_ema":
        raise ValueError("adaptive scan requires an rq_ema checkpoint")
    num_embeddings = [int(value) for value in inferred["num_embeddings_list"]]
    rq_depths = [int(value) for value in inferred.get("rq_depth_list", [])]
    if any(depth != 2 for depth in rq_depths):
        raise ValueError(f"adaptive scan requires RQ depth [2,...], got {rq_depths}")

    threshold_pairs = parse_scale_pairs(
        args.threshold_pairs, len(num_embeddings), "--threshold-pairs"
    )
    target_pairs = parse_scale_pairs(
        args.target_active_rate_pairs,
        len(num_embeddings),
        "--target-active-rate-pairs",
    )
    common_targets = [float(value) for value in args.target_active_rates]
    for value in common_targets:
        if not 0.0 <= value <= 1.0:
            raise ValueError("--target-active-rates values must be in [0,1]")

    loader = get_dataloader(
        root_dir=args.dataset,
        batch_size=1,
        shuffle=False,
        mode="test",
        num_workers=args.num_workers,
        pin_memory=cfg.PIN_MEMORY,
    )
    max_images = None if args.max_images <= 0 else args.max_images
    print("Encoding dataset once and collecting first-stage errors...")
    samples = collect_dense_adaptive_samples(
        model, loader, device, max_images=max_images
    )
    errors = pooled_first_stage_errors(samples)
    scan_points = build_scan_points(
        errors,
        threshold_pairs=threshold_pairs,
        target_active_rate_pairs=target_pairs,
        common_target_active_rates=common_targets,
    )

    results = []
    for point in scan_points:
        print(
            f"[{point['scan_id'] + 1}/{len(scan_points)}] "
            f"{point['source']} thresholds={point['thresholds']}"
        )
        result = evaluate_adaptive_point(
            model, samples, point["thresholds"], num_embeddings
        )
        result.update(point)
        active = [
            scale["active_ratio"] for scale in result["rate"]["per_scale"]
        ]
        print(
            f"  active={active}, ideal_bpp={result['rate']['ideal_bpp']:.8f}, "
            f"PSNR={result['psnr']:.4f}, MS-SSIM={result['ms_ssim']:.6f}"
        )
        results.append(result)

    json_path = _confined_output_path(args.json_output, "adaptive_scan.json")
    csv_path = _confined_output_path(args.csv_output, "adaptive_scan.csv")
    plot_path = (
        None
        if args.no_plot
        else _confined_output_path(args.plot_output, "adaptive_scan.png")
    )
    payload = {
        "schema_version": 1,
        "evaluation": "adaptive_second_stage_rq_stop_scan",
        "checkpoint": str(checkpoint_path),
        "dataset": str(Path(args.dataset).resolve()),
        "num_images": int(sum(sample.image.shape[0] for sample in samples)),
        "image_sizes": [
            list(size)
            for size in sorted({
                (int(sample.image.shape[-2]), int(sample.image.shape[-1]))
                for sample in samples
            })
        ],
        "quantizer_type": inferred["quantizer_type"],
        "num_embeddings_list": num_embeddings,
        "rq_depth_list": rq_depths,
        "activation_rule": "second stage active iff first_stage_error >= threshold",
        "first_stage_error": "channel-mean squared residual after first RQ code",
        "quantile_scope": "global pooled tokens from this evaluation dataset, per scale",
        "scan_implementation": (
            "The encoder and dense second-stage indices are cached once only to "
            "accelerate repeated thresholds; inserting STOP gives the same codes "
            "and reconstruction as thresholded encoding. The core adaptive "
            "quantizer performs an actual sparse second-codebook lookup."
        ),
        "rate_contract": {
            "first_stage": "fixed-width ceil(log2(K)) index at every token",
            "second_stage_ideal": "zero-order joint entropy over {STOP,0,...,K-1}",
            "exact_raw": "one-bit activity mask plus fixed-width active indices",
            "dense_fixed": "fixed-width indices at both RQ depths",
            "overhead_counted": False,
            "overhead_note": (
                "Shape/header/model/threshold/entropy-table signalling overhead is not counted."
            ),
        },
        "results": results,
    }
    safe_payload = _json_safe(payload)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(safe_payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    _write_csv(csv_path, results)
    if plot_path is not None:
        _write_plot(plot_path, results)

    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    if plot_path is not None:
        print(f"PNG:  {plot_path}")
    return payload


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Scan independent per-scale adaptive second-RQ thresholds and "
            "dataset-global target activation quantiles."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--dataset", default="/workspace/yi/work/Kodak-256-transform-resize")
    parser.add_argument(
        "--threshold-pairs",
        nargs="*",
        default=[],
        metavar="T0,T1",
        help="Direct per-scale thresholds, e.g. 0.01,0.02 0.02,0.04.",
    )
    parser.add_argument(
        "--target-active-rates",
        type=float,
        nargs="*",
        default=DEFAULT_TARGET_ACTIVE_RATES,
        metavar="RATE",
        help=(
            "Dataset-global quantile targets applied equally to all scales. "
            "Pass the option with no values to disable this default scan."
        ),
    )
    parser.add_argument(
        "--target-active-rate-pairs",
        nargs="*",
        default=[],
        metavar="R0,R1",
        help="Independent per-scale target active rates, e.g. 0.5,0.2.",
    )
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--plot-output", default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
