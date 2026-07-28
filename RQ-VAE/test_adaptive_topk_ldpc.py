"""Evaluate per-image/per-scale Top-K EMA-RQ over LDPC 1/2+BPSK+AWGN."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from config import Config
from data.datasets import get_dataloader
from evaluation.adaptive import collect_dense_adaptive_samples
from evaluation.adaptive_channel import (
    evaluate_adaptive_ldpc_bpsk,
    summarize_prepared_packets,
)
from evaluation.adaptive_topk import prepare_per_image_topk_packets
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
    / "ldpc_bpsk_explicit_mask_per_image_topk"
)
DEFAULT_TARGET_ACTIVE_RATES = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]
DEFAULT_SNRS = [0, 3, 6, 9, 12]


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


def _sorted_image_names(loader, max_images):
    dataset = loader.dataset
    if hasattr(dataset, "image_files"):
        dataset.image_files.sort()
        names = [str(name) for name in dataset.image_files]
    else:
        names = [f"image_{index + 1:04d}" for index in range(len(dataset))]
    if max_images is not None:
        names = names[: int(max_images)]
    return names


def _threshold_summary(records, num_scales):
    summaries = []
    for scale in range(num_scales):
        scale_records = [
            image["scales"][scale]
            for image in records
        ]
        thresholds = [
            float(record["threshold"]) for record in scale_records
        ]
        summaries.append(
            {
                "scale": scale,
                "threshold_count": len(thresholds),
                "threshold_min": min(thresholds),
                "threshold_mean": sum(thresholds) / len(thresholds),
                "threshold_max": max(thresholds),
                "images_with_boundary_tie_split": sum(
                    int(record["threshold_splits_tie"])
                    for record in scale_records
                ),
                "target_active_count_per_image": int(
                    scale_records[0]["target_active_count"]
                ),
                "token_count_per_image": int(
                    scale_records[0]["token_count"]
                ),
                "actual_active_rate_per_image": float(
                    scale_records[0]["actual_active_rate"]
                ),
            }
        )
    return summaries


def _result_csv_rows(payload):
    rows = []
    for point in payload["results"]:
        threshold_summary = point["threshold_summary"]
        base = {
            "target_active_rate": point["target_active_rates"][0],
            "thresholds_per_point": point["threshold_count"],
            "scale0_threshold_min": threshold_summary[0]["threshold_min"],
            "scale0_threshold_mean": threshold_summary[0]["threshold_mean"],
            "scale0_threshold_max": threshold_summary[0]["threshold_max"],
            "scale1_threshold_min": threshold_summary[1]["threshold_min"],
            "scale1_threshold_mean": threshold_summary[1]["threshold_mean"],
            "scale1_threshold_max": threshold_summary[1]["threshold_max"],
            "scale0_active_count_per_image": threshold_summary[0][
                "target_active_count_per_image"
            ],
            "scale1_active_count_per_image": threshold_summary[1][
                "target_active_count_per_image"
            ],
            "scale0_actual_active_rate": point["rate"]["active_ratios"][0],
            "scale1_actual_active_rate": point["rate"]["active_ratios"][1],
            "source_bpp_explicit_mask": point["rate"]["source_bpp"],
            "coded_bpp_with_ldpc_padding": point["rate"][
                "coded_bpp_with_ldpc_padding"
            ],
            "bpsk_channel_uses_per_rgb_value": point["rate"][
                "bpsk_channel_uses_per_rgb_value"
            ],
            "nochannel_psnr": point["nochannel"]["psnr"],
            "nochannel_ms_ssim": point["nochannel"]["ms_ssim"],
        }
        for channel in point["channel_results"]:
            rows.append(
                {
                    **base,
                    "snr_db": channel["snr_db"],
                    "psnr": channel["psnr"],
                    "ms_ssim": channel["ms_ssim"],
                    "source_ber_after_ldpc": channel[
                        "source_ber_after_ldpc"
                    ],
                    "source_bit_errors": channel["source_bit_errors"],
                    "mask_bit_errors": channel["segment_bit_errors"]["mask"],
                    "images_with_source_errors": channel[
                        "images_with_source_errors"
                    ],
                    "mask_active_count_mismatch_scale_events": channel[
                        "mask_active_count_mismatch_scale_events"
                    ],
                    "zero_filled_second_indices": channel[
                        "zero_filled_second_indices"
                    ],
                    "truncated_second_indices": channel[
                        "truncated_second_indices"
                    ],
                }
            )
    return rows


def _threshold_csv_rows(payload):
    rows = []
    for point in payload["results"]:
        target = point["target_active_rates"][0]
        for image in point["per_image_selection"]:
            row = {
                "target_active_rate": target,
                "image_index": image["image_index"],
                "image_number": image["image_number"],
                "image_name": image["image_name"],
            }
            for scale_record in image["scales"]:
                prefix = f"scale{scale_record['scale']}_"
                for name in (
                    "threshold",
                    "token_count",
                    "target_active_count",
                    "active_count",
                    "actual_active_rate",
                    "strict_above_threshold_count",
                    "threshold_equal_count",
                    "threshold_equal_selected_count",
                    "threshold_splits_tie",
                    "first_inactive_error",
                ):
                    row[prefix + name] = scale_record[name]
            rows.append(row)
    return rows


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows available for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
    if inferred["quantizer_type"] != "rq_ema":
        raise ValueError("adaptive Top-K LDPC evaluation requires rq_ema")
    num_embeddings = [
        int(value) for value in inferred["num_embeddings_list"]
    ]
    rq_depths = [
        int(value) for value in inferred.get("rq_depth_list", [])
    ]
    if (
        len(num_embeddings) != 2
        or len(rq_depths) != len(num_embeddings)
        or any(depth != 2 for depth in rq_depths)
    ):
        raise ValueError(
            "adaptive Top-K evaluation requires exactly two scales with "
            f"RQ depth [2,2], got K={num_embeddings}, depths={rq_depths}"
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
        "Encoding images once and collecting per-image first-stage errors "
        "in sorted filename order..."
    )
    samples = collect_dense_adaptive_samples(
        model, loader, device, max_images=max_images
    )
    image_names = image_names[: len(samples)]
    if len(image_names) != len(samples):
        raise RuntimeError("image-name and encoded-sample counts differ")

    targets = [float(value) for value in args.target_active_rates]
    if any(not 0.0 <= target <= 1.0 for target in targets):
        raise ValueError("target active rates must be in [0,1]")

    from communications.ldpc_coding import get_ldpc_code

    ldpc_code = get_ldpc_code(128, rate=0.5)
    results = []
    for point_index, target in enumerate(targets):
        target_rates = [target] * len(num_embeddings)
        print(
            f"\n[{point_index + 1}/{len(targets)}] "
            f"per-image/per-scale Top-K target={target:.0%}"
        )
        packets, selection_records = prepare_per_image_topk_packets(
            model,
            samples,
            target_rates,
            num_embeddings,
            image_names=image_names,
        )
        threshold_summary = _threshold_summary(
            selection_records, len(num_embeddings)
        )
        threshold_count = sum(
            len(image["scales"]) for image in selection_records
        )
        expected_threshold_count = len(samples) * len(num_embeddings)
        if threshold_count != expected_threshold_count:
            raise RuntimeError(
                "per-image threshold count mismatch: "
                f"{threshold_count} vs {expected_threshold_count}"
            )

        rate = summarize_prepared_packets(packets)
        nochannel = {
            "psnr": sum(packet.clean_psnr for packet in packets)
            / len(packets),
            "ms_ssim": sum(packet.clean_ms_ssim for packet in packets)
            / len(packets),
        }
        print(
            f"  thresholds={threshold_count} "
            f"({len(samples)} images x {len(num_embeddings)} scales), "
            f"active/image={[item['target_active_count_per_image'] for item in threshold_summary]}"
        )
        print(
            f"  actual active={rate['active_ratios']}, "
            f"raw source bpp={rate['source_bpp']:.8f}, "
            f"coded bpp={rate['coded_bpp_with_ldpc_padding']:.8f}, "
            f"no-channel PSNR={nochannel['psnr']:.4f}"
        )

        channel_results = []
        for snr in args.snrs:
            channel = evaluate_adaptive_ldpc_bpsk(
                model,
                packets,
                float(snr),
                ldpc_code,
                device,
                seed=args.seed,
            )
            channel_results.append(channel)
            print(
                f"  SNR={snr:g} dB | PSNR={channel['psnr']:.4f} | "
                f"MS-SSIM={channel['ms_ssim']:.6f} | "
                f"post-LDPC BER={channel['source_ber_after_ldpc']:.8g} | "
                f"mask errors={channel['segment_bit_errors']['mask']}"
            )

        results.append(
            {
                "scan_id": point_index,
                "source": "per_image_per_scale_exact_topk",
                "target_active_rates": target_rates,
                "threshold_count": threshold_count,
                "threshold_summary": threshold_summary,
                "per_image_selection": selection_records,
                "rate": rate,
                "nochannel": nochannel,
                "channel_results": channel_results,
            }
        )

    output = {
        "schema_version": 1,
        "evaluation": (
            "adaptive_per_image_per_scale_topk_explicit_mask_ldpc_bpsk"
        ),
        "checkpoint": str(checkpoint_path),
        "dataset": str(Path(args.dataset).resolve()),
        "image_order": image_names,
        "num_images": len(samples),
        "quantizer_type": inferred["quantizer_type"],
        "num_embeddings_list": num_embeddings,
        "rq_depth_list": rq_depths,
        "selection": {
            "scope": "each image and each scale independently",
            "active_count_rule": "floor(target_active_rate * token_count + 0.5)",
            "ordering": (
                "first-stage error descending; exact ties use ascending "
                "raster/flat token index"
            ),
            "threshold_definition": (
                "minimum selected error (the K-th largest); at K=0 use "
                "nextafter(max_error,+inf)"
            ),
            "mask_is_authoritative_at_boundary_ties": True,
            "thresholds_per_image": len(num_embeddings),
            "thresholds_per_scan_point": len(samples)
            * len(num_embeddings),
            "thresholds_transmitted": False,
            "threshold_bits_counted": False,
            "implementation_note": (
                "Dense depth-two indices are cached once for an exact "
                "rate-distortion scan; the selected output is numerically "
                "equivalent to sparse lookup but this evaluation does not "
                "measure sparse-lookup runtime."
            ),
        },
        "transport": {
            "representation": (
                "fixed first indices + one-bit explicit mask per token + "
                "fixed-width active second indices"
            ),
            "logical_frames_per_image": (
                "first_s0, first_s1, mask_s0, mask_s1, "
                "second_active_s0, second_active_s1"
            ),
            "mask_decode_rule": (
                "Map A_tx decoded refinement indices onto received-mask active "
                "positions in raster order; truncate surplus payload or "
                "zero-fill surplus received active positions."
            ),
            "framing_metadata": (
                "segment boundaries, shapes, K, and original source lengths "
                "are assumed known out of band"
            ),
            "framing_overhead_counted": False,
            "mask_error_note": (
                "Residual mask errors can shift compact active-index "
                "assignment within one scale; independent frames prevent "
                "cross-scale and cross-image propagation."
            ),
        },
        "channel": {
            "ldpc": "Sionna 5G LDPC",
            "k": 128,
            "n": 256,
            "rate": 0.5,
            "modulation": "bpsk",
            "channel": "awgn",
            "snrs_db": [float(value) for value in args.snrs],
            "seed_per_rate_snr_point": int(args.seed),
        },
        "results": results,
    }

    json_path = _confined_output_path(args.json_output, "results.json")
    csv_path = _confined_output_path(args.csv_output, "results.csv")
    thresholds_path = _confined_output_path(
        args.thresholds_csv_output, "per_image_thresholds.csv"
    )
    json_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(csv_path, _result_csv_rows(output))
    _write_csv(thresholds_path, _threshold_csv_rows(output))
    print(f"\nJSON:       {json_path}")
    print(f"CSV:        {csv_path}")
    print(f"Thresholds: {thresholds_path}")
    return output


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Per-image/per-scale exact Top-K rq_ema explicit-mask evaluation "
            "over LDPC 1/2+BPSK."
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
    parser.add_argument(
        "--snrs", type=float, nargs="+", default=DEFAULT_SNRS
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--thresholds-csv-output", default=None)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
