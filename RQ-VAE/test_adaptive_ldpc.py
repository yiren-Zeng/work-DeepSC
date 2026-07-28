"""Evaluate adaptive EMA-RQ with explicit masks over LDPC 1/2+BPSK+AWGN."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from config import Config
from data.datasets import get_dataloader
from evaluation.adaptive import (
    build_scan_points,
    collect_dense_adaptive_samples,
    pooled_first_stage_errors,
)
from evaluation.adaptive_channel import (
    evaluate_adaptive_ldpc_bpsk,
    prepare_adaptive_packets,
    summarize_prepared_packets,
)
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
DEFAULT_THRESHOLD_JSON = (
    PROJECT_ROOT
    / "experiments"
    / "adaptive_eval"
    / EXPERIMENT_NAME
    / "adaptive_scan_0to100_step10.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "adaptive_eval"
    / EXPERIMENT_NAME
    / "ldpc_bpsk_explicit_mask"
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


def _load_threshold_points(path, requested_targets, num_scales):
    path = Path(path).resolve()
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    points_by_target = {}
    for result in payload.get("results", []):
        targets = result.get("target_active_rates")
        thresholds = result.get("thresholds")
        if (
            not isinstance(targets, list)
            or not isinstance(thresholds, list)
            or len(targets) != num_scales
            or len(thresholds) != num_scales
        ):
            continue
        if any(abs(float(value) - float(targets[0])) > 1e-12 for value in targets):
            continue
        points_by_target[round(float(targets[0]), 12)] = {
            "source": "reused_nochannel_quantile_threshold",
            "target_active_rates": [float(value) for value in targets],
            "thresholds": [float(value) for value in thresholds],
        }
    requested = []
    for target in requested_targets:
        key = round(float(target), 12)
        if key not in points_by_target:
            return None
        requested.append(points_by_target[key])
    for index, point in enumerate(requested):
        point["scan_id"] = index
    return requested


def _csv_rows(payload):
    rows = []
    for point in payload["results"]:
        base = {
            "target_active_rate": point["target_active_rates"][0],
            "scale0_threshold": point["thresholds"][0],
            "scale1_threshold": point["thresholds"][1],
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


def run(args):
    setup_seed(42)
    cfg = Config()
    device = torch.device(cfg.DEVICE)
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    print(f"CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Logical device={device}")
    if device.type == "cuda":
        print(f"CUDA device name={torch.cuda.get_device_name(device)}")
    print(f"Loading checkpoint: {checkpoint_path}")
    model, inferred = build_model_from_checkpoint(
        str(checkpoint_path), cfg, device
    )
    if inferred["quantizer_type"] != "rq_ema":
        raise ValueError("adaptive LDPC evaluation requires rq_ema")
    num_embeddings = [int(value) for value in inferred["num_embeddings_list"]]
    rq_depths = [int(value) for value in inferred.get("rq_depth_list", [])]
    if any(depth != 2 for depth in rq_depths):
        raise ValueError(
            f"adaptive LDPC evaluation requires RQ depth 2, got {rq_depths}"
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
    print("Encoding Kodak once and collecting first-stage errors...")
    samples = collect_dense_adaptive_samples(
        model, loader, device, max_images=max_images
    )
    targets = [float(value) for value in args.target_active_rates]
    if any(not 0.0 <= target <= 1.0 for target in targets):
        raise ValueError("target active rates must be in [0,1]")
    threshold_points = None
    if args.threshold_json:
        threshold_points = _load_threshold_points(
            args.threshold_json, targets, len(num_embeddings)
        )
        if threshold_points is not None:
            print(f"Reusing thresholds from: {Path(args.threshold_json).resolve()}")
    if threshold_points is None:
        print("Calibrating thresholds from the current evaluation samples.")
        threshold_points = build_scan_points(
            pooled_first_stage_errors(samples),
            common_target_active_rates=targets,
        )

    from communications.ldpc_coding import get_ldpc_code

    ldpc_code = get_ldpc_code(128, rate=0.5)
    results = []
    for point_index, point in enumerate(threshold_points):
        target = point["target_active_rates"][0]
        print(
            f"\n[{point_index + 1}/{len(threshold_points)}] "
            f"target active={target:.0%}, thresholds={point['thresholds']}"
        )
        packets = prepare_adaptive_packets(
            model,
            samples,
            point["thresholds"],
            num_embeddings,
        )
        rate = summarize_prepared_packets(packets)
        nochannel = {
            "psnr": sum(packet.clean_psnr for packet in packets) / len(packets),
            "ms_ssim": sum(packet.clean_ms_ssim for packet in packets)
            / len(packets),
        }
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
                seed=42,
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
                **point,
                "rate": rate,
                "nochannel": nochannel,
                "channel_results": channel_results,
            }
        )

    output = {
        "schema_version": 1,
        "evaluation": "adaptive_explicit_mask_ldpc_bpsk",
        "checkpoint": str(checkpoint_path),
        "dataset": str(Path(args.dataset).resolve()),
        "num_images": len(samples),
        "quantizer_type": inferred["quantizer_type"],
        "num_embeddings_list": num_embeddings,
        "rq_depth_list": rq_depths,
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
                "Residual mask errors can shift compact active-index assignment "
                "within one scale; independent frames prevent cross-scale and "
                "cross-image propagation."
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
            "seed_per_rate_snr_point": 42,
        },
        "threshold_source": (
            str(Path(args.threshold_json).resolve())
            if args.threshold_json and Path(args.threshold_json).is_file()
            else "calibrated from current evaluation samples"
        ),
        "results": results,
    }
    json_path = _confined_output_path(args.json_output, "results.json")
    csv_path = _confined_output_path(args.csv_output, "results.csv")
    json_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    rows = _csv_rows(output)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nJSON: {json_path}")
    print(f"CSV:  {csv_path}")
    return output


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Adaptive rq_ema explicit-mask evaluation over LDPC 1/2+BPSK."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--dataset", default="/workspace/yi/work/Kodak-256-transform-resize"
    )
    parser.add_argument(
        "--threshold-json", default=str(DEFAULT_THRESHOLD_JSON)
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
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--csv-output", default=None)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
