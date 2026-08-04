"""Evaluate Top-K adaptive EMA-RQ with lossy RLE masks at BPSK SNR 0 dB."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from config import Config
from data.datasets import get_dataloader
from evaluation.adaptive import collect_dense_adaptive_samples
from evaluation.adaptive_rle_channel import (
    evaluate_rle_ldpc_bpsk,
    prepare_per_image_topk_rle_packets,
    summarize_rle_packets,
)
from test_adaptive_topk_ldpc import (
    _sorted_image_names,
    _threshold_csv_rows,
    _threshold_summary,
    _write_csv,
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
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "adaptive_eval"
    / EXPERIMENT_NAME
    / "ldpc_bpsk_rle_mask_per_image_topk_snr0"
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


def _summary_csv_rows(payload):
    rows = []
    for point in payload["results"]:
        rate = point["rate"]
        channel = point["channel"]
        scale0_rate, scale1_rate = rate["per_scale"]
        scale0_rx, scale1_rx = channel["per_scale"]
        rows.append(
            {
                "target_active_rate": point["target_active_rates"][0],
                "scale0_active_count_per_image": scale0_rate[
                    "tx_active_count_per_image"
                ],
                "scale1_active_count_per_image": scale1_rate[
                    "tx_active_count_per_image"
                ],
                "scale0_rle_runs_mean_per_image": scale0_rate[
                    "rle_run_count_mean_per_image"
                ],
                "scale1_rle_runs_mean_per_image": scale1_rate[
                    "rle_run_count_mean_per_image"
                ],
                "scale0_raw_mask_bits_per_image": (
                    scale0_rate["raw_mask_source_bits_reference"]
                    / rate["num_images"]
                ),
                "scale1_raw_mask_bits_per_image": (
                    scale1_rate["raw_mask_source_bits_reference"]
                    / rate["num_images"]
                ),
                "scale0_rle_mask_bits_mean_per_image": scale0_rate[
                    "rle_mask_source_bits_mean_per_image"
                ],
                "scale1_rle_mask_bits_mean_per_image": scale1_rate[
                    "rle_mask_source_bits_mean_per_image"
                ],
                "scale0_rle_mask_source_saving_ratio": scale0_rate[
                    "mask_source_saving_ratio"
                ],
                "scale1_rle_mask_source_saving_ratio": scale1_rate[
                    "mask_source_saving_ratio"
                ],
                "raw_source_bits_mean_per_image": (
                    rate["raw_explicit_source_bits_reference"]
                    / rate["num_images"]
                ),
                "rle_source_bits_mean_per_image": (
                    rate["rle_source_bits"] / rate["num_images"]
                ),
                "raw_source_bpp_reference": rate[
                    "raw_explicit_source_bpp_reference"
                ],
                "rle_source_bpp": rate["rle_source_bpp"],
                "source_saving_ratio_vs_raw": rate[
                    "source_saving_ratio_vs_raw"
                ],
                "raw_coded_bits_mean_per_image": (
                    rate[
                        "raw_explicit_coded_bits_with_ldpc_padding_reference"
                    ]
                    / rate["num_images"]
                ),
                "rle_coded_bits_mean_per_image": (
                    rate["rle_coded_bits_with_ldpc_padding"]
                    / rate["num_images"]
                ),
                "raw_coded_bpp_reference": rate[
                    "raw_explicit_coded_bpp_reference"
                ],
                "rle_coded_bpp": rate["rle_coded_bpp"],
                "rle_bpsk_channel_uses_per_rgb_value": rate[
                    "rle_bpsk_channel_uses_per_rgb_value"
                ],
                "snr_db": channel["snr_db"],
                "psnr": channel["psnr"],
                "ms_ssim": channel["ms_ssim"],
                "source_ber_after_ldpc": channel[
                    "source_ber_after_ldpc"
                ],
                "source_bit_errors": channel["source_bit_errors"],
                "rle_mask_source_bit_errors": channel[
                    "segment_bit_errors"
                ]["mask_rle"],
                "images_with_source_errors": channel[
                    "images_with_source_errors"
                ],
                "scale0_rx_active_ratio": scale0_rx["rx_active_ratio"],
                "scale1_rx_active_ratio": scale1_rx["rx_active_ratio"],
                "scale0_semantic_mask_ber": scale0_rx[
                    "semantic_mask_ber"
                ],
                "scale1_semantic_mask_ber": scale1_rx[
                    "semantic_mask_ber"
                ],
                "scale0_structurally_valid_rle_frame_rate": scale0_rx[
                    "structurally_valid_rle_frame_rate"
                ],
                "scale1_structurally_valid_rle_frame_rate": scale1_rx[
                    "structurally_valid_rle_frame_rate"
                ],
                "scale0_exact_mask_frame_rate": scale0_rx[
                    "exact_mask_frame_rate"
                ],
                "scale1_exact_mask_frame_rate": scale1_rx[
                    "exact_mask_frame_rate"
                ],
            }
        )
    return rows


def _per_scale_csv_rows(payload):
    rows = []
    for point in payload["results"]:
        target = point["target_active_rates"][0]
        for rate_stats, channel_stats in zip(
            point["rate"]["per_scale"], point["channel"]["per_scale"]
        ):
            row = {
                "target_active_rate": target,
                **{
                    f"tx_{key}": value
                    for key, value in rate_stats.items()
                    if key != "scale"
                },
                **{
                    f"rx_{key}": value
                    for key, value in channel_stats.items()
                    if key != "scale"
                },
                "scale": rate_stats["scale"],
            }
            rows.append(row)
    return rows


def _per_image_rle_csv_rows(payload):
    rows = []
    for point in payload["results"]:
        target = point["target_active_rates"][0]
        selections = point["per_image_selection"]
        channel_images = point["channel"]["per_image"]
        for selection, channel_image in zip(selections, channel_images):
            for scale_selection, scale_channel in zip(
                selection["scales"], channel_image["scales"]
            ):
                rows.append(
                    {
                        "target_active_rate": target,
                        "image_index": selection["image_index"],
                        "image_number": selection["image_number"],
                        "image_name": selection["image_name"],
                        "scale": scale_selection["scale"],
                        "threshold": scale_selection["threshold"],
                        "tx_active_count": scale_channel[
                            "tx_active_count"
                        ],
                        "rx_active_count": scale_channel[
                            "rx_active_count"
                        ],
                        "rle_run_count": scale_channel["rle_run_count"],
                        "rle_source_bits": scale_channel[
                            "rle_source_bits"
                        ],
                        "rle_coded_bits": scale_channel["rle_coded_bits"],
                        "rle_source_bit_errors": scale_channel[
                            "rle_source_bit_errors"
                        ],
                        "semantic_mask_bit_errors": scale_channel[
                            "semantic_mask_bit_errors"
                        ],
                        "semantic_mask_ber": scale_channel[
                            "semantic_mask_ber"
                        ],
                        "structurally_valid": scale_channel[
                            "structurally_valid"
                        ],
                        "length_sum_error": scale_channel[
                            "length_sum_error"
                        ],
                        "zero_filled_second_indices": scale_channel[
                            "zero_filled_second_indices"
                        ],
                        "truncated_second_indices": scale_channel[
                            "truncated_second_indices"
                        ],
                        "image_source_bit_errors": channel_image[
                            "source_bit_errors"
                        ],
                        "image_psnr": channel_image["psnr"],
                        "image_ms_ssim": channel_image["ms_ssim"],
                    }
                )
    return rows


def run(args):
    if float(args.snr) != 0.0:
        raise ValueError("this dedicated evaluation is locked to SNR=0 dB")
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
            "RLE adaptive evaluation requires rq_ema K=[4,2], "
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
        "Encoding sorted Kodak images once and caching dense depth-two "
        "indices plus first-stage token errors..."
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
    from communications.ldpc_coding import get_ldpc_code

    ldpc_code = get_ldpc_code(128, rate=0.5)
    results = []
    for point_index, target in enumerate(targets):
        target_rates = [target, target]
        print(
            f"\n[{point_index + 1}/{len(targets)}] "
            f"Top-K target={target:.0%}, fixed-width RLE masks"
        )
        packets, selection_records = prepare_per_image_topk_rle_packets(
            model,
            samples,
            target_rates,
            num_embeddings,
            image_names=image_names,
        )
        rate = summarize_rle_packets(packets)
        threshold_summary = _threshold_summary(selection_records, 2)
        threshold_count = sum(
            len(image["scales"]) for image in selection_records
        )
        if threshold_count != len(samples) * 2:
            raise RuntimeError("per-image threshold count mismatch")
        channel = evaluate_rle_ldpc_bpsk(
            model,
            packets,
            0.0,
            ldpc_code,
            device,
            seed=args.seed,
        )
        print(
            "  active/image="
            f"{[scale['tx_active_count_per_image'] for scale in rate['per_scale']]}"
        )
        print(
            "  RLE mask bits/image="
            f"{[scale['rle_mask_source_bits_mean_per_image'] for scale in rate['per_scale']]} "
            "(raw=[1024,256])"
        )
        print(
            f"  total source bits/image={rate['rle_source_bits']/len(samples):.2f} "
            f"(raw={rate['raw_explicit_source_bits_reference']/len(samples):.2f}), "
            f"coded bits/image={rate['rle_coded_bits_with_ldpc_padding']/len(samples):.2f}"
        )
        print(
            f"  0 dB PSNR={channel['psnr']:.6f}, "
            f"MS-SSIM={channel['ms_ssim']:.9f}, "
            f"post-LDPC BER={channel['source_ber_after_ldpc']:.9g}, "
            f"RLE source errors={channel['segment_bit_errors']['mask_rle']}"
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
                "channel": channel,
            }
        )

    output = {
        "schema_version": 1,
        "evaluation": (
            "adaptive_per_image_per_scale_topk_fixed_width_rle_mask_"
            "ldpc_half_bpsk_snr0"
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
            "active_count_rule": (
                "floor(target_active_rate * token_count + 0.5)"
            ),
            "ordering": (
                "first-stage error descending; ties use ascending raster index"
            ),
            "thresholds_per_scan_point": len(samples) * 2,
            "thresholds_transmitted": False,
            "threshold_bits_counted": False,
        },
        "transport": {
            "mask_scan_order": "row-major raster order",
            "mask_coding": (
                "one start bit plus fixed-width binary (run_length-1) fields"
            ),
            "run_length_bits_per_scale": [10, 8],
            "raw_mask_fallback": False,
            "logical_frames_per_image": (
                "first_s0, first_s1, mask_rle_s0, mask_rle_s1, "
                "second_active_s0, second_active_s1"
            ),
            "corrupt_rle_concealment": (
                "crop decoded overflow; extend the last emitted run value "
                "through decoded underflow"
            ),
            "active_payload_mapping": (
                "map A_tx decoded refinement indices to received-mask active "
                "positions in raster order; truncate surplus payload or "
                "zero-fill surplus received active positions"
            ),
            "framing_metadata": (
                "segment boundaries, shapes, K, source segment lengths, and "
                "A_tx are assumed known out of band"
            ),
            "framing_overhead_counted": False,
        },
        "channel": {
            "ldpc": "Sionna 5G LDPC",
            "k": 128,
            "n": 256,
            "rate": 0.5,
            "modulation": "bpsk",
            "channel": "awgn",
            "snr_db": 0.0,
            "seed_per_activation_point": int(args.seed),
        },
        "results": results,
    }

    json_path = _confined_output_path(args.json_output, "results.json")
    summary_path = _confined_output_path(
        args.csv_output, "results_summary.csv"
    )
    scale_path = _confined_output_path(
        args.per_scale_csv_output, "per_scale_bits_and_channel.csv"
    )
    image_path = _confined_output_path(
        args.per_image_csv_output, "per_image_rle_channel.csv"
    )
    thresholds_path = _confined_output_path(
        args.thresholds_csv_output, "per_image_thresholds.csv"
    )
    json_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(summary_path, _summary_csv_rows(output))
    _write_csv(scale_path, _per_scale_csv_rows(output))
    _write_csv(image_path, _per_image_rle_csv_rows(output))
    _write_csv(thresholds_path, _threshold_csv_rows(output))
    print(f"\nJSON:       {json_path}")
    print(f"Summary:    {summary_path}")
    print(f"Per-scale:  {scale_path}")
    print(f"Per-image:  {image_path}")
    print(f"Thresholds: {thresholds_path}")
    return output


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Lossy per-image/per-scale Top-K RLE-mask evaluation over "
            "Sionna LDPC 1/2, BPSK, AWGN at SNR=0 dB."
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
    parser.add_argument("--snr", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--per-scale-csv-output", default=None)
    parser.add_argument("--per-image-csv-output", default=None)
    parser.add_argument("--thresholds-csv-output", default=None)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
