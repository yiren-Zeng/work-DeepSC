"""Evaluate independent two-stage RAQ-RVQ with Top-K RLE masks.

The existing dense ``test_real.py`` path is deliberately untouched.  This
entry point keeps every first-stage index, selects an exact per-image/per-scale
Top-K subset of second-stage indices, and sends one combined LDPC/modulation
stream per image.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch

from config import Config
from data.datasets import get_dataloader
from evaluation.independent_raq_rvq_adaptive import (
    collect_independent_adaptive_samples,
    evaluate_packets_over_channel,
    prepare_topk_rle_packets,
    summarize_packets,
)
from utils.checkpoint_utils import build_model_from_checkpoint
from utils.reproducibility import setup_seed


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_NAME = (
    "shiyan_independent_raq_rvq_src64-64_trg2-64_d2_curriculum_"
    "rate094_A_patch_ch256-512_unet2_ds8x2_k64"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / EXPERIMENT_NAME / "best_vq_deepsc.pth"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "eval"
    / "independent_raq_rvq_src64_64_d2_adaptive_topk_rle_combined"
)
DEFAULT_TARGET_ACTIVE_RATES = [step / 10 for step in range(11)]
DEFAULT_SNRS = [3.0]


def parse_nested_int_lists(value: str) -> List[List[int]]:
    """Parse ``4,2;8,2`` or JSON nested lists."""

    raw = str(value).strip()
    if not raw:
        raise ValueError("RVQ K lists must not be empty")
    if raw.startswith("["):
        parsed = json.loads(raw)
        if not isinstance(parsed, list) or any(
            not isinstance(row, list) for row in parsed
        ):
            raise ValueError("RVQ K lists JSON must be nested")
        return [[int(item) for item in row] for row in parsed]
    return [
        [int(item.strip()) for item in row.split(",") if item.strip()]
        for row in raw.split(";")
        if row.strip()
    ]


def parse_rate_pairs(
    values: Iterable[str],
    num_scales: int,
) -> List[List[float]]:
    pairs = []
    for raw in values:
        parts = [part.strip() for part in str(raw).split(",") if part.strip()]
        if len(parts) != num_scales:
            raise ValueError(
                f"active-rate pair {raw!r} has {len(parts)} scales; "
                f"expected {num_scales}"
            )
        rates = [float(part) for part in parts]
        if any(not 0.0 <= rate <= 1.0 for rate in rates):
            raise ValueError("active rates must be in [0,1]")
        pairs.append(rates)
    return pairs


def build_activation_points(
    common_rates: Sequence[float],
    pair_values: Sequence[str],
    num_scales: int,
) -> List[List[float]]:
    points = []
    for rate in common_rates:
        rate = float(rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError("active rates must be in [0,1]")
        points.append([rate] * num_scales)
    points.extend(parse_rate_pairs(pair_values, num_scales))
    if not points:
        raise ValueError("at least one activation point is required")
    unique = []
    seen = set()
    for point in points:
        key = tuple(round(value, 12) for value in point)
        if key not in seen:
            seen.add(key)
            unique.append(point)
    return unique


def _output_path(path: str | None, default_name: str) -> Path:
    candidate = Path(path) if path else DEFAULT_OUTPUT_DIR / default_name
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def _image_names(loader, sample_count: int) -> List[str]:
    dataset = getattr(loader, "dataset", None)
    names = getattr(dataset, "image_files", None)
    if isinstance(names, list) and len(names) >= sample_count:
        return [str(name) for name in names[:sample_count]]
    return [f"image_{index + 1:04d}" for index in range(sample_count)]


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(payload: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for point in payload["results"]:
        rate = point["rate"]
        base = {
            "scan_id": point["scan_id"],
            "target_active_rate_scale0": point["target_active_rates"][0],
            "target_active_rate_scale1": point["target_active_rates"][1],
            "actual_active_rate_scale0": rate["per_scale"][0][
                "tx_active_ratio"
            ],
            "actual_active_rate_scale1": rate["per_scale"][1][
                "tx_active_ratio"
            ],
            "nochannel_psnr": point["nochannel"]["psnr"],
            "nochannel_ms_ssim": point["nochannel"]["ms_ssim"],
            "rle_source_bits": rate["rle_source_bits"],
            "rle_source_bpp": rate["rle_source_bpp"],
            "rle_coded_bits": rate["rle_coded_bits"],
            "rle_coded_bpp": rate["rle_coded_bpp"],
            "rle_channel_symbols": rate["rle_channel_symbols"],
            "transmission_ratio_per_rgb_value": rate[
                "rle_transmission_ratio_per_rgb_value"
            ],
            "raw_explicit_source_bits_reference": rate[
                "raw_explicit_reference"
            ]["payload_bits"],
            "dense_two_stage_source_bits_reference": rate[
                "dense_two_stage_reference"
            ]["payload_bits"],
            "source_bits_saved_vs_raw_mask": rate[
                "source_bits_saved_vs_raw_mask"
            ],
            "coded_bits_saved_vs_raw_mask": rate[
                "coded_bits_saved_vs_raw_mask"
            ],
        }
        for channel in point["channel_results"]:
            rows.append(
                {
                    **base,
                    "snr_db": channel["snr_db"],
                    "modulation": channel["modulation"],
                    "psnr": channel["psnr"],
                    "ms_ssim": channel["ms_ssim"],
                    "source_ber_after_ldpc": channel[
                        "source_ber_after_ldpc"
                    ],
                    "source_bit_errors": channel["bit_errors"],
                    "semantic_mask_ber_scale0": channel["per_scale"][0][
                        "semantic_mask_ber"
                    ],
                    "semantic_mask_ber_scale1": channel["per_scale"][1][
                        "semantic_mask_ber"
                    ],
                    "rx_active_rate_scale0": channel["per_scale"][0][
                        "rx_active_ratio"
                    ],
                    "rx_active_rate_scale1": channel["per_scale"][1][
                        "rx_active_ratio"
                    ],
                }
            )
    return rows


def _per_scale_rows(payload: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for point in payload["results"]:
        for rate_scale in point["rate"]["per_scale"]:
            scale = int(rate_scale["scale"])
            base = {
                "scan_id": point["scan_id"],
                "target_active_rate": point["target_active_rates"][scale],
                **rate_scale,
            }
            for channel in point["channel_results"]:
                channel_scale = channel["per_scale"][scale]
                rows.append(
                    {
                        **base,
                        "snr_db": channel["snr_db"],
                        **{
                            f"channel_{key}": value
                            for key, value in channel_scale.items()
                            if key != "scale"
                        },
                    }
                )
    return rows


def _threshold_rows(payload: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for point in payload["results"]:
        for image in point["per_image_selection"]:
            for scale in image["scales"]:
                rows.append(
                    {
                        "scan_id": point["scan_id"],
                        "image_index": image["image_index"],
                        "image_number": image["image_number"],
                        "image_name": image["image_name"],
                        **scale,
                    }
                )
    return rows


def _per_image_rows(payload: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for point in payload["results"]:
        for channel in point["channel_results"]:
            for image in channel["per_image"]:
                base = {
                    "scan_id": point["scan_id"],
                    "target_active_rate_scale0": point[
                        "target_active_rates"
                    ][0],
                    "target_active_rate_scale1": point[
                        "target_active_rates"
                    ][1],
                    "snr_db": channel["snr_db"],
                    "image_index": image["image_index"],
                    "image_number": image["image_number"],
                    "image_name": image["image_name"],
                    "source_bits": image["source_bits"],
                    "coded_bits": image["coded_bits"],
                    "channel_symbols": image["channel_symbols"],
                    "source_bit_errors": image["source_bit_errors"],
                    "source_ber_after_ldpc": image[
                        "source_ber_after_ldpc"
                    ],
                    "ldpc_input_bits": image["ldpc_input_bits"],
                    "ldpc_padding_bits": image["ldpc_padding_bits"],
                    "modulation_padding_bits": image[
                        "modulation_padding_bits"
                    ],
                    "transmitted_bits": image["transmitted_bits"],
                    "psnr": image["psnr"],
                    "ms_ssim": image["ms_ssim"],
                }
                for scale in image["scales"]:
                    prefix = f"scale{scale['scale']}_"
                    for key, value in scale.items():
                        if key != "scale":
                            base[prefix + key] = value
                rows.append(base)
    return rows


@torch.no_grad()
def run(args):
    setup_seed(args.seed)
    if args.rvq_k_lists:
        Config.INDEPENDENT_RAQ_RVQ_K_LISTS = parse_nested_int_lists(
            args.rvq_k_lists
        )
    cfg = Config()
    cfg.validate()
    if not bool(cfg.USE_INDEPENDENT_RAQ_RVQ):
        raise ValueError("SIMVQ_USE_INDEPENDENT_RAQ_RVQ must be 1")
    if int(cfg.INDEPENDENT_RAQ_RVQ_DEPTH) != 2:
        raise ValueError("adaptive test requires independent depth 2")
    rvq_k_lists = [
        [int(value) for value in stages]
        for stages in cfg.INDEPENDENT_RAQ_RVQ_K_LISTS
    ]
    if len(rvq_k_lists) != 2 or any(len(stages) != 2 for stages in rvq_k_lists):
        raise ValueError("this test requires two scales with two stages each")

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    device = torch.device(cfg.DEVICE)
    print(f"CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Logical device={device}")
    if device.type == "cuda":
        print(f"CUDA device={torch.cuda.get_device_name(device)}")
    print(f"Checkpoint={checkpoint}")
    print(f"Independent stage K lists={rvq_k_lists}")
    model, inferred = build_model_from_checkpoint(str(checkpoint), cfg, device)

    loader = get_dataloader(
        root_dir=args.dataset,
        batch_size=1,
        shuffle=False,
        mode="test",
        num_workers=args.num_workers,
        pin_memory=cfg.PIN_MEMORY,
    )
    max_images = None if args.max_images <= 0 else int(args.max_images)
    print("Encoding images once and collecting first-stage token residuals...")
    samples = collect_independent_adaptive_samples(
        model, loader, max_images=max_images
    )
    image_names = _image_names(loader, len(samples))
    activation_points = build_activation_points(
        args.target_active_rates,
        args.target_active_rate_pairs,
        len(rvq_k_lists),
    )

    from communications.ldpc_coding import get_ldpc_code

    requested_k = int(round(args.ldpc_n * args.ldpc_rate))
    if requested_k <= 0 or requested_k >= int(args.ldpc_n):
        raise ValueError("LDPC rate must produce 0 < k < n")
    ldpc_code = get_ldpc_code(requested_k, rate=float(args.ldpc_rate))
    if int(ldpc_code["n"]) != int(args.ldpc_n):
        raise ValueError(
            f"requested LDPC n={args.ldpc_n}, constructed n={ldpc_code['n']}"
        )
    actual_ldpc_rate = int(ldpc_code["k"]) / int(ldpc_code["n"])
    print(
        f"LDPC requested rate={args.ldpc_rate:.12g}, "
        f"actual k/n={actual_ldpc_rate:.12g} "
        f"(k={ldpc_code['k']}, n={ldpc_code['n']})"
    )

    results = []
    for scan_id, target_rates in enumerate(activation_points):
        print(
            f"\n[{scan_id + 1}/{len(activation_points)}] "
            f"per-image/per-scale Top-K rates={target_rates}"
        )
        packets, selection_records = prepare_topk_rle_packets(
            model,
            samples,
            target_rates,
            rvq_k_lists,
            image_names=image_names,
        )
        rate = summarize_packets(
            packets, ldpc_code=ldpc_code, modulation=args.modulation
        )
        nochannel = {
            "psnr": sum(packet.clean_psnr for packet in packets) / len(packets),
            "ms_ssim": (
                sum(packet.clean_ms_ssim for packet in packets) / len(packets)
            ),
        }
        print(
            "  active/image="
            f"{[scale['tx_active_count_per_image'] for scale in rate['per_scale']]}"
        )
        print(
            "  RLE mask bits/image="
            f"{[scale['rle_mask_source_bits_mean_per_image'] for scale in rate['per_scale']]}"
        )
        print(
            f"  combined source={rate['rle_source_bits']/len(packets):.2f} bit/image, "
            f"coded={rate['rle_coded_bits']/len(packets):.2f} bit/image, "
            f"symbols={rate['rle_channel_symbols']/len(packets):.2f}/image, "
            f"no-channel PSNR={nochannel['psnr']:.6f}"
        )

        channel_results = []
        for snr in args.snrs:
            channel = evaluate_packets_over_channel(
                model,
                packets,
                float(snr),
                ldpc_code,
                device,
                args.modulation,
                seed=args.seed,
            )
            channel_results.append(channel)
            print(
                f"  SNR={snr:g} dB | PSNR={channel['psnr']:.6f} | "
                f"MS-SSIM={channel['ms_ssim']:.9f} | "
                f"BER={channel['source_ber_after_ldpc']:.9g} | "
                f"semantic mask BER="
                f"{[scale['semantic_mask_ber'] for scale in channel['per_scale']]}"
            )
        results.append(
            {
                "scan_id": scan_id,
                "target_active_rates": [float(value) for value in target_rates],
                "threshold_count": len(samples) * len(rvq_k_lists),
                "per_image_selection": selection_records,
                "rate": rate,
                "nochannel": nochannel,
                "channel_results": channel_results,
            }
        )

    output = {
        "schema_version": 1,
        "evaluation": (
            "independent_raq_rvq_per_image_per_scale_topk_fixed_width_"
            "rle_mask_combined_ldpc"
        ),
        "checkpoint": str(checkpoint),
        "dataset": str(Path(args.dataset).resolve()),
        "image_order": image_names,
        "num_images": len(samples),
        "source_num_embeddings_list": inferred["num_embeddings_list"],
        "independent_rvq_k_lists": rvq_k_lists,
        "rvq_depth": 2,
        "selection": {
            "scope": "each image and each scale independently",
            "score": "channel-mean squared residual after stage one",
            "active_count_rule": "floor(target_active_rate * token_count + 0.5)",
            "ordering": (
                "first-stage error descending; exact ties use ascending "
                "raster token index"
            ),
            "thresholds_transmitted": False,
            "threshold_bits_counted": False,
        },
        "transport": {
            "stream_packing": "combined",
            "logical_segment_order": (
                "all first-stage scales, all fixed-width RLE masks, then all "
                "compact active second-stage scales"
            ),
            "mask_scan_order": "row-major raster order",
            "mask_coding": (
                "one start bit plus fixed-width binary (run_length-1) fields"
            ),
            "run_length_width_rule": "max(1, ceil(log2(token_count)))",
            "raw_mask_fallback": False,
            "corrupt_rle_concealment": (
                "crop overflow; extend the last emitted run value through underflow"
            ),
            "active_payload_mapping": (
                "map A_tx compact values to received-mask active positions in "
                "raster order; truncate surplus payload or use stage-two index "
                "zero at surplus received active positions"
            ),
            "inactive_second_stage_contribution": "exact zero vector",
            "framing_metadata_counted": False,
        },
        "channel": {
            "ldpc": "Sionna 5G LDPC",
            "requested_n": int(args.ldpc_n),
            "requested_rate": float(args.ldpc_rate),
            "k": int(ldpc_code["k"]),
            "n": int(ldpc_code["n"]),
            "rate": actual_ldpc_rate,
            "actual_rate": actual_ldpc_rate,
            "modulation": args.modulation,
            "channel": "awgn",
            "snrs_db": [float(value) for value in args.snrs],
            "seed_per_activation_snr_point": int(args.seed),
        },
        "results": results,
    }

    json_path = _output_path(args.json_output, "results.json")
    summary_path = _output_path(args.csv_output, "results_summary.csv")
    scale_path = _output_path(
        args.per_scale_csv_output, "per_scale_bits_and_channel.csv"
    )
    image_path = _output_path(
        args.per_image_csv_output, "per_image_channel.csv"
    )
    thresholds_path = _output_path(
        args.thresholds_csv_output, "per_image_thresholds.csv"
    )
    json_path.write_text(
        json.dumps(
            output, indent=2, ensure_ascii=False, allow_nan=False
        ),
        encoding="utf-8",
    )
    _write_csv(summary_path, _summary_rows(output))
    _write_csv(scale_path, _per_scale_rows(output))
    _write_csv(image_path, _per_image_rows(output))
    _write_csv(thresholds_path, _threshold_rows(output))
    print(f"\nJSON:       {json_path}")
    print(f"Summary:    {summary_path}")
    print(f"Per-scale:  {scale_path}")
    print(f"Per-image:  {image_path}")
    print(f"Thresholds: {thresholds_path}")
    return output


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Independent RAQ-RVQ per-image/per-scale exact Top-K fixed-width "
            "RLE-mask evaluation with one combined LDPC stream per image."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--dataset",
        default="/workspace/yi/work/Kodak-256-transform-resize",
    )
    parser.add_argument(
        "--rvq-k-lists",
        default=None,
        help="Override per-scale stage K lists, e.g. '4,2;8,2'.",
    )
    parser.add_argument(
        "--target-active-rates",
        type=float,
        nargs="*",
        default=DEFAULT_TARGET_ACTIVE_RATES,
        help="Common second-stage rate applied to every scale.",
    )
    parser.add_argument(
        "--target-active-rate-pairs",
        nargs="*",
        default=[],
        metavar="R0,R1",
        help="Additional independent per-scale rate points.",
    )
    parser.add_argument("--snrs", type=float, nargs="+", default=DEFAULT_SNRS)
    parser.add_argument(
        "--modulation", choices=["bpsk", "qpsk", "16qam"], default="qpsk"
    )
    parser.add_argument("--ldpc-n", "--ldpc_n", type=int, default=256)
    parser.add_argument(
        "--ldpc-rate", "--ldpc_rate", "--ldpc_k", type=float, default=0.5
    )
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
