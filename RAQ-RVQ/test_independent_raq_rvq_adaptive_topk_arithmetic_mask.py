"""Evaluate Top-K independent RAQ-RVQ with arithmetic-coded masks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import redirect_stdout
from fractions import Fraction
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch

from config import Config
from data.datasets import get_dataloader
from evaluation.independent_raq_rvq_adaptive import (
    collect_independent_adaptive_samples,
)
from evaluation.independent_raq_rvq_adaptive_arithmetic import (
    evaluate_arithmetic_mask_packets_over_channel,
    prepare_topk_arithmetic_mask_packets,
    summarize_arithmetic_mask_packets,
)
from utils.checkpoint_utils import build_model_from_checkpoint
from utils.reproducibility import setup_seed


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_NAME = (
    "shiyan_independent_raq_rvq_src64-64_trg2-64_d2_curriculum_"
    "rate094_A_patch_ch256-512_res6-6_mixture_per_k_unet2_ds8x2_k64"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / EXPERIMENT_NAME / "best_vq_deepsc.pth"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "eval"
    / "independent_raq_rvq_src64_64_d2_res6_6_mixture_per_k_"
    "adaptive_topk_arithmetic_mask_combined"
)
DEFAULT_TARGET_ACTIVE_RATES = [step / 10 for step in range(11)]


def _format_cbr(channel_symbols: int, rgb_values: int) -> str:
    ratio = Fraction(int(channel_symbols), int(rgb_values))
    return f"{float(ratio):.4f} ({ratio.numerator}/{ratio.denominator})"


def _format_transmitted_bits(transmitted_bits: int, num_images: int) -> str:
    """Format the physical bit count sent through the channel."""

    if int(num_images) <= 0:
        raise ValueError("num_images must be positive")
    total = int(transmitted_bits)
    return f"{total} bit total ({total / int(num_images):.2f} bit/image)"


def ms_ssim_to_db(ms_ssim: float) -> float:
    """Convert MS-SSIM to -10*log10(1-MS-SSIM)."""

    value = float(ms_ssim)
    if math.isnan(value):
        return float("nan")
    if value >= 1.0:
        return float("inf")
    return -10.0 * math.log10(1.0 - value)


def _format_ms_ssim(ms_ssim: float, precision: int = 4) -> str:
    return (
        f"{float(ms_ssim):.{precision}f} "
        f"({ms_ssim_to_db(ms_ssim):.4f} dB)"
    )


def _channel_label(channel_type: str, rician_k_factor: float = 10.0) -> str:
    if channel_type == "rician":
        return f"Rician(K={rician_k_factor:g})"
    return "AWGN"


def _normalize_channel_types(channel_types: Sequence[str]) -> List[str]:
    normalized = []
    for value in channel_types or ["awgn"]:
        channel_type = str(value).strip().lower()
        if channel_type not in {"awgn", "rician"}:
            raise ValueError(
                f"unsupported channel type {channel_type!r}; "
                "expected 'awgn' or 'rician'"
            )
        if channel_type not in normalized:
            normalized.append(channel_type)
    return normalized


def parse_nested_int_lists(value: str) -> List[List[int]]:
    """Parse semicolon-separated or JSON nested RVQ K lists."""

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


def _output_path(path: str | None, default_name: str) -> Path:
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


def _summary_rows(payload: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for point in payload["results"]:
        rate = point["rate"]
        scale0, scale1 = rate["per_scale"]
        base = {
            "scan_id": point["scan_id"],
            "target_active_rate_scale0": point["target_active_rates"][0],
            "target_active_rate_scale1": point["target_active_rates"][1],
            "actual_active_rate_scale0": scale0["tx_active_ratio"],
            "actual_active_rate_scale1": scale1["tx_active_ratio"],
            "raw_mask_bits_per_image": (
                rate["raw_mask_bits_reference"] / rate["num_images"]
            ),
            "arithmetic_mask_bits_per_image": (
                rate["arithmetic_mask_bits"] / rate["num_images"]
            ),
            "mask_bits_saved_per_image": (
                (
                    rate["raw_mask_bits_reference"]
                    - rate["arithmetic_mask_bits"]
                )
                / rate["num_images"]
            ),
            "mask_saving_ratio": (
                (
                    rate["raw_mask_bits_reference"]
                    - rate["arithmetic_mask_bits"]
                )
                / rate["raw_mask_bits_reference"]
            ),
            "scale0_raw_mask_bits_per_image": (
                scale0["raw_mask_source_bits_reference"]
                / rate["num_images"]
            ),
            "scale0_arithmetic_mask_bits_per_image": scale0[
                "arithmetic_mask_source_bits_mean_per_image"
            ],
            "scale1_raw_mask_bits_per_image": (
                scale1["raw_mask_source_bits_reference"]
                / rate["num_images"]
            ),
            "scale1_arithmetic_mask_bits_per_image": scale1[
                "arithmetic_mask_source_bits_mean_per_image"
            ],
            "arithmetic_source_bits": rate["arithmetic_source_bits"],
            "arithmetic_source_bpp": rate["arithmetic_source_bpp"],
            "arithmetic_coded_bits": rate["arithmetic_coded_bits"],
            "arithmetic_coded_bpp": rate["arithmetic_coded_bpp"],
            "arithmetic_channel_symbols": rate[
                "arithmetic_channel_symbols"
            ],
            "transmission_ratio_per_rgb_value": rate[
                "arithmetic_transmission_ratio_per_rgb_value"
            ],
            "raw_explicit_source_bits_reference": rate[
                "raw_explicit_reference"
            ]["payload_bits"],
            "raw_explicit_coded_bits_reference": rate[
                "raw_explicit_reference"
            ]["coded_bits"],
            "dense_two_stage_source_bits_reference": rate[
                "dense_two_stage_reference"
            ]["payload_bits"],
        }
        for channel in point["channel_results"]:
            rows.append(
                {
                    **base,
                    "channel_type": channel["channel_type"],
                    "snr_db": channel["snr_db"],
                    "modulation": channel["modulation"],
                    "psnr": channel["psnr"],
                    "ms_ssim": channel["ms_ssim"],
                    "ms_ssim_db": channel["ms_ssim_db"],
                    "source_ber_after_ldpc": channel[
                        "source_ber_after_ldpc"
                    ],
                    "source_bit_errors": channel["bit_errors"],
                    "transmitted_bits": channel["transmitted_bits"],
                    "transmitted_bits_per_image": (
                        channel["transmitted_bits"] / rate["num_images"]
                    ),
                    "arithmetic_mask_bit_errors": channel[
                        "segment_bit_errors"
                    ]["mask_arithmetic"],
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
                        "channel_type": channel["channel_type"],
                        "snr_db": channel["snr_db"],
                        **{
                            f"channel_{key}": value
                            for key, value in channel_scale.items()
                            if key != "scale"
                        },
                    }
                )
    return rows


def _per_image_rows(payload: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for point in payload["results"]:
        for channel in point["channel_results"]:
            for image in channel["per_image"]:
                row = {
                    "scan_id": point["scan_id"],
                    "target_active_rate_scale0": point[
                        "target_active_rates"
                    ][0],
                    "target_active_rate_scale1": point[
                        "target_active_rates"
                    ][1],
                    "channel_type": channel["channel_type"],
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
                    "ms_ssim_db": ms_ssim_to_db(image["ms_ssim"]),
                }
                for scale in image["scales"]:
                    prefix = f"scale{scale['scale']}_"
                    for key, value in scale.items():
                        if key != "scale":
                            row[prefix + key] = value
                rows.append(row)
    return rows


@torch.no_grad()
def run(args):
    channel_types = _normalize_channel_types(args.channels)
    if float(args.rician_k_factor) < 0:
        raise ValueError("Rician K-factor must be non-negative")
    if args.concise:
        with redirect_stdout(StringIO()):
            setup_seed(args.seed)
    else:
        setup_seed(args.seed)
    if args.rvq_k_lists:
        Config.INDEPENDENT_RAQ_RVQ_K_LISTS = parse_nested_int_lists(
            args.rvq_k_lists
        )
    cfg = Config()
    cfg.validate()
    independent_enabled = getattr(cfg, "USE_INDEPENDENT_RAQ_RVQ", None)
    if independent_enabled is False:
        raise ValueError("independent RAQ-RVQ must be enabled")
    if int(cfg.INDEPENDENT_RAQ_RVQ_DEPTH) != 2:
        raise ValueError("adaptive test requires independent depth 2")
    rvq_k_lists = [
        [int(value) for value in stages]
        for stages in cfg.INDEPENDENT_RAQ_RVQ_K_LISTS
    ]
    if len(rvq_k_lists) != 2 or any(
        len(stages) != 2 for stages in rvq_k_lists
    ):
        raise ValueError("this test requires two scales with two stages each")

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    device = torch.device(cfg.DEVICE)
    if not args.concise:
        print(
            "CUDA_VISIBLE_DEVICES="
            f"{__import__('os').environ.get('CUDA_VISIBLE_DEVICES')}"
        )
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
    if not args.concise:
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

    if args.concise:
        with redirect_stdout(StringIO()):
            from communications.ldpc_coding import get_ldpc_code
    else:
        from communications.ldpc_coding import get_ldpc_code

    requested_k = int(round(args.ldpc_n * args.ldpc_rate))
    if requested_k <= 0 or requested_k >= int(args.ldpc_n):
        raise ValueError("LDPC rate must produce 0 < k < n")
    if args.concise:
        with redirect_stdout(StringIO()):
            ldpc_code = get_ldpc_code(
                requested_k, rate=float(args.ldpc_rate)
            )
    else:
        ldpc_code = get_ldpc_code(requested_k, rate=float(args.ldpc_rate))
    if int(ldpc_code["n"]) != int(args.ldpc_n):
        raise ValueError(
            f"requested LDPC n={args.ldpc_n}, constructed n={ldpc_code['n']}"
        )
    actual_ldpc_rate = int(ldpc_code["k"]) / int(ldpc_code["n"])
    if not args.concise:
        print(
            f"LDPC requested rate={args.ldpc_rate:.12g}, "
            f"actual k/n={actual_ldpc_rate:.12g} "
            f"(k={ldpc_code['k']}, n={ldpc_code['n']})"
        )

    results = []
    for scan_id, target_rates in enumerate(activation_points):
        if not args.concise:
            print(
                f"\n[{scan_id + 1}/{len(activation_points)}] "
                f"per-image/per-scale Top-K rates={target_rates}"
            )
        packets, selection_records = prepare_topk_arithmetic_mask_packets(
            model,
            samples,
            target_rates,
            rvq_k_lists,
            image_names=image_names,
            compute_clean_metrics=False,
        )
        rate = summarize_arithmetic_mask_packets(
            packets, ldpc_code=ldpc_code, modulation=args.modulation
        )
        per_scale_bits = [
            scale["arithmetic_mask_source_bits_mean_per_image"]
            for scale in rate["per_scale"]
        ]
        raw_mask_per_image = rate["raw_mask_bits_reference"] / len(packets)
        arithmetic_per_image = rate["arithmetic_mask_bits"] / len(packets)
        saving = (raw_mask_per_image - arithmetic_per_image) / raw_mask_per_image
        if not args.concise:
            print(
                "  active/image="
                f"{[scale['tx_active_count_per_image'] for scale in rate['per_scale']]}"
            )
            print(
                f"  mask raw={raw_mask_per_image:.2f} -> arithmetic="
                f"{arithmetic_per_image:.3f} bit/image "
                f"(scales={per_scale_bits}, saving={saving:.3%})"
            )
            print(
                f"  combined source={rate['arithmetic_source_bits']/len(packets):.2f} bit/image, "
                f"coded={rate['arithmetic_coded_bits']/len(packets):.2f} bit/image, "
                f"symbols={rate['arithmetic_channel_symbols']/len(packets):.2f}/image, "
                "CBR="
                f"{_format_cbr(rate['arithmetic_channel_symbols'], rate['total_rgb_values'])}"
            )

        channel_results = []
        for channel_type in channel_types:
            for snr in args.snrs:
                channel_args = (
                    model,
                    packets,
                    float(snr),
                    ldpc_code,
                    device,
                    args.modulation,
                )
                channel_kwargs = {
                    "seed": args.seed,
                    "channel_type": channel_type,
                    "rician_k_factor": args.rician_k_factor,
                }
                if args.concise:
                    with redirect_stdout(StringIO()):
                        channel = evaluate_arithmetic_mask_packets_over_channel(
                            *channel_args, **channel_kwargs
                        )
                else:
                    channel = evaluate_arithmetic_mask_packets_over_channel(
                        *channel_args, **channel_kwargs
                    )
                channel["ms_ssim_db"] = ms_ssim_to_db(channel["ms_ssim"])
                channel_results.append(channel)
                label = _channel_label(channel_type, args.rician_k_factor)
                if args.concise:
                    rate_label = ",".join(
                        str(float(value)) for value in target_rates
                    )
                    print(
                        f"{label} | rate=[{rate_label}] | CBR="
                        f"{_format_cbr(channel['channel_symbols'], rate['total_rgb_values'])} | "
                        "transmitted="
                        f"{_format_transmitted_bits(channel['transmitted_bits'], len(packets))} | "
                        f"SNR={snr:g} dB | PSNR={channel['psnr']:.4f} | "
                        f"MS-SSIM={_format_ms_ssim(channel['ms_ssim'])} | "
                        f"BER={channel['source_ber_after_ldpc']:.4f}"
                    )
                else:
                    print(
                        f"  {label} | SNR={snr:g} dB | "
                        f"PSNR={channel['psnr']:.6f} | "
                        f"MS-SSIM={_format_ms_ssim(channel['ms_ssim'], 9)} | "
                        f"BER={channel['source_ber_after_ldpc']:.9g} | "
                        "CBR="
                        f"{_format_cbr(channel['channel_symbols'], rate['total_rgb_values'])} | "
                        "transmitted="
                        f"{_format_transmitted_bits(channel['transmitted_bits'], len(packets))} | "
                        "arithmetic-mask bit errors="
                        f"{channel['segment_bit_errors']['mask_arithmetic']}"
                    )
        results.append(
            {
                "scan_id": scan_id,
                "target_active_rates": [float(value) for value in target_rates],
                "threshold_count": len(samples) * len(rvq_k_lists),
                "per_image_selection": selection_records,
                "rate": rate,
                "channel_results": channel_results,
            }
        )

    output = {
        "schema_version": 1,
        "evaluation": (
            "independent_raq_rvq_per_image_per_scale_topk_adaptive_"
            "arithmetic_mask_combined_ldpc"
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
            "active_count_rule": (
                "floor(target_active_rate * token_count + 0.5)"
            ),
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
                "all first-stage scales, all arithmetic mask streams, then "
                "all compact active second-stage scales"
            ),
            "mask_scan_order": "row-major raster order",
            "mask_coding": "32-bit integer adaptive binary arithmetic coder",
            "probability_model": (
                "per-image/per-scale adaptive Laplace counts [1,1], reset "
                "for every mask"
            ),
            "counted_mask_bits": "actual arithmetic payload length",
            "decoder_side_information": (
                "token count and arithmetic segment payload length are "
                "assumed known from framing"
            ),
            "framing_metadata_counted": False,
            "active_payload_mapping": (
                "map A_tx compact values to received-mask active positions "
                "in raster order; truncate surplus payload or use stage-two "
                "index zero at surplus received active positions"
            ),
            "inactive_second_stage_contribution": "exact zero vector",
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
            "channel_types": channel_types,
            "rician_k_factor": float(args.rician_k_factor),
            "snrs_db": [float(value) for value in args.snrs],
            "seed_per_activation_channel_snr_point": int(args.seed),
        },
        "results": results,
    }

    if not args.concise:
        print("\nEvaluation complete. Results were printed only; no files were saved.")
    return output


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Independent RAQ-RVQ Top-K arithmetic-mask transmission with "
            "one combined LDPC stream per image."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--dataset", default="/workspace/yi/work/Kodak-256-transform-resize"
    )
    parser.add_argument("--rvq-k-lists", default=None)
    parser.add_argument(
        "--target-active-rates",
        type=float,
        nargs="*",
        default=DEFAULT_TARGET_ACTIVE_RATES,
    )
    parser.add_argument(
        "--target-active-rate-pairs",
        nargs="*",
        default=[],
        metavar="R0,R1",
    )
    parser.add_argument("--snrs", type=float, nargs="+", default=[0.0])
    parser.add_argument(
        "--modulation", choices=["bpsk", "qpsk", "16qam"], default="bpsk"
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        choices=["awgn", "rician"],
        default=["awgn"],
    )
    parser.add_argument("--rician-k-factor", type=float, default=10.0)
    parser.add_argument("--ldpc-n", "--ldpc_n", type=int, default=256)
    parser.add_argument(
        "--ldpc-rate", "--ldpc_rate", "--ldpc_k", type=float, default=0.5
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--concise",
        action="store_true",
        help=(
            "Print one compact result line per activation rate, channel, "
            "and SNR."
        ),
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
