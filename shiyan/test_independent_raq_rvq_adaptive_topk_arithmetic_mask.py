"""Evaluate Top-K independent RAQ-RVQ with arithmetic-coded masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

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
from test_independent_raq_rvq_adaptive_topk_rle import (
    _image_names,
    _threshold_rows,
    _write_csv,
    build_activation_points,
    parse_nested_int_lists,
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
    / "independent_raq_rvq_src64_64_d2_adaptive_topk_"
    "arithmetic_mask_combined"
)
DEFAULT_TARGET_ACTIVE_RATES = [step / 10 for step in range(11)]


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
            "nochannel_psnr": point["nochannel"]["psnr"],
            "nochannel_ms_ssim": point["nochannel"]["ms_ssim"],
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
                    "snr_db": channel["snr_db"],
                    "modulation": channel["modulation"],
                    "psnr": channel["psnr"],
                    "ms_ssim": channel["ms_ssim"],
                    "source_ber_after_ldpc": channel[
                        "source_ber_after_ldpc"
                    ],
                    "source_bit_errors": channel["bit_errors"],
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
                            row[prefix + key] = value
                rows.append(row)
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
    if len(rvq_k_lists) != 2 or any(
        len(stages) != 2 for stages in rvq_k_lists
    ):
        raise ValueError("this test requires two scales with two stages each")

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    device = torch.device(cfg.DEVICE)
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
        packets, selection_records = prepare_topk_arithmetic_mask_packets(
            model,
            samples,
            target_rates,
            rvq_k_lists,
            image_names=image_names,
        )
        rate = summarize_arithmetic_mask_packets(
            packets, ldpc_code=ldpc_code, modulation=args.modulation
        )
        nochannel = {
            "psnr": sum(packet.clean_psnr for packet in packets) / len(packets),
            "ms_ssim": (
                sum(packet.clean_ms_ssim for packet in packets) / len(packets)
            ),
        }
        per_scale_bits = [
            scale["arithmetic_mask_source_bits_mean_per_image"]
            for scale in rate["per_scale"]
        ]
        raw_mask_per_image = rate["raw_mask_bits_reference"] / len(packets)
        arithmetic_per_image = rate["arithmetic_mask_bits"] / len(packets)
        saving = (raw_mask_per_image - arithmetic_per_image) / raw_mask_per_image
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
            f"no-channel PSNR={nochannel['psnr']:.6f}"
        )

        channel_results = []
        for snr in args.snrs:
            channel = evaluate_arithmetic_mask_packets_over_channel(
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
                "nochannel": nochannel,
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
            "channel": "awgn",
            "snrs_db": [float(value) for value in args.snrs],
            "seed_per_activation_snr_point": int(args.seed),
        },
        "results": results,
    }

    json_path = _output_path(args.json_output, "results.json")
    summary_path = _output_path(args.csv_output, "results_summary.csv")
    scale_path = _output_path(args.per_scale_csv_output, "per_scale.csv")
    image_path = _output_path(args.per_image_csv_output, "per_image.csv")
    thresholds_path = _output_path(
        args.thresholds_csv_output, "thresholds.csv"
    )
    json_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False),
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
