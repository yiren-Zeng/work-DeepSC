#!/usr/bin/env python3
"""Search two-scale arithmetic Top-K mask rates at an exact physical CBR."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from fractions import Fraction
from io import StringIO
import itertools
import math
import os
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from data.datasets import get_dataloader
from evaluation.independent_raq_rvq_adaptive import (
    collect_independent_adaptive_samples,
    reconstruct_from_adaptive_indices,
)
from evaluation.independent_raq_rvq_adaptive_arithmetic import (
    evaluate_arithmetic_mask_packets_over_channel,
    prepare_topk_arithmetic_mask_packets,
    summarize_arithmetic_mask_packets,
)
from utils.checkpoint_utils import build_model_from_checkpoint
from utils.reproducibility import setup_seed


DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / (
    "shiyan_independent_raq_rvq_src64-64_trg2-64_d2_curriculum_"
    "rate094_A_patch_ch256-512_res6-6_mixture_per_k_unet2_ds8x2_k64"
) / "best_vq_deepsc.pth"


def _csv_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _rvq(value: str) -> list[list[int]]:
    rows = [row.strip() for row in value.split(";") if row.strip()]
    parsed = [[int(item) for item in row.split(",")] for row in rows]
    if len(parsed) != 2 or any(len(row) != 2 for row in parsed):
        raise argparse.ArgumentTypeError("RVQ must look like '4,2;8,2'")
    return parsed


def _fraction_float(value: str) -> float:
    return float(Fraction(value))


def _clean_psnr(model, packets) -> float:
    values = []
    for packet in packets:
        reconstructed = reconstruct_from_adaptive_indices(
            model,
            packet.tx_indices,
            packet.feature_shapes,
            packet.codebooks,
        )
        reference = (packet.image.to(reconstructed.device) + 1.0) / 2.0
        estimate = (reconstructed + 1.0) / 2.0
        mse = float(torch.mean((reference - estimate) ** 2).item())
        values.append(100.0 if mse == 0.0 else 10.0 * math.log10(1.0 / mse))
    return sum(values) / len(values)


def _ms_db(value: float) -> float:
    return -10.0 * math.log10(1.0 - value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--dataset", default="/workspace/yi/work/Kodak-256-transform-resize"
    )
    parser.add_argument("--rvq", type=_rvq)
    parser.add_argument(
        "--rvq-configs",
        help=(
            "Search several layouts separated by '|', for example "
            "'2,4;8,2|4,2;4,2'."
        ),
    )
    parser.add_argument(
        "--auto-layouts",
        action="store_true",
        help=(
            "Generate feasible K layouts from powers of two in [2,64]. "
            "First-stage bits must fit the LDPC input budget and dense bits "
            "must be between the budget and budget plus --max-dense-excess."
        ),
    )
    parser.add_argument("--max-dense-excess", type=int, default=2048)
    parser.add_argument("--ldpc-n", type=int, default=256)
    parser.add_argument("--ldpc-rate", required=True, type=_fraction_float)
    parser.add_argument("--modulation", choices=["bpsk", "qpsk", "16qam"], required=True)
    parser.add_argument("--snr", type=float, required=True)
    parser.add_argument("--target-cbr", required=True, type=Fraction)
    parser.add_argument("--rate0-values", required=True)
    parser.add_argument("--rate1-values", required=True)
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evaluate-top",
        type=int,
        default=0,
        help="Evaluate this many clean-ranked matches over the channel; zero means all.",
    )
    parser.add_argument("--top", type=int, default=10)
    return parser


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    setup_seed(args.seed)
    os.environ.setdefault("SIMVQ_TEST_NO_RESIZE", "1")
    requested_k = int(round(args.ldpc_n * args.ldpc_rate))
    modulation_bits = {"bpsk": 1, "qpsk": 2, "16qam": 4}[
        args.modulation
    ]
    per_image_symbols = Fraction(256 * 256 * 3) * args.target_cbr
    source_capacity = (
        per_image_symbols
        * modulation_bits
        * requested_k
        / args.ldpc_n
    )
    if source_capacity.denominator != 1:
        raise ValueError("target settings do not produce an integer source budget")
    source_capacity = int(source_capacity)

    if args.auto_layouts:
        choices = [2, 4, 8, 16, 32, 64]
        rvq_layouts = []
        for first0, second0, first1, second1 in itertools.product(
            choices, choices, choices, choices
        ):
            widths = [
                int(math.log2(value))
                for value in (first0, second0, first1, second1)
            ]
            base_bits = 1024 * widths[0] + 256 * widths[2]
            dense_bits = (
                1024 * (widths[0] + widths[1])
                + 256 * (widths[2] + widths[3])
            )
            if base_bits > source_capacity:
                continue
            if not (
                source_capacity
                <= dense_bits
                <= source_capacity + args.max_dense_excess
            ):
                continue
            rvq_layouts.append(
                [[first0, second0], [first1, second1]]
            )
    elif args.rvq_configs:
        rvq_layouts = [
            _rvq(item) for item in args.rvq_configs.split("|") if item.strip()
        ]
    elif args.rvq:
        rvq_layouts = [args.rvq]
    else:
        raise ValueError("one of --rvq or --rvq-configs is required")
    Config.INDEPENDENT_RAQ_RVQ_K_LISTS = rvq_layouts[0]
    cfg = Config()
    cfg.validate()
    device = torch.device(cfg.DEVICE)
    model, _ = build_model_from_checkpoint(args.checkpoint, cfg, device)
    loader = get_dataloader(
        root_dir=args.dataset,
        batch_size=1,
        shuffle=False,
        mode="test",
        num_workers=args.num_workers,
        pin_memory=cfg.PIN_MEMORY,
    )
    from communications.ldpc_coding import get_ldpc_code

    with redirect_stdout(StringIO()):
        ldpc_code = get_ldpc_code(requested_k, rate=args.ldpc_rate)
    if int(ldpc_code["n"]) != args.ldpc_n:
        raise RuntimeError(
            f"constructed LDPC n={ldpc_code['n']}, expected {args.ldpc_n}"
        )

    rate_pairs = list(itertools.product(
        _csv_floats(args.rate0_values),
        _csv_floats(args.rate1_values),
    ))
    matches = []
    target_symbols = None
    num_samples = None
    for layout_index, rvq_layout in enumerate(rvq_layouts):
        model.independent_raq_rvq_k_lists = rvq_layout
        samples = collect_independent_adaptive_samples(
            model,
            loader,
            max_images=(None if args.max_images <= 0 else args.max_images),
        )
        num_samples = len(samples)
        before = len(matches)
        for rate0, rate1 in rate_pairs:
            packets, _ = prepare_topk_arithmetic_mask_packets(
                model,
                samples,
                [rate0, rate1],
                rvq_layout,
                compute_clean_metrics=False,
            )
            summary = summarize_arithmetic_mask_packets(
                packets,
                ldpc_code=ldpc_code,
                modulation=args.modulation,
            )
            if target_symbols is None:
                exact = Fraction(summary["total_rgb_values"]) * args.target_cbr
                if exact.denominator != 1:
                    raise ValueError(
                        "target CBR does not produce an integer symbol budget"
                    )
                target_symbols = exact.numerator
            if int(summary["arithmetic_channel_symbols"]) != target_symbols:
                continue
            matches.append(
                {
                    "rvq": rvq_layout,
                    "rates": [rate0, rate1],
                    "clean_psnr": _clean_psnr(model, packets),
                    "source_mean": (
                        summary["arithmetic_source_bits"] / len(packets)
                    ),
                    "symbols_mean": (
                        summary["arithmetic_channel_symbols"] / len(packets)
                    ),
                }
            )
        print(
            f"progress={layout_index + 1}/{len(rvq_layouts)} "
            f"rvq={rvq_layout} new_matches={len(matches) - before}",
            flush=True,
        )

    matches.sort(key=lambda row: row["clean_psnr"], reverse=True)
    selected = matches[: args.evaluate_top] if args.evaluate_top > 0 else matches
    channel_rows = []
    for row in selected:
        model.independent_raq_rvq_k_lists = row["rvq"]
        samples = collect_independent_adaptive_samples(
            model,
            loader,
            max_images=(None if args.max_images <= 0 else args.max_images),
        )
        packets, _ = prepare_topk_arithmetic_mask_packets(
            model,
            samples,
            row["rates"],
            row["rvq"],
            compute_clean_metrics=False,
        )
        with redirect_stdout(StringIO()):
            channel = evaluate_arithmetic_mask_packets_over_channel(
                model,
                packets,
                args.snr,
                ldpc_code,
                device,
                args.modulation,
                seed=args.seed,
                channel_type="awgn",
            )
        channel_rows.append(
            {
                **row,
                "psnr": channel["psnr"],
                "ms_ssim": channel["ms_ssim"],
                "ber": channel["source_ber_after_ldpc"],
            }
        )

    channel_rows.sort(key=lambda row: row["psnr"], reverse=True)
    print(
        f"SEARCH layouts={len(rvq_layouts)} ldpc={ldpc_code['k']}/{ldpc_code['n']} "
        f"mod={args.modulation} snr={args.snr:g} target_cbr={args.target_cbr} "
        f"images={num_samples} pairs={len(rate_pairs)} matches={len(matches)} "
        f"evaluated={len(channel_rows)}",
        flush=True,
    )
    for row in channel_rows[: args.top]:
        print(
            f"PSNR={row['psnr']:.4f} MS-SSIM={row['ms_ssim']:.4f} "
            f"({ _ms_db(row['ms_ssim']):.4f} dB) BER={row['ber']:.6f} "
            f"clean={row['clean_psnr']:.4f} rvq={row['rvq']} rates={row['rates']} "
            f"source={row['source_mean']:.2f} symbols={row['symbols_mean']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main(build_parser().parse_args())
