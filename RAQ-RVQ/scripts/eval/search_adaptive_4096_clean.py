#!/usr/bin/env python3
"""Screen adaptive RAQ-RVQ layouts for an exact coded-bit budget.

The search is channel-free on purpose: layouts that cannot reach the target
without channel errors cannot reach it over the 0 dB channel either.  For
every layout and Top-K pair, the script finds all LDPC information lengths k
for which every image occupies the requested number of n=256 codewords.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
from pathlib import Path
import sys

import numpy as np
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
    prepare_topk_arithmetic_mask_packets,
)
from utils.checkpoint_utils import build_model_from_checkpoint
from utils.reproducibility import setup_seed


DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / (
    "shiyan_independent_raq_rvq_src64-64_trg2-64_d2_curriculum_"
    "rate094_A_patch_ch256-512_res6-6_mixture_per_k_unet2_ds8x2_k64"
) / "best_vq_deepsc.pth"


def _csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _csv_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _valid_k_values(
    source_bits: list[int], blocks: int, n: int = 256
) -> list[int]:
    return [
        k
        for k in range(1, n)
        if all(math.ceil(bits / k) == blocks for bits in source_bits)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--dataset", default="/workspace/yi/work/Kodak-256-transform-resize"
    )
    parser.add_argument("--first-k0-values", default="4")
    parser.add_argument("--first-k1-values", default="2,4")
    parser.add_argument("--second-k0-values", default="2,4,8,16,32,64")
    parser.add_argument("--second-k1-values", default="2,4,8,16,32,64")
    parser.add_argument(
        "--rate0-values",
        default="0,0.002,0.005,0.01,0.02,0.03,0.05,0.08,0.1,0.15,0.2",
    )
    parser.add_argument(
        "--rate1-values",
        default="0,0.01,0.02,0.05,0.08,0.1,0.15,0.2,0.3,0.5",
    )
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument(
        "--fixed-k",
        type=int,
        default=None,
        help=(
            "Require this exact LDPC information length. For example, "
            "--fixed-k 128 with --blocks 16 retains per-image source "
            "payloads in 1921..2048 bits."
        ),
    )
    parser.add_argument(
        "--average-block-budget",
        action="store_true",
        help=(
            "Match the reported average coded-bit budget instead of requiring "
            "every image to occupy the same number of LDPC blocks."
        ),
    )
    parser.add_argument(
        "--max-k-min",
        type=int,
        default=255,
        help="Discard layouts whose strongest exact-4096 LDPC k is larger.",
    )
    parser.add_argument("--top", type=int, default=20)
    return parser


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    setup_seed(args.seed)
    # Match the production evaluation shell: the Kodak directory already
    # contains 256x256 images and must not be resized back to 768x512.
    os.environ.setdefault("SIMVQ_TEST_NO_RESIZE", "1")
    first_k0_values = _csv_ints(args.first_k0_values)
    first_k1_values = _csv_ints(args.first_k1_values)
    second_k0_values = _csv_ints(args.second_k0_values)
    second_k1_values = _csv_ints(args.second_k1_values)
    rate0_values = _csv_floats(args.rate0_values)
    rate1_values = _csv_floats(args.rate1_values)

    initial = [
        [first_k0_values[0], second_k0_values[0]],
        [first_k1_values[0], second_k1_values[0]],
    ]
    Config.INDEPENDENT_RAQ_RVQ_K_LISTS = initial
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

    layouts = list(itertools.product(
        first_k0_values,
        first_k1_values,
        second_k0_values,
        second_k1_values,
    ))
    activation_pairs = list(itertools.product(rate0_values, rate1_values))
    rows: list[dict[str, object]] = []
    for layout_index, (first0, first1, second0, second1) in enumerate(layouts):
        rvq_k_lists = [[first0, second0], [first1, second1]]
        model.independent_raq_rvq_k_lists = rvq_k_lists
        samples = collect_independent_adaptive_samples(
            model,
            loader,
            max_images=(None if args.max_images <= 0 else args.max_images),
        )
        last_source_bits: list[int] = []
        for rate0, rate1 in activation_pairs:
            packets, _ = prepare_topk_arithmetic_mask_packets(
                model,
                samples,
                [rate0, rate1],
                rvq_k_lists,
                compute_clean_metrics=False,
            )
            source_bits = [
                int(packet.metadata["source_bits"]) for packet in packets
            ]
            last_source_bits = source_bits
            if args.fixed_k is not None and args.average_block_budget:
                total_blocks = sum(
                    math.ceil(bits / args.fixed_k) for bits in source_bits
                )
                valid_ks = (
                    [args.fixed_k]
                    if total_blocks == args.blocks * len(source_bits)
                    else []
                )
            else:
                valid_ks = _valid_k_values(source_bits, args.blocks)
                if args.fixed_k is not None:
                    valid_ks = [k for k in valid_ks if k == args.fixed_k]
            valid_ks = [k for k in valid_ks if k <= args.max_k_min]
            if not valid_ks:
                continue
            clean_psnr_values = []
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
                clean_psnr_values.append(
                    100.0 if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
                )
            rows.append(
                {
                    "rvq": rvq_k_lists,
                    "rates": [rate0, rate1],
                    "k_min": min(valid_ks),
                    "k_max": max(valid_ks),
                    "source_min": min(source_bits),
                    "source_max": max(source_bits),
                    "source_mean": float(np.mean(source_bits)),
                    "psnr": float(np.mean(clean_psnr_values)),
                }
            )
        print(
            f"progress={layout_index + 1}/{len(layouts)} "
            f"rvq={rvq_k_lists} retained={len(rows)} "
            f"last_source={min(last_source_bits)}..{max(last_source_bits)}",
            flush=True,
        )

    rows.sort(key=lambda row: float(row["psnr"]), reverse=True)
    print(f"TOP {min(args.top, len(rows))} OF {len(rows)}", flush=True)
    for row in rows[: args.top]:
        print(
            f"PSNR={row['psnr']:.4f} "
            f"RVQ={row['rvq']} rates={row['rates']} "
            f"source={row['source_min']}..{row['source_max']} "
            f"(mean={row['source_mean']:.2f}) "
            f"valid_k={row['k_min']}..{row['k_max']} "
            f"transmitted={args.blocks * 256} bit/image",
            flush=True,
        )


if __name__ == "__main__":
    main(build_parser().parse_args())
