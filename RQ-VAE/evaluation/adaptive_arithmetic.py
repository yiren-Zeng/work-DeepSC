"""Mask-only arithmetic-code statistics for per-image Top-K selection."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from evaluation.adaptive import AdaptiveSample
from evaluation.adaptive_topk import select_per_image_topk_masks
from utils.adaptive_arithmetic_coding import (
    decode_adaptive_binary_mask,
    encode_adaptive_binary_mask,
)


def evaluate_topk_arithmetic_masks(
    samples: Sequence[AdaptiveSample],
    target_active_rates: Sequence[float],
    image_names: Sequence[str] | None = None,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Encode/decode every image-scale mask and report payload lengths only."""

    if not samples:
        raise ValueError("samples must not be empty")
    if image_names is not None and len(image_names) != len(samples):
        raise ValueError("image_names must match samples")

    num_scales = len(samples[0].first_stage_errors)
    if len(target_active_rates) != num_scales:
        raise ValueError("one target active rate is required per scale")
    per_scale: List[Dict[str, object]] = [
        {
            "scale": scale,
            "num_images": len(samples),
            "token_count": 0,
            "active_count": 0,
            "raw_mask_bits": 0,
            "arithmetic_mask_bits": 0,
            "arithmetic_bits_per_image": [],
        }
        for scale in range(num_scales)
    ]
    per_image: List[Dict[str, object]] = []

    for image_index, sample in enumerate(samples):
        if len(sample.first_stage_errors) != num_scales:
            raise ValueError("all samples must have the same scale count")
        masks, selection = select_per_image_topk_masks(
            sample.first_stage_errors, target_active_rates
        )
        if len(selection) != 1:
            raise ValueError("mask-only evaluation requires batch_size=1")

        image_record: Dict[str, object] = {
            "image_index": image_index,
            "image_number": image_index + 1,
            "image_name": (
                str(image_names[image_index])
                if image_names is not None
                else f"image_{image_index + 1:04d}"
            ),
            "scales": [],
        }
        for scale, mask_tensor in enumerate(masks):
            mask = (
                mask_tensor[0].detach().to(device="cpu").numpy()
                .reshape(-1).astype(np.uint8)
            )
            encoded, metadata = encode_adaptive_binary_mask(mask)
            decoded = decode_adaptive_binary_mask(encoded, mask.size)
            if not np.array_equal(decoded, mask):
                raise RuntimeError(
                    f"arithmetic mask roundtrip failed for image "
                    f"{image_index}, scale {scale}"
                )

            raw_bits = int(mask.size)
            arithmetic_bits = int(encoded.size)
            stats = per_scale[scale]
            stats["token_count"] += raw_bits
            stats["active_count"] += int(mask.sum())
            stats["raw_mask_bits"] += raw_bits
            stats["arithmetic_mask_bits"] += arithmetic_bits
            stats["arithmetic_bits_per_image"].append(arithmetic_bits)
            image_record["scales"].append(
                {
                    **selection[0]["scales"][scale],
                    "raw_mask_bits": raw_bits,
                    "arithmetic_mask_bits": arithmetic_bits,
                    "mask_bits_saved": raw_bits - arithmetic_bits,
                    "mask_saving_ratio": (
                        (raw_bits - arithmetic_bits) / raw_bits
                    ),
                    "arithmetic_roundtrip_exact": True,
                    "zero_count": metadata["zero_count"],
                    "one_count": metadata["one_count"],
                }
            )
        per_image.append(image_record)

    for stats in per_scale:
        bits_per_image = stats.pop("arithmetic_bits_per_image")
        raw_bits = int(stats["raw_mask_bits"])
        arithmetic_bits = int(stats["arithmetic_mask_bits"])
        token_count = int(stats["token_count"])
        active_count = int(stats["active_count"])
        stats.update(
            {
                "token_count_per_image": token_count // len(samples),
                "active_count_per_image": active_count // len(samples),
                "actual_active_ratio": active_count / token_count,
                "raw_mask_bits_mean_per_image": raw_bits / len(samples),
                "arithmetic_mask_bits_mean_per_image": (
                    arithmetic_bits / len(samples)
                ),
                "arithmetic_mask_bits_min_per_image": min(bits_per_image),
                "arithmetic_mask_bits_max_per_image": max(bits_per_image),
                "mask_bits_saved": raw_bits - arithmetic_bits,
                "mask_saving_ratio": (
                    (raw_bits - arithmetic_bits) / raw_bits
                ),
                "all_roundtrips_exact": True,
            }
        )

    raw_total = sum(int(scale["raw_mask_bits"]) for scale in per_scale)
    arithmetic_total = sum(
        int(scale["arithmetic_mask_bits"]) for scale in per_scale
    )
    summary: Dict[str, object] = {
        "num_images": len(samples),
        "target_active_rates": [float(rate) for rate in target_active_rates],
        "raw_mask_bits": raw_total,
        "arithmetic_mask_bits": arithmetic_total,
        "raw_mask_bits_mean_per_image": raw_total / len(samples),
        "arithmetic_mask_bits_mean_per_image": (
            arithmetic_total / len(samples)
        ),
        "mask_bits_saved": raw_total - arithmetic_total,
        "mask_saving_ratio": (raw_total - arithmetic_total) / raw_total,
        "all_roundtrips_exact": True,
        "per_scale": per_scale,
    }
    return summary, per_image


__all__ = ["evaluate_topk_arithmetic_masks"]
