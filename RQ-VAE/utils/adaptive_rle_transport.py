"""Fixed-width run-length transport for two-depth adaptive RQ masks.

This module is deliberately independent from :mod:`utils.adaptive_transport`.
The legacy explicit one-bit-per-token mask remains unchanged.  The new packet
layout keeps the same first-stage and compact active-refinement payloads, but
replaces each raw mask segment with:

* one start-value bit;
* one fixed-width ``run_length - 1`` field per raster-order run.

The run-length width is ``ceil(log2(token_count))`` (10 bits for 1024 shallow
tokens and 8 bits for 256 deep tokens).  Logical segment source lengths remain
framing metadata, matching the existing explicit-mask evaluation contract.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .adaptive_transport import (
    AdaptiveTransportSegment,
    pack_explicit_mask_segments,
)


def rle_length_width(token_count: int) -> int:
    """Return the fixed width used to encode ``run_length - 1``."""

    token_count = int(token_count)
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    return max(1, int(math.ceil(math.log2(token_count))))


def _values_to_bits(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    width = int(width)
    if width <= 0:
        raise ValueError("width must be positive")
    if values.size == 0:
        return np.empty(0, dtype=np.uint8)
    if int(values.min()) < 0 or int(values.max()) >= (1 << width):
        raise ValueError("value does not fit the requested bit width")
    shifts = np.arange(width - 1, -1, -1, dtype=np.int64)
    return ((values[:, None] >> shifts) & 1).reshape(-1).astype(np.uint8)


def _bits_to_values(bits: np.ndarray, width: int) -> np.ndarray:
    normalized = (np.asarray(bits).reshape(-1) != 0).astype(np.int64)
    width = int(width)
    if width <= 0:
        raise ValueError("width must be positive")
    if normalized.size % width:
        raise ValueError("bit count must be divisible by width")
    if normalized.size == 0:
        return np.empty(0, dtype=np.int64)
    grouped = normalized.reshape(-1, width)
    powers = 1 << np.arange(width - 1, -1, -1, dtype=np.int64)
    return np.sum(grouped * powers, axis=1, dtype=np.int64)


def encode_fixed_width_rle_mask(
    mask: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Encode one non-empty raster-order binary mask."""

    flat = (np.asarray(mask).reshape(-1) != 0).astype(np.uint8)
    if flat.size == 0:
        raise ValueError("mask must not be empty")
    token_count = int(flat.size)
    width = rle_length_width(token_count)
    change_points = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            change_points.astype(np.int64),
            np.array([token_count], dtype=np.int64),
        ]
    )
    run_lengths = np.diff(boundaries)
    if (
        run_lengths.size == 0
        or int(run_lengths.min()) <= 0
        or int(run_lengths.sum()) != token_count
    ):
        raise RuntimeError("invalid internal RLE partition")
    bits = np.concatenate(
        [
            np.array([flat[0]], dtype=np.uint8),
            _values_to_bits(run_lengths - 1, width),
        ]
    )
    metadata: Dict[str, object] = {
        "token_count": token_count,
        "run_length_bits": width,
        "run_count": int(run_lengths.size),
        "start_value": int(flat[0]),
        "run_lengths": run_lengths.astype(np.int64).tolist(),
        "source_bits": int(bits.size),
    }
    return bits, metadata


def decode_fixed_width_rle_mask(
    bits: np.ndarray,
    token_count: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Decode a possibly corrupted RLE source segment deterministically.

    Fixed-width fields preserve field synchronization.  Corrupted lengths may
    no longer sum to ``token_count``.  Overflow is cropped to the target grid;
    underflow extends the last emitted run value to the end of the grid.
    """

    token_count = int(token_count)
    width = rle_length_width(token_count)
    normalized = (np.asarray(bits).reshape(-1) != 0).astype(np.uint8)
    if normalized.size < 1:
        raise ValueError("RLE segment must contain a start bit")
    if (normalized.size - 1) % width:
        raise ValueError(
            "RLE segment after the start bit must contain complete length fields"
        )

    start_value = int(normalized[0])
    encoded_lengths = _bits_to_values(normalized[1:], width) + 1
    decoded_length_sum = int(encoded_lengths.sum())
    output = np.empty(token_count, dtype=np.uint8)
    position = 0
    value = start_value
    used_runs = 0
    last_emitted_value = start_value

    for length in encoded_lengths:
        if position >= token_count:
            break
        emitted = min(int(length), token_count - position)
        output[position : position + emitted] = value
        position += emitted
        last_emitted_value = value
        value = 1 - value
        used_runs += 1

    underflow = max(token_count - decoded_length_sum, 0)
    overflow = max(decoded_length_sum - token_count, 0)
    if position < token_count:
        output[position:] = last_emitted_value

    stats: Dict[str, object] = {
        "token_count": token_count,
        "run_length_bits": width,
        "decoded_run_count": int(encoded_lengths.size),
        "used_run_count": used_runs,
        "decoded_start_value": start_value,
        "decoded_length_sum": decoded_length_sum,
        "length_sum_error": decoded_length_sum - token_count,
        "underflow_tokens": underflow,
        "overflow_tokens": overflow,
        "structurally_valid": decoded_length_sum == token_count,
        "rx_active_count": int(output.sum()),
    }
    return output, stats


def _decode_fixed_width_values(
    bits: np.ndarray,
    count: int,
    width: int,
    num_embeddings: int,
) -> np.ndarray:
    count = int(count)
    width = int(width)
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return np.empty(0, dtype=np.int64)
    normalized = (np.asarray(bits).reshape(-1) != 0).astype(np.int64)
    required = count * width
    if normalized.size < required:
        normalized = np.pad(normalized, (0, required - normalized.size))
    else:
        normalized = normalized[:required]
    grouped = normalized.reshape(count, width)
    powers = 1 << np.arange(width - 1, -1, -1, dtype=np.int64)
    values = np.sum(grouped * powers, axis=1, dtype=np.int64)
    return np.clip(values, 0, int(num_embeddings) - 1)


def pack_rle_mask_segments(
    indices_list: Sequence[torch.Tensor],
    num_embeddings_list: Sequence[int],
) -> Tuple[List[AdaptiveTransportSegment], Dict[str, object]]:
    """Pack adaptive indices with fixed-width RLE activity masks."""

    explicit_segments, explicit_metadata = pack_explicit_mask_segments(
        indices_list, num_embeddings_list
    )
    by_name = {segment.name: segment for segment in explicit_segments}
    first_segments: List[AdaptiveTransportSegment] = []
    mask_segments: List[AdaptiveTransportSegment] = []
    second_segments: List[AdaptiveTransportSegment] = []
    scale_metadata: List[Dict[str, object]] = []

    for explicit_scale in explicit_metadata["scales"]:
        scale = int(explicit_scale["scale"])
        names = explicit_scale["segment_names"]
        first_segment = by_name[names["first"]]
        second_segment = by_name[names["second"]]
        raw_mask = by_name[names["mask"]].bits
        rle_bits, rle_metadata = encode_fixed_width_rle_mask(raw_mask)
        rle_name = f"mask_rle_s{scale}"
        first_segments.append(first_segment)
        mask_segments.append(
            AdaptiveTransportSegment(
                rle_name, "mask_rle", scale, rle_bits
            )
        )
        second_segments.append(second_segment)
        scale_metadata.append(
            {
                **explicit_scale,
                "segment_names": {
                    "first": names["first"],
                    "mask_rle": rle_name,
                    "second": names["second"],
                },
                "segment_source_bits": {
                    "first": int(first_segment.bits.size),
                    "mask_rle": int(rle_bits.size),
                    "second_active": int(second_segment.bits.size),
                },
                "raw_mask_source_bits_reference": int(raw_mask.size),
                "raw_explicit_source_bits_reference": int(
                    first_segment.bits.size
                    + raw_mask.size
                    + second_segment.bits.size
                ),
                "rle": rle_metadata,
            }
        )

    segments = first_segments + mask_segments + second_segments
    first_bits = int(sum(segment.bits.size for segment in first_segments))
    rle_mask_bits = int(sum(segment.bits.size for segment in mask_segments))
    active_second_bits = int(
        sum(segment.bits.size for segment in second_segments)
    )
    raw_mask_bits_reference = int(
        sum(
            scale["raw_mask_source_bits_reference"]
            for scale in scale_metadata
        )
    )
    metadata: Dict[str, object] = {
        "schema": "adaptive_fixed_width_rle_mask_v1",
        "framing": (
            "independent logical segments: all first-depth scales, all fixed-"
            "width RLE masks, then all active second-depth payloads"
        ),
        "framing_overhead_counted": False,
        "scales": scale_metadata,
        "source_bits": first_bits + rle_mask_bits + active_second_bits,
        "first_stage_bits": first_bits,
        "rle_mask_bits": rle_mask_bits,
        "raw_mask_bits_reference": raw_mask_bits_reference,
        "active_second_bits": active_second_bits,
        "raw_explicit_source_bits_reference": (
            first_bits + raw_mask_bits_reference + active_second_bits
        ),
    }
    return segments, metadata


def unpack_rle_mask_segments(
    decoded_segments: Dict[str, np.ndarray],
    metadata: Dict[str, object],
) -> Tuple[List[torch.Tensor], Dict[str, object]]:
    """Recover adaptive indices from received RLE masks and compact payloads."""

    if metadata.get("schema") != "adaptive_fixed_width_rle_mask_v1":
        raise ValueError("unsupported adaptive RLE transport metadata")
    recovered: List[torch.Tensor] = []
    per_scale_stats: List[Dict[str, object]] = []

    for scale_metadata in metadata["scales"]:
        scale = int(scale_metadata["scale"])
        height, width_spatial = [
            int(value) for value in scale_metadata["shape"]
        ]
        token_count = int(scale_metadata["token_count"])
        num_embeddings = int(scale_metadata["num_embeddings"])
        index_width = int(scale_metadata["bits_per_index"])
        tx_active_count = int(scale_metadata["tx_active_count"])
        names = scale_metadata["segment_names"]

        first_values = _decode_fixed_width_values(
            decoded_segments.get(
                names["first"], np.empty(0, dtype=np.uint8)
            ),
            token_count,
            index_width,
            num_embeddings,
        )
        mask, rle_stats = decode_fixed_width_rle_mask(
            decoded_segments.get(
                names["mask_rle"], np.empty(0, dtype=np.uint8)
            ),
            token_count,
        )
        active_values = _decode_fixed_width_values(
            decoded_segments.get(
                names["second"], np.empty(0, dtype=np.uint8)
            ),
            tx_active_count,
            index_width,
            num_embeddings,
        )

        rx_active_positions = np.flatnonzero(mask)
        rx_active_count = int(rx_active_positions.size)
        mapped_count = min(rx_active_count, tx_active_count)
        zero_filled = max(rx_active_count - tx_active_count, 0)
        truncated = max(tx_active_count - rx_active_count, 0)
        second_values = np.full(token_count, -1, dtype=np.int64)
        if mapped_count:
            second_values[rx_active_positions[:mapped_count]] = active_values[
                :mapped_count
            ]
        if zero_filled:
            second_values[rx_active_positions[mapped_count:]] = 0

        stacked = np.stack([first_values, second_values], axis=-1).reshape(
            1, height, width_spatial, 2
        )
        recovered.append(torch.from_numpy(stacked).long())
        per_scale_stats.append(
            {
                "scale": scale,
                "tx_active_count": tx_active_count,
                "rx_active_count": rx_active_count,
                "active_count_mismatch": rx_active_count - tx_active_count,
                "mapped_second_indices": mapped_count,
                "zero_filled_second_indices": zero_filled,
                "truncated_second_indices": truncated,
                **rle_stats,
            }
        )

    return recovered, {"per_scale": per_scale_stats}


__all__ = [
    "decode_fixed_width_rle_mask",
    "encode_fixed_width_rle_mask",
    "pack_rle_mask_segments",
    "rle_length_width",
    "unpack_rle_mask_segments",
]
