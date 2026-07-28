"""Explicit-mask serialization for two-depth adaptive residual quantization.

The transport representation is intentionally separate from the in-memory
``STOP=-1`` convention:

* every first-depth index is sent at fixed width;
* every token has one explicit activity-mask bit;
* only active second-depth indices are sent, in raster order.

The returned logical segments let a channel layer frame and protect each
component independently.  Segment boundaries, shapes, and original segment
lengths are framing metadata and are not counted by this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .bit_utils import bits_per_index


@dataclass(frozen=True)
class AdaptiveTransportSegment:
    """One logical source-bit segment before channel coding."""

    name: str
    kind: str
    scale: int
    bits: np.ndarray


def _values_to_bits(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    width = int(width)
    if width <= 0:
        raise ValueError("width must be positive")
    if values.size == 0:
        return np.empty(0, dtype=np.uint8)
    shifts = np.arange(width - 1, -1, -1, dtype=np.int64)
    return ((values[:, None] >> shifts) & 1).reshape(-1).astype(np.uint8)


def _bits_to_values(
    bits: np.ndarray,
    count: int,
    width: int,
    num_embeddings: int,
) -> np.ndarray:
    count = int(count)
    width = int(width)
    num_embeddings = int(num_embeddings)
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return np.empty(0, dtype=np.int64)
    required = count * width
    normalized = (np.asarray(bits).reshape(-1) != 0).astype(np.int64)
    if normalized.size < required:
        normalized = np.pad(normalized, (0, required - normalized.size))
    else:
        normalized = normalized[:required]
    grouped = normalized.reshape(count, width)
    powers = 1 << np.arange(width - 1, -1, -1, dtype=np.int64)
    values = np.sum(grouped * powers, axis=1, dtype=np.int64)
    return np.clip(values, 0, num_embeddings - 1)


def _validate_adaptive_indices(
    indices: torch.Tensor,
    num_embeddings: int,
    scale: int,
) -> None:
    if not isinstance(indices, torch.Tensor):
        raise TypeError(f"indices_list[{scale}] must be a torch.Tensor")
    if indices.ndim != 4 or indices.shape[0] != 1 or indices.shape[-1] != 2:
        raise ValueError(
            f"indices_list[{scale}] must have shape [1,H,W,2], "
            f"got {tuple(indices.shape)}"
        )
    first = indices[..., 0]
    second = indices[..., 1]
    if first.numel() and (
        int(first.min().item()) < 0
        or int(first.max().item()) >= int(num_embeddings)
    ):
        raise ValueError(
            f"scale {scale} first-depth indices are invalid for K={num_embeddings}"
        )
    if second.numel() and (
        int(second.min().item()) < -1
        or int(second.max().item()) >= int(num_embeddings)
    ):
        raise ValueError(
            f"scale {scale} second-depth indices must be STOP=-1 or valid "
            f"for K={num_embeddings}"
        )


def pack_explicit_mask_segments(
    indices_list: Sequence[torch.Tensor],
    num_embeddings_list: Sequence[int],
) -> Tuple[List[AdaptiveTransportSegment], Dict[str, object]]:
    """Serialize adaptive BHWD indices into six logical segment classes.

    Segment order is first-depth scales, mask scales, then active refinement
    scales.  Empty active-refinement segments are retained in metadata but
    carry zero source bits and therefore need no physical channel frame.
    """

    if len(indices_list) != len(num_embeddings_list):
        raise ValueError(
            "indices_list and num_embeddings_list must have equal lengths"
        )
    if not indices_list:
        raise ValueError("at least one scale is required")

    first_segments: List[AdaptiveTransportSegment] = []
    mask_segments: List[AdaptiveTransportSegment] = []
    second_segments: List[AdaptiveTransportSegment] = []
    scale_metadata: List[Dict[str, object]] = []

    for scale, (indices, num_embeddings) in enumerate(
        zip(indices_list, num_embeddings_list)
    ):
        num_embeddings = int(num_embeddings)
        _validate_adaptive_indices(indices, num_embeddings, scale)
        width = bits_per_index(num_embeddings)
        indices_cpu = indices.detach().to(device="cpu", dtype=torch.long)
        height, width_spatial = (
            int(indices_cpu.shape[1]),
            int(indices_cpu.shape[2]),
        )
        first = indices_cpu[..., 0].reshape(-1).numpy().astype(np.int64)
        second = indices_cpu[..., 1].reshape(-1).numpy().astype(np.int64)
        active_mask = second >= 0
        active_values = second[active_mask]

        first_bits = _values_to_bits(first, width)
        mask_bits = active_mask.astype(np.uint8)
        second_bits = _values_to_bits(active_values, width)
        names = {
            "first": f"first_s{scale}",
            "mask": f"mask_s{scale}",
            "second": f"second_active_s{scale}",
        }
        first_segments.append(
            AdaptiveTransportSegment(
                names["first"], "first", scale, first_bits
            )
        )
        mask_segments.append(
            AdaptiveTransportSegment(names["mask"], "mask", scale, mask_bits)
        )
        second_segments.append(
            AdaptiveTransportSegment(
                names["second"], "second_active", scale, second_bits
            )
        )
        scale_metadata.append(
            {
                "scale": scale,
                "shape": [height, width_spatial],
                "num_embeddings": num_embeddings,
                "bits_per_index": width,
                "token_count": int(first.size),
                "tx_active_count": int(active_mask.sum()),
                "segment_names": names,
                "segment_source_bits": {
                    "first": int(first_bits.size),
                    "mask": int(mask_bits.size),
                    "second_active": int(second_bits.size),
                },
            }
        )

    segments = first_segments + mask_segments + second_segments
    source_bits = int(sum(segment.bits.size for segment in segments))
    first_bits = int(
        sum(
            metadata["segment_source_bits"]["first"]
            for metadata in scale_metadata
        )
    )
    mask_bits = int(
        sum(
            metadata["segment_source_bits"]["mask"]
            for metadata in scale_metadata
        )
    )
    second_bits = source_bits - first_bits - mask_bits
    metadata: Dict[str, object] = {
        "schema": "adaptive_explicit_mask_v1",
        "framing": (
            "independent logical segments: all first-depth scales, all masks, "
            "then all active second-depth payloads"
        ),
        "framing_overhead_counted": False,
        "scales": scale_metadata,
        "source_bits": source_bits,
        "first_stage_bits": first_bits,
        "mask_bits": mask_bits,
        "active_second_bits": second_bits,
    }
    return segments, metadata


def unpack_explicit_mask_segments(
    decoded_segments: Dict[str, np.ndarray],
    metadata: Dict[str, object],
) -> Tuple[List[torch.Tensor], Dict[str, object]]:
    """Recover adaptive indices using only received masks and frame metadata.

    The compact active-index payload contains ``A_tx`` values.  If the received
    mask contains more active positions than ``A_tx``, missing tail indices are
    concealed with code zero.  If it contains fewer, unused tail payload values
    are discarded.  The transmitter's original mask is never used for
    positioning at the receiver.
    """

    if metadata.get("schema") != "adaptive_explicit_mask_v1":
        raise ValueError("unsupported adaptive transport metadata")
    recovered: List[torch.Tensor] = []
    per_scale_stats: List[Dict[str, int]] = []

    for scale_metadata in metadata["scales"]:
        scale = int(scale_metadata["scale"])
        height, width_spatial = (
            int(scale_metadata["shape"][0]),
            int(scale_metadata["shape"][1]),
        )
        token_count = int(scale_metadata["token_count"])
        num_embeddings = int(scale_metadata["num_embeddings"])
        width = int(scale_metadata["bits_per_index"])
        tx_active_count = int(scale_metadata["tx_active_count"])
        names = scale_metadata["segment_names"]

        first_values = _bits_to_values(
            decoded_segments.get(names["first"], np.empty(0, dtype=np.uint8)),
            token_count,
            width,
            num_embeddings,
        )
        mask_bits = (
            np.asarray(
                decoded_segments.get(
                    names["mask"], np.empty(0, dtype=np.uint8)
                )
            ).reshape(-1)
            != 0
        )
        if mask_bits.size < token_count:
            mask_bits = np.pad(
                mask_bits,
                (0, token_count - mask_bits.size),
                constant_values=False,
            )
        else:
            mask_bits = mask_bits[:token_count]

        active_values = _bits_to_values(
            decoded_segments.get(
                names["second"], np.empty(0, dtype=np.uint8)
            ),
            tx_active_count,
            width,
            num_embeddings,
        )
        rx_active_positions = np.flatnonzero(mask_bits)
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
            }
        )

    return recovered, {"per_scale": per_scale_stats}


def ldpc_segment_lengths(
    source_bits: int,
    information_block_bits: int = 128,
    coded_block_bits: int = 256,
) -> Dict[str, int]:
    """Return padding and coded lengths for one independently framed segment."""

    source_bits = int(source_bits)
    information_block_bits = int(information_block_bits)
    coded_block_bits = int(coded_block_bits)
    if source_bits < 0:
        raise ValueError("source_bits must be non-negative")
    if information_block_bits <= 0 or coded_block_bits <= 0:
        raise ValueError("LDPC block lengths must be positive")
    if source_bits == 0:
        return {
            "source_bits": 0,
            "information_blocks": 0,
            "padded_information_bits": 0,
            "padding_bits": 0,
            "coded_bits": 0,
        }
    information_blocks = int(
        math.ceil(source_bits / information_block_bits)
    )
    padded = information_blocks * information_block_bits
    return {
        "source_bits": source_bits,
        "information_blocks": information_blocks,
        "padded_information_bits": padded,
        "padding_bits": padded - source_bits,
        "coded_bits": information_blocks * coded_block_bits,
    }


__all__ = [
    "AdaptiveTransportSegment",
    "ldpc_segment_lengths",
    "pack_explicit_mask_segments",
    "unpack_explicit_mask_segments",
]
