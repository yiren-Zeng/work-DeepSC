"""Adaptive Top-K/RLE transport for trained independent two-stage RAQ-RVQ.

This module is intentionally isolated from :mod:`evaluation.quality`.  The
existing dense/per-stage/combined evaluators keep their original contracts.
For every image and scale, the first RAQ-RVQ stage is always transmitted and
an exact Top-K subset of second-stage tokens is selected from the clean
first-stage residual.  Each second-stage activity mask is serialized as one
start bit plus fixed-width ``run_length - 1`` fields.  All logical segments
are concatenated before one LDPC/modulation operation per image.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from communications.channel import awgn_channel
from communications.modulation import (
    bpsk_llr,
    bpsk_modulate,
    qam16_llr,
    qam16_modulate,
    qpsk_llr,
    qpsk_modulate,
)
from utils.metrics import calculate_ms_ssim


_MODULATORS = {
    "bpsk": (bpsk_modulate, bpsk_llr, 1),
    "qpsk": (qpsk_modulate, qpsk_llr, 2),
    "16qam": (qam16_modulate, qam16_llr, 4),
}


@dataclass(frozen=True)
class AdaptiveSegment:
    """One logical segment inside an image's combined source payload."""

    name: str
    kind: str
    scale: int
    bits: np.ndarray


@dataclass
class IndependentAdaptiveSample:
    """CPU cache for one densely encoded image."""

    image: torch.Tensor
    feature_shapes: List[Tuple[int, int]]
    dense_indices: List[List[torch.Tensor]]
    codebooks: List[List[torch.Tensor]]
    first_stage_errors: List[torch.Tensor]


@dataclass
class PreparedAdaptivePacket:
    """One image's selected source payload and clean reconstruction."""

    image: torch.Tensor
    feature_shapes: List[Tuple[int, int]]
    tx_indices: List[List[torch.Tensor]]
    codebooks: List[List[torch.Tensor]]
    active_masks: List[torch.Tensor]
    segments: List[AdaptiveSegment]
    metadata: Dict[str, object]
    selection: Dict[str, object]
    clean_psnr: float
    clean_ms_ssim: float


def bits_per_index(num_embeddings: int) -> int:
    """Return the exact fixed index width for a power-of-two codebook."""

    num_embeddings = int(num_embeddings)
    if num_embeddings < 2 or num_embeddings & (num_embeddings - 1):
        raise ValueError(
            "independent RAQ-RVQ stage K must be a power of two >= 2"
        )
    return num_embeddings.bit_length() - 1


def rounded_active_count(token_count: int, target_active_rate: float) -> int:
    """Return ``floor(rate * N + 0.5)`` clamped to ``[0, N]``."""

    token_count = int(token_count)
    target_active_rate = float(target_active_rate)
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if not 0.0 <= target_active_rate <= 1.0:
        raise ValueError("target_active_rate must be in [0, 1]")
    count = int(math.floor(target_active_rate * token_count + 0.5))
    return min(token_count, max(0, count))


def _select_one_error_grid(
    errors: torch.Tensor,
    target_active_rate: float,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    errors = torch.as_tensor(errors).detach().float()
    if errors.ndim != 2:
        raise ValueError(
            f"one image error grid must be [H,W], got {tuple(errors.shape)}"
        )
    flat = errors.reshape(-1)
    if flat.numel() == 0:
        raise ValueError("error grid must not be empty")
    if not bool(torch.isfinite(flat).all()):
        raise ValueError("first-stage residual errors must be finite")

    token_count = int(flat.numel())
    active_count = rounded_active_count(token_count, target_active_rate)
    active = torch.zeros(token_count, dtype=torch.bool, device=flat.device)
    if active_count == 0:
        maximum = flat.max()
        threshold = torch.nextafter(
            maximum, torch.full_like(maximum, float("inf"))
        )
        strict_above = 0
        equal_count = 0
        equal_selected = 0
        first_inactive = float(maximum.item())
        splits_tie = False
    else:
        order = torch.argsort(flat, descending=True, stable=True)
        selected = order[:active_count]
        active[selected] = True
        threshold = flat[selected[-1]]
        equal = flat == threshold
        strict_above = int((flat > threshold).sum().item())
        equal_count = int(equal.sum().item())
        equal_selected = int((equal & active).sum().item())
        first_inactive = (
            None
            if active_count == token_count
            else float(flat[order[active_count]].item())
        )
        splits_tie = equal_selected < equal_count

    metadata = {
        "threshold": float(threshold.item()),
        "target_active_rate": float(target_active_rate),
        "token_count": token_count,
        "target_active_count": active_count,
        "active_count": int(active.sum().item()),
        "actual_active_rate": active_count / token_count,
        "strict_above_threshold_count": strict_above,
        "threshold_equal_count": equal_count,
        "threshold_equal_selected_count": equal_selected,
        "threshold_splits_tie": bool(splits_tie),
        "first_inactive_error": first_inactive,
    }
    return active.reshape_as(errors), metadata


def select_per_image_topk_masks(
    first_stage_errors: Sequence[torch.Tensor],
    target_active_rates: Sequence[float],
) -> Tuple[List[torch.Tensor], List[Dict[str, object]]]:
    """Select an exact second-stage Top-K mask per image and scale."""

    if len(first_stage_errors) != len(target_active_rates):
        raise ValueError("error scale count must match active-rate count")
    if not first_stage_errors:
        raise ValueError("at least one scale is required")

    normalized: List[torch.Tensor] = []
    batch_size = None
    for scale, errors in enumerate(first_stage_errors):
        errors = torch.as_tensor(errors).detach().float()
        if errors.ndim != 3:
            raise ValueError(
                f"scale {scale} errors must be [B,H,W], got {tuple(errors.shape)}"
            )
        if batch_size is None:
            batch_size = int(errors.shape[0])
            if batch_size <= 0:
                raise ValueError("batch size must be positive")
        elif int(errors.shape[0]) != batch_size:
            raise ValueError("all scales must have the same batch size")
        if not bool(torch.isfinite(errors).all()):
            raise ValueError(f"scale {scale} residual errors contain NaN/Inf")
        normalized.append(errors)

    masks = [torch.zeros_like(errors, dtype=torch.bool) for errors in normalized]
    records = [
        {"batch_index": image_index, "scales": []}
        for image_index in range(int(batch_size))
    ]
    for scale, (errors, rate) in enumerate(zip(normalized, target_active_rates)):
        for image_index in range(int(batch_size)):
            mask, metadata = _select_one_error_grid(
                errors[image_index], float(rate)
            )
            masks[scale][image_index] = mask
            records[image_index]["scales"].append(
                {"scale": scale, **metadata}
            )
    return masks, records


def rle_length_width(token_count: int) -> int:
    """Width of each fixed ``run_length - 1`` field."""

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


def _bits_to_values(
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


def encode_fixed_width_rle_mask(
    mask: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Encode one non-empty raster-order binary mask."""

    flat = (np.asarray(mask).reshape(-1) != 0).astype(np.uint8)
    if flat.size == 0:
        raise ValueError("mask must not be empty")
    token_count = int(flat.size)
    width = rle_length_width(token_count)
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            changes.astype(np.int64),
            np.array([token_count], dtype=np.int64),
        ]
    )
    run_lengths = np.diff(boundaries)
    if int(run_lengths.min()) <= 0 or int(run_lengths.sum()) != token_count:
        raise RuntimeError("invalid internal RLE partition")
    bits = np.concatenate(
        [
            np.array([flat[0]], dtype=np.uint8),
            _values_to_bits(run_lengths - 1, width),
        ]
    )
    return bits, {
        "token_count": token_count,
        "run_length_bits": width,
        "run_count": int(run_lengths.size),
        "start_value": int(flat[0]),
        "run_lengths": run_lengths.astype(np.int64).tolist(),
        "source_bits": int(bits.size),
    }


def decode_fixed_width_rle_mask(
    bits: np.ndarray,
    token_count: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Decode a possibly corrupted fixed-width RLE mask deterministically."""

    token_count = int(token_count)
    width = rle_length_width(token_count)
    normalized = (np.asarray(bits).reshape(-1) != 0).astype(np.uint8)
    if normalized.size < 1:
        raise ValueError("RLE segment must contain a start bit")
    if (normalized.size - 1) % width:
        raise ValueError("RLE length fields must be complete")

    start_value = int(normalized[0])
    field_bits = normalized[1:]
    if field_bits.size:
        grouped = field_bits.reshape(-1, width).astype(np.int64)
        powers = 1 << np.arange(width - 1, -1, -1, dtype=np.int64)
        lengths = np.sum(grouped * powers, axis=1, dtype=np.int64) + 1
    else:
        lengths = np.empty(0, dtype=np.int64)
    decoded_sum = int(lengths.sum())
    output = np.empty(token_count, dtype=np.uint8)
    position = 0
    value = start_value
    last_value = start_value
    used_runs = 0
    for length in lengths:
        if position >= token_count:
            break
        emitted = min(int(length), token_count - position)
        output[position : position + emitted] = value
        position += emitted
        last_value = value
        value = 1 - value
        used_runs += 1
    if position < token_count:
        output[position:] = last_value

    return output, {
        "token_count": token_count,
        "run_length_bits": width,
        "decoded_run_count": int(lengths.size),
        "used_run_count": used_runs,
        "decoded_start_value": start_value,
        "decoded_length_sum": decoded_sum,
        "length_sum_error": decoded_sum - token_count,
        "underflow_tokens": max(token_count - decoded_sum, 0),
        "overflow_tokens": max(decoded_sum - token_count, 0),
        "structurally_valid": decoded_sum == token_count,
        "rx_active_count": int(output.sum()),
    }


def _validate_dense_layout(
    indices_by_scale: Sequence[Sequence[torch.Tensor]],
    active_masks: Sequence[torch.Tensor],
    rvq_k_lists: Sequence[Sequence[int]],
) -> None:
    if not (
        len(indices_by_scale) == len(active_masks) == len(rvq_k_lists)
    ):
        raise ValueError("indices, masks, and K lists must have equal scale counts")
    if not indices_by_scale:
        raise ValueError("at least one scale is required")
    for scale, (stages, mask, stage_ks) in enumerate(
        zip(indices_by_scale, active_masks, rvq_k_lists)
    ):
        if len(stages) != 2 or len(stage_ks) != 2:
            raise ValueError(f"scale {scale} must contain exactly two stages")
        first, second = stages
        if first.ndim != 3 or int(first.shape[0]) != 1:
            raise ValueError(f"scale {scale} indices must be [1,H,W]")
        if tuple(second.shape) != tuple(first.shape):
            raise ValueError(f"scale {scale} stage shapes differ")
        if tuple(mask.shape) != tuple(first.shape):
            raise ValueError(f"scale {scale} mask shape differs from indices")
        for stage, (indices, stage_k) in enumerate(zip(stages, stage_ks)):
            bits_per_index(stage_k)
            if indices.numel() and (
                int(indices.min().item()) < 0
                or int(indices.max().item()) >= int(stage_k)
            ):
                raise ValueError(
                    f"scale {scale} stage {stage} indices invalid for K={stage_k}"
                )


def pack_topk_rle_segments(
    indices_by_scale: Sequence[Sequence[torch.Tensor]],
    active_masks: Sequence[torch.Tensor],
    rvq_k_lists: Sequence[Sequence[int]],
) -> Tuple[List[AdaptiveSegment], Dict[str, object]]:
    """Pack two-stage nested indices as one combined adaptive source payload."""

    _validate_dense_layout(indices_by_scale, active_masks, rvq_k_lists)
    first_segments: List[AdaptiveSegment] = []
    mask_segments: List[AdaptiveSegment] = []
    second_segments: List[AdaptiveSegment] = []
    scales: List[Dict[str, object]] = []

    for scale, (stages, mask, stage_ks) in enumerate(
        zip(indices_by_scale, active_masks, rvq_k_lists)
    ):
        first_k, second_k = [int(value) for value in stage_ks]
        first_width = bits_per_index(first_k)
        second_width = bits_per_index(second_k)
        first = stages[0].detach().cpu().long().reshape(-1).numpy()
        second = stages[1].detach().cpu().long().reshape(-1).numpy()
        mask_flat = mask.detach().cpu().bool().reshape(-1).numpy()
        active_second = second[mask_flat]
        first_bits = _values_to_bits(first, first_width)
        mask_bits, rle = encode_fixed_width_rle_mask(mask_flat)
        second_bits = _values_to_bits(active_second, second_width)
        names = {
            "first": f"first_s{scale}",
            "mask_rle": f"mask_rle_s{scale}",
            "second": f"second_active_s{scale}",
        }
        first_segments.append(
            AdaptiveSegment(names["first"], "first", scale, first_bits)
        )
        mask_segments.append(
            AdaptiveSegment(names["mask_rle"], "mask_rle", scale, mask_bits)
        )
        second_segments.append(
            AdaptiveSegment(
                names["second"], "second_active", scale, second_bits
            )
        )
        token_count = int(first.size)
        scales.append(
            {
                "scale": scale,
                "shape": [int(stages[0].shape[1]), int(stages[0].shape[2])],
                "token_count": token_count,
                "first_num_embeddings": first_k,
                "second_num_embeddings": second_k,
                "first_bits_per_index": first_width,
                "second_bits_per_index": second_width,
                "tx_active_count": int(mask_flat.sum()),
                "segment_names": names,
                "segment_source_bits": {
                    "first": int(first_bits.size),
                    "mask_rle": int(mask_bits.size),
                    "second_active": int(second_bits.size),
                },
                "raw_mask_source_bits_reference": token_count,
                "rle": rle,
            }
        )

    segments = first_segments + mask_segments + second_segments
    offsets = []
    offset = 0
    for segment in segments:
        end = offset + int(segment.bits.size)
        offsets.append(
            {
                "name": segment.name,
                "kind": segment.kind,
                "scale": int(segment.scale),
                "offset": offset,
                "end": end,
                "source_bits": int(segment.bits.size),
            }
        )
        offset = end

    first_bits = sum(
        int(scale["segment_source_bits"]["first"]) for scale in scales
    )
    mask_bits = sum(
        int(scale["segment_source_bits"]["mask_rle"]) for scale in scales
    )
    second_bits = sum(
        int(scale["segment_source_bits"]["second_active"]) for scale in scales
    )
    raw_mask_bits = sum(
        int(scale["raw_mask_source_bits_reference"]) for scale in scales
    )
    dense_second_bits = sum(
        int(scale["token_count"]) * int(scale["second_bits_per_index"])
        for scale in scales
    )
    metadata = {
        "schema": "independent_raq_rvq_topk_rle_combined_v1",
        "stream_packing": "combined",
        "logical_segment_order": [segment.name for segment in segments],
        "segments": offsets,
        "scales": scales,
        "source_bits": first_bits + mask_bits + second_bits,
        "first_stage_bits": first_bits,
        "rle_mask_bits": mask_bits,
        "active_second_bits": second_bits,
        "raw_mask_bits_reference": raw_mask_bits,
        "raw_explicit_source_bits_reference": (
            first_bits + raw_mask_bits + second_bits
        ),
        "dense_two_stage_source_bits_reference": (
            first_bits + dense_second_bits
        ),
        "framing_metadata_counted": False,
    }
    if int(metadata["source_bits"]) != offset:
        raise RuntimeError("combined source length accounting mismatch")
    return segments, metadata


def combined_bits_from_segments(segments: Sequence[AdaptiveSegment]) -> np.ndarray:
    """Concatenate logical segments without adding per-segment padding."""

    if not segments:
        raise ValueError("segments must not be empty")
    return np.concatenate(
        [np.asarray(segment.bits, dtype=np.uint8).reshape(-1) for segment in segments]
    ).astype(np.uint8, copy=False)


def _split_combined_bits(
    decoded_bits: np.ndarray,
    metadata: Dict[str, object],
) -> Dict[str, np.ndarray]:
    expected = int(metadata["source_bits"])
    normalized = (np.asarray(decoded_bits).reshape(-1) != 0).astype(np.uint8)
    if normalized.size < expected:
        normalized = np.pad(normalized, (0, expected - normalized.size))
    else:
        normalized = normalized[:expected]
    return {
        str(segment["name"]): normalized[
            int(segment["offset"]) : int(segment["end"])
        ]
        for segment in metadata["segments"]
    }


def unpack_topk_rle_combined(
    decoded_bits: np.ndarray,
    metadata: Dict[str, object],
) -> Tuple[List[List[torch.Tensor]], List[torch.Tensor], Dict[str, object]]:
    """Recover nested indices using received masks and TX framing metadata."""

    if metadata.get("schema") != "independent_raq_rvq_topk_rle_combined_v1":
        raise ValueError("unsupported adaptive combined transport schema")
    decoded = _split_combined_bits(decoded_bits, metadata)
    recovered: List[List[torch.Tensor]] = []
    masks: List[torch.Tensor] = []
    per_scale = []

    for scale_metadata in metadata["scales"]:
        scale = int(scale_metadata["scale"])
        height, width = [int(value) for value in scale_metadata["shape"]]
        token_count = int(scale_metadata["token_count"])
        first_k = int(scale_metadata["first_num_embeddings"])
        second_k = int(scale_metadata["second_num_embeddings"])
        first_width = int(scale_metadata["first_bits_per_index"])
        second_width = int(scale_metadata["second_bits_per_index"])
        tx_active_count = int(scale_metadata["tx_active_count"])
        names = scale_metadata["segment_names"]

        first = _bits_to_values(
            decoded[names["first"]], token_count, first_width, first_k
        )
        mask, rle_stats = decode_fixed_width_rle_mask(
            decoded[names["mask_rle"]], token_count
        )
        active_values = _bits_to_values(
            decoded[names["second"]],
            tx_active_count,
            second_width,
            second_k,
        )
        rx_positions = np.flatnonzero(mask)
        rx_active_count = int(rx_positions.size)
        mapped = min(rx_active_count, tx_active_count)
        zero_filled = max(rx_active_count - tx_active_count, 0)
        truncated = max(tx_active_count - rx_active_count, 0)
        second = np.full(token_count, -1, dtype=np.int64)
        if mapped:
            second[rx_positions[:mapped]] = active_values[:mapped]
        if zero_filled:
            second[rx_positions[mapped:]] = 0

        first_tensor = torch.from_numpy(first.reshape(1, height, width)).long()
        second_tensor = torch.from_numpy(second.reshape(1, height, width)).long()
        mask_tensor = second_tensor >= 0
        recovered.append([first_tensor, second_tensor])
        masks.append(mask_tensor)
        per_scale.append(
            {
                "scale": scale,
                "tx_active_count": tx_active_count,
                "rx_active_count": rx_active_count,
                "active_count_mismatch": rx_active_count - tx_active_count,
                "mapped_second_indices": mapped,
                "zero_filled_second_indices": zero_filled,
                "truncated_second_indices": truncated,
                **rle_stats,
            }
        )
    return recovered, masks, {"per_scale": per_scale}


def reconstruct_from_adaptive_indices(
    model,
    indices_by_scale: Sequence[Sequence[torch.Tensor]],
    feature_shapes: Sequence[Tuple[int, int]],
    codebooks: Sequence[Sequence[torch.Tensor]],
) -> torch.Tensor:
    """Reconstruct with true zero contribution at inactive second stages.

    The model's normal nested reconstruction does not accept ``-1`` and its
    independent stage-two codebooks have no reserved zero.  A temporary zero
    row is prepended only at this dequantization boundary; physical K and bit
    widths remain unchanged.
    """

    if not (len(indices_by_scale) == len(feature_shapes) == len(codebooks)):
        raise ValueError("indices, feature shapes, and codebooks must align")
    fallback_device = getattr(model, "device", torch.device("cpu"))
    encoder_device = torch.device(
        getattr(model, "encoder_device", fallback_device)
    )
    shifted_indices = []
    augmented_codebooks = []
    for scale, (stages, stage_codebooks) in enumerate(
        zip(indices_by_scale, codebooks)
    ):
        if len(stages) != 2 or len(stage_codebooks) != 2:
            raise ValueError(f"scale {scale} must have exactly two stages")
        first = stages[0].to(encoder_device, non_blocking=True)
        second = stages[1].to(encoder_device, non_blocking=True)
        if second.numel() and int(second.min().item()) < -1:
            raise ValueError("second-stage sentinel must be -1")
        first_codebook = stage_codebooks[0].to(
            encoder_device, non_blocking=True
        )
        second_codebook = stage_codebooks[1].to(
            encoder_device, non_blocking=True
        )
        if first.numel() and (
            int(first.min().item()) < 0
            or int(first.max().item()) >= int(first_codebook.shape[0])
        ):
            raise ValueError(f"scale {scale} first-stage index out of range")
        if second.numel() and int(second.max().item()) >= int(
            second_codebook.shape[0]
        ):
            raise ValueError(f"scale {scale} second-stage index out of range")
        augmented_second = torch.cat(
            [torch.zeros_like(second_codebook[:1]), second_codebook], dim=0
        )
        shifted_indices.append([first, second + 1])
        augmented_codebooks.append([first_codebook, augmented_second])
    return model.reconstruct_from_indices(
        shifted_indices,
        feature_shapes=list(feature_shapes),
        codebooks=augmented_codebooks,
    )


def _quality_metrics(
    real_image: torch.Tensor,
    reconstructed: torch.Tensor,
) -> Tuple[float, float]:
    if real_image.device != reconstructed.device:
        real_image = real_image.to(reconstructed.device, non_blocking=True)
    reference = (real_image + 1.0) / 2.0
    estimate = (reconstructed + 1.0) / 2.0
    ms_ssim = float(calculate_ms_ssim(reference, estimate))
    mse = float(torch.mean((reference - estimate) ** 2).item())
    psnr = 100.0 if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))
    return ms_ssim, psnr


@torch.no_grad()
def collect_independent_adaptive_samples(
    model,
    loader,
    max_images: int | None = None,
) -> List[IndependentAdaptiveSample]:
    """Encode once and cache dense indices plus per-token stage-one errors."""

    if not bool(getattr(model, "use_independent_raq_rvq", False)):
        raise ValueError("model must enable the trained independent RAQ-RVQ branch")
    if int(getattr(model, "independent_raq_rvq_depth", 0)) != 2:
        raise ValueError("adaptive masking requires independent RAQ-RVQ depth 2")
    encoder = getattr(model, "_forward_test_independent_raq_rvq", None)
    if not callable(encoder):
        raise RuntimeError("model lacks the independent RAQ-RVQ test encoder")

    model.eval()
    samples: List[IndependentAdaptiveSample] = []
    images_seen = 0
    for image in loader:
        if max_images is not None and images_seen >= int(max_images):
            break
        if int(image.shape[0]) != 1:
            raise ValueError("adaptive combined evaluation requires batch_size=1")
        encoder_image = model._to_encoder_device(image)
        features = model.semantic_encoder(encoder_image)
        features[-1] = model.bottleneck_attention(features[-1])
        out = encoder(features)
        if out.get("branch") != "independent_raq_rvq":
            raise RuntimeError("unexpected model branch during adaptive encoding")
        scale_count = len(features)
        if not (
            len(out["indices"])
            == len(out["codebooks"])
            == len(out["feature_shapes"])
            == scale_count
        ):
            raise ValueError(
                "feature, index, codebook, and shape scale counts must align"
            )
        errors = []
        for scale, (feature, stages, stage_codebooks) in enumerate(
            zip(features, out["indices"], out["codebooks"])
        ):
            if len(stages) != 2 or len(stage_codebooks) != 2:
                raise ValueError(f"scale {scale} is not a two-stage layout")
            first_quantized = F.embedding(
                stages[0].reshape(-1), stage_codebooks[0]
            ).view(
                int(stages[0].shape[0]),
                int(stages[0].shape[1]),
                int(stages[0].shape[2]),
                int(stage_codebooks[0].shape[1]),
            ).permute(0, 3, 1, 2).contiguous()
            errors.append(
                (feature.detach() - first_quantized.detach())
                .pow(2)
                .mean(dim=1)
                .cpu()
            )
        samples.append(
            IndependentAdaptiveSample(
                image=image.detach().cpu(),
                feature_shapes=[tuple(shape) for shape in out["feature_shapes"]],
                dense_indices=[
                    [indices.detach().cpu() for indices in stages]
                    for stages in out["indices"]
                ],
                codebooks=[
                    [codebook.detach().cpu() for codebook in stages]
                    for stages in out["codebooks"]
                ],
                first_stage_errors=errors,
            )
        )
        images_seen += int(image.shape[0])
    if not samples:
        raise ValueError("adaptive evaluation loader produced no images")
    return samples


@torch.no_grad()
def prepare_topk_rle_packets(
    model,
    samples: Sequence[IndependentAdaptiveSample],
    target_active_rates: Sequence[float],
    rvq_k_lists: Sequence[Sequence[int]],
    image_names: Sequence[str] | None = None,
) -> Tuple[List[PreparedAdaptivePacket], List[Dict[str, object]]]:
    """Apply Top-K and prepare one combined RLE packet per image."""

    if not samples:
        raise ValueError("samples must not be empty")
    if image_names is not None and len(image_names) != len(samples):
        raise ValueError("image_names must align with samples")
    num_scales = len(rvq_k_lists)
    if num_scales <= 0 or len(target_active_rates) != num_scales:
        raise ValueError("active rates and K lists must have equal scale counts")
    packets = []
    selection_records = []
    for image_index, sample in enumerate(samples):
        if not (
            len(sample.first_stage_errors)
            == len(sample.dense_indices)
            == len(sample.codebooks)
            == len(sample.feature_shapes)
            == num_scales
        ):
            raise ValueError(
                f"sample {image_index} adaptive scale layouts do not align"
            )
        for scale, (stage_codebooks, stage_ks) in enumerate(
            zip(sample.codebooks, rvq_k_lists)
        ):
            if len(stage_codebooks) != 2 or len(stage_ks) != 2:
                raise ValueError(f"scale {scale} must have exactly two stages")
            for stage, (codebook, expected_k) in enumerate(
                zip(stage_codebooks, stage_ks)
            ):
                actual_k = int(codebook.shape[0])
                if actual_k != int(expected_k):
                    raise ValueError(
                        f"scale {scale} stage {stage} codebook has K={actual_k}, "
                        f"but configured K={int(expected_k)}"
                    )
        masks, batch_records = select_per_image_topk_masks(
            sample.first_stage_errors, target_active_rates
        )
        if len(batch_records) != 1:
            raise RuntimeError("expected one selection record per sample")
        tx_indices = []
        for stages, mask in zip(sample.dense_indices, masks):
            second = torch.where(
                mask,
                stages[1],
                torch.full_like(stages[1], -1),
            )
            tx_indices.append([stages[0].clone(), second])
        segments, metadata = pack_topk_rle_segments(
            sample.dense_indices, masks, rvq_k_lists
        )
        reconstructed = reconstruct_from_adaptive_indices(
            model,
            tx_indices,
            sample.feature_shapes,
            sample.codebooks,
        )
        clean_ms_ssim, clean_psnr = _quality_metrics(
            sample.image, reconstructed
        )
        selection = {
            "image_index": image_index,
            "image_number": image_index + 1,
            "image_name": (
                str(image_names[image_index])
                if image_names is not None
                else f"image_{image_index + 1:04d}"
            ),
            "scales": batch_records[0]["scales"],
        }
        packets.append(
            PreparedAdaptivePacket(
                image=sample.image,
                feature_shapes=list(sample.feature_shapes),
                tx_indices=tx_indices,
                codebooks=sample.codebooks,
                active_masks=masks,
                segments=segments,
                metadata=metadata,
                selection=selection,
                clean_psnr=clean_psnr,
                clean_ms_ssim=clean_ms_ssim,
            )
        )
        selection_records.append(selection)
    return packets, selection_records


def combined_stream_lengths(
    payload_bits: int,
    ldpc_code: Dict[str, object],
    modulation_bits: int,
) -> Dict[str, int]:
    """Calculate the two padding layers for one combined image stream."""

    payload_bits = int(payload_bits)
    modulation_bits = int(modulation_bits)
    if payload_bits < 0 or modulation_bits <= 0:
        raise ValueError("invalid stream length input")
    k = int(ldpc_code["k"])
    n = int(ldpc_code["n"])
    blocks = (payload_bits + k - 1) // k if payload_bits else 0
    ldpc_input = blocks * k
    coded = blocks * n
    modulation_padding = (-coded) % modulation_bits
    transmitted = coded + modulation_padding
    return {
        "payload_bits": payload_bits,
        "ldpc_input_bits": ldpc_input,
        "ldpc_padding_bits": ldpc_input - payload_bits,
        "coded_bits": coded,
        "modulation_padding_bits": modulation_padding,
        "transmitted_bits": transmitted,
        "channel_symbols": transmitted // modulation_bits,
    }


def summarize_packets(
    packets: Sequence[PreparedAdaptivePacket],
    ldpc_code: Dict[str, object],
    modulation: str,
) -> Dict[str, object]:
    """Summarize source rate and true combined-stream physical lengths."""

    if not packets:
        raise ValueError("packets must not be empty")
    if modulation not in _MODULATORS:
        raise ValueError(f"unsupported modulation: {modulation}")
    modulation_bits = _MODULATORS[modulation][2]
    num_scales = len(packets[0].metadata["scales"])
    per_scale = [
        {
            "scale": scale,
            "token_count": 0,
            "tx_active_count": 0,
            "first_stage_source_bits": 0,
            "rle_mask_source_bits": 0,
            "raw_mask_source_bits_reference": 0,
            "active_second_source_bits": 0,
            "rle_run_count": 0,
            "rle_source_bits_per_image": [],
            "run_count_per_image": [],
            "token_count_values": [],
            "active_count_values": [],
        }
        for scale in range(num_scales)
    ]
    totals = {
        "rle": {key: 0 for key in combined_stream_lengths(0, ldpc_code, modulation_bits)},
        "raw": {key: 0 for key in combined_stream_lengths(0, ldpc_code, modulation_bits)},
        "dense": {key: 0 for key in combined_stream_lengths(0, ldpc_code, modulation_bits)},
    }
    total_pixels = 0
    total_values = 0
    for packet in packets:
        if len(packet.metadata["scales"]) != num_scales:
            raise ValueError("all packets must have the same scale count")
        total_pixels += int(
            packet.image.shape[0]
            * packet.image.shape[-2]
            * packet.image.shape[-1]
        )
        total_values += int(packet.image.numel())
        references = {
            "rle": int(packet.metadata["source_bits"]),
            "raw": int(packet.metadata["raw_explicit_source_bits_reference"]),
            "dense": int(packet.metadata["dense_two_stage_source_bits_reference"]),
        }
        for name, payload in references.items():
            lengths = combined_stream_lengths(payload, ldpc_code, modulation_bits)
            for key, value in lengths.items():
                totals[name][key] += int(value)
        for scale_metadata in packet.metadata["scales"]:
            stats = per_scale[int(scale_metadata["scale"])]
            segment_bits = scale_metadata["segment_source_bits"]
            stats["token_count"] += int(scale_metadata["token_count"])
            stats["tx_active_count"] += int(scale_metadata["tx_active_count"])
            stats["first_stage_source_bits"] += int(segment_bits["first"])
            stats["rle_mask_source_bits"] += int(segment_bits["mask_rle"])
            stats["raw_mask_source_bits_reference"] += int(
                scale_metadata["raw_mask_source_bits_reference"]
            )
            stats["active_second_source_bits"] += int(
                segment_bits["second_active"]
            )
            stats["rle_run_count"] += int(scale_metadata["rle"]["run_count"])
            stats["rle_source_bits_per_image"].append(
                int(segment_bits["mask_rle"])
            )
            stats["run_count_per_image"].append(
                int(scale_metadata["rle"]["run_count"])
            )
            stats["token_count_values"].append(
                int(scale_metadata["token_count"])
            )
            stats["active_count_values"].append(
                int(scale_metadata["tx_active_count"])
            )

    for stats in per_scale:
        token_count = int(stats["token_count"])
        active_count = int(stats["tx_active_count"])
        mask_values = stats.pop("rle_source_bits_per_image")
        run_values = stats.pop("run_count_per_image")
        token_values = stats.pop("token_count_values")
        active_values = stats.pop("active_count_values")
        raw_mask = int(stats["raw_mask_source_bits_reference"])
        rle_mask = int(stats["rle_mask_source_bits"])
        stats.update(
            {
                "token_count_per_image": token_count / len(packets),
                "token_count_min_per_image": min(token_values),
                "token_count_max_per_image": max(token_values),
                "tx_active_count_per_image": active_count / len(packets),
                "tx_active_count_min_per_image": min(active_values),
                "tx_active_count_max_per_image": max(active_values),
                "tx_active_ratio": active_count / token_count,
                "rle_run_count_mean_per_image": sum(run_values) / len(packets),
                "rle_run_count_min_per_image": min(run_values),
                "rle_run_count_max_per_image": max(run_values),
                "rle_mask_source_bits_mean_per_image": (
                    sum(mask_values) / len(packets)
                ),
                "rle_mask_source_bits_min_per_image": min(mask_values),
                "rle_mask_source_bits_max_per_image": max(mask_values),
                "mask_source_bits_saved_vs_raw": raw_mask - rle_mask,
                "mask_source_saving_ratio": (
                    (raw_mask - rle_mask) / raw_mask if raw_mask else 0.0
                ),
            }
        )

    rle = totals["rle"]
    raw = totals["raw"]
    dense = totals["dense"]
    return {
        "num_images": len(packets),
        "total_image_pixels": total_pixels,
        "total_rgb_values": total_values,
        "first_stage_bits": sum(
            int(scale["first_stage_source_bits"]) for scale in per_scale
        ),
        "rle_mask_bits": sum(
            int(scale["rle_mask_source_bits"]) for scale in per_scale
        ),
        "active_second_bits": sum(
            int(scale["active_second_source_bits"]) for scale in per_scale
        ),
        "rle_source_bits": rle["payload_bits"],
        "rle_source_bpp": rle["payload_bits"] / total_pixels,
        "rle_coded_bits": rle["coded_bits"],
        "rle_coded_bpp": rle["coded_bits"] / total_pixels,
        "rle_channel_symbols": rle["channel_symbols"],
        "rle_channel_uses_per_pixel": rle["channel_symbols"] / total_pixels,
        "rle_transmission_ratio_per_rgb_value": (
            rle["channel_symbols"] / total_values
        ),
        "rle_ldpc_padding_bits": rle["ldpc_padding_bits"],
        "rle_modulation_padding_bits": rle["modulation_padding_bits"],
        "raw_explicit_reference": raw,
        "dense_two_stage_reference": dense,
        "source_bits_saved_vs_raw_mask": raw["payload_bits"] - rle["payload_bits"],
        "coded_bits_saved_vs_raw_mask": raw["coded_bits"] - rle["coded_bits"],
        "source_bits_delta_vs_dense": rle["payload_bits"] - dense["payload_bits"],
        "coded_bits_delta_vs_dense": rle["coded_bits"] - dense["coded_bits"],
        "per_scale": per_scale,
        "combined_stream": True,
        "framing_metadata_counted": False,
    }


def _reset_channel_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))


def transmit_combined_stream(
    source_bits: np.ndarray,
    snr_db: float,
    ldpc_code: Dict[str, object],
    device: torch.device,
    modulation: str,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Transmit exactly one image-level combined stream."""

    if modulation not in _MODULATORS:
        raise ValueError(f"unsupported modulation: {modulation}")
    from communications.ldpc_coding import ldpc_decode, ldpc_encode

    modulate, calculate_llr, modulation_bits = _MODULATORS[modulation]
    source = (np.asarray(source_bits).reshape(-1) != 0).astype(np.uint8)
    lengths = combined_stream_lengths(
        int(source.size), ldpc_code, modulation_bits
    )
    coded = np.asarray(ldpc_encode(source, code=ldpc_code)).reshape(-1)
    if int(coded.size) != lengths["coded_bits"]:
        raise RuntimeError("LDPC encoded length differs from combined accounting")
    if lengths["modulation_padding_bits"]:
        transmitted = np.pad(
            coded, (0, lengths["modulation_padding_bits"]), "constant"
        )
    else:
        transmitted = coded
    transmitted_tensor = torch.from_numpy(transmitted).float().to(device)
    symbols = modulate(transmitted_tensor)
    noisy = awgn_channel(symbols, float(snr_db))
    llrs = calculate_llr(noisy, float(snr_db), device).reshape(-1)
    decoded = np.asarray(
        ldpc_decode(
            llrs[: lengths["coded_bits"]].detach().cpu().numpy(),
            ldpc_code,
        )
    ).reshape(-1)[: source.size]
    if decoded.size < source.size:
        decoded = np.pad(decoded, (0, source.size - decoded.size))
    decoded = (decoded != 0).astype(np.uint8)
    return decoded, {
        **lengths,
        "bit_errors": int(np.count_nonzero(decoded != source)),
    }


@torch.no_grad()
def evaluate_packets_over_channel(
    model,
    packets: Sequence[PreparedAdaptivePacket],
    snr_db: float,
    ldpc_code: Dict[str, object],
    device: torch.device,
    modulation: str,
    seed: int = 42,
    transmit_fn: Callable[..., Tuple[np.ndarray, Dict[str, int]]] | None = None,
) -> Dict[str, object]:
    """Evaluate prepared packets through one combined channel stream per image."""

    if not packets:
        raise ValueError("packets must not be empty")
    _reset_channel_seed(seed)
    transmit_fn = transmit_fn or transmit_combined_stream
    num_scales = len(packets[0].metadata["scales"])
    kind_bits = {"first": 0, "mask_rle": 0, "second_active": 0}
    kind_errors = {"first": 0, "mask_rle": 0, "second_active": 0}
    per_scale = [
        {
            "scale": scale,
            "tx_active_count": 0,
            "rx_active_count": 0,
            "mask_token_count": 0,
            "semantic_mask_bit_errors": 0,
            "structurally_valid_rle_frames": 0,
            "exact_mask_frames": 0,
            "active_count_mismatch_frames": 0,
            "zero_filled_second_indices": 0,
            "truncated_second_indices": 0,
            "first_source_bits": 0,
            "first_source_bit_errors": 0,
            "rle_source_bit_errors": 0,
            "second_source_bits": 0,
            "second_source_bit_errors": 0,
            "rle_source_bits": 0,
            "rle_start_bit_errors": 0,
            "rle_length_bit_errors": 0,
        }
        for scale in range(num_scales)
    ]
    totals = {
        "payload_bits": 0,
        "ldpc_input_bits": 0,
        "ldpc_padding_bits": 0,
        "coded_bits": 0,
        "modulation_padding_bits": 0,
        "transmitted_bits": 0,
        "channel_symbols": 0,
        "bit_errors": 0,
    }
    psnr_values = []
    ms_ssim_values = []
    per_image = []
    images_with_errors = 0

    for image_index, packet in enumerate(packets):
        if len(packet.metadata["scales"]) != num_scales:
            raise ValueError("all packets must have the same scale count")
        source = combined_bits_from_segments(packet.segments)
        decoded, channel_stats = transmit_fn(
            source,
            float(snr_db),
            ldpc_code,
            device,
            modulation,
        )
        for key in totals:
            totals[key] += int(channel_stats[key])
        packet_errors = int(np.count_nonzero(decoded != source))
        if int(channel_stats["bit_errors"]) != packet_errors:
            raise RuntimeError(
                "transmit function bit_errors disagrees with decoded payload"
            )
        if packet_errors:
            images_with_errors += 1
        decoded_segments = _split_combined_bits(decoded, packet.metadata)
        source_segments = {segment.name: segment for segment in packet.segments}
        segment_records = {}
        for segment_metadata in packet.metadata["segments"]:
            name = str(segment_metadata["name"])
            kind = str(segment_metadata["kind"])
            scale = int(segment_metadata["scale"])
            sent = source_segments[name].bits
            received = decoded_segments[name]
            errors = int(np.count_nonzero(sent != received))
            bits = int(sent.size)
            kind_bits[kind] += bits
            kind_errors[kind] += errors
            segment_records[name] = {
                "source_bits": bits,
                "source_bit_errors": errors,
            }
            if kind == "first":
                per_scale[scale]["first_source_bits"] += bits
                per_scale[scale]["first_source_bit_errors"] += errors
            elif kind == "mask_rle":
                stats = per_scale[scale]
                stats["rle_source_bits"] += bits
                stats["rle_source_bit_errors"] += errors
                stats["rle_start_bit_errors"] += int(
                    bool(bits) and sent[0] != received[0]
                )
                if bits > 1:
                    stats["rle_length_bit_errors"] += int(
                        np.count_nonzero(sent[1:] != received[1:])
                    )
            else:
                per_scale[scale]["second_source_bits"] += bits
                per_scale[scale]["second_source_bit_errors"] += errors

        recovered, rx_masks, decode_stats = unpack_topk_rle_combined(
            decoded, packet.metadata
        )
        image_scale_records = []
        for scale, (tx_mask, rx_mask, scale_decode) in enumerate(
            zip(packet.active_masks, rx_masks, decode_stats["per_scale"])
        ):
            tx_flat = tx_mask.detach().cpu().numpy().reshape(-1)
            rx_flat = rx_mask.detach().cpu().numpy().reshape(-1)
            semantic_errors = int(np.count_nonzero(tx_flat != rx_flat))
            stats = per_scale[scale]
            stats["tx_active_count"] += int(tx_flat.sum())
            stats["rx_active_count"] += int(rx_flat.sum())
            stats["mask_token_count"] += int(tx_flat.size)
            stats["semantic_mask_bit_errors"] += semantic_errors
            stats["structurally_valid_rle_frames"] += int(
                scale_decode["structurally_valid"]
            )
            stats["exact_mask_frames"] += int(semantic_errors == 0)
            stats["active_count_mismatch_frames"] += int(
                scale_decode["active_count_mismatch"] != 0
            )
            stats["zero_filled_second_indices"] += int(
                scale_decode["zero_filled_second_indices"]
            )
            stats["truncated_second_indices"] += int(
                scale_decode["truncated_second_indices"]
            )
            names = packet.metadata["scales"][scale]["segment_names"]
            first_record = segment_records[names["first"]]
            mask_record = segment_records[names["mask_rle"]]
            second_record = segment_records[names["second"]]
            scale_source_bits = sum(
                int(record["source_bits"])
                for record in (first_record, mask_record, second_record)
            )
            scale_source_errors = sum(
                int(record["source_bit_errors"])
                for record in (first_record, mask_record, second_record)
            )
            image_scale_records.append(
                {
                    "scale": scale,
                    "tx_active_count": int(tx_flat.sum()),
                    "rx_active_count": int(rx_flat.sum()),
                    "semantic_mask_bit_errors": semantic_errors,
                    "semantic_mask_ber": semantic_errors / int(tx_flat.size),
                    "first_source_bits": first_record["source_bits"],
                    "first_source_bit_errors": first_record[
                        "source_bit_errors"
                    ],
                    "first_source_ber_after_ldpc": (
                        first_record["source_bit_errors"]
                        / first_record["source_bits"]
                        if first_record["source_bits"]
                        else 0.0
                    ),
                    "rle_source_bits": mask_record["source_bits"],
                    "rle_source_bit_errors": mask_record[
                        "source_bit_errors"
                    ],
                    "rle_source_ber_after_ldpc": (
                        mask_record["source_bit_errors"]
                        / mask_record["source_bits"]
                        if mask_record["source_bits"]
                        else 0.0
                    ),
                    "second_source_bits": second_record["source_bits"],
                    "second_source_bit_errors": second_record[
                        "source_bit_errors"
                    ],
                    "second_source_ber_after_ldpc": (
                        second_record["source_bit_errors"]
                        / second_record["source_bits"]
                        if second_record["source_bits"]
                        else 0.0
                    ),
                    "source_bits": scale_source_bits,
                    "source_bit_errors": scale_source_errors,
                    "source_ber_after_ldpc": (
                        scale_source_errors / scale_source_bits
                        if scale_source_bits
                        else 0.0
                    ),
                    "rle_run_count": int(
                        packet.metadata["scales"][scale]["rle"]["run_count"]
                    ),
                    "structurally_valid": bool(
                        scale_decode["structurally_valid"]
                    ),
                    "length_sum_error": int(scale_decode["length_sum_error"]),
                    "zero_filled_second_indices": int(
                        scale_decode["zero_filled_second_indices"]
                    ),
                    "truncated_second_indices": int(
                        scale_decode["truncated_second_indices"]
                    ),
                }
            )

        reconstructed = reconstruct_from_adaptive_indices(
            model,
            recovered,
            packet.feature_shapes,
            packet.codebooks,
        )
        ms_ssim, psnr = _quality_metrics(packet.image, reconstructed)
        psnr_values.append(psnr)
        ms_ssim_values.append(ms_ssim)
        per_image.append(
            {
                "image_index": image_index,
                "image_number": image_index + 1,
                "image_name": packet.selection["image_name"],
                "source_bits": int(source.size),
                "coded_bits": int(channel_stats["coded_bits"]),
                "channel_symbols": int(channel_stats["channel_symbols"]),
                "source_bit_errors": packet_errors,
                "source_ber_after_ldpc": packet_errors / int(source.size),
                "ldpc_input_bits": int(channel_stats["ldpc_input_bits"]),
                "ldpc_padding_bits": int(channel_stats["ldpc_padding_bits"]),
                "modulation_padding_bits": int(
                    channel_stats["modulation_padding_bits"]
                ),
                "transmitted_bits": int(channel_stats["transmitted_bits"]),
                "psnr": psnr,
                "ms_ssim": ms_ssim,
                "scales": image_scale_records,
            }
        )

    for stats in per_scale:
        token_count = int(stats["mask_token_count"])
        first_bits = int(stats["first_source_bits"])
        rle_bits = int(stats["rle_source_bits"])
        second_bits = int(stats["second_source_bits"])
        source_bits = first_bits + rle_bits + second_bits
        source_errors = (
            int(stats["first_source_bit_errors"])
            + int(stats["rle_source_bit_errors"])
            + int(stats["second_source_bit_errors"])
        )
        stats.update(
            {
                "tx_active_ratio": stats["tx_active_count"] / token_count,
                "rx_active_ratio": stats["rx_active_count"] / token_count,
                "semantic_mask_ber": (
                    stats["semantic_mask_bit_errors"] / token_count
                ),
                "rle_source_ber_after_ldpc": (
                    stats["rle_source_bit_errors"] / rle_bits
                    if rle_bits
                    else 0.0
                ),
                "first_source_ber_after_ldpc": (
                    stats["first_source_bit_errors"] / first_bits
                    if first_bits
                    else 0.0
                ),
                "second_source_ber_after_ldpc": (
                    stats["second_source_bit_errors"] / second_bits
                    if second_bits
                    else 0.0
                ),
                "source_bits": source_bits,
                "source_bit_errors": source_errors,
                "source_ber_after_ldpc": (
                    source_errors / source_bits if source_bits else 0.0
                ),
                "structurally_valid_rle_frame_rate": (
                    stats["structurally_valid_rle_frames"] / len(packets)
                ),
                "exact_mask_frame_rate": (
                    stats["exact_mask_frames"] / len(packets)
                ),
            }
        )

    total_pixels = sum(
        int(
            packet.image.shape[0]
            * packet.image.shape[-2]
            * packet.image.shape[-1]
        )
        for packet in packets
    )
    total_values = sum(int(packet.image.numel()) for packet in packets)
    return {
        "snr_db": float(snr_db),
        "modulation": modulation,
        "psnr": float(np.mean(psnr_values)),
        "ms_ssim": float(np.mean(ms_ssim_values)),
        **{key: int(value) for key, value in totals.items()},
        "source_bpp": totals["payload_bits"] / total_pixels,
        "coded_bpp": totals["coded_bits"] / total_pixels,
        "channel_uses_per_pixel": totals["channel_symbols"] / total_pixels,
        "transmission_ratio_per_rgb_value": (
            totals["channel_symbols"] / total_values
        ),
        "source_ber_after_ldpc": (
            totals["bit_errors"] / totals["payload_bits"]
        ),
        "segment_source_bits": kind_bits,
        "segment_bit_errors": kind_errors,
        "images_with_source_errors": images_with_errors,
        "per_scale": per_scale,
        "per_image": per_image,
    }


__all__ = [
    "AdaptiveSegment",
    "IndependentAdaptiveSample",
    "PreparedAdaptivePacket",
    "bits_per_index",
    "collect_independent_adaptive_samples",
    "combined_bits_from_segments",
    "combined_stream_lengths",
    "decode_fixed_width_rle_mask",
    "encode_fixed_width_rle_mask",
    "evaluate_packets_over_channel",
    "pack_topk_rle_segments",
    "prepare_topk_rle_packets",
    "reconstruct_from_adaptive_indices",
    "rle_length_width",
    "rounded_active_count",
    "select_per_image_topk_masks",
    "summarize_packets",
    "transmit_combined_stream",
    "unpack_topk_rle_combined",
]
