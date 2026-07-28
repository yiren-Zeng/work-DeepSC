"""Per-image, per-scale exact Top-K selection for adaptive EMA-RQ.

This module is deliberately separate from the existing dataset-global
quantile path.  For every image and scale it selects an exact number of
second-stage tokens, breaking equal-error ties by raster order.  The cutoff
error is retained as diagnostic metadata; the explicit activity mask remains
the authoritative transport representation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch

from evaluation.adaptive import AdaptiveSample, _quality_metrics
from evaluation.adaptive_channel import PreparedAdaptivePacket
from utils.adaptive_transport import pack_explicit_mask_segments


def rounded_active_count(token_count: int, target_active_rate: float) -> int:
    """Return ``floor(rate * N + 0.5)`` clamped to ``[0, N]``."""

    token_count = int(token_count)
    target_active_rate = float(target_active_rate)
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if not 0.0 <= target_active_rate <= 1.0:
        raise ValueError("target_active_rate must be in [0, 1]")
    active_count = int(math.floor(target_active_rate * token_count + 0.5))
    return min(token_count, max(0, active_count))


def _select_one_error_grid(
    errors: torch.Tensor,
    target_active_rate: float,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Select exact Top-K errors with stable raster-order tie breaking."""

    errors = torch.as_tensor(errors).detach().float()
    if errors.ndim != 2:
        raise ValueError(
            f"one image's error grid must be [H,W], got {tuple(errors.shape)}"
        )
    flat_errors = errors.reshape(-1)
    if flat_errors.numel() == 0:
        raise ValueError("error grid must not be empty")
    if not bool(torch.isfinite(flat_errors).all()):
        raise ValueError("first-stage errors must be finite")

    token_count = int(flat_errors.numel())
    active_count = rounded_active_count(token_count, target_active_rate)
    active_flat = torch.zeros(
        token_count, dtype=torch.bool, device=flat_errors.device
    )

    if active_count == 0:
        maximum = flat_errors.max()
        cutoff = torch.nextafter(
            maximum, torch.full_like(maximum, float("inf"))
        )
        strict_above_count = 0
        cutoff_equal_count = 0
        cutoff_equal_selected_count = 0
        first_inactive_error = float(maximum.item())
        cutoff_splits_tie = False
    else:
        # stable=True keeps the original flattened/raster order for exact ties.
        order = torch.argsort(
            flat_errors, descending=True, stable=True
        )
        selected = order[:active_count]
        active_flat[selected] = True
        cutoff = flat_errors[selected[-1]]
        equal_cutoff = flat_errors == cutoff
        strict_above_count = int((flat_errors > cutoff).sum().item())
        cutoff_equal_count = int(equal_cutoff.sum().item())
        cutoff_equal_selected_count = int(
            (equal_cutoff & active_flat).sum().item()
        )
        first_inactive_error = (
            None
            if active_count == token_count
            else float(flat_errors[order[active_count]].item())
        )
        cutoff_splits_tie = bool(
            cutoff_equal_selected_count < cutoff_equal_count
        )

    metadata: Dict[str, object] = {
        "threshold": float(cutoff.item()),
        "target_active_rate": float(target_active_rate),
        "token_count": token_count,
        "target_active_count": active_count,
        "active_count": int(active_flat.sum().item()),
        "actual_active_rate": float(active_count / token_count),
        "strict_above_threshold_count": strict_above_count,
        "threshold_equal_count": cutoff_equal_count,
        "threshold_equal_selected_count": cutoff_equal_selected_count,
        "threshold_splits_tie": cutoff_splits_tie,
        "first_inactive_error": first_inactive_error,
    }
    return active_flat.reshape_as(errors), metadata


def select_per_image_topk_masks(
    first_stage_errors: Sequence[torch.Tensor],
    target_active_rates: Sequence[float],
) -> Tuple[List[torch.Tensor], List[Dict[str, object]]]:
    """Select one exact Top-K activity mask per image and scale.

    Parameters
    ----------
    first_stage_errors:
        One ``[B,H,W]`` tensor per scale.
    target_active_rates:
        One requested second-stage activity rate per scale.

    Returns
    -------
    masks:
        Boolean ``[B,H,W]`` tensors where ``True`` means depth two is sent.
    per_image:
        JSON-serializable diagnostic records.  Each image has one cutoff
        threshold and count record per scale.
    """

    if len(first_stage_errors) != len(target_active_rates):
        raise ValueError(
            "error tensor and target active-rate counts must match"
        )
    if not first_stage_errors:
        raise ValueError("at least one scale is required")

    normalized_errors: List[torch.Tensor] = []
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
            raise ValueError(
                f"first-stage errors at scale {scale} contain NaN/Inf"
            )
        normalized_errors.append(errors)

    masks = [
        torch.zeros_like(errors, dtype=torch.bool)
        for errors in normalized_errors
    ]
    per_image: List[Dict[str, object]] = [
        {"batch_index": image_index, "scales": []}
        for image_index in range(int(batch_size))
    ]

    for scale, (errors, target_rate) in enumerate(
        zip(normalized_errors, target_active_rates)
    ):
        for image_index in range(int(batch_size)):
            mask, metadata = _select_one_error_grid(
                errors[image_index], float(target_rate)
            )
            masks[scale][image_index] = mask
            per_image[image_index]["scales"].append(
                {"scale": scale, **metadata}
            )

    return masks, per_image


def apply_adaptive_topk(
    dense_indices: Sequence[torch.Tensor],
    first_stage_errors: Sequence[torch.Tensor],
    target_active_rates: Sequence[float],
) -> Tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    List[Dict[str, object]],
]:
    """Apply per-image Top-K masks while leaving all first indices intact."""

    if len(dense_indices) != len(first_stage_errors):
        raise ValueError("dense indices and error scale counts must match")
    active_masks, per_image = select_per_image_topk_masks(
        first_stage_errors, target_active_rates
    )
    adaptive_indices: List[torch.Tensor] = []
    stop_masks: List[torch.Tensor] = []

    for scale, (indices, errors, active_mask) in enumerate(
        zip(dense_indices, first_stage_errors, active_masks)
    ):
        if indices.ndim != 4 or indices.shape[-1] != 2:
            raise ValueError(
                f"scale {scale} dense indices must be [B,H,W,2]"
            )
        if tuple(indices.shape[:3]) != tuple(errors.shape):
            raise ValueError(
                f"scale {scale} error grid does not match dense indices"
            )
        if tuple(active_mask.shape) != tuple(errors.shape):
            raise RuntimeError("internal Top-K mask shape mismatch")
        adaptive = indices.clone()
        adaptive[..., 1] = torch.where(
            active_mask,
            indices[..., 1],
            torch.full_like(indices[..., 1], -1),
        )
        adaptive_indices.append(adaptive)
        stop_masks.append(~active_mask)

    return adaptive_indices, stop_masks, per_image


@torch.no_grad()
def prepare_per_image_topk_packets(
    model,
    samples: Sequence[AdaptiveSample],
    target_active_rates: Sequence[float],
    num_embeddings_list: Sequence[int],
    image_names: Sequence[str] | None = None,
) -> Tuple[List[PreparedAdaptivePacket], List[Dict[str, object]]]:
    """Prepare explicit-mask packets and per-image threshold records."""

    if not samples:
        raise ValueError("samples must not be empty")
    if image_names is not None and len(image_names) != len(samples):
        raise ValueError("image_names must match the number of samples")

    packets: List[PreparedAdaptivePacket] = []
    selection_records: List[Dict[str, object]] = []
    for image_index, sample in enumerate(samples):
        if int(sample.image.shape[0]) != 1:
            raise ValueError(
                "per-image Top-K packet preparation requires batch_size=1"
            )
        tx_indices, _, batch_records = apply_adaptive_topk(
            sample.dense_indices,
            sample.first_stage_errors,
            target_active_rates,
        )
        if len(batch_records) != 1:
            raise RuntimeError("expected exactly one Top-K image record")

        image_record = {
            "image_index": image_index,
            "image_number": image_index + 1,
            "image_name": (
                str(image_names[image_index])
                if image_names is not None
                else f"image_{image_index + 1:04d}"
            ),
            "scales": batch_records[0]["scales"],
        }
        segments, metadata = pack_explicit_mask_segments(
            tx_indices, num_embeddings_list
        )
        reconstructed = model.reconstruct_from_adaptive_indices(
            tx_indices, feature_shapes=sample.feature_shapes
        )
        clean_ms_ssim, clean_psnr = _quality_metrics(
            sample.image, reconstructed
        )
        packets.append(
            PreparedAdaptivePacket(
                image=sample.image,
                feature_shapes=list(sample.feature_shapes),
                tx_indices=tx_indices,
                segments=segments,
                metadata=metadata,
                clean_psnr=float(clean_psnr),
                clean_ms_ssim=float(clean_ms_ssim),
            )
        )
        selection_records.append(image_record)

    return packets, selection_records


__all__ = [
    "apply_adaptive_topk",
    "prepare_per_image_topk_packets",
    "rounded_active_count",
    "select_per_image_topk_masks",
]
