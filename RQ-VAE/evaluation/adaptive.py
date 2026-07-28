"""Independent adaptive second-stage RQ evaluation helpers.

This module does not alter the legacy evaluation path. It consumes the
adaptive model contract exposed by DeepSC and evaluates rate/distortion by
turning the second residual-quantizer depth on only where the first-stage
quantization error exceeds a scale-specific threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch

from utils.bit_utils import AdaptiveRQBitAccumulator
from utils.metrics import calculate_ms_ssim


@dataclass
class AdaptiveSample:
    """CPU cache for one encoded image batch."""

    image: torch.Tensor
    dense_indices: List[torch.Tensor]
    first_stage_errors: List[torch.Tensor]
    feature_shapes: List[tuple]


def _require_adaptive_contract(model):
    missing = [
        name
        for name in (
            "forward_test_adaptive",
            "reconstruct_from_adaptive_indices",
        )
        if not callable(getattr(model, name, None))
    ]
    if missing:
        raise RuntimeError(
            "The checkpoint model does not expose the adaptive RQ contract: "
            + ", ".join(missing)
            + ". Expected forward_test_adaptive(x, thresholds) and "
            "reconstruct_from_adaptive_indices(indices, feature_shapes)."
        )


def _validate_dense_output(output, num_scales):
    required = {"indices", "stop_masks", "first_stage_errors", "feature_shapes"}
    missing = sorted(required.difference(output))
    if missing:
        raise RuntimeError(f"adaptive model output is missing keys: {missing}")
    for name in required:
        if len(output[name]) != num_scales:
            raise RuntimeError(
                f"adaptive output {name!r} has {len(output[name])} scales; "
                f"expected {num_scales}"
            )
    for scale, (indices, errors) in enumerate(
        zip(output["indices"], output["first_stage_errors"])
    ):
        if indices.ndim != 4 or indices.shape[-1] != 2:
            raise RuntimeError(
                f"adaptive indices at scale {scale} must be [B,H,W,2], "
                f"got {tuple(indices.shape)}"
            )
        if errors.ndim == 4 and errors.shape[-1] == 1:
            errors = errors.squeeze(-1)
            output["first_stage_errors"][scale] = errors
        if errors.ndim != 3 or tuple(errors.shape) != tuple(indices.shape[:3]):
            raise RuntimeError(
                f"first_stage_errors at scale {scale} must be [B,H,W] matching "
                f"indices; got {tuple(errors.shape)} vs {tuple(indices.shape)}"
            )


@torch.no_grad()
def collect_dense_adaptive_samples(model, loader, device, max_images=None):
    """Encode once with every refinement active and cache tensors on CPU.

    Passing negative-infinity thresholds makes the model return valid
    second-depth indices everywhere while still exposing the first-stage
    errors used for later threshold and quantile scans.
    """
    _require_adaptive_contract(model)
    if getattr(model, "quantizer_type", None) != "rq_ema":
        raise ValueError("adaptive refinement evaluation requires quantizer_type='rq_ema'")
    depths = list(getattr(model, "rq_depth_list", []))
    if not depths or any(int(depth) != 2 for depth in depths):
        raise ValueError(
            "adaptive STOP evaluation currently requires RQ depth exactly 2 at every scale"
        )

    model.eval()
    num_scales = len(depths)
    dense_thresholds = [float("-inf")] * num_scales
    samples = []
    images_seen = 0

    for image in loader:
        if max_images is not None and images_seen >= int(max_images):
            break
        if max_images is not None:
            remaining = int(max_images) - images_seen
            image = image[:remaining]
        input_image = image.to(device, non_blocking=True)
        output = model.forward_test_adaptive(input_image, dense_thresholds)
        _validate_dense_output(output, num_scales)
        if any(bool((indices[..., 1] < 0).any()) for indices in output["indices"]):
            raise RuntimeError(
                "forward_test_adaptive(..., thresholds=-inf) returned STOP symbols; "
                "the adaptive threshold contract must activate all finite-error refinements"
            )
        samples.append(
            AdaptiveSample(
                image=image.detach().cpu(),
                dense_indices=[indices.detach().cpu() for indices in output["indices"]],
                first_stage_errors=[
                    errors.detach().float().cpu()
                    for errors in output["first_stage_errors"]
                ],
                feature_shapes=[tuple(shape) for shape in output["feature_shapes"]],
            )
        )
        images_seen += int(image.shape[0])

    if not samples:
        raise ValueError("adaptive evaluation loader produced no images")
    return samples


def pooled_first_stage_errors(samples):
    """Return one flattened finite error tensor per scale."""
    if not samples:
        raise ValueError("samples must not be empty")
    num_scales = len(samples[0].first_stage_errors)
    pooled = []
    for scale in range(num_scales):
        values = torch.cat(
            [sample.first_stage_errors[scale].reshape(-1) for sample in samples]
        )
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"first-stage errors at scale {scale} contain NaN/Inf")
        pooled.append(values)
    return pooled


def quantile_thresholds(error_values, target_active_rates):
    """Calibrate global thresholds for error >= threshold activation."""
    if len(error_values) != len(target_active_rates):
        raise ValueError("target active-rate count must match number of scales")
    thresholds = []
    for scale, (values, target) in enumerate(zip(error_values, target_active_rates)):
        target = float(target)
        if not 0.0 <= target <= 1.0:
            raise ValueError(f"target active rate at scale {scale} must be in [0,1]")
        values = torch.as_tensor(values).detach().float().reshape(-1)
        if values.numel() == 0:
            raise ValueError(f"no calibration errors for scale {scale}")
        if target == 0.0:
            maximum = values.max()
            positive_infinity = torch.full_like(maximum, float("inf"))
            # Keep the nextafter operation in the error tensor's dtype.  A
            # Python float successor would round back to the float32 maximum
            # when compared by torch, incorrectly leaving max-error tokens on.
            threshold = float(torch.nextafter(maximum, positive_infinity).item())
        elif target == 1.0:
            threshold = float(values.min().item())
        else:
            threshold = float(torch.quantile(values, 1.0 - target).item())
        thresholds.append(threshold)
    return thresholds


def apply_adaptive_thresholds(dense_indices, first_stage_errors, thresholds):
    """Insert STOP=-1 wherever first_stage_error < threshold."""
    if not (
        len(dense_indices) == len(first_stage_errors) == len(thresholds)
    ):
        raise ValueError("indices, errors, and thresholds must have equal scale counts")
    adaptive_indices = []
    stop_masks = []
    for scale, (indices, errors, threshold) in enumerate(
        zip(dense_indices, first_stage_errors, thresholds)
    ):
        if indices.ndim != 4 or indices.shape[-1] != 2:
            raise ValueError(f"scale {scale} dense indices must be [B,H,W,2]")
        if tuple(errors.shape) != tuple(indices.shape[:3]):
            raise ValueError(f"scale {scale} error grid does not match its indices")
        active_mask = errors >= float(threshold)
        stop_mask = ~active_mask
        adaptive = indices.clone()
        adaptive[..., 1] = torch.where(
            active_mask,
            adaptive[..., 1],
            torch.full_like(adaptive[..., 1], -1),
        )
        adaptive_indices.append(adaptive)
        stop_masks.append(stop_mask)
    return adaptive_indices, stop_masks


def _quality_metrics(real_image, reconstructed_image):
    if real_image.device != reconstructed_image.device:
        real_image = real_image.to(reconstructed_image.device, non_blocking=True)
    reference = (real_image + 1.0) / 2.0
    reconstruction = (reconstructed_image + 1.0) / 2.0
    score = float(calculate_ms_ssim(reference, reconstruction))
    mse = float(torch.mean((reference - reconstruction) ** 2).item())
    psnr = 100.0 if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))
    return score, psnr


@torch.no_grad()
def evaluate_adaptive_point(
    model,
    samples: Sequence[AdaptiveSample],
    thresholds: Sequence[float],
    num_embeddings_list: Sequence[int],
):
    """Evaluate one scale-specific threshold vector."""
    _require_adaptive_contract(model)
    accumulator = AdaptiveRQBitAccumulator(num_embeddings_list)
    ms_ssim_scores = []
    psnr_scores = []
    total_pixels = 0

    for sample in samples:
        adaptive_indices, _ = apply_adaptive_thresholds(
            sample.dense_indices, sample.first_stage_errors, thresholds
        )
        reconstructed = model.reconstruct_from_adaptive_indices(
            adaptive_indices, feature_shapes=sample.feature_shapes
        )
        ms_ssim, psnr = _quality_metrics(sample.image, reconstructed)
        ms_ssim_scores.append(ms_ssim)
        psnr_scores.append(psnr)
        accumulator.update(adaptive_indices)
        total_pixels += int(
            sample.image.shape[0] * sample.image.shape[-2] * sample.image.shape[-1]
        )

    rate = accumulator.summary(total_pixels)
    return {
        "thresholds": [float(value) for value in thresholds],
        "psnr": float(np.mean(psnr_scores)),
        "ms_ssim": float(np.mean(ms_ssim_scores)),
        "rate": rate,
    }


def parse_scale_pairs(values: Optional[Iterable[str]], num_scales: int, name: str):
    """Parse CLI values such as '0.1,0.2' into scale vectors."""
    parsed = []
    for raw in values or []:
        parts = [part.strip() for part in str(raw).split(",") if part.strip()]
        if len(parts) != num_scales:
            raise ValueError(
                f"{name} entry {raw!r} has {len(parts)} values; expected {num_scales}"
            )
        parsed.append([float(part) for part in parts])
    return parsed


def build_scan_points(
    error_values,
    threshold_pairs=None,
    target_active_rate_pairs=None,
    common_target_active_rates=None,
):
    """Build direct-threshold and dataset-quantile scan specifications."""
    num_scales = len(error_values)
    points = []
    for thresholds in threshold_pairs or []:
        if len(thresholds) != num_scales:
            raise ValueError("threshold vector length must match scale count")
        points.append(
            {
                "source": "direct_threshold",
                "thresholds": [float(value) for value in thresholds],
                "target_active_rates": None,
            }
        )

    target_pairs = list(target_active_rate_pairs or [])
    for target in common_target_active_rates or []:
        target_pairs.append([float(target)] * num_scales)
    for targets in target_pairs:
        thresholds = quantile_thresholds(error_values, targets)
        points.append(
            {
                "source": "global_error_quantile",
                "thresholds": thresholds,
                "target_active_rates": [float(value) for value in targets],
            }
        )
    if not points:
        raise ValueError("no adaptive threshold or target-active-rate scan points requested")
    for index, point in enumerate(points):
        point["scan_id"] = index
    return points


__all__ = [
    "AdaptiveSample",
    "apply_adaptive_thresholds",
    "build_scan_points",
    "collect_dense_adaptive_samples",
    "evaluate_adaptive_point",
    "parse_scale_pairs",
    "pooled_first_stage_errors",
    "quantile_thresholds",
]
