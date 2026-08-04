"""Shared sample collection and quality helpers for adaptive Top-K RQ."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch

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
    """Encode once with every refinement active for later Top-K selection."""
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


def _quality_metrics(real_image, reconstructed_image):
    if real_image.device != reconstructed_image.device:
        real_image = real_image.to(reconstructed_image.device, non_blocking=True)
    reference = (real_image + 1.0) / 2.0
    reconstruction = (reconstructed_image + 1.0) / 2.0
    score = float(calculate_ms_ssim(reference, reconstruction))
    mse = float(torch.mean((reference - reconstruction) ** 2).item())
    psnr = 100.0 if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))
    return score, psnr


__all__ = [
    "AdaptiveSample",
    "collect_dense_adaptive_samples",
]
