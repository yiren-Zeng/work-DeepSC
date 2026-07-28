"""Evaluation-only shared-codebook RQ depth extension helpers.

These helpers deliberately live outside the training/model implementation.
They first require a normally loaded, eval-mode EMA-RQ model and then add
additional references to each scale's already trained shared codebook.  No
checkpoint tensor is created, copied, or updated.
"""

from collections.abc import Sequence

import numpy as np
import torch

from models.rq_ema_quantizer import RQEMAQuantizer
from utils.bit_utils import count_index_bits
from utils.metrics import calculate_ms_ssim


def _normalize_depth_list(depths, num_scales, name):
    if isinstance(depths, int):
        values = [int(depths)] * int(num_scales)
    elif isinstance(depths, Sequence) and not isinstance(depths, (str, bytes)):
        values = [int(value) for value in depths]
    else:
        raise TypeError(f"{name} must be an integer or one integer per scale")
    if len(values) != int(num_scales):
        raise ValueError(
            f"{name} length ({len(values)}) must equal the scale count "
            f"({num_scales})"
        )
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} entries must be positive")
    return values


def _require_eval_shared_rq(model):
    if model.training:
        raise RuntimeError("RQ depth extension is eval-only; call model.eval() first")
    if getattr(model, "quantizer_type", None) != "rq_ema":
        raise ValueError("RQ depth extension requires quantizer_type='rq_ema'")

    quantizers = list(getattr(model, "vector_quantizers", []))
    if not quantizers:
        raise ValueError("model has no vector quantizers")
    for scale, quantizer in enumerate(quantizers):
        if not isinstance(quantizer, RQEMAQuantizer):
            raise TypeError(
                f"scale {scale} is not an RQEMAQuantizer: "
                f"{type(quantizer).__name__}"
            )
        if not quantizer.shared_codebook:
            raise ValueError(f"scale {scale} does not use a shared RQ codebook")
        if not quantizer.codebooks:
            raise ValueError(f"scale {scale} has no codebook")
        first = quantizer.codebook
        if any(codebook is not first for codebook in quantizer.codebooks):
            raise ValueError(
                f"scale {scale} contains independent codebooks; zero-training "
                "reference extension is only defined for a shared codebook"
            )
    return quantizers


def extend_shared_rq_depth_for_eval(model, target_depths):
    """Extend a loaded EMA-RQ model by reusing each scale's codebook object.

    The checkpoint must be loaded normally before this function is called.
    ``target_depths`` can be a scalar shared by all scales or one value per
    scale.  Targets may not be smaller than the currently active depth; use
    :func:`set_shared_rq_depth_for_eval` to select a shorter prefix later.
    """

    quantizers = _require_eval_shared_rq(model)
    loaded_depths = [int(quantizer.rq_depth) for quantizer in quantizers]
    targets = _normalize_depth_list(
        target_depths, len(quantizers), "target_depths"
    )
    for scale, (quantizer, loaded, target) in enumerate(
        zip(quantizers, loaded_depths, targets)
    ):
        if len(quantizer.codebooks) != loaded:
            raise ValueError(
                f"scale {scale} has active depth {loaded}, but its prepared "
                f"codebook-reference count is {len(quantizer.codebooks)}"
            )
        if target < loaded:
            raise ValueError(
                f"scale {scale} target depth {target} is smaller than loaded "
                f"depth {loaded}"
            )

    parameter_ids_before = {id(parameter) for parameter in model.parameters()}
    buffer_ids_before = {id(buffer) for buffer in model.buffers()}
    added_references = []
    for quantizer, target in zip(quantizers, targets):
        first = quantizer.codebook
        added = max(0, target - len(quantizer.codebooks))
        for _ in range(added):
            quantizer.codebooks.append(first)
        quantizer.rq_depth = target
        added_references.append(added)
    model.rq_depth_list = list(targets)
    if {id(parameter) for parameter in model.parameters()} != parameter_ids_before:
        raise RuntimeError("depth extension unexpectedly changed model parameters")
    if {id(buffer) for buffer in model.buffers()} != buffer_ids_before:
        raise RuntimeError("depth extension unexpectedly changed model buffers")

    return {
        "loaded_rq_depth_list": loaded_depths,
        "runtime_rq_depth_list": list(targets),
        "added_codebook_references_per_scale": added_references,
        "shared_object_per_scale": [
            all(codebook is quantizer.codebook for codebook in quantizer.codebooks)
            for quantizer in quantizers
        ],
    }


def set_shared_rq_depth_for_eval(model, active_depths):
    """Select an already available shared-codebook prefix for fixed inference."""

    quantizers = _require_eval_shared_rq(model)
    depths = _normalize_depth_list(
        active_depths, len(quantizers), "active_depths"
    )
    for scale, (quantizer, depth) in enumerate(zip(quantizers, depths)):
        if depth > len(quantizer.codebooks):
            raise ValueError(
                f"scale {scale} depth {depth} exceeds the prepared maximum "
                f"{len(quantizer.codebooks)}"
            )
    for quantizer, depth in zip(quantizers, depths):
        quantizer.rq_depth = depth
    model.rq_depth_list = list(depths)
    return list(depths)


def truncate_rq_indices(indices_list, active_depths):
    """Return prefix views of fixed RQ index tensors without changing values."""

    depths = _normalize_depth_list(
        active_depths, len(indices_list), "active_depths"
    )
    truncated = []
    for scale, (indices, depth) in enumerate(zip(indices_list, depths)):
        if not isinstance(indices, torch.Tensor):
            raise TypeError(f"indices_list[{scale}] must be a torch.Tensor")
        if indices.ndim != 4:
            raise ValueError(
                f"indices_list[{scale}] must have shape [B, H, W, D]"
            )
        if depth > indices.shape[-1]:
            raise ValueError(
                f"scale {scale} requested depth {depth}, but indices only have "
                f"depth {indices.shape[-1]}"
            )
        truncated.append(indices[..., :depth])
    return truncated


def _image_quality(real_image, reconstructed_image):
    if real_image.device != reconstructed_image.device:
        real_image = real_image.to(
            reconstructed_image.device, non_blocking=True
        )
    reference = (real_image + 1.0) / 2.0
    reconstruction = (reconstructed_image + 1.0) / 2.0
    ms_ssim = calculate_ms_ssim(reference, reconstruction)
    mse = torch.mean((reference - reconstruction) ** 2)
    psnr = 100.0 if float(mse.item()) == 0.0 else float(
        10.0 * torch.log10(1.0 / mse).item()
    )
    return float(ms_ssim), psnr


@torch.no_grad()
def evaluate_no_channel_depth_sweep(
    model,
    loader,
    device,
    depths=(1, 2, 3, 4),
    num_embeddings_list=None,
    ldpc_rate=0.5,
    modulation_bits=1,
    image_channels=3,
):
    """Evaluate fixed shared-codebook prefixes from one maximum-depth encode.

    Encoding once at the maximum depth guarantees that all shorter cases use
    exactly the corresponding prefix indices.  The model remains at the
    maximum requested depth when this function returns.
    """

    quantizers = _require_eval_shared_rq(model)
    requested_depths = sorted({int(depth) for depth in depths})
    if not requested_depths or requested_depths[0] <= 0:
        raise ValueError("depths must contain positive integers")
    max_depth = requested_depths[-1]
    if num_embeddings_list is None:
        num_embeddings_list = [
            int(quantizer.num_embeddings) for quantizer in quantizers
        ]
    num_embeddings_list = [int(value) for value in num_embeddings_list]
    if len(num_embeddings_list) != len(quantizers):
        raise ValueError("num_embeddings_list length must equal the scale count")

    set_shared_rq_depth_for_eval(model, max_depth)
    quality = {
        depth: {"psnr": [], "ms_ssim": [], "source_bits": []}
        for depth in requested_depths
    }
    residual_norm_sq_sums = [
        np.zeros(max_depth, dtype=np.float64) for _ in quantizers
    ]
    sample_count = 0
    total_image_pixels = 0

    for real_image in loader:
        if int(real_image.shape[0]) != 1:
            raise ValueError("depth sweep evaluation requires batch_size=1")
        real_image = real_image.to(device)
        set_shared_rq_depth_for_eval(model, max_depth)
        encoded = model.forward_test(real_image)
        full_indices = encoded["indices"]

        for scale, quantizer in enumerate(quantizers):
            residual_norms = (
                quantizer.get_last_diagnostics()["residual_norm_per_depth"]
                .detach()
                .to(device="cpu", dtype=torch.float64)
                .numpy()
            )
            if residual_norms.shape != (max_depth,):
                raise RuntimeError(
                    f"scale {scale} returned residual diagnostics with shape "
                    f"{residual_norms.shape}, expected {(max_depth,)}"
                )
            residual_norm_sq_sums[scale] += residual_norms ** 2

        image_pixels = int(real_image.shape[-2] * real_image.shape[-1])
        total_image_pixels += image_pixels
        sample_count += 1

        for depth in requested_depths:
            active_depths = [depth] * len(quantizers)
            prefix_indices = truncate_rq_indices(full_indices, active_depths)
            set_shared_rq_depth_for_eval(model, active_depths)
            reconstruction = model.reconstruct_from_indices(
                prefix_indices, feature_shapes=encoded.get("feature_shapes")
            )
            ms_ssim, psnr = _image_quality(real_image, reconstruction)
            bit_stats = count_index_bits(
                prefix_indices, num_embeddings_list
            )
            quality[depth]["ms_ssim"].append(ms_ssim)
            quality[depth]["psnr"].append(psnr)
            quality[depth]["source_bits"].append(bit_stats["total_bits"])

    set_shared_rq_depth_for_eval(model, max_depth)
    if sample_count == 0:
        raise ValueError("evaluation loader produced no images")

    residual_rms_by_scale = [
        np.sqrt(values / sample_count).tolist()
        for values in residual_norm_sq_sums
    ]
    results = []
    for depth in requested_depths:
        source_bits_values = quality[depth]["source_bits"]
        if len(set(source_bits_values)) != 1:
            raise RuntimeError(
                f"source bit count changed across images at depth {depth}: "
                f"{sorted(set(source_bits_values))}"
            )
        source_bits_per_image = int(source_bits_values[0])
        source_bpp = float(
            sum(source_bits_values) / total_image_pixels
        )
        transmission_ratio = float(
            source_bpp
            / (float(ldpc_rate) * int(modulation_bits) * int(image_channels))
        )
        results.append(
            {
                "depth": depth,
                "rq_depth_list": [depth] * len(quantizers),
                "num_images": sample_count,
                "source_bits_per_image": source_bits_per_image,
                "source_bpp": source_bpp,
                "ldpc_bpsk_rgb_transmission_ratio": transmission_ratio,
                "psnr": float(np.mean(quality[depth]["psnr"])),
                "ms_ssim": float(np.mean(quality[depth]["ms_ssim"])),
                "final_residual_rms_per_scale": [
                    float(scale_values[depth - 1])
                    for scale_values in residual_rms_by_scale
                ],
            }
        )

    return {
        "depth_results": results,
        "residual_rms_by_scale": residual_rms_by_scale,
        "num_images": sample_count,
        "total_image_pixels": total_image_pixels,
        "max_runtime_rq_depth_list": [max_depth] * len(quantizers),
    }


__all__ = [
    "evaluate_no_channel_depth_sweep",
    "extend_shared_rq_depth_for_eval",
    "set_shared_rq_depth_for_eval",
    "truncate_rq_indices",
]
