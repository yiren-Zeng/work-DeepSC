"""Evaluation-only depth sweeps for a trained shared-codebook Residual-SimVQ.

Residual-SimVQ registers one projected codebook per scale and recursively
reuses that same object at every residual depth.  These helpers only change
the integer ``rq_depth`` used by fixed inference; they never add parameters,
buffers, state-dict keys, or checkpoint files.
"""

from collections.abc import Sequence

import numpy as np
import torch

from models.residual_simvq_quantizer import ResidualSimVQQuantizer
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


def _require_eval_residual_simvq(model):
    if model.training:
        raise RuntimeError(
            "Residual-SimVQ depth extension is eval-only; call model.eval() first"
        )
    if getattr(model, "quantizer_type", None) != "residual_simvq":
        raise ValueError(
            "Residual-SimVQ depth extension requires "
            "quantizer_type='residual_simvq'"
        )

    quantizers = list(getattr(model, "vector_quantizers", []))
    if not quantizers:
        raise ValueError("model has no vector quantizers")
    for scale, quantizer in enumerate(quantizers):
        if not isinstance(quantizer, ResidualSimVQQuantizer):
            raise TypeError(
                f"scale {scale} is not a ResidualSimVQQuantizer: "
                f"{type(quantizer).__name__}"
            )
        if not quantizer.shared_codebook:
            raise ValueError(f"scale {scale} does not use a shared codebook")
        if any(
            codebook is not quantizer.codebook
            for codebook in quantizer.codebooks
        ):
            raise ValueError(f"scale {scale} does not reuse one codebook object")
    return quantizers


def extend_residual_simvq_depth_for_eval(model, target_depths):
    """Select a larger logical depth without registering any new state."""

    quantizers = _require_eval_residual_simvq(model)
    loaded_depths = [int(quantizer.rq_depth) for quantizer in quantizers]
    targets = _normalize_depth_list(
        target_depths, len(quantizers), "target_depths"
    )
    for scale, (loaded, target) in enumerate(zip(loaded_depths, targets)):
        if target < loaded:
            raise ValueError(
                f"scale {scale} target depth {target} is smaller than loaded "
                f"depth {loaded}"
            )

    parameter_ids_before = {id(value) for value in model.parameters()}
    buffer_ids_before = {id(value) for value in model.buffers()}
    state_keys_before = tuple(model.state_dict())
    for quantizer, target in zip(quantizers, targets):
        quantizer.rq_depth = target
    model.rq_depth_list = list(targets)

    if {id(value) for value in model.parameters()} != parameter_ids_before:
        raise RuntimeError("depth extension unexpectedly changed model parameters")
    if {id(value) for value in model.buffers()} != buffer_ids_before:
        raise RuntimeError("depth extension unexpectedly changed model buffers")
    if tuple(model.state_dict()) != state_keys_before:
        raise RuntimeError("depth extension unexpectedly changed state-dict keys")

    return {
        "loaded_rq_depth_list": loaded_depths,
        "runtime_rq_depth_list": list(targets),
        "added_registered_parameters": 0,
        "added_registered_buffers": 0,
        "added_state_dict_keys": 0,
        "same_projected_codebook_object_per_scale": [
            all(
                codebook is quantizer.codebook
                for codebook in quantizer.codebooks
            )
            for quantizer in quantizers
        ],
    }


def set_residual_simvq_depth_for_eval(model, active_depths):
    """Select a positive fixed residual depth for every model scale."""

    quantizers = _require_eval_residual_simvq(model)
    depths = _normalize_depth_list(
        active_depths, len(quantizers), "active_depths"
    )
    for quantizer, depth in zip(quantizers, depths):
        quantizer.rq_depth = depth
    model.rq_depth_list = list(depths)
    return list(depths)


def truncate_residual_indices(indices_list, active_depths):
    """Return prefix views of BHWD Residual-SimVQ index tensors."""

    depths = _normalize_depth_list(
        active_depths, len(indices_list), "active_depths"
    )
    prefixes = []
    for scale, (indices, depth) in enumerate(zip(indices_list, depths)):
        if not isinstance(indices, torch.Tensor):
            raise TypeError(f"indices_list[{scale}] must be a torch.Tensor")
        if indices.ndim != 4:
            raise ValueError(
                f"indices_list[{scale}] must have shape [B, H, W, D]"
            )
        if depth > int(indices.shape[-1]):
            raise ValueError(
                f"scale {scale} requested depth {depth}, but indices only have "
                f"depth {indices.shape[-1]}"
            )
        prefixes.append(indices[..., :depth])
    return prefixes


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
def evaluate_residual_simvq_no_channel_depth_sweep(
    model,
    loader,
    device,
    depths=(1, 2, 3, 4),
    num_embeddings_list=None,
    ldpc_rate=0.5,
    modulation_bits=1,
    image_channels=3,
):
    """Evaluate depth prefixes from one maximum-depth encoding per image."""

    quantizers = _require_eval_residual_simvq(model)
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

    set_residual_simvq_depth_for_eval(model, max_depth)
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
        set_residual_simvq_depth_for_eval(model, max_depth)
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

        total_image_pixels += int(
            real_image.shape[-2] * real_image.shape[-1]
        )
        sample_count += 1

        for depth in requested_depths:
            active_depths = [depth] * len(quantizers)
            prefix_indices = truncate_residual_indices(
                full_indices, active_depths
            )
            set_residual_simvq_depth_for_eval(model, active_depths)
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

    set_residual_simvq_depth_for_eval(model, max_depth)
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
        source_bpp = float(sum(source_bits_values) / total_image_pixels)
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
                    float(values[depth - 1])
                    for values in residual_rms_by_scale
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
    "evaluate_residual_simvq_no_channel_depth_sweep",
    "extend_residual_simvq_depth_for_eval",
    "set_residual_simvq_depth_for_eval",
    "truncate_residual_indices",
]
