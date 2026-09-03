"""Stateless stage-wise residual quantization for independent RAQ codebooks."""

import torch
import torch.nn.functional as F


def quantize_independent_raq_rvq(
    source_quantizer,
    inputs: torch.Tensor,
    embed_weights,
):
    """Quantize successive residuals with one independent codebook per stage.

    The loss and gradient contract intentionally matches Residual-SimVQ:
    nearest-neighbour targets are detached, codebook losses supervise the
    current residual, commitment losses supervise the cumulative
    reconstruction, and exactly one straight-through bridge is applied after
    all stages have been summed.
    """
    embed_weights = list(embed_weights)
    if not embed_weights:
        raise ValueError("independent RAQ-RVQ requires at least one codebook")
    if inputs.ndim != 4:
        raise ValueError(f"expected NCHW inputs, got shape {tuple(inputs.shape)}")

    inputs_bhwc = inputs.permute(0, 2, 3, 1).contiguous()
    batch, height, width, channels = inputs_bhwc.shape
    residual = inputs_bhwc.detach().clone()
    cumulative_quant = torch.zeros_like(inputs_bhwc)
    codebook_losses = []
    commitment_losses = []
    residual_mse_per_depth = []
    indices_per_depth = []

    for stage_index, embed_weight in enumerate(embed_weights):
        if embed_weight.ndim != 2 or embed_weight.shape[1] != channels:
            raise ValueError(
                f"stage {stage_index} codebook must have shape [K,{channels}], "
                f"got {tuple(embed_weight.shape)}"
            )
        encoding_idx = source_quantizer._nearest_code_indices(
            residual.reshape(-1, channels),
            embed_weight,
        )
        quantized = F.embedding(encoding_idx, embed_weight).view(
            batch, height, width, channels
        )
        codebook_losses.append(F.mse_loss(quantized, residual.detach()))
        cumulative_quant = cumulative_quant + quantized
        commitment_losses.append(
            F.mse_loss(inputs_bhwc, cumulative_quant.detach())
        )
        residual = residual - quantized.detach()
        residual_mse_per_depth.append(residual.pow(2).mean())
        indices_per_depth.append(encoding_idx.view(batch, height, width))

    codebook_loss_per_depth = torch.stack(codebook_losses)
    commitment_loss_per_depth = torch.stack(commitment_losses)
    codebook_loss = codebook_loss_per_depth.mean()
    commitment_loss = commitment_loss_per_depth.mean()
    vq_loss = (
        codebook_loss
        + source_quantizer.commitment_cost * commitment_loss
    )

    quantized_raw = cumulative_quant.permute(0, 3, 1, 2).contiguous()
    quantized_ste = (
        inputs_bhwc + (cumulative_quant - inputs_bhwc).detach()
    ).permute(0, 3, 1, 2).contiguous()
    return {
        "loss": vq_loss,
        "quantized": quantized_ste,
        "indices": indices_per_depth,
        "quantized_raw": quantized_raw,
        "residual_mse_per_depth": torch.stack(residual_mse_per_depth),
    }
