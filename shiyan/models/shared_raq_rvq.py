"""Stateless shared-codebook residual quantization for RAQ-generated codebooks."""

import torch
import torch.nn.functional as F


def quantize_shared_raq_rvq(
    source_quantizer,
    inputs: torch.Tensor,
    embed_weight: torch.Tensor,
    depth: int = 2,
):
    """Quantize ``inputs`` repeatedly with one externally generated codebook.

    The function deliberately owns no parameters or buffers.  ``embed_weight``
    is generated once by the scale's RAQ module and reused at every residual
    depth.  Residual targets and nearest-neighbour decisions are detached, while
    selected codewords retain their gradient path to the RAQ generator.

    A single straight-through bridge is applied to the final accumulated
    quantization.  Per-depth codebook and commitment losses are averaged, which
    matches the shared-codebook Residual-SimVQ training contract.
    """
    depth = int(depth)
    if depth < 1:
        raise ValueError("shared RAQ-RVQ depth must be positive")
    if inputs.ndim != 4:
        raise ValueError(f"expected NCHW inputs, got shape {tuple(inputs.shape)}")

    inputs_bhwc = inputs.permute(0, 2, 3, 1).contiguous()
    batch, height, width, channels = inputs_bhwc.shape
    if embed_weight.ndim != 2 or embed_weight.shape[1] != channels:
        raise ValueError(
            "dynamic codebook must have shape [K,C] matching the input channels; "
            f"got {tuple(embed_weight.shape)} for C={channels}"
        )

    residual = inputs_bhwc.detach().clone()
    cumulative_quant = torch.zeros_like(inputs_bhwc)
    codebook_losses = []
    commitment_losses = []
    residual_mse_per_depth = []
    indices_per_depth = []

    for _ in range(depth):
        flat_residual = residual.reshape(-1, channels)
        encoding_idx = source_quantizer._nearest_code_indices(
            flat_residual, embed_weight
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

    encoding_indices = torch.stack(indices_per_depth, dim=-1)
    quantized_raw = cumulative_quant.permute(0, 3, 1, 2).contiguous()
    quantized_ste = (
        inputs_bhwc + (cumulative_quant - inputs_bhwc).detach()
    ).permute(0, 3, 1, 2).contiguous()

    return {
        "loss": vq_loss,
        "quantized": quantized_ste,
        "indices": encoding_indices,
        "quantized_raw": quantized_raw,
        "codebook_loss": codebook_loss,
        "commitment_loss": commitment_loss,
        "codebook_loss_per_depth": codebook_loss_per_depth,
        "commitment_loss_per_depth": commitment_loss_per_depth,
        "residual_mse_per_depth": torch.stack(residual_mse_per_depth),
    }
