"""Shared-codebook residual SimVQ for direct NCHW feature quantization.

The implementation reuses :class:`ProjectedEmbedding` and the exact chunked
nearest-neighbour lookup of the existing :class:`VectorQuantizer`.  A scale
owns one frozen-base/projected codebook, and every residual depth recursively
uses that same object.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .vector_quantizer import VectorQuantizer


class ResidualSimVQQuantizer(VectorQuantizer):
    """Residual SimVQ with one projected codebook shared across all depths.

    ``forward`` returns ``(loss, ste_quantized, codes)`` where codes use BHWD
    layout.  The two loss components are averaged independently over depth:

    * ``Q_d = MSE(q_d, residual_{d-1}.detach())``
    * ``C_d = MSE(input, cumulative_q_d.detach())``

    The returned loss is ``mean(Q_d) + commitment_cost * mean(C_d)``.
    Reconstruction gradients pass through the final straight-through estimator
    to the encoder input, but never directly into the projected codebook.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
        rq_depth: int = 2,
        shared_codebook: bool = True,
        decay: float = 0.99,
        eps: float = 1e-5,
    ) -> None:
        if int(rq_depth) <= 0:
            raise ValueError("rq_depth must be positive")
        if not shared_codebook:
            raise ValueError(
                "ResidualSimVQQuantizer requires one codebook shared by all depths"
            )
        # ``decay`` and ``eps`` are accepted through the legacy constructor
        # surface only.  VectorQuantizer does not create EMA state from them.
        super().__init__(
            int(num_embeddings),
            int(embedding_dim),
            float(commitment_cost),
            decay=float(decay),
            eps=float(eps),
        )
        self.rq_depth = int(rq_depth)
        self.shared_codebook = True
        self.last_diagnostics: Dict[str, object] = {}

    @property
    def codebooks(self):
        """Expose the repeated shared object without duplicate registration."""
        return tuple(self.codebook for _ in range(self.rq_depth))

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        if inputs.ndim != 4:
            raise ValueError(f"expected NCHW input, got shape {tuple(inputs.shape)}")
        if inputs.shape[1] != self.embedding_dim:
            raise ValueError(
                f"expected {self.embedding_dim} channels, got {inputs.shape[1]}"
            )

    def forward(self, inputs: torch.Tensor):
        self._validate_inputs(inputs)
        inputs_bhwc = inputs.permute(0, 2, 3, 1).contiguous()
        embed_weight = self.transformed_weight()

        # Residual targets and NN decisions are deliberately detached.  Each
        # projected code receives gradients only from its own Q_d term.
        residual = inputs_bhwc.detach().clone()
        cumulative_quant = torch.zeros_like(inputs_bhwc)
        codebook_loss_per_depth = []
        commitment_loss_per_depth = []
        residual_norm_per_depth = []
        codes = []

        for _ in range(self.rq_depth):
            quant, code = self._lookup_bhwc(
                residual, embed_weight=embed_weight
            )
            codebook_loss_per_depth.append(
                F.mse_loss(quant, residual.detach())
            )
            cumulative_quant = cumulative_quant + quant
            commitment_loss_per_depth.append(
                F.mse_loss(inputs_bhwc, cumulative_quant.detach())
            )
            residual = residual - quant.detach()
            residual_norm_per_depth.append(residual.pow(2).mean().sqrt())
            codes.append(code.unsqueeze(-1))

        codebook_loss_per_depth_tensor = torch.stack(codebook_loss_per_depth)
        commitment_loss_per_depth_tensor = torch.stack(
            commitment_loss_per_depth
        )
        codebook_loss = codebook_loss_per_depth_tensor.mean()
        commitment_loss = commitment_loss_per_depth_tensor.mean()
        loss = codebook_loss + self.commitment_cost * commitment_loss
        encoding_indices = torch.cat(codes, dim=-1)

        final_quant = cumulative_quant
        quantized_bhwc = inputs_bhwc + (final_quant - inputs_bhwc).detach()
        quantized = quantized_bhwc.permute(0, 3, 1, 2).contiguous()

        per_depth_stats = [
            self.compute_codebook_stats(
                encoding_indices[..., depth], self.num_embeddings
            )
            for depth in range(self.rq_depth)
        ]
        aggregate_stats = self.compute_codebook_stats(
            encoding_indices, self.num_embeddings
        )
        diagnostic_device = inputs.device
        self.last_diagnostics = {
            "vq_loss": loss.detach(),
            "codebook_loss": codebook_loss.detach(),
            "commitment_loss": commitment_loss.detach(),
            "codebook_loss_per_depth": (
                codebook_loss_per_depth_tensor.detach()
            ),
            "codebook_per_depth": codebook_loss_per_depth_tensor.detach(),
            "commitment_loss_per_depth": (
                commitment_loss_per_depth_tensor.detach()
            ),
            # Keep the shorter key aligned with the existing RQ monitor.
            "commitment_per_depth": commitment_loss_per_depth_tensor.detach(),
            "residual_norm_per_depth": torch.stack(
                residual_norm_per_depth
            ).detach(),
            "usage_per_depth": torch.tensor(
                [stats["active_ratio"] for stats in per_depth_stats],
                device=diagnostic_device,
                dtype=torch.float32,
            ),
            "perplexity_per_depth": torch.tensor(
                [stats["perplexity"] for stats in per_depth_stats],
                device=diagnostic_device,
                dtype=torch.float32,
            ),
            "usage_counts_per_depth": torch.stack(
                [stats["usage_counts"] for stats in per_depth_stats]
            ).detach(),
            "aggregate_usage": float(aggregate_stats["active_ratio"]),
            "aggregate_perplexity": float(aggregate_stats["perplexity"]),
            "aggregate_usage_counts": aggregate_stats[
                "usage_counts"
            ].detach(),
            "dead_codes_per_depth": torch.tensor(
                [stats["dead_count"] for stats in per_depth_stats],
                device=diagnostic_device,
                dtype=torch.long,
            ),
            # Residual-SimVQ has no EMA lifecycle and therefore never restarts.
            "restarted_codes_per_depth": torch.zeros(
                self.rq_depth,
                device=diagnostic_device,
                dtype=torch.long,
            ),
        }
        return loss, quantized, encoding_indices

    @torch.no_grad()
    def get_quantized_features(
        self,
        encoding_indices: torch.Tensor,
        output_spatial_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Decode all residual depths with the same projected weight and sum."""
        if encoding_indices.ndim == 3:
            encoding_indices = encoding_indices.unsqueeze(0)
        if encoding_indices.ndim != 4:
            raise ValueError(
                "encoding_indices must have shape [B, H, W, rq_depth]"
            )
        batch, height, width, depth = encoding_indices.shape
        if depth != self.rq_depth:
            raise ValueError(f"expected RQ depth {self.rq_depth}, got {depth}")
        if output_spatial_size is not None:
            requested_size = tuple(int(value) for value in output_spatial_size)
            if requested_size != (height, width):
                raise ValueError(
                    "ResidualSimVQQuantizer performs no spatial resize: "
                    f"requested {requested_size}, index grid is {(height, width)}"
                )

        weight = self.transformed_weight()
        indices = encoding_indices.to(device=weight.device).long()
        if indices.numel():
            minimum = int(indices.min().item())
            maximum = int(indices.max().item())
            if minimum < 0 or maximum >= self.num_embeddings:
                raise ValueError(
                    f"code index range [{minimum}, {maximum}] is invalid for "
                    f"K={self.num_embeddings}"
                )

        quantized_bhwc = torch.zeros(
            batch,
            height,
            width,
            self.embedding_dim,
            device=weight.device,
            dtype=weight.dtype,
        )
        for depth_index in range(self.rq_depth):
            quantized_bhwc.add_(
                F.embedding(indices[..., depth_index], weight)
            )
        return quantized_bhwc.permute(0, 3, 1, 2).contiguous()

    def get_last_diagnostics(self) -> Dict[str, object]:
        return self.last_diagnostics


__all__ = ["ResidualSimVQQuantizer"]
