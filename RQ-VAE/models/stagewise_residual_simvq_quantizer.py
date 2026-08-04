"""Stage-wise residual SimVQ with an independent codebook at every depth.

Unlike :class:`ResidualSimVQQuantizer`, this module allows each residual stage
to have a different vocabulary size.  Every stage owns its own frozen base
embedding and trainable SimVQ projection.
"""

from numbers import Integral
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .vector_quantizer import ProjectedEmbedding, VectorQuantizer


class StagewiseResidualSimVQQuantizer(nn.Module):
    """Residual SimVQ whose depth-specific codebooks may have different sizes.

    ``forward`` returns ``(loss, ste_quantized, codes)``.  Codes use BHWD
    layout, with ``codes[..., depth]`` ranging from zero to
    ``num_embeddings_per_depth[depth] - 1``.

    The loss matches :class:`ResidualSimVQQuantizer`:

    * ``Q_d = MSE(q_d, residual_{d-1}.detach())``
    * ``C_d = MSE(input, cumulative_q_d.detach())``
    * ``loss = mean(Q_d) + commitment_cost * mean(C_d)``

    Reconstruction gradients pass through the final straight-through
    estimator to the encoder input.  Each projected codebook receives
    gradients only from its own ``Q_d`` term.
    """

    def __init__(
        self,
        num_embeddings_per_depth: Sequence[int],
        embedding_dim: int,
        commitment_cost: float,
    ) -> None:
        super().__init__()
        try:
            sizes = tuple(num_embeddings_per_depth)
        except TypeError as error:
            raise TypeError(
                "num_embeddings_per_depth must be a non-empty sequence"
            ) from error
        if not sizes:
            raise ValueError("num_embeddings_per_depth must not be empty")
        if any(
            isinstance(size, bool) or not isinstance(size, Integral)
            for size in sizes
        ):
            raise TypeError(
                "every num_embeddings_per_depth value must be an integer"
            )
        if any(int(size) <= 0 for size in sizes):
            raise ValueError(
                "every num_embeddings_per_depth value must be positive"
            )
        if (
            isinstance(embedding_dim, bool)
            or not isinstance(embedding_dim, Integral)
        ):
            raise TypeError("embedding_dim must be an integer")
        if int(embedding_dim) <= 0:
            raise ValueError("embedding_dim must be positive")

        self.num_embeddings_per_depth = tuple(int(size) for size in sizes)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        self.rq_depth = len(self.num_embeddings_per_depth)
        self.shared_codebook = False
        self.codebooks = nn.ModuleList(
            [
                ProjectedEmbedding(size, self.embedding_dim)
                for size in self.num_embeddings_per_depth
            ]
        )
        self.last_diagnostics: Dict[str, object] = {}

    def transformed_weight(self, depth_index: int = 0) -> torch.Tensor:
        """Return the projected weight for one depth.

        Depth zero is the default so generic codebook inspection code that
        expects the legacy no-argument method can still inspect this module.
        Quantization and decoding always pass an explicit depth.
        """
        depth_index = int(depth_index)
        if depth_index < 0 or depth_index >= self.rq_depth:
            raise IndexError(
                f"depth index {depth_index} is outside [0, {self.rq_depth})"
            )
        return self.codebooks[depth_index].projected_weight()

    def transformed_weights(self) -> Tuple[torch.Tensor, ...]:
        """Return projected weights for all independent residual stages."""
        return tuple(
            self.transformed_weight(depth_index)
            for depth_index in range(self.rq_depth)
        )

    @staticmethod
    def compute_codebook_stats(
        encoding_indices: torch.Tensor, num_embeddings: int
    ) -> Dict[str, object]:
        """Use the existing SimVQ usage-statistics schema."""
        return VectorQuantizer.compute_codebook_stats(
            encoding_indices, num_embeddings
        )

    @staticmethod
    def compute_min_l2_distance(
        embed_weight: torch.Tensor,
        collapse_threshold: float = 0.1,
        max_reference_codes: int = 4096,
    ) -> Dict[str, object]:
        """Expose the existing projected-code distance diagnostics."""
        return VectorQuantizer.compute_min_l2_distance(
            embed_weight,
            collapse_threshold=collapse_threshold,
            max_reference_codes=max_reference_codes,
        )

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        if inputs.ndim != 4:
            raise ValueError(f"expected NCHW input, got shape {tuple(inputs.shape)}")
        if inputs.shape[1] != self.embedding_dim:
            raise ValueError(
                f"expected {self.embedding_dim} channels, got {inputs.shape[1]}"
            )

    @staticmethod
    def _lookup_bhwc(
        inputs_bhwc: torch.Tensor, embed_weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, height, width, channels = inputs_bhwc.shape
        flat = inputs_bhwc.view(-1, channels)
        encoding_indices = VectorQuantizer._nearest_code_indices(
            flat, embed_weight
        )
        quantized = F.embedding(encoding_indices, embed_weight).view(
            batch, height, width, channels
        )
        return quantized, encoding_indices.view(batch, height, width)

    @staticmethod
    def _aggregate_independent_stats(
        per_depth_stats: Sequence[Dict[str, object]],
    ) -> Dict[str, object]:
        """Aggregate counts while keeping each depth's code namespace distinct."""
        usage_counts = torch.cat(
            [stats["usage_counts"] for stats in per_depth_stats]
        )
        active_count = int((usage_counts > 0).sum().item())
        total_codes = int(usage_counts.numel())
        total_observations = int(usage_counts.sum().item())
        if total_observations:
            probabilities = usage_counts / total_observations
            probabilities = probabilities[probabilities > 0]
            entropy = -(probabilities * torch.log2(probabilities)).sum()
            perplexity = float(torch.pow(2.0, entropy).item())
        else:
            perplexity = 1.0
        return {
            "active_ratio": active_count / total_codes,
            "perplexity": perplexity,
            "active_count": active_count,
            "dead_count": total_codes - active_count,
            "usage_counts": usage_counts,
        }

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_inputs(inputs)
        inputs_bhwc = inputs.permute(0, 2, 3, 1).contiguous()

        residual = inputs_bhwc.detach().clone()
        cumulative_quant = torch.zeros_like(inputs_bhwc)
        codebook_loss_per_depth = []
        commitment_loss_per_depth = []
        residual_norm_per_depth = []
        codes = []

        for depth_index in range(self.rq_depth):
            embed_weight = self.transformed_weight(depth_index)
            quantized_at_depth, code = self._lookup_bhwc(
                residual, embed_weight
            )
            codebook_loss_per_depth.append(
                F.mse_loss(quantized_at_depth, residual.detach())
            )
            cumulative_quant = cumulative_quant + quantized_at_depth
            commitment_loss_per_depth.append(
                F.mse_loss(inputs_bhwc, cumulative_quant.detach())
            )
            residual = residual - quantized_at_depth.detach()
            residual_norm_per_depth.append(
                residual.pow(2).mean().sqrt()
            )
            codes.append(code.unsqueeze(-1))

        codebook_loss_per_depth_tensor = torch.stack(
            codebook_loss_per_depth
        )
        commitment_loss_per_depth_tensor = torch.stack(
            commitment_loss_per_depth
        )
        codebook_loss = codebook_loss_per_depth_tensor.mean()
        commitment_loss = commitment_loss_per_depth_tensor.mean()
        loss = codebook_loss + self.commitment_cost * commitment_loss
        encoding_indices = torch.cat(codes, dim=-1)

        quantized_bhwc = inputs_bhwc + (
            cumulative_quant - inputs_bhwc
        ).detach()
        quantized = quantized_bhwc.permute(0, 3, 1, 2).contiguous()

        per_depth_stats = [
            self.compute_codebook_stats(
                encoding_indices[..., depth_index],
                self.num_embeddings_per_depth[depth_index],
            )
            for depth_index in range(self.rq_depth)
        ]
        aggregate_stats = self._aggregate_independent_stats(per_depth_stats)
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
            "commitment_per_depth": (
                commitment_loss_per_depth_tensor.detach()
            ),
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
            # A list is required because K can differ between depths.
            "usage_counts_per_depth": [
                stats["usage_counts"].detach()
                for stats in per_depth_stats
            ],
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
        """Decode and sum one independently projected code per depth."""
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
                    "StagewiseResidualSimVQQuantizer performs no spatial "
                    f"resize: requested {requested_size}, index grid is "
                    f"{(height, width)}"
                )

        first_weight = self.transformed_weight(0)
        indices = encoding_indices.to(device=first_weight.device).long()
        quantized_bhwc = torch.zeros(
            batch,
            height,
            width,
            self.embedding_dim,
            device=first_weight.device,
            dtype=first_weight.dtype,
        )
        for depth_index, codebook_size in enumerate(
            self.num_embeddings_per_depth
        ):
            depth_indices = indices[..., depth_index]
            if depth_indices.numel():
                minimum = int(depth_indices.min().item())
                maximum = int(depth_indices.max().item())
                if minimum < 0 or maximum >= codebook_size:
                    raise ValueError(
                        f"depth {depth_index} code index range "
                        f"[{minimum}, {maximum}] is invalid for "
                        f"K={codebook_size}"
                    )
            quantized_bhwc.add_(
                F.embedding(
                    depth_indices,
                    self.transformed_weight(depth_index),
                )
            )
        return quantized_bhwc.permute(0, 3, 1, 2).contiguous()

    def get_last_diagnostics(self) -> Dict[str, object]:
        return self.last_diagnostics


__all__ = ["StagewiseResidualSimVQQuantizer"]
