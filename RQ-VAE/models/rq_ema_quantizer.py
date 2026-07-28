# Copyright (c) 2022-present, Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified for the RQ-VAE project from Kakao Brain's RQ-VAE implementation at
# commit 341395e562ac347f5eb62db9f5f08b9f2cc42a60.  The modifications adapt
# the bottleneck to direct NCHW feature quantization, expose project-compatible
# monitoring methods, and record diagnostics without changing the EMA/RQ core.

"""Shared-codebook residual quantization with official-style EMA updates.

The quantizer deliberately performs no projection, resizing, or feature
transformation.  Its only layout conversion is NCHW <-> NHWC around the
residual-quantization core.  Fixed-depth :meth:`forward` remains the training
and legacy evaluation API.  The separate eval-only adaptive API can stop after
the first code on a per-token basis without changing fixed-depth behavior.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


class VQEmbedding(nn.Embedding):
    """VQ embedding table with the EMA update used by official RQ-VAE.

    The final row is a padding embedding.  It is saved in ``weight`` but is
    excluded from nearest-neighbour search and all EMA statistics.
    """

    def __init__(
        self,
        n_embed: int,
        embed_dim: int,
        ema: bool = True,
        decay: float = 0.99,
        restart_unused_codes: bool = True,
        eps: float = 1e-5,
    ) -> None:
        if int(n_embed) <= 0:
            raise ValueError("n_embed must be positive")
        if int(embed_dim) <= 0:
            raise ValueError("embed_dim must be positive")
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError("decay must be in [0, 1)")

        super().__init__(int(n_embed) + 1, int(embed_dim), padding_idx=int(n_embed))
        self.ema = bool(ema)
        self.decay = float(decay)
        self.eps = float(eps)
        self.restart_unused_codes = bool(restart_unused_codes)
        self.n_embed = int(n_embed)

        if self.ema:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            # The padding row is intentionally absent from both EMA buffers.
            self.register_buffer("cluster_size_ema", torch.zeros(self.n_embed))
            self.register_buffer("embed_ema", self.weight[:-1].detach().clone())

        self.last_update_diagnostics: Dict[str, int] = {
            "dead_codes": 0,
            "restarted_codes": 0,
            "remaining_dead_codes": 0,
        }

    @torch.no_grad()
    def compute_distances(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return squared L2 distances to the K non-padding embeddings."""
        codebook_t = self.weight[:-1].t()
        embed_dim, _ = codebook_t.shape
        if inputs.shape[-1] != embed_dim:
            raise ValueError(
                f"expected feature dimension {embed_dim}, got {inputs.shape[-1]}"
            )

        inputs_shape = inputs.shape
        inputs_flat = inputs.reshape(-1, embed_dim)
        inputs_norm_sq = inputs_flat.pow(2.0).sum(dim=1, keepdim=True)
        codebook_norm_sq = codebook_t.pow(2.0).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_norm_sq,
            inputs_flat,
            codebook_t,
            alpha=-2.0,
        )
        return distances.reshape(*inputs_shape[:-1], self.n_embed)

    @torch.no_grad()
    def find_nearest_embedding(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.compute_distances(inputs).argmin(dim=-1)

    @torch.no_grad()
    def _tile_with_noise(self, vectors: torch.Tensor, target_n: int) -> torch.Tensor:
        n_vectors, embed_dim = vectors.shape
        if n_vectors == 0:
            raise ValueError("cannot restart codes from an empty input batch")
        n_repeats = (int(target_n) + n_vectors - 1) // n_vectors
        std = vectors.new_ones(embed_dim) * 0.01 / np.sqrt(embed_dim)
        tiled = vectors.repeat(n_repeats, 1)
        # This intentionally follows the official positive uniform noise.
        return tiled + torch.rand_like(tiled) * std

    @torch.no_grad()
    def _update_buffers(
        self, vectors: torch.Tensor, indices: torch.Tensor
    ) -> Dict[str, int]:
        n_embed = self.weight.shape[0] - 1
        embed_dim = self.weight.shape[-1]
        vectors = vectors.reshape(-1, embed_dim)
        indices = indices.reshape(-1)
        n_vectors = vectors.shape[0]

        one_hot_indices = vectors.new_zeros(n_embed, n_vectors)
        one_hot_indices.scatter_(
            dim=0,
            index=indices.unsqueeze(0),
            src=vectors.new_ones(1, n_vectors),
        )
        cluster_size = one_hot_indices.sum(dim=1)
        vectors_sum_per_cluster = one_hot_indices @ vectors

        if dist.is_initialized():
            dist.all_reduce(vectors_sum_per_cluster, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)

        self.cluster_size_ema.mul_(self.decay).add_(
            cluster_size, alpha=1.0 - self.decay
        )
        self.embed_ema.mul_(self.decay).add_(
            vectors_sum_per_cluster, alpha=1.0 - self.decay
        )

        dead_mask = self.cluster_size_ema < 1
        dead_codes = int(dead_mask.sum().item())
        restarted_codes = 0

        if self.restart_unused_codes:
            restart_vectors = vectors
            if n_vectors < n_embed:
                restart_vectors = self._tile_with_noise(restart_vectors, n_embed)
                n_vectors = restart_vectors.shape[0]
            random_vectors = restart_vectors[
                torch.randperm(n_vectors, device=restart_vectors.device)
            ][:n_embed]
            if dist.is_initialized():
                dist.broadcast(random_vectors, 0)

            usage = (self.cluster_size_ema.view(-1, 1) >= 1).float()
            self.embed_ema.mul_(usage).add_(random_vectors * (1.0 - usage))
            self.cluster_size_ema.mul_(usage.view(-1))
            self.cluster_size_ema.add_(
                torch.ones_like(self.cluster_size_ema) * (1.0 - usage).view(-1)
            )
            restarted_codes = dead_codes

        remaining_dead_codes = int((self.cluster_size_ema < 1).sum().item())
        return {
            # ``dead_codes`` is measured immediately before optional restart.
            "dead_codes": dead_codes,
            "restarted_codes": restarted_codes,
            "remaining_dead_codes": remaining_dead_codes,
        }

    @torch.no_grad()
    def _update_embedding(self) -> None:
        n_embed = self.weight.shape[0] - 1
        total_count = self.cluster_size_ema.sum()
        normalized_cluster_size = (
            total_count
            * (self.cluster_size_ema + self.eps)
            / (total_count + n_embed * self.eps)
        )
        self.weight[:-1].copy_(
            self.embed_ema / normalized_cluster_size.reshape(-1, 1)
        )

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding_indices = self.find_nearest_embedding(inputs)

        if self.training and self.ema:
            self.last_update_diagnostics = self._update_buffers(
                inputs, embedding_indices
            )
        elif self.ema:
            self.last_update_diagnostics = {
                "dead_codes": int((self.cluster_size_ema < 1).sum().item()),
                "restarted_codes": 0,
                "remaining_dead_codes": int(
                    (self.cluster_size_ema < 1).sum().item()
                ),
            }
        else:
            self.last_update_diagnostics = {
                "dead_codes": 0,
                "restarted_codes": 0,
                "remaining_dead_codes": 0,
            }

        # Official ordering: return this lookup from the pre-update weight,
        # then expose the newly updated weight to the next RQ depth/call.
        embeddings = self.embed(embedding_indices)
        if self.training and self.ema:
            self._update_embedding()
        return embeddings, embedding_indices

    def embed(self, indices: torch.Tensor) -> torch.Tensor:
        return super().forward(indices)


class RQEMAQuantizer(nn.Module):
    """Direct NCHW residual quantizer with one EMA codebook shared by depth.

    ``forward`` returns ``(raw_commitment, ste_quantized, codes)``.  The raw
    commitment is intentionally not multiplied by the outer loss coefficient.
    Residual norms in :attr:`last_diagnostics` are RMS values measured after
    each cumulative RQ depth.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        rq_depth: int = 2,
        decay: float = 0.99,
        restart_unused_codes: bool = True,
        shared_codebook: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if int(rq_depth) <= 0:
            raise ValueError("rq_depth must be positive")
        if not shared_codebook:
            raise ValueError("RQEMAQuantizer requires one codebook shared by all depths")

        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.rq_depth = int(rq_depth)
        self.decay = float(decay)
        self.restart_unused_codes = bool(restart_unused_codes)
        self.shared_codebook = True
        self.eps = float(eps)

        codebook = VQEmbedding(
            self.num_embeddings,
            self.embedding_dim,
            ema=True,
            decay=self.decay,
            restart_unused_codes=self.restart_unused_codes,
            eps=self.eps,
        )
        # Repeated references are intentional: every depth updates and reads
        # exactly the same object, in sequence, as in official RQ-VAE.
        self.codebooks = nn.ModuleList([codebook for _ in range(self.rq_depth)])
        self.last_diagnostics: Dict[str, object] = {}

    @property
    def codebook(self) -> VQEmbedding:
        return self.codebooks[0]

    def transformed_weight(self) -> torch.Tensor:
        """Return only the K real EMA embeddings, excluding padding."""
        return self.codebook.weight[:-1]

    @staticmethod
    def compute_codebook_stats(
        encoding_idx: torch.Tensor, num_embeddings: int
    ) -> Dict[str, object]:
        """Compute usage statistics with the existing quantizer key schema."""
        num_embeddings = int(num_embeddings)
        flat_idx = encoding_idx.reshape(-1).long()
        if flat_idx.numel() > 0:
            min_index = int(flat_idx.min().item())
            max_index = int(flat_idx.max().item())
            if min_index < 0 or max_index >= num_embeddings:
                raise ValueError(
                    f"code index range [{min_index}, {max_index}] is invalid for "
                    f"K={num_embeddings}"
                )
        usage_counts = torch.bincount(flat_idx, minlength=num_embeddings).float()
        active_count = int((usage_counts > 0).sum().item())
        dead_count = num_embeddings - active_count
        active_ratio = active_count / num_embeddings

        total = flat_idx.numel()
        if total:
            probabilities = usage_counts / total
            probabilities = probabilities[probabilities > 0]
            entropy = -(probabilities * torch.log2(probabilities)).sum()
            perplexity = float(torch.pow(2.0, entropy).item())
        else:
            perplexity = 1.0

        return {
            "active_ratio": active_ratio,
            "perplexity": perplexity,
            "active_count": active_count,
            "dead_count": dead_count,
            "usage_counts": usage_counts,
        }

    @staticmethod
    @torch.no_grad()
    def compute_min_l2_distance(
        embed_weight: torch.Tensor,
        collapse_threshold: float = 0.1,
        max_reference_codes: int = 4096,
    ) -> Dict[str, object]:
        """Compute nearest-code distance metrics with bounded memory."""
        if embed_weight.ndim != 2 or embed_weight.shape[0] == 0:
            raise ValueError("embed_weight must have shape [K, C] with K > 0")
        codebook_size = embed_weight.shape[0]
        if codebook_size == 1:
            return {
                "min_l2_dist": float("inf"),
                "collapse_count": 0,
                "collapse_ratio": 0.0,
                "distance_reference_count": 1,
                "distance_stats_exact": True,
            }

        if codebook_size > int(max_reference_codes):
            reference_indices = torch.linspace(
                0,
                codebook_size - 1,
                steps=int(max_reference_codes),
                device=embed_weight.device,
            ).long()
        else:
            reference_indices = torch.arange(
                codebook_size, device=embed_weight.device
            )
        reference_weight = embed_weight[reference_indices]
        norm_sq = embed_weight.pow(2).sum(dim=1)
        chunk_size = max(1, 16_777_216 // codebook_size)
        nearest_chunks = []

        for start in range(0, reference_weight.shape[0], chunk_size):
            end = min(start + chunk_size, reference_weight.shape[0])
            chunk = reference_weight[start:end]
            distance_sq = (
                chunk.pow(2).sum(dim=1, keepdim=True)
                + norm_sq.unsqueeze(0)
                - 2.0 * torch.matmul(chunk, embed_weight.t())
            )
            rows = torch.arange(end - start, device=embed_weight.device)
            columns = reference_indices[start:end]
            distance_sq[rows, columns] = float("inf")
            nearest_chunks.append(distance_sq.min(dim=1).values)

        nearest_distance_sq = torch.cat(nearest_chunks)
        nearest_distance = torch.sqrt(nearest_distance_sq.clamp(min=0))
        min_l2_distance = float(nearest_distance.min().item())
        sampled_collapse_count = int(
            (nearest_distance < float(collapse_threshold)).sum().item()
        )
        reference_count = reference_weight.shape[0]
        collapse_ratio = sampled_collapse_count / reference_count
        collapse_count = round(collapse_ratio * codebook_size)
        return {
            "min_l2_dist": min_l2_distance,
            "collapse_count": int(collapse_count),
            "collapse_ratio": collapse_ratio,
            "distance_reference_count": reference_count,
            "distance_stats_exact": reference_count == codebook_size,
        }

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        if inputs.ndim != 4:
            raise ValueError(f"expected NCHW input, got shape {tuple(inputs.shape)}")
        if inputs.shape[1] != self.embedding_dim:
            raise ValueError(
                f"expected {self.embedding_dim} channels, got {inputs.shape[1]}"
            )

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_inputs(inputs)
        inputs_bhwc = inputs.permute(0, 2, 3, 1).contiguous()

        residual = inputs_bhwc.detach().clone()
        aggregated_quants = torch.zeros_like(inputs_bhwc)
        cumulative_quants = []
        codes = []
        residual_norms = []
        dead_codes_per_depth = []
        restarted_codes_per_depth = []
        remaining_dead_codes_per_depth = []

        for depth in range(self.rq_depth):
            quant, code = self.codebooks[depth](residual)
            residual.sub_(quant)
            aggregated_quants.add_(quant)
            cumulative_quants.append(aggregated_quants.clone())
            codes.append(code.unsqueeze(-1))
            residual_norms.append(residual.pow(2).mean().sqrt())

            update_stats = self.codebooks[depth].last_update_diagnostics
            dead_codes_per_depth.append(int(update_stats["dead_codes"]))
            restarted_codes_per_depth.append(int(update_stats["restarted_codes"]))
            remaining_dead_codes_per_depth.append(
                int(update_stats["remaining_dead_codes"])
            )

        commitment_per_depth = torch.stack(
            [
                (inputs_bhwc - cumulative.detach()).pow(2).mean()
                for cumulative in cumulative_quants
            ]
        )
        commitment_loss = commitment_per_depth.mean()
        encoding_indices = torch.cat(codes, dim=-1)

        final_quant = cumulative_quants[-1]
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
            "commitment_loss": commitment_loss.detach(),
            "commitment_per_depth": commitment_per_depth.detach(),
            "residual_norm_per_depth": torch.stack(residual_norms).detach(),
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
            "aggregate_usage_counts": aggregate_stats["usage_counts"].detach(),
            "dead_codes_per_depth": torch.tensor(
                dead_codes_per_depth, device=diagnostic_device, dtype=torch.long
            ),
            "restarted_codes_per_depth": torch.tensor(
                restarted_codes_per_depth,
                device=diagnostic_device,
                dtype=torch.long,
            ),
            "remaining_dead_codes_per_depth": torch.tensor(
                remaining_dead_codes_per_depth,
                device=diagnostic_device,
                dtype=torch.long,
            ),
            # These totals count observations/updates across sequential depths.
            "dead_codes": int(sum(dead_codes_per_depth)),
            "restarted_codes": int(sum(restarted_codes_per_depth)),
            "remaining_dead_codes": int(
                (self.codebook.cluster_size_ema < 1).sum().item()
            ),
        }
        return commitment_loss, quantized, encoding_indices

    def _require_two_stage_eval(self) -> None:
        if self.training:
            raise RuntimeError(
                "adaptive RQ is eval-only; call quantizer.eval() before using it"
            )
        if self.rq_depth != 2:
            raise ValueError(
                "adaptive RQ currently requires rq_depth=2, "
                f"got rq_depth={self.rq_depth}"
            )

    @staticmethod
    def _normalize_need_second_mask(
        need_second_mask: torch.Tensor,
        batch: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not torch.is_tensor(need_second_mask):
            raise TypeError("need_second_mask must be a torch.Tensor")
        if need_second_mask.dtype != torch.bool:
            raise TypeError("need_second_mask must have dtype torch.bool")
        if need_second_mask.ndim == 2 and batch == 1:
            need_second_mask = need_second_mask.unsqueeze(0)
        expected_shape = (batch, height, width)
        if tuple(need_second_mask.shape) != expected_shape:
            raise ValueError(
                "need_second_mask must have shape [B, H, W]; "
                f"expected {expected_shape}, got {tuple(need_second_mask.shape)}"
            )
        return need_second_mask.to(device=device, non_blocking=True)

    @torch.no_grad()
    def forward_adaptive(
        self,
        inputs: torch.Tensor,
        threshold: Optional[float] = None,
        *,
        need_second_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        """Quantize two-stage RQ features with per-token early stopping.

        Selection is based on the channel-mean squared residual after the first
        code: a token STOPs exactly when ``first_stage_error < threshold`` and
        therefore uses stage two when the error is greater than or equal to the
        threshold.  An explicit boolean ``need_second_mask`` may be supplied
        instead; when present, it is authoritative and ``threshold`` is only
        retained as optional metadata.

        STOP tokens carry ``-1`` in ``indices[..., 1]`` and receive no second
        embedding contribution.  Active second-stage indices remain in
        ``[0, K)``.  This method is intentionally separate from :meth:`forward`
        so training and all fixed-rate callers retain their exact old contract.
        """
        self._require_two_stage_eval()
        self._validate_inputs(inputs)
        inputs_bhwc = inputs.permute(0, 2, 3, 1).contiguous()
        batch, height, width, _ = inputs_bhwc.shape

        threshold_value = None
        if threshold is not None:
            if torch.is_tensor(threshold):
                if threshold.numel() != 1:
                    raise ValueError("threshold tensor must contain one value")
                threshold_value = float(threshold.detach().cpu().item())
            else:
                threshold_value = float(threshold)
            if np.isnan(threshold_value):
                raise ValueError("threshold must not be NaN")

        first_quant, first_code = self.codebooks[0](inputs_bhwc)
        first_residual = inputs_bhwc - first_quant
        first_stage_error = first_residual.pow(2).mean(dim=-1)

        if need_second_mask is None:
            if threshold_value is None:
                raise ValueError(
                    "provide threshold or an explicit need_second_mask"
                )
            # Strict STOP rule: equality is active and receives stage two.
            # +/-inf are useful exact controls for all-active/all-STOP tests.
            normalized_mask = first_stage_error >= threshold_value
            selection_mode = "threshold"
        else:
            normalized_mask = self._normalize_need_second_mask(
                need_second_mask, batch, height, width, inputs.device
            )
            selection_mode = "explicit_mask"

        second_quant = torch.zeros_like(inputs_bhwc)
        # -1 is an unambiguous STOP sentinel and is never passed to embedding.
        second_code = torch.full_like(first_code, -1)
        if bool(normalized_mask.any().item()):
            active_second_quant, active_second_code = self.codebooks[1](
                first_residual[normalized_mask]
            )
            second_quant[normalized_mask] = active_second_quant
            second_code[normalized_mask] = active_second_code

        final_quant = first_quant + second_quant
        final_residual = inputs_bhwc - final_quant
        final_stage_error = final_residual.pow(2).mean(dim=-1)
        commitment_per_depth = torch.stack(
            [first_stage_error.mean(), final_stage_error.mean()]
        )
        commitment_loss = commitment_per_depth.mean()
        quantized_bhwc = inputs_bhwc + (final_quant - inputs_bhwc).detach()
        quantized = quantized_bhwc.permute(0, 3, 1, 2).contiguous()
        encoding_indices = torch.stack([first_code, second_code], dim=-1)
        stop_mask = ~normalized_mask
        token_count = normalized_mask.numel()
        second_token_count = int(normalized_mask.sum().item())

        return {
            "commitment_loss": commitment_loss,
            "commitment_per_depth": commitment_per_depth,
            "quantized": quantized,
            "indices": encoding_indices,
            # Alias for callers that use RQ terminology instead of the legacy
            # DeepSC ``indices`` name.  Both refer to the same BHWD tensor.
            "codes": encoding_indices,
            "need_second_mask": normalized_mask,
            "stop_mask": stop_mask,
            "first_stage_error": first_stage_error,
            "final_stage_error": final_stage_error,
            "threshold": threshold_value,
            "selection_mode": selection_mode,
            "second_token_count": second_token_count,
            "stop_token_count": token_count - second_token_count,
            "second_token_ratio": second_token_count / token_count,
            "stop_token_ratio": (token_count - second_token_count) / token_count,
        }

    @torch.no_grad()
    def get_adaptive_quantized_features(
        self,
        encoding_indices: torch.Tensor,
        need_second_mask: Optional[torch.Tensor] = None,
        output_spatial_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Decode BHWD adaptive codes while suppressing every STOP token.

        If ``need_second_mask`` is omitted it is inferred from the required
        ``-1`` STOP sentinel in the second code plane.  Supplying the mask is
        useful after a transport has encoded it separately; consistency with
        the sentinel representation is still validated.
        """
        self._require_two_stage_eval()
        if encoding_indices.ndim == 3:
            encoding_indices = encoding_indices.unsqueeze(0)
        if encoding_indices.ndim != 4:
            raise ValueError(
                "encoding_indices must have shape [B, H, W, 2]"
            )
        batch, height, width, depth = encoding_indices.shape
        if depth != 2:
            raise ValueError(f"expected RQ depth 2, got {depth}")
        if output_spatial_size is not None:
            requested_size = tuple(int(value) for value in output_spatial_size)
            if requested_size != (height, width):
                raise ValueError(
                    "RQEMAQuantizer performs no spatial resize: requested "
                    f"{requested_size}, index grid is {(height, width)}"
                )

        indices = encoding_indices.to(device=self.codebook.weight.device).long()
        first_code = indices[..., 0]
        second_code = indices[..., 1]
        if first_code.numel():
            first_min = int(first_code.min().item())
            first_max = int(first_code.max().item())
            if first_min < 0 or first_max >= self.num_embeddings:
                raise ValueError(
                    f"first-stage index range [{first_min}, {first_max}] is "
                    f"invalid for K={self.num_embeddings}"
                )

        inferred_mask = second_code != -1
        if need_second_mask is None:
            normalized_mask = inferred_mask
        else:
            normalized_mask = self._normalize_need_second_mask(
                need_second_mask,
                batch,
                height,
                width,
                indices.device,
            )
            if not torch.equal(normalized_mask, inferred_mask):
                raise ValueError(
                    "need_second_mask is inconsistent with second-stage -1 "
                    "STOP sentinels"
                )

        if bool(normalized_mask.any().item()):
            active_codes = second_code[normalized_mask]
            active_min = int(active_codes.min().item())
            active_max = int(active_codes.max().item())
            if active_min < 0 or active_max >= self.num_embeddings:
                raise ValueError(
                    f"active second-stage index range [{active_min}, "
                    f"{active_max}] is invalid for K={self.num_embeddings}"
                )

        weight = self.transformed_weight()
        quantized_bhwc = F.embedding(first_code, weight)
        if bool(normalized_mask.any().item()):
            quantized_bhwc[normalized_mask] += F.embedding(
                second_code[normalized_mask], weight
            )
        return quantized_bhwc.permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def get_quantized_features(
        self,
        encoding_indices: torch.Tensor,
        output_spatial_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Decode every RQ index with the same table and sum over depth."""
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
                    "RQEMAQuantizer performs no spatial resize: requested "
                    f"{requested_size}, index grid is {(height, width)}"
                )

        indices = encoding_indices.long()
        if indices.numel():
            min_index = int(indices.min().item())
            max_index = int(indices.max().item())
            if min_index < 0 or max_index >= self.num_embeddings:
                raise ValueError(
                    f"code index range [{min_index}, {max_index}] is invalid for "
                    f"K={self.num_embeddings}"
                )

        weight = self.transformed_weight()
        quantized_bhwc = torch.zeros(
            batch,
            height,
            width,
            self.embedding_dim,
            device=weight.device,
            dtype=weight.dtype,
        )
        for rq_index in range(self.rq_depth):
            quantized_bhwc.add_(F.embedding(indices[..., rq_index], weight))
        return quantized_bhwc.permute(0, 3, 1, 2).contiguous()

    def get_last_diagnostics(self) -> Dict[str, object]:
        return self.last_diagnostics


__all__ = ["RQEMAQuantizer", "VQEmbedding"]
