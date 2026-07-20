"""Variable-rate residual adaptive quantization (RAQ) codebook generator.

This module intentionally does not reuse :mod:`models.transformer`.  The
legacy generator adds positional encodings to the source codebook, which
would make the result depend on an arbitrary ordering of source codewords.
Here the source is treated as a set: learned target queries attend to every
source codeword and explicitly form a convex aggregation ``S_K``.  A small
target-side Transformer then predicts only a residual ``Delta W``.

One :class:`VariableRateRAQGenerator` is shared by all profiles.  It contains
two scale-specific layer generators because the U-Net scales can have
different feature dimensions, while their rate condition is produced by one
shared full-profile embedding network.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


Profile = Union[Sequence[int], torch.Tensor]


def _as_int_tuple(values: Profile, *, name: str) -> Tuple[int, ...]:
    """Convert a profile-like value without silently truncating floats."""

    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional, got {tuple(values.shape)}")
        values = values.detach().cpu().tolist()
    converted: List[int] = []
    for value in values:
        integer = int(value)
        if isinstance(value, float) and value != integer:
            raise ValueError(f"{name} entries must be integers, got {value!r}")
        converted.append(integer)
    return tuple(converted)


def _default_allowed_sizes(source_size: int, minimum_size: int) -> Tuple[int, ...]:
    if minimum_size < 1:
        raise ValueError("minimum target size must be positive")
    if source_size < minimum_size:
        raise ValueError("source codebook cannot be smaller than the minimum target size")
    if source_size & (source_size - 1):
        raise ValueError(
            "default variable-rate profiles require power-of-two source codebooks; "
            "pass allowed_target_sizes explicitly for another layout"
        )
    sizes = []
    value = 1
    while value < minimum_size:
        value *= 2
    while value <= source_size:
        sizes.append(value)
        value *= 2
    return tuple(sizes)


def _normalise_allowed_sizes(
    allowed_target_sizes: Optional[Sequence[Union[int, Sequence[int]]]],
    source_sizes: Tuple[int, ...],
    minimum_size: int,
) -> Tuple[Tuple[int, ...], ...]:
    """Accept one shared rate list or one list per U-Net scale."""

    if allowed_target_sizes is None:
        return tuple(_default_allowed_sizes(size, minimum_size) for size in source_sizes)

    raw = list(allowed_target_sizes)
    if not raw:
        raise ValueError("allowed_target_sizes cannot be empty")
    if all(isinstance(item, int) for item in raw):
        per_layer = [raw for _ in source_sizes]
    else:
        if len(raw) != len(source_sizes):
            raise ValueError(
                "nested allowed_target_sizes must contain one sequence per scale"
            )
        per_layer = [list(item) for item in raw]  # type: ignore[arg-type]

    result = []
    for layer_index, (values, source_size) in enumerate(zip(per_layer, source_sizes)):
        sizes = tuple(sorted(set(int(value) for value in values)))
        if not sizes or sizes[0] < 1 or sizes[-1] > source_size:
            raise ValueError(
                f"invalid target sizes for layer {layer_index}: {sizes}; "
                f"expected values in [1, {source_size}]"
            )
        if source_size not in sizes:
            raise ValueError(
                f"layer {layer_index} target sizes must include source size {source_size}"
            )
        result.append(sizes)
    return tuple(result)


def _compatible_num_heads(model_dim: int, requested_heads: int) -> int:
    """Choose the largest requested-or-smaller head count dividing model_dim."""

    for heads in range(min(model_dim, requested_heads), 0, -1):
        if model_dim % heads == 0:
            return heads
    return 1


class RateProfileEmbedding(nn.Module):
    """Embed the complete rate profile ``[log2(K_0), log2(K_1)]`` once.

    The raw log-rates are divided by their per-scale maxima before the MLP for
    numerical conditioning.  No information is discarded: the maxima are
    fixed known constants, and ``raw_log_rates`` is exposed for diagnostics.
    """

    def __init__(
        self,
        source_sizes: Sequence[int],
        embedding_dim: int = 128,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        source_sizes = _as_int_tuple(source_sizes, name="source_sizes")
        if len(source_sizes) != 2:
            raise ValueError(
                "the current variable-rate design requires exactly two U-Net scales"
            )
        if embedding_dim < 1:
            raise ValueError("rate embedding dimension must be positive")
        hidden_dim = int(hidden_dim or embedding_dim)
        self.source_sizes = source_sizes
        self.embedding_dim = int(embedding_dim)
        max_log_rates = torch.tensor([math.log2(size) for size in source_sizes])
        self.register_buffer("max_log_rates", max_log_rates, persistent=True)
        self.network = nn.Sequential(
            nn.Linear(len(source_sizes), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
        )

    def raw_log_rates(self, profile: Profile) -> torch.Tensor:
        profile_tuple = _as_int_tuple(profile, name="profile")
        if len(profile_tuple) != len(self.source_sizes):
            raise ValueError(
                f"profile must have {len(self.source_sizes)} entries, got {profile_tuple}"
            )
        if any(value <= 0 for value in profile_tuple):
            raise ValueError(f"profile entries must be positive, got {profile_tuple}")
        return self.max_log_rates.new_tensor(
            [math.log2(value) for value in profile_tuple]
        )

    def forward(self, profile: Profile) -> torch.Tensor:
        raw_rates = self.raw_log_rates(profile)
        normalised = raw_rates / self.max_log_rates.clamp_min(1.0)
        return self.network(normalised.unsqueeze(0))


class LayerResidualCodebookGenerator(nn.Module):
    """Generate one scale's ``W_K = S_K(W_src) + Delta W_K``.

    Cross-attention is deliberately implemented as a target-query softmax over
    all source codewords.  Multiplying those weights by the *unprojected*
    source codebook makes ``S_K`` an explicit differentiable aggregation in
    the original feature space.  There is no source positional encoding.
    """

    def __init__(
        self,
        embedding_dim: int,
        source_size: int,
        rate_embedding_dim: int,
        *,
        model_dim: Optional[int] = None,
        attention_dim: int = 64,
        transformer_depth: int = 2,
        transformer_heads: int = 8,
        feedforward_multiplier: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim < 1 or source_size < 2:
            raise ValueError("embedding_dim must be positive and source_size at least two")
        if attention_dim < 1 or transformer_depth < 1:
            raise ValueError("attention_dim and transformer_depth must be positive")
        self.embedding_dim = int(embedding_dim)
        self.source_size = int(source_size)
        self.model_dim = int(model_dim or embedding_dim)
        self.attention_dim = int(attention_dim)

        # Target identities/order are carried by learned queries.  Source
        # codewords receive only content projections, never positional terms.
        self.pool_queries = nn.Embedding(self.source_size, self.attention_dim)
        nn.init.normal_(self.pool_queries.weight, std=self.attention_dim ** -0.5)
        self.source_to_key = nn.Linear(self.embedding_dim, self.attention_dim, bias=False)
        self.key_norm = nn.LayerNorm(self.attention_dim)
        self.rate_to_pool_query = nn.Linear(
            rate_embedding_dim, self.attention_dim, bias=False
        )

        self.baseline_to_model = nn.Linear(self.embedding_dim, self.model_dim)
        self.pool_query_to_model = nn.Linear(self.attention_dim, self.model_dim, bias=False)
        self.rate_to_model = nn.Linear(rate_embedding_dim, self.model_dim, bias=False)
        heads = _compatible_num_heads(self.model_dim, int(transformer_heads))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=heads,
            dim_feedforward=max(
                self.model_dim, int(round(self.model_dim * feedforward_multiplier))
            ),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.residual_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(transformer_depth),
            norm=nn.LayerNorm(self.model_dim),
        )
        self.delta_projection = nn.Linear(self.model_dim, self.embedding_dim)

        # Start from the meaningful aggregation S_K.  The generator learns a
        # residual without injecting arbitrary output noise at initialisation.
        nn.init.zeros_(self.delta_projection.weight)
        nn.init.zeros_(self.delta_projection.bias)

    def _validate_inputs(
        self,
        source_codebook: torch.Tensor,
        target_size: int,
        rate_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if source_codebook.ndim != 2:
            raise ValueError(
                "source_codebook must have shape [K_src, D], got "
                f"{tuple(source_codebook.shape)}"
            )
        expected = (self.source_size, self.embedding_dim)
        if tuple(source_codebook.shape) != expected:
            raise ValueError(
                f"source codebook shape mismatch: expected {expected}, "
                f"got {tuple(source_codebook.shape)}"
            )
        if not 1 <= int(target_size) < self.source_size:
            raise ValueError(
                f"layer generator only handles 1 <= K < {self.source_size}; "
                f"K={target_size} must use the caller's hard bypass"
            )
        if rate_embedding.ndim == 1:
            rate_embedding = rate_embedding.unsqueeze(0)
        if rate_embedding.ndim != 2 or rate_embedding.shape[0] != 1:
            raise ValueError(
                "a codebook is global for the batch, so rate_embedding must have "
                f"shape [1, R], got {tuple(rate_embedding.shape)}"
            )
        if source_codebook.device != self.pool_queries.weight.device:
            raise ValueError(
                "source codebook and RAQ generator must be on the same device; "
                f"got {source_codebook.device} and {self.pool_queries.weight.device}"
            )
        return rate_embedding

    def forward(
        self,
        source_codebook: torch.Tensor,
        target_size: int,
        rate_embedding: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        target_size = int(target_size)
        rate_embedding = self._validate_inputs(
            source_codebook, target_size, rate_embedding
        )

        target_queries = self.pool_queries.weight[:target_size]
        target_queries = target_queries + self.rate_to_pool_query(rate_embedding)
        source_keys = self.key_norm(self.source_to_key(source_codebook))

        # [K, A] @ [A, K_src] -> [K, K_src].  Every target query sees every
        # source codeword, and the softmax weights form a convex aggregation.
        logits = torch.matmul(target_queries, source_keys.transpose(0, 1))
        logits = logits / math.sqrt(self.attention_dim)
        attention = torch.softmax(logits, dim=-1)
        aggregation = torch.matmul(attention, source_codebook)

        target_tokens = self.baseline_to_model(aggregation)
        target_tokens = target_tokens + self.pool_query_to_model(target_queries)
        target_tokens = target_tokens + self.rate_to_model(rate_embedding)
        transformed = self.residual_transformer(target_tokens.unsqueeze(0)).squeeze(0)
        residual = self.delta_projection(transformed)
        codebook = aggregation + residual
        return {
            "codebook": codebook,
            "aggregation": aggregation,
            "residual": residual,
            "attention": attention if return_attention else None,
        }


class VariableRateRAQGenerator(nn.Module):
    """Unified two-scale generator shared by every atomic rate profile."""

    def __init__(
        self,
        embedding_dims: Sequence[int],
        source_num_embeddings: Sequence[int] = (2048, 2048),
        *,
        rate_embedding_dim: int = 128,
        rate_hidden_dim: Optional[int] = None,
        generator_model_dims: Optional[Union[int, Sequence[int]]] = None,
        attention_dim: int = 64,
        transformer_depth: int = 2,
        transformer_heads: int = 8,
        feedforward_multiplier: float = 4.0,
        dropout: float = 0.1,
        minimum_target_size: int = 2,
        allowed_target_sizes: Optional[
            Sequence[Union[int, Sequence[int]]]
        ] = None,
    ) -> None:
        super().__init__()
        self.embedding_dims = _as_int_tuple(embedding_dims, name="embedding_dims")
        self.source_num_embeddings = _as_int_tuple(
            source_num_embeddings, name="source_num_embeddings"
        )
        if len(self.embedding_dims) != 2 or len(self.source_num_embeddings) != 2:
            raise ValueError(
                "VariableRateRAQGenerator requires exactly two scale dimensions and sizes"
            )
        if any(dim < 1 for dim in self.embedding_dims):
            raise ValueError("all embedding dimensions must be positive")
        self.allowed_target_sizes = _normalise_allowed_sizes(
            allowed_target_sizes,
            self.source_num_embeddings,
            int(minimum_target_size),
        )

        if generator_model_dims is None:
            model_dims = self.embedding_dims
        elif isinstance(generator_model_dims, int):
            model_dims = (int(generator_model_dims),) * 2
        else:
            model_dims = _as_int_tuple(generator_model_dims, name="generator_model_dims")
            if len(model_dims) != 2:
                raise ValueError("generator_model_dims must contain two entries")

        self.rate_conditioner = RateProfileEmbedding(
            self.source_num_embeddings,
            embedding_dim=int(rate_embedding_dim),
            hidden_dim=rate_hidden_dim,
        )
        self.layer_generators = nn.ModuleList(
            [
                LayerResidualCodebookGenerator(
                    embedding_dim=embedding_dim,
                    source_size=source_size,
                    rate_embedding_dim=int(rate_embedding_dim),
                    model_dim=model_dim,
                    attention_dim=int(attention_dim),
                    transformer_depth=int(transformer_depth),
                    transformer_heads=int(transformer_heads),
                    feedforward_multiplier=float(feedforward_multiplier),
                    dropout=float(dropout),
                )
                for embedding_dim, source_size, model_dim in zip(
                    self.embedding_dims, self.source_num_embeddings, model_dims
                )
            ]
        )

    def validate_profile(self, profile: Profile) -> Tuple[int, int]:
        profile_tuple = _as_int_tuple(profile, name="profile")
        if len(profile_tuple) != 2:
            raise ValueError(f"profile must contain exactly two rates, got {profile_tuple}")
        for layer_index, (target_size, allowed) in enumerate(
            zip(profile_tuple, self.allowed_target_sizes)
        ):
            if target_size not in allowed:
                raise ValueError(
                    f"K[{layer_index}]={target_size} is not supported; allowed={allowed}"
                )
        return profile_tuple  # type: ignore[return-value]

    def _validate_source_codebooks(
        self, source_codebooks: Sequence[torch.Tensor]
    ) -> None:
        if len(source_codebooks) != 2:
            raise ValueError(
                f"expected two source codebooks, got {len(source_codebooks)}"
            )
        first_device = source_codebooks[0].device
        for layer_index, (codebook, source_size, embedding_dim) in enumerate(
            zip(source_codebooks, self.source_num_embeddings, self.embedding_dims)
        ):
            expected = (source_size, embedding_dim)
            if codebook.ndim != 2 or tuple(codebook.shape) != expected:
                raise ValueError(
                    f"source codebook {layer_index} must have shape {expected}, "
                    f"got {tuple(codebook.shape)}"
                )
            if codebook.device != first_device:
                raise ValueError("all source codebooks must be on the same device")

    def _generate_one(
        self,
        layer_index: int,
        target_size: int,
        source_codebook: torch.Tensor,
        rate_embedding: torch.Tensor,
        *,
        return_attention: bool,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if target_size == self.source_num_embeddings[layer_index]:
            # Strict layer-wise identity: do not invoke, checkpoint, or even
            # inspect the scale generator on this path.  Consequently it gets
            # no gradient for a [2048, 2048] profile, as mathematically required.
            return {
                "codebook": source_codebook,
                "aggregation": source_codebook,
                "residual": torch.zeros_like(source_codebook),
                "attention": None,
            }
        return self.layer_generators[layer_index](
            source_codebook,
            target_size,
            rate_embedding,
            return_attention=return_attention,
        )

    def forward(
        self,
        source_codebooks: Sequence[torch.Tensor],
        profile: Profile,
        *,
        generate_hierarchy: bool = False,
        return_attention: bool = False,
    ) -> Dict[str, object]:
        profile_tuple = self.validate_profile(profile)
        self._validate_source_codebooks(source_codebooks)
        rate_embedding = self.rate_conditioner(profile_tuple)

        codebooks: List[torch.Tensor] = []
        aggregations: List[torch.Tensor] = []
        residuals: List[torch.Tensor] = []
        attentions: List[Optional[torch.Tensor]] = []
        bypass_flags: List[bool] = []
        for layer_index, (target_size, source_codebook) in enumerate(
            zip(profile_tuple, source_codebooks)
        ):
            generated = self._generate_one(
                layer_index,
                target_size,
                source_codebook,
                rate_embedding,
                return_attention=return_attention,
            )
            codebooks.append(generated["codebook"])  # type: ignore[arg-type]
            aggregations.append(generated["aggregation"])  # type: ignore[arg-type]
            residuals.append(generated["residual"])  # type: ignore[arg-type]
            attentions.append(generated["attention"])
            bypass_flags.append(target_size == self.source_num_embeddings[layer_index])

        hierarchy_codebooks: List[Optional[torch.Tensor]] = [None, None]
        hierarchy_aggregations: List[Optional[torch.Tensor]] = [None, None]
        hierarchy_residuals: List[Optional[torch.Tensor]] = [None, None]
        hierarchy_profiles: List[Optional[Tuple[int, int]]] = [None, None]
        hierarchy_bypass_flags: List[Optional[bool]] = [None, None]
        if generate_hierarchy:
            for layer_index, (target_size, source_codebook) in enumerate(
                zip(profile_tuple, source_codebooks)
            ):
                parent_size = target_size * 2
                if parent_size not in self.allowed_target_sizes[layer_index]:
                    continue
                parent_profile_list = list(profile_tuple)
                parent_profile_list[layer_index] = parent_size
                parent_profile = tuple(parent_profile_list)
                parent_rate_embedding = self.rate_conditioner(parent_profile)
                generated_parent = self._generate_one(
                    layer_index,
                    parent_size,
                    source_codebook,
                    parent_rate_embedding,
                    return_attention=False,
                )
                hierarchy_codebooks[layer_index] = generated_parent["codebook"]
                hierarchy_aggregations[layer_index] = generated_parent["aggregation"]
                hierarchy_residuals[layer_index] = generated_parent["residual"]
                hierarchy_profiles[layer_index] = parent_profile  # type: ignore[assignment]
                hierarchy_bypass_flags[layer_index] = (
                    parent_size == self.source_num_embeddings[layer_index]
                )

        return {
            "profile": profile_tuple,
            "raw_log_rates": self.rate_conditioner.raw_log_rates(profile_tuple),
            "rate_embedding": rate_embedding,
            "codebooks": codebooks,
            "aggregation_codebooks": aggregations,
            "residual_codebooks": residuals,
            "attention_weights": attentions,
            "bypass_flags": bypass_flags,
            "hierarchy_codebooks": hierarchy_codebooks,
            "hierarchy_aggregation_codebooks": hierarchy_aggregations,
            "hierarchy_residual_codebooks": hierarchy_residuals,
            "hierarchy_profiles": hierarchy_profiles,
            "hierarchy_bypass_flags": hierarchy_bypass_flags,
        }


__all__ = [
    "LayerResidualCodebookGenerator",
    "Profile",
    "RateProfileEmbedding",
    "VariableRateRAQGenerator",
]
