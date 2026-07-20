"""Two-scale DeepSC student with one unified variable-rate RAQ generator.

The model owns one semantic encoder, one decoder, and two 2048-entry source
SimVQ codebooks.  A complete rate profile conditions both scale generators
and identity-initialised FiLM adapters.  ``forward_src`` is deliberately kept
as a clean source-model path: it does not call RAQ or FiLM and can therefore
be used to verify/load a conventional SRC checkpoint without hidden changes.
"""

from __future__ import annotations

import inspect
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .attention import BottleneckAttentionStack
from .channel import FiniteBlocklengthChannel
from .semantic_decoder import SemanticDecoder
from .semantic_encoder import SemanticEncoder
from .swinir_enhance import SwinIREnhance
from .variable_rate_raq import Profile, VariableRateRAQGenerator
from .vector_quantizer import VectorQuantizer


class ConditionalAffine(nn.Module):
    """A lightweight, exactly identity-initialised FiLM adapter.

    ``condition`` is a single full-profile embedding shared by the image batch.
    The learned affine parameters are channel-wise and broadcast spatially.
    """

    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__()
        if channels < 1 or condition_dim < 1:
            raise ValueError("channels and condition_dim must be positive")
        self.channels = int(channels)
        self.condition_dim = int(condition_dim)
        self.to_affine = nn.Linear(self.condition_dim, 2 * self.channels)
        nn.init.zeros_(self.to_affine.weight)
        nn.init.zeros_(self.to_affine.bias)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(
                f"FiLM features must have shape [B,C,H,W], got {tuple(features.shape)}"
            )
        if features.shape[1] != self.channels:
            raise ValueError(
                f"FiLM expected {self.channels} channels, got {features.shape[1]}"
            )
        if condition.ndim == 1:
            condition = condition.unsqueeze(0)
        if condition.ndim != 2 or condition.shape[-1] != self.condition_dim:
            raise ValueError(
                f"condition must have shape [1 or B,{self.condition_dim}], "
                f"got {tuple(condition.shape)}"
            )
        if condition.shape[0] not in (1, features.shape[0]):
            raise ValueError(
                "condition batch dimension must be one or match the image batch; "
                f"got {condition.shape[0]} and {features.shape[0]}"
            )
        affine = self.to_affine(condition)
        scale, shift = affine.chunk(2, dim=-1)
        scale = scale.view(scale.shape[0], self.channels, 1, 1)
        shift = shift.view(shift.shape[0], self.channels, 1, 1)
        return features * (1.0 + scale) + shift


# FiLM is a widely recognised name; keep it as a readable public alias.
FiLM = ConditionalAffine


class VariableRateDeepSC(nn.Module):
    """Shared encoder/decoder variable-rate semantic communication student.

    Args are intentionally explicit so a ``VariableRateConfig`` dataclass or
    mapping can be passed through :meth:`from_config`.  ``embedding_dims`` must
    match the channels emitted by the semantic encoder.  If omitted, it is
    derived as ``base_channels * 2**(layer+1)``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_downsample_blocks: int = 2,
        base_channels: int = 256,
        source_num_embeddings: Sequence[int] = (2048, 2048),
        embedding_dims: Optional[Sequence[int]] = None,
        commitment_cost: float = 0.25,
        *,
        strides: Optional[Sequence[int]] = None,
        skip_dropout_p: Optional[Sequence[float]] = None,
        norm_type: str = "batch",
        norm_groups: int = 32,
        activation: str = "prelu",
        encoder_res_blocks: int = 1,
        decoder_res_blocks: int = 1,
        upsample_mode: str = "nearest",
        use_cascade_downsample: bool = True,
        use_bottleneck_attention: bool = False,
        bottleneck_attention_blocks: int = 1,
        use_swinir_enhance: bool = False,
        swinir_enhance_blocks: int = 4,
        rate_embedding_dim: int = 128,
        rate_hidden_dim: Optional[int] = None,
        generator_model_dims: Optional[Union[int, Sequence[int]]] = None,
        generator_attention_dim: int = 64,
        generator_transformer_depth: int = 2,
        generator_transformer_heads: int = 8,
        generator_feedforward_multiplier: float = 4.0,
        generator_dropout: float = 0.1,
        minimum_target_size: int = 2,
        allowed_target_sizes: Optional[
            Sequence[Union[int, Sequence[int]]]
        ] = None,
        freeze_source_codebooks: bool = False,
        channel_coding_rate_train: float = 0.5,
        channel_coding_rate_val: float = 0.5,
        block_length: int = 256,
        snr_range_db: Sequence[float] = (0.0, 15.0),
        channel_prob: float = 1.0,
    ) -> None:
        super().__init__()
        if int(num_downsample_blocks) != 2:
            raise ValueError(
                "the requested full-profile variable-rate design currently requires "
                "exactly two U-Net scales"
            )
        self.num_downsample_blocks = 2
        self.base_channels = int(base_channels)
        if self.base_channels < 1:
            raise ValueError("base_channels must be positive")

        expected_dims = tuple(
            self.base_channels * (2 ** (layer_index + 1))
            for layer_index in range(self.num_downsample_blocks)
        )
        if embedding_dims is None:
            embedding_dims_tuple = expected_dims
        else:
            embedding_dims_tuple = tuple(int(value) for value in embedding_dims)
        source_sizes = tuple(int(value) for value in source_num_embeddings)
        if len(embedding_dims_tuple) != 2 or len(source_sizes) != 2:
            raise ValueError("embedding_dims and source_num_embeddings must have length two")
        if embedding_dims_tuple != expected_dims:
            raise ValueError(
                "embedding_dims must match SemanticEncoder output channels; expected "
                f"{expected_dims} for base_channels={self.base_channels}, got "
                f"{embedding_dims_tuple}"
            )
        if any(size < 2 for size in source_sizes):
            raise ValueError("source codebook sizes must be at least two")
        if commitment_cost < 0:
            raise ValueError("commitment_cost cannot be negative")

        strides_tuple = (
            tuple(int(value) for value in strides)
            if strides is not None
            else (2, 2)
        )
        if len(strides_tuple) != 2 or any(value < 1 for value in strides_tuple):
            raise ValueError("strides must contain two positive integers")
        if skip_dropout_p is not None and len(skip_dropout_p) < 1:
            raise ValueError("two-scale decoder skip_dropout_p must contain one value")

        self.embedding_dims = embedding_dims_tuple
        # Legacy-friendly aliases make checkpoint/trainer integration explicit.
        self.embedding_dim_list = list(embedding_dims_tuple)
        self.source_num_embeddings = source_sizes
        self.num_embeddings_list = list(source_sizes)
        self.strides = strides_tuple
        self.commitment_cost = float(commitment_cost)

        self.semantic_encoder = SemanticEncoder(
            int(in_channels),
            self.num_downsample_blocks,
            self.base_channels,
            strides=list(strides_tuple),
            norm_type=norm_type,
            num_groups=int(norm_groups),
            activation=activation,
            num_res_blocks=int(encoder_res_blocks),
            use_cascade_downsample=bool(use_cascade_downsample),
        )
        self.bottleneck_attention: nn.Module
        if use_bottleneck_attention:
            self.bottleneck_attention = BottleneckAttentionStack(
                embedding_dims_tuple[-1],
                num_blocks=int(bottleneck_attention_blocks),
                num_groups=int(norm_groups),
            )
        else:
            self.bottleneck_attention = nn.Identity()

        self.semantic_decoder = SemanticDecoder(
            list(embedding_dims_tuple),
            int(out_channels),
            up_mode=upsample_mode,
            skip_dropout_p=(list(skip_dropout_p) if skip_dropout_p is not None else None),
            upsample_scales=list(reversed(strides_tuple)),
            norm_type=norm_type,
            num_groups=int(norm_groups),
            activation=activation,
            num_res_blocks=int(decoder_res_blocks),
        )
        if use_swinir_enhance:
            self.swinir_enhance: nn.Module = SwinIREnhance(
                embed_dim=48,
                num_rstb=int(swinir_enhance_blocks),
                window_size=8,
                num_heads=4,
            )
        else:
            self.swinir_enhance = nn.Identity()

        # The student, not the RAQ generator, owns the two source codebooks.
        self.vector_quantizers = nn.ModuleList(
            [
                VectorQuantizer(source_size, embedding_dim, float(commitment_cost))
                for source_size, embedding_dim in zip(source_sizes, embedding_dims_tuple)
            ]
        )
        self.raq_generator = VariableRateRAQGenerator(
            embedding_dims_tuple,
            source_sizes,
            rate_embedding_dim=int(rate_embedding_dim),
            rate_hidden_dim=rate_hidden_dim,
            generator_model_dims=generator_model_dims,
            attention_dim=int(generator_attention_dim),
            transformer_depth=int(generator_transformer_depth),
            transformer_heads=int(generator_transformer_heads),
            feedforward_multiplier=float(generator_feedforward_multiplier),
            dropout=float(generator_dropout),
            minimum_target_size=int(minimum_target_size),
            allowed_target_sizes=allowed_target_sizes,
        )

        # Encoder output and decoder input each get their own identity adapter.
        self.encoder_rate_affines = nn.ModuleList(
            [ConditionalAffine(dim, int(rate_embedding_dim)) for dim in embedding_dims_tuple]
        )
        self.decoder_rate_affines = nn.ModuleList(
            [ConditionalAffine(dim, int(rate_embedding_dim)) for dim in embedding_dims_tuple]
        )

        snr_values = tuple(float(value) for value in snr_range_db)
        if len(snr_values) != 2 or snr_values[0] > snr_values[1]:
            raise ValueError("snr_range_db must be an ordered (minimum, maximum) pair")
        if not 0 < float(channel_coding_rate_train) <= 1:
            raise ValueError("channel_coding_rate_train must be in (0, 1]")
        if not 0 < float(channel_coding_rate_val) <= 1:
            raise ValueError("channel_coding_rate_val must be in (0, 1]")
        self.channel_coding_rate_train = float(channel_coding_rate_train)
        self.channel_coding_rate_val = float(channel_coding_rate_val)
        self.snr_range_db = snr_values
        self.channel = FiniteBlocklengthChannel(
            channel_coding_rate=self.channel_coding_rate_train,
            coded_block_length_bits=int(block_length),
            device=torch.device("cpu"),
        )
        self.channel_prob = 1.0
        self.set_channel_prob(channel_prob)
        self.source_codebooks_frozen = False
        if freeze_source_codebooks:
            self.freeze_source_codebooks()

        # Store only JSON/checkpoint-friendly constructor state.  Training code
        # can persist this under checkpoint["model_config"].
        self._model_config: Dict[str, Any] = {
            "in_channels": int(in_channels),
            "out_channels": int(out_channels),
            "num_downsample_blocks": self.num_downsample_blocks,
            "base_channels": self.base_channels,
            "source_num_embeddings": list(source_sizes),
            "embedding_dims": list(embedding_dims_tuple),
            "commitment_cost": float(commitment_cost),
            "strides": list(strides_tuple),
            "skip_dropout_p": (
                list(skip_dropout_p) if skip_dropout_p is not None else None
            ),
            "norm_type": str(norm_type),
            "norm_groups": int(norm_groups),
            "activation": str(activation),
            "encoder_res_blocks": int(encoder_res_blocks),
            "decoder_res_blocks": int(decoder_res_blocks),
            "upsample_mode": str(upsample_mode),
            "use_cascade_downsample": bool(use_cascade_downsample),
            "use_bottleneck_attention": bool(use_bottleneck_attention),
            "bottleneck_attention_blocks": int(bottleneck_attention_blocks),
            "use_swinir_enhance": bool(use_swinir_enhance),
            "swinir_enhance_blocks": int(swinir_enhance_blocks),
            "rate_embedding_dim": int(rate_embedding_dim),
            "rate_hidden_dim": rate_hidden_dim,
            "generator_model_dims": (
                list(generator_model_dims)
                if isinstance(generator_model_dims, (list, tuple))
                else generator_model_dims
            ),
            "generator_attention_dim": int(generator_attention_dim),
            "generator_transformer_depth": int(generator_transformer_depth),
            "generator_transformer_heads": int(generator_transformer_heads),
            "generator_feedforward_multiplier": float(generator_feedforward_multiplier),
            "generator_dropout": float(generator_dropout),
            "minimum_target_size": int(minimum_target_size),
            "allowed_target_sizes": (
                [
                    list(value) if isinstance(value, (list, tuple)) else int(value)
                    for value in allowed_target_sizes
                ]
                if allowed_target_sizes is not None
                else None
            ),
            "freeze_source_codebooks": bool(freeze_source_codebooks),
            "channel_coding_rate_train": self.channel_coding_rate_train,
            "channel_coding_rate_val": self.channel_coding_rate_val,
            "block_length": int(block_length),
            "snr_range_db": list(snr_values),
            "channel_prob": float(channel_prob),
        }

    @classmethod
    def from_config(cls, config: Union[Mapping[str, Any], object]) -> "VariableRateDeepSC":
        """Build from a lowercase mapping/dataclass or the legacy Config class.

        Unknown training-only fields are ignored.  This keeps the model entry
        point independent of the experiment runner while still accepting a
        future ``VariableRateConfig`` directly.
        """

        if isinstance(config, Mapping):
            values = dict(config)
        else:
            values = {
                name: getattr(config, name)
                for name in dir(config)
                if not name.startswith("_") and not callable(getattr(config, name))
            }
        aliases = {
            "IN_CHANNELS": "in_channels",
            "OUT_CHANNELS": "out_channels",
            "NUM_DOWNSAMPLE_BLOCKS": "num_downsample_blocks",
            "UNET_DEPTH": "num_downsample_blocks",
            "BASE_CHANNELS": "base_channels",
            "NUM_EMBEDDINGS_LIST": "source_num_embeddings",
            "SOURCE_NUM_EMBEDDINGS": "source_num_embeddings",
            "EMBEDDING_DIM_LIST": "embedding_dims",
            "EMBEDDING_DIMS": "embedding_dims",
            "COMMITMENT_COST": "commitment_cost",
            "DOWNSAMPLE_STRIDES": "strides",
            "SKIP_DROPOUT_P_INIT": "skip_dropout_p",
            "NORM_TYPE": "norm_type",
            "GROUP_NORM_GROUPS": "norm_groups",
            "ACTIVATION": "activation",
            "ENCODER_RES_BLOCKS": "encoder_res_blocks",
            "DECODER_RES_BLOCKS": "decoder_res_blocks",
            "UPSAMPLE_MODE": "upsample_mode",
            "USE_CASCADE_DOWNSAMPLE": "use_cascade_downsample",
            "USE_BOTTLENECK_ATTENTION": "use_bottleneck_attention",
            "BOTTLENECK_ATTENTION_BLOCKS": "bottleneck_attention_blocks",
            "USE_SWINIR_ENHANCE": "use_swinir_enhance",
            "SWINIR_ENHANCE_BLOCKS": "swinir_enhance_blocks",
            "RATE_EMBED_DIM": "rate_embedding_dim",
            "RATE_HIDDEN_DIM": "rate_hidden_dim",
            "RAQ_GENERATOR_MODEL_DIMS": "generator_model_dims",
            "RAQ_GENERATOR_ATTENTION_DIM": "generator_attention_dim",
            "RAQ_GENERATOR_FEEDFORWARD_MULTIPLIER": "generator_feedforward_multiplier",
            "RAQ_TRANSFORMER_DIM": "generator_model_dims",
            "RAQ_TRANSFORMER_HEADS": "generator_transformer_heads",
            "RAQ_TRANSFORMER_LAYERS": "generator_transformer_depth",
            "RAQ_TRANSFORMER_DROPOUT": "generator_dropout",
            "SUPPORTED_K_VALUES": "allowed_target_sizes",
            "CHANNEL_CODING_RATE_TRAIN": "channel_coding_rate_train",
            "CHANNEL_CODING_RATE_VAL": "channel_coding_rate_val",
            "BLOCK_LENGTH": "block_length",
            "SNR_RANGE_DB": "snr_range_db",
            # Lowercase legacy/config-dump spellings.
            "unet_depth": "num_downsample_blocks",
            "num_embeddings_list": "source_num_embeddings",
            "embedding_dim_list": "embedding_dims",
            "downsample_strides": "strides",
            "group_norm_groups": "norm_groups",
        }
        normalised: Dict[str, Any] = {}
        for key, value in values.items():
            target = aliases.get(key, key)
            # Explicit lowercase VariableRateConfig values win over aliases.
            if target not in normalised or key == target:
                normalised[target] = value
        parameters = inspect.signature(cls.__init__).parameters
        kwargs = {
            key: value
            for key, value in normalised.items()
            if key in parameters and key != "self"
        }
        return cls(**kwargs)

    def get_model_config(self) -> Dict[str, Any]:
        """Return a copy suitable for checkpoint metadata."""

        copied: Dict[str, Any] = {}
        for key, value in self._model_config.items():
            copied[key] = list(value) if isinstance(value, list) else value
        return copied

    @property
    def model_config(self) -> Dict[str, Any]:
        """Checkpoint-friendly constructor metadata (returned as a copy)."""

        return self.get_model_config()

    def export_constructor_config(self) -> Dict[str, Any]:
        """Explicit alias used by training/checkpoint pipelines."""

        return self.get_model_config()

    def set_channel_prob(self, probability: float) -> None:
        probability = float(probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("channel probability must be in [0, 1]")
        self.channel_prob = probability

    def freeze_source_codebooks(self) -> "VariableRateDeepSC":
        """Freeze both student-owned source SimVQ codebooks in-place."""

        for quantizer in self.vector_quantizers:
            for parameter in quantizer.parameters():
                parameter.requires_grad_(False)
        self.source_codebooks_frozen = True
        if hasattr(self, "_model_config"):
            self._model_config["freeze_source_codebooks"] = True
        return self

    def unfreeze_source_codebooks(self) -> "VariableRateDeepSC":
        """Unfreeze trainable source projections (base SimVQ embeddings stay frozen)."""

        for quantizer in self.vector_quantizers:
            for parameter in quantizer.codebook.proj.parameters():
                parameter.requires_grad_(True)
        self.source_codebooks_frozen = False
        if hasattr(self, "_model_config"):
            self._model_config["freeze_source_codebooks"] = False
        return self

    def source_codebooks(self) -> List[torch.Tensor]:
        return [quantizer.transformed_weight() for quantizer in self.vector_quantizers]

    @staticmethod
    def _sample_modulation_bits(snr_db: float) -> int:
        if snr_db < 4.0:
            return random.choice((1, 2))
        if snr_db < 8.0:
            return random.choice((1, 2, 4))
        return random.choice((2, 4))

    def _encode(self, images: torch.Tensor) -> List[torch.Tensor]:
        if images.ndim != 4:
            raise ValueError(
                f"images must have shape [B,C,H,W], got {tuple(images.shape)}"
            )
        features = list(self.semantic_encoder(images))
        if len(features) != 2:
            raise RuntimeError(f"semantic encoder returned {len(features)} scales, expected two")
        features[-1] = self.bottleneck_attention(features[-1])
        for layer_index, (feature, expected_dim) in enumerate(
            zip(features, self.embedding_dims)
        ):
            if feature.ndim != 4 or feature.shape[1] != expected_dim:
                raise RuntimeError(
                    f"encoder feature {layer_index} must be [B,{expected_dim},H,W], "
                    f"got {tuple(feature.shape)}"
                )
        return features

    def _decode(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != 2:
            raise ValueError(f"decoder needs two feature tensors, got {len(features)}")
        return self.swinir_enhance(self.semantic_decoder(list(features)))

    @staticmethod
    def _sum_vq_losses(vq_losses: Sequence[torch.Tensor]) -> torch.Tensor:
        if not vq_losses:
            raise RuntimeError("at least one VQ loss is required")
        return torch.stack([loss.reshape(()) for loss in vq_losses]).sum()

    def _channel_settings(
        self,
        reference: torch.Tensor,
        use_channel: bool,
        snr_db: Optional[float],
        channel_coding_rate: Optional[float],
        mod_bits: Optional[int],
    ) -> Tuple[bool, Optional[float], Optional[torch.Tensor], float, Optional[int]]:
        channel_active = bool(use_channel) and random.random() < self.channel_prob
        default_rate = (
            self.channel_coding_rate_train
            if self.training
            else self.channel_coding_rate_val
        )
        coding_rate = float(
            default_rate if channel_coding_rate is None else channel_coding_rate
        )
        if not 0 < coding_rate <= 1:
            raise ValueError("channel_coding_rate must be in (0, 1]")
        if not channel_active:
            return False, None, None, coding_rate, None
        selected_snr = (
            random.uniform(*self.snr_range_db) if snr_db is None else float(snr_db)
        )
        selected_mod_bits = (
            self._sample_modulation_bits(selected_snr)
            if mod_bits is None
            else int(mod_bits)
        )
        if selected_mod_bits < 1:
            raise ValueError("mod_bits must be positive")
        snr_tensor = reference.new_tensor(selected_snr)
        # FiniteBlocklengthChannel stores its construction device as a plain
        # attribute rather than a buffer; keep it aligned after model.to(...).
        self.channel.device = reference.device
        return True, selected_snr, snr_tensor, coding_rate, selected_mod_bits

    def _base_result(
        self,
        *,
        reconstruction: torch.Tensor,
        vq_losses: List[torch.Tensor],
        indices: List[torch.Tensor],
        raw_quantized_features: List[torch.Tensor],
        encoder_features: List[torch.Tensor],
        decoder_features: List[torch.Tensor],
        codebooks: List[torch.Tensor],
        source_codebooks: List[torch.Tensor],
        profile: Tuple[int, int],
        bypass_flags: List[bool],
        hierarchy_codebooks: List[Optional[torch.Tensor]],
        channel_used: bool,
        current_snr: Optional[float],
        bers: List[Optional[torch.Tensor]],
    ) -> Dict[str, Any]:
        vq_loss = self._sum_vq_losses(vq_losses)
        result: Dict[str, Any] = {
            "reconstructed_images": reconstruction,
            "reconstruction": reconstruction,
            "vq_losses": vq_losses,
            "vq_loss": vq_loss,
            "indices": indices,
            "raw_quantized_features": raw_quantized_features,
            "student_features": raw_quantized_features,
            "quantized_features": raw_quantized_features,
            "encoder_features": encoder_features,
            "decoder_features": decoder_features,
            "codebooks": codebooks,
            "source_codebooks": source_codebooks,
            "bypass_flags": bypass_flags,
            "hierarchy_codebooks": hierarchy_codebooks,
            "profile": profile,
            "raq_target_list": list(profile),
            "channel_used": channel_used,
            "current_snr": current_snr,
            "bers": bers,
            "channel_prob": self.channel_prob,
            # Compatibility aliases for existing monitoring/loss utilities.
            "W_trg_list": codebooks,
            "source_codebooks_list": source_codebooks,
            "z_q_raq_list": raw_quantized_features,
        }
        return result

    def forward_profile(
        self,
        images: torch.Tensor,
        profile: Profile,
        use_channel: bool = False,
        generate_hierarchy: bool = False,
        *,
        snr_db: Optional[float] = None,
        channel_coding_rate: Optional[float] = None,
        mod_bits: Optional[int] = None,
        return_attention: bool = False,
    ) -> Dict[str, Any]:
        """Run one atomic profile through the shared student.

        Reconstruction uses ``dual`` VQ gradient bridging unconditionally:
        nearest-neighbour decisions remain discrete, while reconstruction
        gradients reach the selected generated codewords and RAQ Transformer.
        """

        profile_tuple = self.raq_generator.validate_profile(profile)
        source_codebooks = self.source_codebooks()
        generation = self.raq_generator(
            source_codebooks,
            profile_tuple,
            generate_hierarchy=bool(generate_hierarchy),
            return_attention=bool(return_attention),
        )
        rate_embedding = generation["rate_embedding"]
        if not isinstance(rate_embedding, torch.Tensor):
            raise RuntimeError("RAQ generator did not return a rate embedding tensor")

        unconditioned_features = self._encode(images)
        encoder_features = [
            affine(feature, rate_embedding)
            for affine, feature in zip(self.encoder_rate_affines, unconditioned_features)
        ]
        codebooks = generation["codebooks"]
        if not isinstance(codebooks, list) or len(codebooks) != 2:
            raise RuntimeError("RAQ generator must return two target codebooks")

        channel_active, selected_snr, snr_tensor, coding_rate, selected_mod_bits = (
            self._channel_settings(
                images,
                use_channel,
                snr_db,
                channel_coding_rate,
                mod_bits,
            )
        )

        vq_losses: List[torch.Tensor] = []
        indices: List[torch.Tensor] = []
        raw_quantized_features: List[torch.Tensor] = []
        decoder_quantized_features: List[torch.Tensor] = []
        bers: List[Optional[torch.Tensor]] = []
        for layer_index, (feature, codebook, quantizer, target_size) in enumerate(
            zip(encoder_features, codebooks, self.vector_quantizers, profile_tuple)
        ):
            if not isinstance(codebook, torch.Tensor):
                raise RuntimeError(f"target codebook {layer_index} is not a tensor")
            vq_loss, quantized_clean, encoding_indices, quantized_raw = (
                quantizer.forward_raq(
                    feature,
                    codebook,
                    return_raw=True,
                    recon_grad_mode="dual",
                )
            )
            vq_losses.append(vq_loss)
            indices.append(encoding_indices)
            raw_quantized_features.append(quantized_raw)

            if channel_active:
                if snr_tensor is None or selected_mod_bits is None:
                    raise RuntimeError("active channel is missing sampled settings")
                corrupted_indices, ber = self.channel.apply_channel_noise(
                    encoding_indices,
                    int(target_size),
                    snr_tensor,
                    rc=coding_rate,
                    mod_bits=selected_mod_bits,
                )
                noisy_raw = quantizer.get_quantized_features(
                    corrupted_indices,
                    output_spatial_size=feature.shape[-2:],
                    codebook_weight=codebook,
                )
                # Numerically this is noisy_raw.  The clean dual path supplies
                # gradients to encoder, selected codewords, and generator.
                quantized_for_decoder = quantized_clean + (
                    noisy_raw - quantized_clean
                ).detach()
                bers.append(ber)
            else:
                quantized_for_decoder = quantized_clean
                bers.append(None)
            decoder_quantized_features.append(quantized_for_decoder)

        decoder_features = [
            affine(feature, rate_embedding)
            for affine, feature in zip(
                self.decoder_rate_affines, decoder_quantized_features
            )
        ]
        reconstruction = self._decode(decoder_features)
        hierarchy_codebooks = generation["hierarchy_codebooks"]
        if not isinstance(hierarchy_codebooks, list):
            raise RuntimeError("RAQ hierarchy_codebooks must be a list")
        bypass_flags = generation["bypass_flags"]
        if not isinstance(bypass_flags, list):
            raise RuntimeError("RAQ bypass_flags must be a list")

        result = self._base_result(
            reconstruction=reconstruction,
            vq_losses=vq_losses,
            indices=indices,
            raw_quantized_features=raw_quantized_features,
            encoder_features=encoder_features,
            decoder_features=decoder_features,
            codebooks=codebooks,
            source_codebooks=source_codebooks,
            profile=profile_tuple,
            bypass_flags=bypass_flags,
            hierarchy_codebooks=hierarchy_codebooks,
            channel_used=channel_active,
            current_snr=selected_snr,
            bers=bers,
        )
        # Preserve generator internals required by identity/hierarchy losses and
        # diagnostics without duplicating the large codebook tensors.
        for key in (
            "raw_log_rates",
            "rate_embedding",
            "aggregation_codebooks",
            "residual_codebooks",
            "attention_weights",
            "hierarchy_aggregation_codebooks",
            "hierarchy_residual_codebooks",
            "hierarchy_profiles",
            "hierarchy_bypass_flags",
        ):
            result[key] = generation[key]
        return result

    def forward_src(
        self,
        images: torch.Tensor,
        profile: Profile = (2048, 2048),
    ) -> Dict[str, Any]:
        """Clean source identity path with no RAQ generation and no FiLM."""

        profile_tuple = self.raq_generator.validate_profile(profile)
        if profile_tuple != self.source_num_embeddings:
            raise ValueError(
                "forward_src is the maximum source-codebook identity path; expected "
                f"{self.source_num_embeddings}, got {profile_tuple}"
            )
        encoder_features = self._encode(images)
        source_codebooks = self.source_codebooks()
        vq_losses: List[torch.Tensor] = []
        indices: List[torch.Tensor] = []
        raw_quantized_features: List[torch.Tensor] = []
        quantized_features: List[torch.Tensor] = []
        for feature, quantizer in zip(encoder_features, self.vector_quantizers):
            vq_loss, quantized, encoding_indices, quantized_raw = quantizer(
                feature, return_raw=True
            )
            vq_losses.append(vq_loss)
            indices.append(encoding_indices)
            raw_quantized_features.append(quantized_raw)
            quantized_features.append(quantized)
        reconstruction = self._decode(quantized_features)
        result = self._base_result(
            reconstruction=reconstruction,
            vq_losses=vq_losses,
            indices=indices,
            raw_quantized_features=raw_quantized_features,
            encoder_features=encoder_features,
            decoder_features=quantized_features,
            codebooks=source_codebooks,
            source_codebooks=source_codebooks,
            profile=profile_tuple,
            bypass_flags=[True, True],
            hierarchy_codebooks=[None, None],
            channel_used=False,
            current_snr=None,
            bers=[None, None],
        )
        result.update(
            {
                "rate_embedding": None,
                "raw_log_rates": self.raq_generator.rate_conditioner.raw_log_rates(
                    profile_tuple
                ),
                "aggregation_codebooks": source_codebooks,
                "residual_codebooks": [
                    torch.zeros_like(codebook) for codebook in source_codebooks
                ],
                "attention_weights": [None, None],
                "hierarchy_aggregation_codebooks": [None, None],
                "hierarchy_residual_codebooks": [None, None],
                "hierarchy_profiles": [None, None],
                "hierarchy_bypass_flags": [None, None],
                "source_identity": True,
            }
        )
        return result

    def forward(
        self,
        images: torch.Tensor,
        profile: Profile = (2048, 2048),
        use_channel: bool = False,
        generate_hierarchy: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self.forward_profile(
            images,
            profile,
            use_channel=use_channel,
            generate_hierarchy=generate_hierarchy,
            **kwargs,
        )


__all__ = ["ConditionalAffine", "FiLM", "VariableRateDeepSC"]
