"""Losses for single-teacher, two-layer variable-rate RAQ.

The image target remains the primary supervision.  Teacher output/feature terms
are auxiliary and become stronger monotonically with normalized log-rate.
Sampled diversity indexes only ``P`` pairs, never a ``K x K`` distance matrix.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deepsc_loss import ms_ssim_loss


Tensor = torch.Tensor
Profile = tuple[int, int]


class LPIPSReconstructionLoss(nn.Module):
    """Differentiable learned perceptual loss for optional training use.

    The dependency and VGG weights are loaded only when the configured LPIPS
    weight is non-zero.  Inputs already use the native LPIPS ``[-1, 1]`` range.
    """

    def __init__(self) -> None:
        super().__init__()
        try:
            import lpips
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "SIMVQ_RAQ_RECON_LPIPS_WEIGHT>0 requires the lpips package"
            ) from exc
        self.metric = lpips.LPIPS(net="vgg").eval()
        for parameter in self.metric.parameters():
            parameter.requires_grad_(False)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return self.metric(prediction.clamp(-1, 1), target.clamp(-1, 1)).mean()


def _as_profile(profile: Sequence[int]) -> Profile:
    if len(profile) != 2:
        raise ValueError(f"profile must contain exactly two K values, got {profile!r}")
    result = int(profile[0]), int(profile[1])
    if any(k < 2 or k & (k - 1) for k in result):
        raise ValueError(f"profile K values must be powers of two >= 2, got {result}")
    return result


def normalized_log_rate(
    profile: Sequence[int],
    *,
    min_k: int = 2,
    max_k: int = 2048,
) -> float:
    """Mean normalized log2-rate in ``[0, 1]`` for a two-layer profile."""

    k0, k1 = _as_profile(profile)
    if min_k < 2 or max_k <= min_k:
        raise ValueError("expected 2 <= min_k < max_k")
    if k0 < min_k or k1 < min_k or k0 > max_k or k1 > max_k:
        raise ValueError(f"profile {(k0, k1)} is outside [{min_k}, {max_k}]")
    min_bits = math.log2(min_k)
    bit_span = math.log2(max_k) - min_bits
    return (
        (math.log2(k0) - min_bits) + (math.log2(k1) - min_bits)
    ) / (2.0 * bit_span)


def rate_aware_weight(
    profile: Sequence[int],
    low: float,
    high: float,
    gamma: float,
    *,
    min_k: int = 2,
    max_k: int = 2048,
) -> float:
    """``low + (high-low) * normalized_log_rate(profile)^gamma``."""

    if low < 0 or high < 0 or low > high:
        raise ValueError("rate-aware weights require 0 <= low <= high")
    if gamma <= 0:
        raise ValueError("rate-aware gamma must be positive")
    score = normalized_log_rate(profile, min_k=min_k, max_k=max_k)
    return float(low + (high - low) * (score**gamma))


def merge_adjacent_codewords(codebook: Tensor) -> Tensor:
    """Merge consecutive pairs without assuming a first-K nesting relation."""

    if codebook.ndim != 2:
        raise ValueError(f"codebook must be [K,D], got {tuple(codebook.shape)}")
    if codebook.shape[0] % 2 != 0:
        raise ValueError("the larger hierarchy codebook must contain an even K")
    return codebook.reshape(codebook.shape[0] // 2, 2, codebook.shape[1]).mean(dim=1)


def sampled_margin_diversity_loss(
    codebook: Tensor,
    *,
    margin: float,
    num_pairs: int,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, int]:
    """Margin loss over independently sampled distinct pairs in ``O(PD)`` memory.

    Distances are RMS-normalized by embedding dimension, which keeps one margin
    meaningful for layers with different D. Sampling with replacement makes the
    allocated index/feature storage depend on ``num_pairs``, not ``K**2``.
    """

    if codebook.ndim != 2:
        raise ValueError(f"codebook must be [K,D], got {tuple(codebook.shape)}")
    if margin < 0 or num_pairs < 0:
        raise ValueError("margin and num_pairs must be non-negative")
    k, embedding_dim = codebook.shape
    actual_pairs = int(num_pairs) if k >= 2 and num_pairs > 0 else 0
    if actual_pairs == 0:
        return codebook.new_zeros(()), 0

    first = torch.randint(
        0,
        k,
        (actual_pairs,),
        device=codebook.device,
        generator=generator,
    )
    # Draw from K-1 values then skip the first index. This guarantees distinct
    # pairs without materializing a pairwise mask or K-by-K tensor.
    second = torch.randint(
        0,
        k - 1,
        (actual_pairs,),
        device=codebook.device,
        generator=generator,
    )
    second = second + (second >= first).to(second.dtype)
    delta = codebook[first] - codebook[second]
    distance = torch.linalg.vector_norm(delta, dim=-1) / math.sqrt(max(1, embedding_dim))
    loss = F.relu(codebook.new_tensor(margin) - distance).square().mean()
    return loss, actual_pairs


class VariableRateRAQLoss(nn.Module):
    """Independent composite loss for one atomic ``(K0, K1)`` profile."""

    def __init__(
        self,
        *,
        layer_vq_weights: Sequence[float] = (0.25, 0.5),
        feature_layer_weights: Sequence[float] = (1.0, 1.0),
        mse_weight: float = 1.0,
        ms_ssim_weight: float = 0.0,
        lpips_weight: float = 0.0,
        vq_weight: float = 1.0,
        output_distill_low: float = 0.02,
        output_distill_high: float = 0.20,
        output_distill_gamma: float = 2.0,
        feature_distill_low: float = 0.01,
        feature_distill_high: float = 0.10,
        feature_distill_gamma: float = 2.0,
        identity_weight: float = 1.0,
        hierarchy_weight: float = 0.05,
        diversity_weight: float = 0.01,
        diversity_margin: float = 0.5,
        diversity_num_pairs: int = 4096,
        min_k: int = 2,
        max_k: int = 2048,
        diversity_generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.layer_vq_weights = tuple(float(weight) for weight in layer_vq_weights)
        self.feature_layer_weights = tuple(float(weight) for weight in feature_layer_weights)
        if len(self.layer_vq_weights) != 2 or len(self.feature_layer_weights) != 2:
            raise ValueError("VQ and feature layer weights must each contain two entries")

        scalar_weights = {
            "mse_weight": mse_weight,
            "ms_ssim_weight": ms_ssim_weight,
            "lpips_weight": lpips_weight,
            "vq_weight": vq_weight,
            "identity_weight": identity_weight,
            "hierarchy_weight": hierarchy_weight,
            "diversity_weight": diversity_weight,
            "diversity_margin": diversity_margin,
        }
        for name, value in scalar_weights.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if any(weight < 0 for weight in self.layer_vq_weights):
            raise ValueError("layer_vq_weights must be non-negative")
        if any(weight < 0 for weight in self.feature_layer_weights):
            raise ValueError("feature_layer_weights must be non-negative")
        if diversity_num_pairs < 0:
            raise ValueError("diversity_num_pairs must be non-negative")

        # Validate the rate schedules once; forward only evaluates them.
        rate_aware_weight((min_k, min_k), output_distill_low, output_distill_high, output_distill_gamma, min_k=min_k, max_k=max_k)
        rate_aware_weight((min_k, min_k), feature_distill_low, feature_distill_high, feature_distill_gamma, min_k=min_k, max_k=max_k)

        self.mse_weight = float(mse_weight)
        self.ms_ssim_weight = float(ms_ssim_weight)
        self.lpips_weight = float(lpips_weight)
        self.vq_weight = float(vq_weight)
        self.output_distill_low = float(output_distill_low)
        self.output_distill_high = float(output_distill_high)
        self.output_distill_gamma = float(output_distill_gamma)
        self.feature_distill_low = float(feature_distill_low)
        self.feature_distill_high = float(feature_distill_high)
        self.feature_distill_gamma = float(feature_distill_gamma)
        self.identity_weight = float(identity_weight)
        self.hierarchy_weight = float(hierarchy_weight)
        self.diversity_weight = float(diversity_weight)
        self.diversity_margin = float(diversity_margin)
        self.diversity_num_pairs = int(diversity_num_pairs)
        self.min_k = int(min_k)
        self.max_k = int(max_k)
        self.diversity_generator = diversity_generator
        self.lpips_loss = LPIPSReconstructionLoss() if self.lpips_weight > 0 else None

    @classmethod
    def from_config(cls, config: object) -> "VariableRateRAQLoss":
        """Build from :class:`config_variable_rate.VariableRateConfig` names."""

        return cls(
            layer_vq_weights=getattr(config, "RAQ_LAYER_VQ_WEIGHTS"),
            feature_layer_weights=getattr(config, "FEATURE_LAYER_WEIGHTS"),
            mse_weight=getattr(config, "MSE_LOSS_WEIGHT"),
            ms_ssim_weight=getattr(config, "MS_SSIM_LOSS_WEIGHT"),
            lpips_weight=getattr(config, "LPIPS_LOSS_WEIGHT"),
            vq_weight=getattr(config, "RAQ_VQ_WEIGHT"),
            output_distill_low=getattr(config, "OUTPUT_DISTILL_WEIGHT_LOW"),
            output_distill_high=getattr(config, "OUTPUT_DISTILL_WEIGHT_HIGH"),
            output_distill_gamma=getattr(config, "OUTPUT_DISTILL_GAMMA"),
            feature_distill_low=getattr(config, "FEATURE_DISTILL_WEIGHT_LOW"),
            feature_distill_high=getattr(config, "FEATURE_DISTILL_WEIGHT_HIGH"),
            feature_distill_gamma=getattr(config, "FEATURE_DISTILL_GAMMA"),
            identity_weight=getattr(config, "IDENTITY_WEIGHT"),
            hierarchy_weight=getattr(config, "HIERARCHY_WEIGHT"),
            diversity_weight=getattr(config, "DIVERSITY_WEIGHT"),
            diversity_margin=getattr(config, "DIVERSITY_MARGIN"),
            diversity_num_pairs=getattr(config, "DIVERSITY_NUM_PAIRS"),
            min_k=min(getattr(config, "SUPPORTED_K_VALUES")),
            max_k=max(getattr(config, "SUPPORTED_K_VALUES")),
        )

    def _image_loss(self, prediction: Tensor, target: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if prediction.shape != target.shape:
            raise ValueError(
                f"image shapes must match, got {tuple(prediction.shape)} and {tuple(target.shape)}"
            )
        target = target.to(prediction.device, non_blocking=True)
        mse = F.mse_loss(prediction, target)
        zero = prediction.new_zeros(())
        ms_ssim = ms_ssim_loss(prediction, target) if self.ms_ssim_weight > 0 else zero
        perceptual = (
            self.lpips_loss(prediction, target)
            if self.lpips_loss is not None
            else zero
        )
        total = (
            self.mse_weight * mse
            + self.ms_ssim_weight * ms_ssim
            + self.lpips_weight * perceptual
        )
        return total, {
            "mse": mse,
            "ms_ssim_loss": ms_ssim,
            "perceptual_loss": perceptual,
        }

    def _vq_loss(self, vq_losses: Sequence[Tensor] | Tensor | None, device: torch.device) -> Tensor:
        if vq_losses is None:
            return torch.zeros((), device=device)
        if torch.is_tensor(vq_losses):
            return vq_losses.to(device)
        if len(vq_losses) != len(self.layer_vq_weights):
            raise ValueError(
                f"expected {len(self.layer_vq_weights)} VQ losses, got {len(vq_losses)}"
            )
        total = torch.zeros((), device=device)
        for weight, layer_loss in zip(self.layer_vq_weights, vq_losses):
            if not torch.is_tensor(layer_loss) or layer_loss.numel() != 1:
                raise ValueError("each VQ loss must be a scalar tensor")
            total = total + weight * layer_loss.to(device, non_blocking=True)
        return total

    def _feature_loss(
        self,
        student_features: Sequence[Tensor] | None,
        teacher_features: Sequence[Tensor] | None,
        device: torch.device,
    ) -> tuple[Tensor, list[Tensor]]:
        if student_features is None or teacher_features is None:
            raise ValueError(
                "active feature distillation requires student and frozen-teacher features"
            )
        if len(student_features) != 2 or len(teacher_features) != 2:
            raise ValueError("feature distillation requires two student and teacher layers")
        if sum(self.feature_layer_weights) <= 0:
            zero = torch.zeros((), device=device)
            return zero, [zero, zero]

        layer_losses: list[Tensor] = []
        weighted = torch.zeros((), device=device)
        for weight, student, teacher in zip(
            self.feature_layer_weights, student_features, teacher_features
        ):
            teacher_target = teacher.detach().to(device, non_blocking=True)
            student_prediction = student.to(device, non_blocking=True)
            if student_prediction.shape != teacher_target.shape:
                raise ValueError(
                    "student/teacher feature shapes must match, got "
                    f"{tuple(student_prediction.shape)} and {tuple(teacher_target.shape)}"
                )
            layer_loss = F.mse_loss(student_prediction, teacher_target)
            layer_losses.append(layer_loss)
            weighted = weighted + weight * layer_loss
        return weighted / sum(self.feature_layer_weights), layer_losses

    def _identity_loss(
        self,
        profile: Profile,
        raq_codebooks: Sequence[Tensor] | None,
        source_codebooks: Sequence[Tensor] | None,
        device: torch.device,
    ) -> tuple[Tensor, int]:
        active_layers = sum(k == self.max_k for k in profile)
        if active_layers == 0:
            return torch.zeros((), device=device), 0
        if raq_codebooks is None or source_codebooks is None:
            raise ValueError("max-K identity protection requires RAQ and source codebooks")
        if len(raq_codebooks) != 2 or len(source_codebooks) != 2:
            raise ValueError("identity protection requires two codebooks")

        total = torch.zeros((), device=device)
        for layer, k in enumerate(profile):
            if k != self.max_k:
                continue
            raq = raq_codebooks[layer].to(device, non_blocking=True)
            source = source_codebooks[layer].detach().to(device, non_blocking=True)
            if raq.shape != source.shape or raq.shape[0] != self.max_k:
                raise ValueError(
                    f"max-K layer {layer} must directly expose a [{self.max_k},D] source codebook"
                )
            total = total + F.mse_loss(raq, source)
        return total / active_layers, active_layers

    def _hierarchy_loss(
        self,
        profile: Profile,
        raq_codebooks: Sequence[Tensor] | None,
        hierarchy_codebooks: Sequence[Tensor | None] | None,
        device: torch.device,
    ) -> tuple[Tensor, int]:
        active_layers = sum(k < self.max_k for k in profile)
        if active_layers == 0:
            return torch.zeros((), device=device), 0
        if raq_codebooks is None or hierarchy_codebooks is None:
            raise ValueError(
                "active hierarchy loss requires W_K and corresponding W_2K codebooks"
            )
        if len(raq_codebooks) != 2 or len(hierarchy_codebooks) != 2:
            raise ValueError("hierarchy loss requires two codebook slots")

        total = torch.zeros((), device=device)
        used = 0
        for layer, k in enumerate(profile):
            if k >= self.max_k:
                continue
            small = raq_codebooks[layer].to(device, non_blocking=True)
            larger = hierarchy_codebooks[layer]
            if larger is None:
                raise ValueError(f"missing hierarchy W_2K codebook for layer {layer}, K={k}")
            larger = larger.detach().to(device, non_blocking=True)
            if small.ndim != 2 or small.shape[0] != k:
                raise ValueError(f"layer {layer} W_K must have shape [{k},D]")
            if larger.ndim != 2 or larger.shape != (2 * k, small.shape[1]):
                raise ValueError(
                    f"layer {layer} hierarchy parent must have shape "
                    f"[{2 * k},{small.shape[1]}], got {tuple(larger.shape)}"
                )
            total = total + F.mse_loss(small, merge_adjacent_codewords(larger))
            used += 1
        return total / max(1, used), used

    def _diversity_loss(
        self,
        profile: Profile,
        raq_codebooks: Sequence[Tensor] | None,
        device: torch.device,
    ) -> tuple[Tensor, int, int]:
        generated_layers = sum(k < self.max_k for k in profile)
        if generated_layers == 0 or self.diversity_num_pairs == 0:
            return torch.zeros((), device=device), 0, 0
        if raq_codebooks is None or len(raq_codebooks) != 2:
            raise ValueError("active diversity loss requires two RAQ codebooks")

        total = torch.zeros((), device=device)
        actual_pairs = 0
        used_layers = 0
        for layer, k in enumerate(profile):
            if k >= self.max_k:
                continue
            codebook = raq_codebooks[layer].to(device, non_blocking=True)
            if codebook.ndim != 2 or codebook.shape[0] != k:
                raise ValueError(f"layer {layer} RAQ codebook must have shape [{k},D]")
            layer_loss, layer_pairs = sampled_margin_diversity_loss(
                codebook,
                margin=self.diversity_margin,
                num_pairs=self.diversity_num_pairs,
                generator=self.diversity_generator,
            )
            total = total + layer_loss
            actual_pairs += layer_pairs
            used_layers += 1
        return total / max(1, used_layers), actual_pairs, used_layers

    def forward(
        self,
        *,
        target: Tensor,
        reconstruction: Tensor,
        profile: Sequence[int],
        vq_losses: Sequence[Tensor] | Tensor | None = None,
        teacher_reconstruction: Tensor | None = None,
        student_features: Sequence[Tensor] | None = None,
        teacher_features: Sequence[Tensor] | None = None,
        raq_codebooks: Sequence[Tensor] | None = None,
        source_codebooks: Sequence[Tensor] | None = None,
        hierarchy_codebooks: Sequence[Tensor | None] | None = None,
        return_details: bool = True,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Compute one profile loss; the trainer averages across sandwich profiles."""

        parsed_profile = _as_profile(profile)
        rate_score = normalized_log_rate(
            parsed_profile, min_k=self.min_k, max_k=self.max_k
        )
        output_weight = rate_aware_weight(
            parsed_profile,
            self.output_distill_low,
            self.output_distill_high,
            self.output_distill_gamma,
            min_k=self.min_k,
            max_k=self.max_k,
        )
        feature_weight = rate_aware_weight(
            parsed_profile,
            self.feature_distill_low,
            self.feature_distill_high,
            self.feature_distill_gamma,
            min_k=self.min_k,
            max_k=self.max_k,
        )

        reconstruction_loss, reconstruction_parts = self._image_loss(reconstruction, target)
        device = reconstruction_loss.device
        vq_raw = self._vq_loss(vq_losses, device)

        if output_weight > 0:
            if teacher_reconstruction is None:
                raise ValueError(
                    "active output distillation requires the frozen teacher reconstruction"
                )
            output_distill_raw, _ = self._image_loss(
                reconstruction,
                teacher_reconstruction.detach(),
            )
        else:
            output_distill_raw = torch.zeros((), device=device)

        if feature_weight > 0:
            feature_distill_raw, feature_layer_losses = self._feature_loss(
                student_features, teacher_features, device
            )
        else:
            feature_distill_raw = torch.zeros((), device=device)
            feature_layer_losses = [torch.zeros((), device=device) for _ in range(2)]

        if self.identity_weight > 0:
            identity_raw, identity_layers = self._identity_loss(
                parsed_profile, raq_codebooks, source_codebooks, device
            )
        else:
            identity_raw, identity_layers = torch.zeros((), device=device), 0

        if self.hierarchy_weight > 0:
            hierarchy_raw, hierarchy_layers = self._hierarchy_loss(
                parsed_profile, raq_codebooks, hierarchy_codebooks, device
            )
        else:
            hierarchy_raw, hierarchy_layers = torch.zeros((), device=device), 0

        if self.diversity_weight > 0:
            diversity_raw, diversity_pairs, diversity_layers = self._diversity_loss(
                parsed_profile, raq_codebooks, device
            )
        else:
            diversity_raw, diversity_pairs, diversity_layers = (
                torch.zeros((), device=device),
                0,
                0,
            )

        vq_term = self.vq_weight * vq_raw
        output_term = output_weight * output_distill_raw
        feature_term = feature_weight * feature_distill_raw
        identity_term = self.identity_weight * identity_raw
        hierarchy_term = self.hierarchy_weight * hierarchy_raw
        diversity_term = self.diversity_weight * diversity_raw
        total = (
            reconstruction_loss
            + vq_term
            + output_term
            + feature_term
            + identity_term
            + hierarchy_term
            + diversity_term
        )

        if not return_details:
            return total
        details = {
            "total_loss": total.detach(),
            "reconstruction_loss": reconstruction_loss.detach(),
            "reconstruction_mse": reconstruction_parts["mse"].detach(),
            "reconstruction_ms_ssim_loss": reconstruction_parts["ms_ssim_loss"].detach(),
            "reconstruction_perceptual_loss": reconstruction_parts["perceptual_loss"].detach(),
            "vq_raw_loss": vq_raw.detach(),
            "vq_loss": vq_term.detach(),
            "output_distill_raw_loss": output_distill_raw.detach(),
            "output_distill_loss": output_term.detach(),
            "output_distill_weight": torch.tensor(output_weight, device=device),
            "feature_distill_raw_loss": feature_distill_raw.detach(),
            "feature_distill_loss": feature_term.detach(),
            "feature_distill_weight": torch.tensor(feature_weight, device=device),
            "feature_distill_layer0_loss": feature_layer_losses[0].detach(),
            "feature_distill_layer1_loss": feature_layer_losses[1].detach(),
            "identity_raw_loss": identity_raw.detach(),
            "identity_loss": identity_term.detach(),
            "identity_layer_count": torch.tensor(identity_layers, device=device),
            "hierarchy_raw_loss": hierarchy_raw.detach(),
            "hierarchy_loss": hierarchy_term.detach(),
            "hierarchy_layer_count": torch.tensor(hierarchy_layers, device=device),
            "diversity_raw_loss": diversity_raw.detach(),
            "diversity_loss": diversity_term.detach(),
            "diversity_pair_count": torch.tensor(diversity_pairs, device=device),
            "diversity_layer_count": torch.tensor(diversity_layers, device=device),
            "rate_score": torch.tensor(rate_score, device=device),
            "profile_k0": torch.tensor(parsed_profile[0], device=device),
            "profile_k1": torch.tensor(parsed_profile[1], device=device),
        }
        return total, details

    def forward_from_outputs(
        self,
        target: Tensor,
        student_output: Mapping[str, object],
        teacher_output: Mapping[str, object],
        *,
        return_details: bool = True,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Adapter for ``VariableRateDeepSC.forward_profile`` output mappings."""

        reconstruction = student_output.get(
            "reconstructed_images", student_output.get("reconstruction")
        )
        teacher_reconstruction = teacher_output.get(
            "reconstructed_images", teacher_output.get("reconstruction")
        )
        if not torch.is_tensor(reconstruction):
            raise ValueError("student output is missing reconstructed_images")
        if not torch.is_tensor(teacher_reconstruction):
            raise ValueError("teacher output is missing reconstructed_images")
        profile = student_output.get("profile")
        if not isinstance(profile, Sequence):
            raise ValueError("student output is missing its atomic profile")

        student_features = student_output.get(
            "student_features", student_output.get("raw_quantized_features")
        )
        teacher_features = teacher_output.get(
            "student_features", teacher_output.get("raw_quantized_features")
        )
        return self.forward(
            target=target,
            reconstruction=reconstruction,
            profile=profile,
            vq_losses=student_output.get("vq_losses"),
            teacher_reconstruction=teacher_reconstruction,
            student_features=student_features if isinstance(student_features, Sequence) else None,
            teacher_features=teacher_features if isinstance(teacher_features, Sequence) else None,
            raq_codebooks=(
                student_output.get("codebooks")
                if isinstance(student_output.get("codebooks"), Sequence)
                else None
            ),
            source_codebooks=(
                student_output.get("source_codebooks")
                if isinstance(student_output.get("source_codebooks"), Sequence)
                else None
            ),
            hierarchy_codebooks=(
                student_output.get("hierarchy_codebooks")
                if isinstance(student_output.get("hierarchy_codebooks"), Sequence)
                else None
            ),
            return_details=return_details,
        )


__all__ = [
    "LPIPSReconstructionLoss",
    "VariableRateRAQLoss",
    "merge_adjacent_codewords",
    "normalized_log_rate",
    "rate_aware_weight",
    "sampled_margin_diversity_loss",
]
