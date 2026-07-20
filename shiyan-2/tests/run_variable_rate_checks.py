#!/usr/bin/env python3
"""Standalone required checks that do not depend on pytest being installed."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_variable_rate import ALL_PROFILES, SUPPORTED_K_VALUES
from losses.variable_rate_raq_loss import (
    VariableRateRAQLoss,
    sampled_margin_diversity_loss,
)
from models.deepsc import DeepSC
from models.variable_rate_deepsc import VariableRateDeepSC
from training.frozen_teacher import assert_teacher_has_no_grad
from training.profile_sampler import ProfileSampler


def build_actual_k_model() -> VariableRateDeepSC:
    return VariableRateDeepSC(
        in_channels=3,
        out_channels=3,
        num_downsample_blocks=2,
        base_channels=4,
        source_num_embeddings=(2048, 2048),
        embedding_dims=(8, 16),
        commitment_cost=0.25,
        strides=(2, 2),
        norm_type="group",
        norm_groups=4,
        activation="silu",
        encoder_res_blocks=1,
        decoder_res_blocks=1,
        use_cascade_downsample=False,
        use_bottleneck_attention=False,
        rate_embedding_dim=16,
        rate_hidden_dim=16,
        generator_model_dims=16,
        generator_attention_dim=8,
        generator_transformer_depth=1,
        generator_transformer_heads=4,
        generator_dropout=0.0,
        channel_prob=0.0,
    )


def check_identity_mixed_shapes_and_gradients() -> None:
    torch.manual_seed(5)
    model = build_actual_k_model()
    model.freeze_source_codebooks()
    images = torch.randn(2, 3, 16, 16)

    model.eval()
    with torch.no_grad():
        source = model.forward_src(images)
        maximum = model.forward_profile(images, (2048, 2048), use_channel=False)
    assert maximum["bypass_flags"] == [True, True]
    assert maximum["codebooks"][0] is maximum["source_codebooks"][0]
    assert maximum["codebooks"][1] is maximum["source_codebooks"][1]
    assert torch.equal(maximum["reconstructed_images"], source["reconstructed_images"])

    expected = {
        (2048, 16): ([True, False], [(2048, 8), (16, 16)]),
        (16, 2048): ([False, True], [(16, 8), (2048, 16)]),
        (16, 2): ([False, False], [(16, 8), (2, 16)]),
    }
    with torch.no_grad():
        for profile, (flags, shapes) in expected.items():
            output = model.forward_profile(
                images, profile, use_channel=False, generate_hierarchy=True
            )
            assert output["bypass_flags"] == flags
            assert [tuple(codebook.shape) for codebook in output["codebooks"]] == shapes
            assert tuple(output["reconstructed_images"].shape) == tuple(images.shape)

    model.train()
    model.zero_grad(set_to_none=True)
    low_output = model.forward_profile(images, (16, 2), use_channel=False)
    low_output["reconstructed_images"].square().mean().backward()
    for layer, generator in enumerate(model.raq_generator.layer_generators):
        magnitude = sum(
            float(parameter.grad.abs().sum())
            for parameter in generator.parameters()
            if parameter.grad is not None
        )
        assert magnitude > 0.0, f"layer {layer} has no dual reconstruction gradient"
    for quantizer in model.vector_quantizers:
        assert all(parameter.grad is None for parameter in quantizer.parameters())

    fresh = build_actual_k_model().train()
    fresh.zero_grad(set_to_none=True)
    max_output = fresh.forward_profile(images, (2048, 2048), use_channel=False)
    max_output["reconstructed_images"].square().mean().backward()
    for generator in fresh.raq_generator.layer_generators:
        assert all(parameter.grad is None for parameter in generator.parameters())

    frozen_teacher = copy.deepcopy(fresh).eval()
    for parameter in frozen_teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    with torch.no_grad():
        frozen_teacher.forward_src(images)
    assert_teacher_has_no_grad(frozen_teacher)


def check_atomic_sampler_coverage_and_restore() -> None:
    assert len(ALL_PROFILES) == 121
    sampler = ProfileSampler(
        ALL_PROFILES,
        num_random=1,
        min_profile=(2, 2),
        max_profile=(2048, 2048),
        supported_k=SUPPORTED_K_VALUES,
        seed=17,
    )
    for _ in range(len(ALL_PROFILES) - 2):
        profiles = sampler.sample_profiles()
        assert profiles[0] == (2048, 2048)
        assert profiles[1] == (2, 2)
        assert all(isinstance(profile, tuple) and len(profile) == 2 for profile in profiles)
    assert all(count > 0 for count in sampler.counts.values())

    restored = ProfileSampler(
        ALL_PROFILES,
        num_random=1,
        min_profile=(2, 2),
        max_profile=(2048, 2048),
        supported_k=SUPPORTED_K_VALUES,
        seed=17,
    )
    restored.load_state_dict(copy.deepcopy(sampler.state_dict()))
    assert restored.sample_profiles() == sampler.sample_profiles()
    assert restored.counts == sampler.counts


def check_distillation_and_sampled_diversity() -> None:
    target = torch.zeros(1, 3, 4, 4)
    student_image = torch.ones_like(target, requires_grad=True)
    teacher_image = torch.zeros_like(target, requires_grad=True)
    student_features = [
        torch.ones(1, 8, 2, 2, requires_grad=True),
        torch.ones(1, 16, 1, 1, requires_grad=True),
    ]
    teacher_features = [
        torch.zeros(1, 8, 2, 2, requires_grad=True),
        torch.zeros(1, 16, 1, 1, requires_grad=True),
    ]
    criterion = VariableRateRAQLoss(
        layer_vq_weights=(1.0, 1.0),
        feature_layer_weights=(1.0, 1.0),
        vq_weight=0.0,
        output_distill_low=1.0,
        output_distill_high=1.0,
        feature_distill_low=1.0,
        feature_distill_high=1.0,
        identity_weight=0.0,
        hierarchy_weight=0.0,
        diversity_weight=0.0,
    )
    total, _ = criterion(
        target=target,
        reconstruction=student_image,
        profile=(16, 2),
        teacher_reconstruction=teacher_image,
        student_features=student_features,
        teacher_features=teacher_features,
    )
    total.backward()
    assert student_image.grad is not None and student_image.grad.abs().sum() > 0
    assert all(feature.grad is not None for feature in student_features)
    assert teacher_image.grad is None
    assert all(feature.grad is None for feature in teacher_features)

    original_cdist, original_pdist = torch.cdist, torch.pdist

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a full pairwise distance operation was attempted")

    torch.cdist = forbidden
    torch.pdist = forbidden
    try:
        codebook = torch.randn(2048, 32, requires_grad=True)
        diversity, pair_count = sampled_margin_diversity_loss(
            codebook,
            margin=2.0,
            num_pairs=97,
            generator=torch.Generator().manual_seed(9),
        )
        assert pair_count == 97 and diversity.ndim == 0
        diversity.backward()
        assert codebook.grad is not None and torch.count_nonzero(codebook.grad) > 0
    finally:
        torch.cdist, torch.pdist = original_cdist, original_pdist


def check_legacy_path_still_runs() -> None:
    model = DeepSC(
        in_channels=3,
        out_channels=3,
        num_downsample_blocks=2,
        base_channels=4,
        num_embeddings_list=[8, 8],
        embedding_dim_list=[8, 16],
        commitment_cost=0.25,
        device=torch.device("cpu"),
        strides=[2, 2],
        skip_dropout_p=[0.0],
        norm_type="group",
        norm_groups=4,
        activation="silu",
        encoder_res_blocks=1,
        decoder_res_blocks=1,
        use_cascade_downsample=False,
        use_bottleneck_attention=False,
        quantizer_type="simvq",
        quantizer_axis_list=["patch", "patch"],
        cvq_codeword_shapes=[None, None],
        use_raq=False,
    )
    model.set_channel_prob(0.0)
    output = model.forward_train(torch.randn(2, 3, 16, 16))
    assert tuple(output["reconstructed_images"].shape) == (2, 3, 16, 16)
    assert len(output["vq_losses"]) == 2


def main() -> int:
    checks = (
        ("identity/mixed/shape/gradient/teacher", check_identity_mixed_shapes_and_gradients),
        ("atomic sampler coverage/state", check_atomic_sampler_coverage_and_restore),
        ("distillation detach/sampled diversity", check_distillation_and_sampled_diversity),
        ("legacy backward compatibility", check_legacy_path_still_runs),
    )
    for name, function in checks:
        function()
        print(f"[PASS] {name}")
    print(f"[PASS] all {len(checks)} standalone variable-rate checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
