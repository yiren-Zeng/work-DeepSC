import torch

from losses.variable_rate_raq_loss import (
    VariableRateRAQLoss,
    merge_adjacent_codewords,
    rate_aware_weight,
    sampled_margin_diversity_loss,
)


def _loss_without_optional_terms(**overrides):
    kwargs = dict(
        layer_vq_weights=(1.0, 1.0),
        feature_layer_weights=(1.0, 1.0),
        mse_weight=1.0,
        ms_ssim_weight=0.0,
        lpips_weight=0.0,
        vq_weight=0.0,
        output_distill_low=0.0,
        output_distill_high=0.0,
        feature_distill_low=0.0,
        feature_distill_high=0.0,
        identity_weight=0.0,
        hierarchy_weight=0.0,
        diversity_weight=0.0,
    )
    kwargs.update(overrides)
    return VariableRateRAQLoss(**kwargs)


def test_rate_aware_distillation_is_monotonic_and_hits_endpoints():
    profiles = [(2, 2), (16, 2), (64, 16), (512, 64), (2048, 2048)]
    weights = [rate_aware_weight(profile, 0.02, 0.2, 2.0) for profile in profiles]
    assert weights == sorted(weights)
    assert weights[0] == 0.02
    assert weights[-1] == 0.2


def test_real_image_reconstruction_remains_primary_term():
    criterion = _loss_without_optional_terms()
    target = torch.zeros(1, 3, 4, 4)
    reconstruction = torch.ones_like(target, requires_grad=True)
    total, details = criterion(
        target=target,
        reconstruction=reconstruction,
        profile=(16, 2),
    )
    assert torch.allclose(total, torch.tensor(1.0))
    assert torch.allclose(details["reconstruction_loss"], torch.tensor(1.0))
    total.backward()
    assert reconstruction.grad is not None
    assert reconstruction.grad.abs().sum() > 0


def test_teacher_targets_are_detached_for_output_and_per_layer_features():
    criterion = _loss_without_optional_terms(
        output_distill_low=1.0,
        output_distill_high=1.0,
        feature_distill_low=1.0,
        feature_distill_high=1.0,
    )
    target = torch.zeros(1, 3, 4, 4)
    student_image = torch.ones_like(target, requires_grad=True)
    teacher_image = torch.zeros_like(target, requires_grad=True)
    student_features = [
        torch.ones(1, 3, 2, 2, requires_grad=True),
        torch.ones(1, 4, 1, 1, requires_grad=True),
    ]
    teacher_features = [
        torch.zeros(1, 3, 2, 2, requires_grad=True),
        torch.zeros(1, 4, 1, 1, requires_grad=True),
    ]

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


def test_max_profile_identity_is_exactly_zero_for_bypassed_codebooks():
    criterion = _loss_without_optional_terms(identity_weight=3.0)
    target = torch.zeros(1, 3, 2, 2)
    source_codebooks = [torch.randn(2048, 3), torch.randn(2048, 4)]
    # Structural bypass returns these exact tensors rather than regenerating K=2048.
    raq_codebooks = source_codebooks
    total, details = criterion(
        target=target,
        reconstruction=target.clone(),
        profile=(2048, 2048),
        raq_codebooks=raq_codebooks,
        source_codebooks=source_codebooks,
    )
    assert total.item() == 0.0
    assert details["identity_raw_loss"].item() == 0.0
    assert details["identity_loss"].item() == 0.0
    assert details["identity_layer_count"].item() == 2


def test_hierarchy_uses_adjacent_average_merge_of_2k_parent():
    parent0 = torch.arange(32 * 3, dtype=torch.float32).reshape(32, 3)
    parent1 = torch.arange(4 * 5, dtype=torch.float32).reshape(4, 5)
    small0 = merge_adjacent_codewords(parent0).clone().requires_grad_()
    small1 = merge_adjacent_codewords(parent1).clone().requires_grad_()
    criterion = _loss_without_optional_terms(hierarchy_weight=1.0)
    target = torch.zeros(1, 3, 2, 2)

    total, details = criterion(
        target=target,
        reconstruction=target.clone(),
        profile=(16, 2),
        raq_codebooks=[small0, small1],
        hierarchy_codebooks=[parent0, parent1],
    )
    assert total.item() == 0.0
    assert details["hierarchy_raw_loss"].item() == 0.0
    assert details["hierarchy_layer_count"].item() == 2


def test_sampled_diversity_reports_actual_pairs_and_never_calls_pairwise_ops(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a full pairwise distance operation was attempted")

    monkeypatch.setattr(torch, "cdist", forbidden)
    monkeypatch.setattr(torch, "pdist", forbidden)
    codebook = torch.randn(2048, 32, requires_grad=True)
    loss, actual_pairs = sampled_margin_diversity_loss(
        codebook,
        margin=2.0,
        num_pairs=37,
        generator=torch.Generator().manual_seed(9),
    )
    assert loss.ndim == 0
    assert actual_pairs == 37
    loss.backward()
    assert codebook.grad is not None
    assert 0 < torch.count_nonzero(codebook.grad).item() <= 37 * 2 * 32


def test_forward_from_variable_rate_deepsc_output_and_pair_count():
    criterion = _loss_without_optional_terms(
        hierarchy_weight=1.0,
        diversity_weight=1.0,
        diversity_margin=2.0,
        diversity_num_pairs=13,
    )
    target = torch.zeros(1, 3, 2, 2)
    parent0 = torch.randn(32, 3)
    parent1 = torch.randn(4, 4)
    student_output = {
        "reconstructed_images": target.clone(),
        "vq_losses": [torch.tensor(0.0), torch.tensor(0.0)],
        "raw_quantized_features": [torch.zeros(1, 3, 1, 1), torch.zeros(1, 4, 1, 1)],
        "codebooks": [merge_adjacent_codewords(parent0), merge_adjacent_codewords(parent1)],
        "hierarchy_codebooks": [parent0, parent1],
        "source_codebooks": [torch.randn(2048, 3), torch.randn(2048, 4)],
        "profile": (16, 2),
    }
    teacher_output = {
        "reconstructed_images": target.clone(),
        "raw_quantized_features": [torch.zeros(1, 3, 1, 1), torch.zeros(1, 4, 1, 1)],
    }

    total, details = criterion.forward_from_outputs(target, student_output, teacher_output)
    assert torch.isfinite(total)
    assert details["hierarchy_raw_loss"].item() == 0.0
    assert details["diversity_pair_count"].item() == 26

