"""Core contract tests for stage-wise independent Residual-SimVQ."""

import torch

from models.stagewise_residual_simvq_quantizer import (
    StagewiseResidualSimVQQuantizer,
)


torch.set_num_threads(1)


def _assert_raises(error_type, function, *args):
    try:
        function(*args)
    except error_type as error:
        return error
    raise AssertionError(f"expected {error_type.__name__}")


def test_stagewise_constructor_creates_independent_frozen_base_codebooks():
    quantizer = StagewiseResidualSimVQQuantizer(
        num_embeddings_per_depth=[8, 2],
        embedding_dim=3,
        commitment_cost=0.25,
    )

    assert quantizer.num_embeddings_per_depth == (8, 2)
    assert quantizer.rq_depth == 2
    assert quantizer.shared_codebook is False
    assert len(quantizer.codebooks) == 2
    assert quantizer.codebooks[0] is not quantizer.codebooks[1]
    assert tuple(quantizer.transformed_weight(0).shape) == (8, 3)
    assert tuple(quantizer.transformed_weight(1).shape) == (2, 3)
    assert quantizer.codebooks[0].embed.weight.requires_grad is False
    assert quantizer.codebooks[1].embed.weight.requires_grad is False
    assert quantizer.codebooks[0].proj.weight.requires_grad is True
    assert quantizer.codebooks[1].proj.weight.requires_grad is True
    assert (
        quantizer.codebooks[0].embed.weight.data_ptr()
        != quantizer.codebooks[1].embed.weight.data_ptr()
    )
    assert (
        quantizer.codebooks[0].proj.weight.data_ptr()
        != quantizer.codebooks[1].proj.weight.data_ptr()
    )

    state_keys = tuple(quantizer.state_dict())
    assert state_keys == (
        "codebooks.0.embed.weight",
        "codebooks.0.proj.weight",
        "codebooks.1.embed.weight",
        "codebooks.1.proj.weight",
    )
    assert not any(
        "ema" in key.lower() or "cluster" in key.lower()
        for key in state_keys
    )


def test_stagewise_exact_losses_codes_decode_and_diagnostics():
    quantizer = StagewiseResidualSimVQQuantizer(
        num_embeddings_per_depth=[8, 2],
        embedding_dim=1,
        commitment_cost=0.25,
    ).eval()
    with torch.no_grad():
        quantizer.codebooks[0].embed.weight.copy_(
            torch.arange(8, dtype=torch.float32).unsqueeze(1)
        )
        quantizer.codebooks[0].proj.weight.fill_(1.0)
        quantizer.codebooks[1].embed.weight.copy_(
            torch.tensor([[0.0], [0.5]])
        )
        quantizer.codebooks[1].proj.weight.fill_(1.0)

    inputs = torch.tensor([[[[0.0, 0.4, 1.2, 1.8]]]])
    loss, quantized, indices = quantizer(inputs)
    decoded = quantizer.get_quantized_features(indices)

    expected_indices = torch.tensor(
        [[[[0, 0], [0, 1], [1, 0], [2, 0]]]]
    )
    expected_quantized = torch.tensor([[[[0.0, 0.5, 1.0, 2.0]]]])
    expected_per_depth = torch.tensor([0.06, 0.0225])
    expected_component = expected_per_depth.mean()

    assert tuple(indices.shape) == (1, 1, 4, 2)
    assert indices.dtype == torch.long
    assert torch.equal(indices, expected_indices)
    assert torch.allclose(quantized, expected_quantized)
    assert torch.allclose(decoded, expected_quantized)
    assert torch.allclose(
        loss, expected_component + 0.25 * expected_component
    )

    diagnostics = quantizer.get_last_diagnostics()
    required = {
        "vq_loss",
        "codebook_loss",
        "commitment_loss",
        "codebook_per_depth",
        "commitment_per_depth",
        "residual_norm_per_depth",
        "usage_per_depth",
        "perplexity_per_depth",
        "usage_counts_per_depth",
        "aggregate_usage",
        "aggregate_perplexity",
        "aggregate_usage_counts",
        "dead_codes_per_depth",
        "restarted_codes_per_depth",
    }
    assert required.issubset(diagnostics)
    assert torch.allclose(
        diagnostics["codebook_per_depth"], expected_per_depth
    )
    assert torch.allclose(
        diagnostics["commitment_per_depth"], expected_per_depth
    )
    usage_counts = diagnostics["usage_counts_per_depth"]
    assert isinstance(usage_counts, list)
    assert [tuple(counts.shape) for counts in usage_counts] == [(8,), (2,)]
    assert torch.equal(
        usage_counts[0], torch.tensor([2, 1, 1, 0, 0, 0, 0, 0]).float()
    )
    assert torch.equal(usage_counts[1], torch.tensor([3, 1]).float())
    assert tuple(diagnostics["aggregate_usage_counts"].shape) == (10,)
    assert diagnostics["aggregate_usage"] == 0.5
    assert torch.equal(
        diagnostics["dead_codes_per_depth"], torch.tensor([5, 0])
    )
    assert torch.all(diagnostics["restarted_codes_per_depth"] == 0)


def test_stagewise_ste_and_each_projection_gradient_boundary():
    torch.manual_seed(51)
    quantizer = StagewiseResidualSimVQQuantizer([8, 2], 3, 0.25)
    inputs = torch.randn(2, 3, 2, 2, requires_grad=True)

    _, quantized, _ = quantizer(inputs)
    quantized.sum().backward()
    assert torch.allclose(inputs.grad, torch.ones_like(inputs))
    for codebook in quantizer.codebooks:
        assert codebook.proj.weight.grad is None
        assert codebook.embed.weight.grad is None

    quantizer.zero_grad(set_to_none=True)
    inputs.grad = None
    loss, _, _ = quantizer(inputs)
    loss.backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert int(torch.count_nonzero(inputs.grad)) > 0
    for codebook in quantizer.codebooks:
        assert codebook.proj.weight.grad is not None
        assert torch.isfinite(codebook.proj.weight.grad).all()
        assert int(torch.count_nonzero(codebook.proj.weight.grad)) > 0
        assert codebook.embed.weight.requires_grad is False
        assert codebook.embed.weight.grad is None


def test_stagewise_decode_checks_each_depth_specific_index_range():
    quantizer = StagewiseResidualSimVQQuantizer([8, 2], 2, 0.25).eval()
    valid = torch.tensor([[[[7, 1], [0, 0]]]])
    decoded = quantizer.get_quantized_features(valid)
    assert tuple(decoded.shape) == (1, 2, 1, 2)

    invalid_first = valid.clone()
    invalid_first[..., 0] = 8
    error = _assert_raises(
        ValueError, quantizer.get_quantized_features, invalid_first
    )
    assert "depth 0" in str(error)
    assert "K=8" in str(error)

    invalid_second = valid.clone()
    invalid_second[..., 1] = 2
    error = _assert_raises(
        ValueError, quantizer.get_quantized_features, invalid_second
    )
    assert "depth 1" in str(error)
    assert "K=2" in str(error)

    negative_second = valid.clone()
    negative_second[..., 1] = -1
    error = _assert_raises(
        ValueError, quantizer.get_quantized_features, negative_second
    )
    assert "depth 1" in str(error)

    _assert_raises(
        ValueError,
        quantizer.get_quantized_features,
        valid[..., :1],
    )
    _assert_raises(
        ValueError,
        quantizer.get_quantized_features,
        valid,
        (2, 1),
    )


def test_stagewise_rejects_invalid_configuration_and_input_shape():
    _assert_raises(
        ValueError,
        StagewiseResidualSimVQQuantizer,
        [],
        3,
        0.25,
    )
    _assert_raises(
        ValueError,
        StagewiseResidualSimVQQuantizer,
        [8, 0],
        3,
        0.25,
    )
    _assert_raises(
        TypeError,
        StagewiseResidualSimVQQuantizer,
        [8, 2.5],
        3,
        0.25,
    )
    _assert_raises(
        ValueError,
        StagewiseResidualSimVQQuantizer,
        [8, 2],
        0,
        0.25,
    )

    quantizer = StagewiseResidualSimVQQuantizer([8, 2], 3, 0.25)
    _assert_raises(ValueError, quantizer, torch.randn(1, 3, 4))
    _assert_raises(ValueError, quantizer, torch.randn(1, 2, 4, 4))
