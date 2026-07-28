"""Core contract tests for shared-codebook Residual-SimVQ."""

import torch
import torch.nn.functional as F
from torch import nn

from models.residual_simvq_quantizer import ResidualSimVQQuantizer
from models.semantic_encoder import SemanticEncoder
from models.vector_quantizer import VectorQuantizer


torch.set_num_threads(1)


def _assert_index_range(indices, codebook_size):
    assert indices.dtype == torch.long
    assert int(indices.min()) >= 0
    assert int(indices.max()) < codebook_size


def test_actual_256_two_scale_shapes_ranges_and_shared_codebooks():
    """Exercise the production 8x2 feature and BHWD index contract."""
    torch.manual_seed(40)
    encoder = SemanticEncoder(
        in_channels=3,
        num_downsample_blocks=2,
        base_channels=128,
        strides=[8, 2],
        norm_type="group",
        num_groups=32,
        activation="silu",
        num_res_blocks=0,
        use_cascade_downsample=False,
    ).eval()
    quantizers = [
        ResidualSimVQQuantizer(4, 256, 0.25, rq_depth=2).eval(),
        ResidualSimVQQuantizer(2, 512, 0.25, rq_depth=2).eval(),
    ]

    with torch.no_grad():
        features = encoder(torch.randn(1, 3, 256, 256))
        outputs = [
            quantizer(feature)
            for quantizer, feature in zip(quantizers, features)
        ]

    assert [tuple(feature.shape) for feature in features] == [
        (1, 256, 32, 32),
        (1, 512, 16, 16),
    ]
    for scale, ((loss, quantized, indices), feature, expected_shape) in enumerate(
        zip(
            outputs,
            features,
            [(1, 32, 32, 2), (1, 16, 16, 2)],
        )
    ):
        assert loss.ndim == 0
        assert quantized.shape == feature.shape
        assert tuple(indices.shape) == expected_shape
        _assert_index_range(indices, (4, 2)[scale])
        assert quantizers[scale].codebooks[0] is quantizers[scale].codebooks[1]

    assert quantizers[0].codebook is not quantizers[1].codebook
    assert (
        quantizers[0].codebook.embed.weight.data_ptr()
        != quantizers[1].codebook.embed.weight.data_ptr()
    )
    assert (
        quantizers[0].codebook.proj.weight.data_ptr()
        != quantizers[1].codebook.proj.weight.data_ptr()
    )


def _legacy_forward_reference(quantizer, inputs):
    """Literal reference for VectorQuantizer.forward before helper extraction."""
    inputs_bhwc = inputs.permute(0, 2, 3, 1).contiguous()
    embed_weight = quantizer.transformed_weight()
    batch, height, width, channels = inputs_bhwc.shape
    flat = inputs_bhwc.view(-1, channels)
    indices = quantizer._nearest_code_indices(flat, embed_weight)
    quantized = F.embedding(indices, embed_weight).view(
        batch, height, width, channels
    )
    commitment = F.mse_loss(quantized.detach(), inputs_bhwc)
    codebook = F.mse_loss(quantized, inputs_bhwc.detach())
    loss = codebook + quantizer.commitment_cost * commitment
    ste = inputs_bhwc + (quantized - inputs_bhwc).detach()
    return (
        loss,
        ste.permute(0, 3, 1, 2).contiguous(),
        indices.view(batch, height, width),
    )


def test_vector_quantizer_helper_preserves_legacy_forward_exactly():
    torch.manual_seed(41)
    quantizer = VectorQuantizer(7, 5, commitment_cost=0.25).eval()
    inputs = torch.randn(2, 5, 3, 4)

    expected = _legacy_forward_reference(quantizer, inputs)
    actual = quantizer(inputs)

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert torch.equal(actual[2], expected[2])
    assert tuple(actual[2].shape) == (2, 3, 4)


def test_residual_simvq_exact_losses_codes_decode_and_diagnostics():
    quantizer = ResidualSimVQQuantizer(
        num_embeddings=2,
        embedding_dim=1,
        commitment_cost=0.25,
        rq_depth=2,
    ).eval()
    with torch.no_grad():
        quantizer.codebook.embed.weight.copy_(torch.tensor([[0.0], [1.0]]))
        quantizer.codebook.proj.weight.fill_(1.0)

    inputs = torch.tensor([[[[0.0, 0.4, 1.0, 1.8]]]])
    loss, quantized, indices = quantizer(inputs)
    decoded = quantizer.get_quantized_features(indices)

    expected_indices = torch.tensor(
        [[[[0, 0], [0, 0], [1, 0], [1, 1]]]]
    )
    expected_quantized = torch.tensor([[[[0.0, 0.0, 1.0, 2.0]]]])
    expected_per_depth = torch.tensor([0.20, 0.05])
    expected_component = expected_per_depth.mean()

    assert quantizer.codebooks[0] is quantizer.codebooks[1]
    assert tuple(quantizer.transformed_weight().shape) == (2, 1)
    assert torch.equal(indices, expected_indices)
    assert torch.allclose(quantized, expected_quantized)
    assert torch.allclose(decoded, expected_quantized)
    assert torch.allclose(
        loss, expected_component + 0.25 * expected_component
    )

    diagnostics = quantizer.get_last_diagnostics()
    required = {
        "codebook_loss",
        "commitment_loss",
        "codebook_per_depth",
        "commitment_per_depth",
        "residual_norm_per_depth",
        "usage_per_depth",
        "perplexity_per_depth",
        "aggregate_usage",
        "aggregate_perplexity",
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
    assert torch.all(diagnostics["restarted_codes_per_depth"] == 0)


def test_residual_simvq_gradient_boundaries_and_non_ema_state():
    torch.manual_seed(42)
    quantizer = ResidualSimVQQuantizer(4, 3, 0.25, rq_depth=2)
    inputs = torch.randn(1, 3, 2, 2, requires_grad=True)

    _, quantized, _ = quantizer(inputs)
    quantized.sum().backward()
    assert torch.allclose(inputs.grad, torch.ones_like(inputs))
    assert quantizer.codebook.proj.weight.grad is None
    assert quantizer.codebook.embed.weight.grad is None

    quantizer.zero_grad(set_to_none=True)
    inputs.grad = None
    loss, _, _ = quantizer(inputs)
    loss.backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert quantizer.codebook.proj.weight.grad is not None
    assert torch.isfinite(quantizer.codebook.proj.weight.grad).all()
    assert quantizer.codebook.embed.weight.requires_grad is False
    assert quantizer.codebook.embed.weight.grad is None

    state_keys = tuple(quantizer.state_dict())
    assert state_keys == (
        "codebook.embed.weight",
        "codebook.proj.weight",
    )
    assert not any(
        "ema" in key.lower() or "cluster" in key.lower()
        for key in state_keys
    )


def test_encoder_receives_ste_reconstruction_and_commitment_gradients():
    torch.manual_seed(43)
    encoder = nn.Conv2d(3, 4, kernel_size=1, bias=False)
    quantizer = ResidualSimVQQuantizer(4, 4, 0.25, rq_depth=2)
    image = torch.randn(1, 3, 4, 4, requires_grad=True)

    vq_loss, quantized, _ = quantizer(encoder(image))
    (quantized.square().mean() + vq_loss).backward()

    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert int(torch.count_nonzero(image.grad)) > 0
    assert encoder.weight.grad is not None
    assert torch.isfinite(encoder.weight.grad).all()
    assert int(torch.count_nonzero(encoder.weight.grad)) > 0
    assert quantizer.codebook.proj.weight.grad is not None
    assert torch.isfinite(quantizer.codebook.proj.weight.grad).all()
    assert quantizer.codebook.embed.weight.grad is None
