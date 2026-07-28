import torch
import torch.nn.functional as F
from torch import nn

from models.rq_ema_quantizer import RQEMAQuantizer
from models.semantic_encoder import SemanticEncoder


torch.set_num_threads(1)


def _assert_index_range(indices, codebook_size):
    assert indices.dtype == torch.long
    assert int(indices.min()) >= 0
    assert int(indices.max()) < codebook_size


def test_actual_256_encoder_shapes_and_rq_index_ranges():
    """The production 8x2 spatial plan is exercised on a real 256px tensor."""
    torch.manual_seed(1)
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
        RQEMAQuantizer(4, 256, rq_depth=2, restart_unused_codes=False).eval(),
        RQEMAQuantizer(2, 512, rq_depth=2, restart_unused_codes=False).eval(),
    ]

    with torch.no_grad():
        features = encoder(torch.randn(1, 3, 256, 256))
        outputs = [quantizer(feature) for quantizer, feature in zip(quantizers, features)]

    assert [tuple(feature.shape) for feature in features] == [
        (1, 256, 32, 32),
        (1, 512, 16, 16),
    ]
    expected_index_shapes = [(1, 32, 32, 2), (1, 16, 16, 2)]
    for scale, ((loss, quantized, indices), feature, expected_shape) in enumerate(
        zip(outputs, features, expected_index_shapes)
    ):
        assert loss.ndim == 0
        assert quantized.shape == feature.shape
        assert tuple(indices.shape) == expected_shape
        _assert_index_range(indices, (4, 2)[scale])


def test_depths_share_one_object_but_scales_are_independent():
    scale0 = RQEMAQuantizer(4, 8, rq_depth=2)
    scale1 = RQEMAQuantizer(2, 16, rq_depth=2)

    assert scale0.codebooks[0] is scale0.codebooks[1]
    assert scale1.codebooks[0] is scale1.codebooks[1]
    assert scale0.codebook is not scale1.codebook
    assert scale0.codebook.weight.data_ptr() != scale1.codebook.weight.data_ptr()


def test_decode_is_sum_of_shared_embeddings_and_commitment_is_raw_cumulative_mean():
    torch.manual_seed(2)
    quantizer = RQEMAQuantizer(
        num_embeddings=4,
        embedding_dim=3,
        rq_depth=2,
        restart_unused_codes=False,
    ).eval()
    inputs = torch.randn(2, 3, 4, 5)

    raw_commitment, ste_quantized, indices = quantizer(inputs)
    weight = quantizer.transformed_weight()
    depth0 = F.embedding(indices[..., 0], weight)
    depth1 = F.embedding(indices[..., 1], weight)
    cumulative0 = depth0
    cumulative1 = depth0 + depth1
    inputs_bhwc = inputs.permute(0, 2, 3, 1)
    expected_commitment = torch.stack(
        [
            F.mse_loss(inputs_bhwc, cumulative0.detach()),
            F.mse_loss(inputs_bhwc, cumulative1.detach()),
        ]
    ).mean()
    expected_nchw = cumulative1.permute(0, 3, 1, 2).contiguous()

    decoded = quantizer.get_quantized_features(indices)
    assert torch.allclose(decoded, expected_nchw)
    assert torch.allclose(ste_quantized, expected_nchw)
    assert torch.allclose(raw_commitment, expected_commitment)
    assert torch.allclose(
        quantizer.last_diagnostics["commitment_per_depth"],
        torch.stack(
            [
                F.mse_loss(inputs_bhwc, cumulative0.detach()),
                F.mse_loss(inputs_bhwc, cumulative1.detach()),
            ]
        ),
    )


def test_ste_propagates_to_input_and_encoder_but_not_codebook():
    torch.manual_seed(3)
    quantizer = RQEMAQuantizer(
        num_embeddings=4,
        embedding_dim=4,
        rq_depth=2,
        restart_unused_codes=False,
    ).eval()

    direct_input = torch.randn(1, 4, 3, 3, requires_grad=True)
    _, direct_quantized, _ = quantizer(direct_input)
    direct_quantized.sum().backward()
    assert torch.allclose(direct_input.grad, torch.ones_like(direct_input))

    encoder = nn.Conv2d(3, 4, kernel_size=1, bias=False)
    image = torch.randn(1, 3, 4, 4, requires_grad=True)
    _, quantized, _ = quantizer(encoder(image))
    quantized.square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert int(torch.count_nonzero(image.grad)) > 0
    assert encoder.weight.grad is not None and torch.isfinite(encoder.weight.grad).all()
    assert int(torch.count_nonzero(encoder.weight.grad)) > 0
    assert all(not parameter.requires_grad for parameter in quantizer.codebook.parameters())
    assert all(parameter.grad is None for parameter in quantizer.codebook.parameters())


def test_ema_has_no_gradient_and_updates_in_train_mode_only():
    torch.manual_seed(4)
    quantizer = RQEMAQuantizer(
        num_embeddings=4,
        embedding_dim=3,
        rq_depth=2,
        decay=0.9,
        restart_unused_codes=True,
    )
    inputs = torch.randn(1, 3, 3, 3)
    assert quantizer.codebook.weight.requires_grad is False
    assert quantizer.codebook.cluster_size_ema.requires_grad is False
    assert quantizer.codebook.embed_ema.requires_grad is False

    quantizer.eval()
    eval_weight = quantizer.codebook.weight.detach().clone()
    eval_cluster_size = quantizer.codebook.cluster_size_ema.detach().clone()
    eval_embed_ema = quantizer.codebook.embed_ema.detach().clone()
    with torch.no_grad():
        quantizer(inputs)
    assert torch.equal(quantizer.codebook.weight, eval_weight)
    assert torch.equal(quantizer.codebook.cluster_size_ema, eval_cluster_size)
    assert torch.equal(quantizer.codebook.embed_ema, eval_embed_ema)

    quantizer.train()
    with torch.no_grad():
        quantizer(inputs)
    assert not torch.equal(quantizer.codebook.cluster_size_ema, eval_cluster_size)
    assert not torch.equal(quantizer.codebook.embed_ema, eval_embed_ema)
    assert not torch.equal(quantizer.codebook.weight, eval_weight)
