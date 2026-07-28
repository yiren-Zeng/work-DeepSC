import torch

from evaluation.rq_depth_extension import (
    extend_shared_rq_depth_for_eval,
    set_shared_rq_depth_for_eval,
    truncate_rq_indices,
)
from models.deepsc import DeepSC
from models.rq_ema_quantizer import VQEmbedding
from utils.bit_utils import bits_to_indices, indices_to_bits


def _make_model(quantizer_type="rq_ema"):
    return DeepSC(
        in_channels=3,
        out_channels=3,
        num_downsample_blocks=2,
        base_channels=4,
        num_embeddings_list=[4, 2],
        embedding_dim_list=[8, 16],
        commitment_cost=0.25,
        device=torch.device("cpu"),
        strides=[2, 2],
        skip_dropout_p=[0.0],
        norm_type="group",
        norm_groups=4,
        activation="silu",
        encoder_res_blocks=0,
        decoder_res_blocks=0,
        upsample_mode="nearest",
        use_cascade_downsample=False,
        use_bottleneck_attention=False,
        quantizer_type=quantizer_type,
        quantizer_axis_list=["patch", "patch"],
        cvq_codeword_shapes=[None, None],
        rq_depth_list=[2, 2],
        rq_ema_decay=0.99,
        rq_restart_unused_codes=False,
        rq_shared_codebook=True,
    ).eval()


def _assert_raises(error_type, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def test_eval_extension_reuses_aliases_without_new_parameters_or_buffers():
    model = _make_model()
    parameter_ids = {id(value) for value in model.parameters()}
    buffer_ids = {id(value) for value in model.buffers()}
    weights = [
        quantizer.transformed_weight().detach().clone()
        for quantizer in model.vector_quantizers
    ]

    report = extend_shared_rq_depth_for_eval(model, [4, 4])

    assert report["loaded_rq_depth_list"] == [2, 2]
    assert report["runtime_rq_depth_list"] == [4, 4]
    assert report["added_codebook_references_per_scale"] == [2, 2]
    assert report["shared_object_per_scale"] == [True, True]
    assert model.rq_depth_list == [4, 4]
    assert {id(value) for value in model.parameters()} == parameter_ids
    assert {id(value) for value in model.buffers()} == buffer_ids
    for quantizer, expected_weight in zip(model.vector_quantizers, weights):
        assert quantizer.rq_depth == 4
        assert len(quantizer.codebooks) == 4
        assert all(
            codebook is quantizer.codebook for codebook in quantizer.codebooks
        )
        assert torch.equal(quantizer.transformed_weight(), expected_weight)


def test_depth4_prefix_indices_and_depth2_reconstruction_are_exactly_compatible():
    torch.manual_seed(121)
    model = _make_model()
    image = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        original_depth2 = model.forward_test(image)
        original_reconstruction = model.reconstruct_from_indices(
            original_depth2["indices"],
            feature_shapes=original_depth2["feature_shapes"],
        )
        extend_shared_rq_depth_for_eval(model, 4)
        full_depth4 = model.forward_test(image)

    assert [tuple(value.shape) for value in full_depth4["indices"]] == [
        (1, 8, 8, 4),
        (1, 4, 4, 4),
    ]
    for original, extended in zip(
        original_depth2["indices"], full_depth4["indices"]
    ):
        assert torch.equal(original, extended[..., :2])

    prefixes = truncate_rq_indices(full_depth4["indices"], [2, 2])
    set_shared_rq_depth_for_eval(model, 2)
    with torch.no_grad():
        prefix_reconstruction = model.reconstruct_from_indices(
            prefixes, feature_shapes=full_depth4["feature_shapes"]
        )
    assert torch.allclose(
        prefix_reconstruction,
        original_reconstruction,
        atol=1e-6,
        rtol=1e-5,
    )

    set_shared_rq_depth_for_eval(model, 4)
    with torch.no_grad():
        depth4_reconstruction = model.reconstruct_from_indices(
            full_depth4["indices"],
            feature_shapes=full_depth4["feature_shapes"],
        )
    assert depth4_reconstruction.shape == image.shape
    assert torch.isfinite(depth4_reconstruction).all()


def test_depth4_fixed_bitstream_is_9216_bits_and_roundtrips():
    torch.manual_seed(122)
    indices = [
        torch.randint(0, 4, (1, 32, 32, 4), dtype=torch.long),
        torch.randint(0, 2, (1, 16, 16, 4), dtype=torch.long),
    ]
    bitstream, shapes, codebook_sizes, stats = indices_to_bits(
        indices, [4, 2], return_stats=True
    )
    recovered = bits_to_indices(bitstream, shapes, codebook_sizes)

    assert stats["per_scale_bits"] == [8192, 1024]
    assert stats["total_bits"] == 9216
    assert len(bitstream) == 9216
    assert shapes == [(32, 32, 4), (16, 16, 4)]
    assert all(
        torch.equal(actual, expected.squeeze(0))
        for actual, expected in zip(recovered, indices)
    )
    assert stats["total_bits"] / (256 * 256) == 0.140625
    assert (stats["total_bits"] / (256 * 256)) / (0.5 * 1 * 3) == 0.09375


def test_extension_guards_training_wrong_type_independent_tables_and_bad_depths():
    training_model = _make_model().train()
    _assert_raises(
        RuntimeError, extend_shared_rq_depth_for_eval, training_model, 4
    )

    simvq_model = _make_model("simvq")
    _assert_raises(
        ValueError, extend_shared_rq_depth_for_eval, simvq_model, 4
    )

    independent_model = _make_model()
    quantizer = independent_model.vector_quantizers[0]
    quantizer.codebooks[1] = VQEmbedding(
        quantizer.num_embeddings,
        quantizer.embedding_dim,
        ema=True,
        decay=quantizer.decay,
        restart_unused_codes=False,
    )
    _assert_raises(
        ValueError, extend_shared_rq_depth_for_eval, independent_model, 4
    )

    model = _make_model()
    _assert_raises(ValueError, extend_shared_rq_depth_for_eval, model, 1)
    _assert_raises(ValueError, extend_shared_rq_depth_for_eval, model, [4])
    _assert_raises(ValueError, set_shared_rq_depth_for_eval, model, 3)


def test_extension_is_instance_local_and_adaptive_depth2_guard_remains_active():
    model = _make_model()
    untouched = _make_model()
    extend_shared_rq_depth_for_eval(model, 4)

    assert model.rq_depth_list == [4, 4]
    assert untouched.rq_depth_list == [2, 2]
    assert [len(value.codebooks) for value in model.vector_quantizers] == [4, 4]
    assert [len(value.codebooks) for value in untouched.vector_quantizers] == [2, 2]
    _assert_raises(
        ValueError,
        model.forward_test_adaptive,
        torch.randn(1, 3, 16, 16),
        [0.0, 0.0],
    )
