"""CPU contract tests for per-token adaptive two-depth EMA-RQ inference.

These tests deliberately use a one-dimensional, hand-written codebook so
that every first-stage residual, STOP decision, code, and reconstruction has
an exact expected value.  They do not depend on a trained checkpoint.
"""

import torch
import torch.nn.functional as F

from models.rq_ema_quantizer import RQEMAQuantizer
from models.deepsc import DeepSC
from utils.bit_utils import (
    AdaptiveRQBitAccumulator,
    binary_entropy,
    entropy_from_counts,
)


torch.set_num_threads(1)


def _small_two_scale_model():
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
        quantizer_type="rq_ema",
        quantizer_axis_list=["patch", "patch"],
        cvq_codeword_shapes=[None, None],
        rq_depth_list=[2, 2],
        rq_ema_decay=0.99,
        rq_restart_unused_codes=False,
        rq_shared_codebook=True,
    ).eval()


def _deterministic_quantizer():
    quantizer = RQEMAQuantizer(
        num_embeddings=2,
        embedding_dim=1,
        rq_depth=2,
        restart_unused_codes=False,
    ).eval()
    with torch.no_grad():
        # K real entries followed by the implementation's padding row.
        quantizer.codebook.weight.copy_(torch.tensor([[0.0], [1.0], [0.0]]))
    return quantizer


def _mixed_error_input():
    # First-stage codes:       0,    0,   1,   1
    # First-stage errors:      0, 0.16,   0, 0.64
    # With threshold 0.2:   STOP, STOP, STOP, depth-2(code 1)
    return torch.tensor([[[[0.0, 0.4, 1.0, 1.8]]]])


def _first_stage_quantized(quantizer, fixed_indices):
    weight = quantizer.transformed_weight()
    quantized_bhwc = F.embedding(fixed_indices[..., 0], weight)
    return quantized_bhwc.permute(0, 3, 1, 2).contiguous()


def test_default_fixed_forward_is_unchanged_by_adaptive_inference():
    """Calling the opt-in path must not change the original two-depth path."""
    quantizer = _deterministic_quantizer()
    inputs = _mixed_error_input()

    with torch.no_grad():
        loss_before, quantized_before, indices_before = quantizer(inputs)
        quantizer.forward_adaptive(inputs, threshold=0.2)
        loss_after, quantized_after, indices_after = quantizer(inputs)

    assert torch.equal(indices_before, indices_after)
    assert torch.equal(quantized_before, quantized_after)
    assert torch.equal(loss_before, loss_after)
    assert tuple(indices_after.shape) == (1, 1, 4, 2)
    assert bool((indices_after >= 0).all())


def test_two_scale_deepsc_all_active_matches_legacy_test_path():
    """The new model-level API is opt-in and lossless at 100% depth two."""
    torch.manual_seed(31)
    model = _small_two_scale_model()
    image = torch.randn(1, 3, 16, 16)

    with torch.no_grad():
        fixed_before = model.forward_test(image)
        fixed_reconstruction = model.reconstruct_from_indices(
            fixed_before["indices"], fixed_before["feature_shapes"]
        )
        adaptive = model.forward_test_adaptive(
            image, thresholds=[float("-inf"), float("-inf")]
        )
        adaptive_reconstruction = model.reconstruct_from_adaptive_indices(adaptive)
        fixed_after = model.forward_test(image)

    assert len(adaptive["indices"]) == 2
    assert len(adaptive["need_second_masks"]) == 2
    for fixed_indices, adaptive_indices, need_second, stop_mask in zip(
        fixed_before["indices"],
        adaptive["indices"],
        adaptive["need_second_masks"],
        adaptive["stop_masks"],
    ):
        assert torch.equal(adaptive_indices, fixed_indices)
        assert bool(need_second.all())
        assert not bool(stop_mask.any())
    assert all(
        torch.equal(before, after)
        for before, after in zip(fixed_before["indices"], fixed_after["indices"])
    )
    assert torch.allclose(
        adaptive_reconstruction, fixed_reconstruction, atol=1e-6, rtol=1e-5
    )


def test_two_scale_deepsc_all_stop_marks_both_refinement_planes():
    torch.manual_seed(32)
    model = _small_two_scale_model()
    image = torch.randn(1, 3, 16, 16)

    with torch.no_grad():
        adaptive = model.forward_test_adaptive(
            image, thresholds=[float("inf"), float("inf")]
        )
        reconstruction = model.reconstruct_from_adaptive_indices(adaptive)

    assert [tuple(indices.shape) for indices in adaptive["indices"]] == [
        (1, 8, 8, 2),
        (1, 4, 4, 2),
    ]
    for indices, metadata in zip(
        adaptive["indices"], adaptive["adaptive_metadata"]
    ):
        assert bool((indices[..., 1] == -1).all())
        assert metadata["second_token_count"] == 0
        assert metadata["stop_token_count"] == indices[..., 1].numel()
        assert metadata["second_token_ratio"] == 0.0
        assert metadata["stop_token_ratio"] == 1.0
    assert reconstruction.shape == image.shape
    assert torch.isfinite(reconstruction).all()


def test_all_active_is_exactly_the_original_two_depth_quantizer():
    quantizer = _deterministic_quantizer()
    inputs = _mixed_error_input()

    with torch.no_grad():
        _, fixed_quantized, fixed_indices = quantizer(inputs)
        adaptive = quantizer.forward_adaptive(inputs, threshold=float("-inf"))
        decoded = quantizer.get_adaptive_quantized_features(adaptive["indices"])

    assert torch.equal(adaptive["indices"], fixed_indices)
    assert torch.equal(adaptive["quantized"], fixed_quantized)
    assert torch.equal(decoded, fixed_quantized)
    assert adaptive["stop_mask"].dtype == torch.bool
    assert adaptive["need_second_mask"].dtype == torch.bool
    assert not bool(adaptive["stop_mask"].any())
    assert bool(adaptive["need_second_mask"].all())
    assert torch.equal(
        adaptive["need_second_mask"], ~adaptive["stop_mask"]
    )


def test_all_stop_equals_first_stage_and_uses_minus_one_sentinel():
    quantizer = _deterministic_quantizer()
    inputs = _mixed_error_input()

    with torch.no_grad():
        _, _, fixed_indices = quantizer(inputs)
        expected_first_stage = _first_stage_quantized(quantizer, fixed_indices)
        adaptive = quantizer.forward_adaptive(inputs, threshold=float("inf"))
        decoded = quantizer.get_adaptive_quantized_features(adaptive["indices"])

    assert torch.equal(adaptive["indices"][..., 0], fixed_indices[..., 0])
    assert bool((adaptive["indices"][..., 1] == -1).all())
    assert bool(adaptive["stop_mask"].all())
    assert not bool(adaptive["need_second_mask"].any())
    assert torch.equal(adaptive["quantized"], expected_first_stage)
    assert torch.equal(decoded, expected_first_stage)


def test_per_token_threshold_mask_codes_errors_and_sparse_second_lookup():
    quantizer = _deterministic_quantizer()
    inputs = _mixed_error_input()
    lookup_token_counts = []

    def record_lookup_size(_module, args):
        lookup_inputs = args[0]
        lookup_token_counts.append(lookup_inputs.numel() // lookup_inputs.shape[-1])

    hook = quantizer.codebook.register_forward_pre_hook(record_lookup_size)
    try:
        with torch.no_grad():
            adaptive = quantizer.forward_adaptive(inputs, threshold=0.2)
    finally:
        hook.remove()

    expected_stop = torch.tensor([[[True, True, True, False]]])
    expected_indices = torch.tensor([[[[0, -1], [0, -1], [1, -1], [1, 1]]]])
    expected_error = torch.tensor([[[0.0, 0.16, 0.0, 0.64]]])
    expected_quantized = torch.tensor([[[[0.0, 0.0, 1.0, 2.0]]]])

    assert lookup_token_counts == [4, 1]
    assert torch.equal(adaptive["stop_mask"], expected_stop)
    assert torch.equal(adaptive["need_second_mask"], ~expected_stop)
    assert torch.equal(adaptive["indices"], expected_indices)
    assert torch.allclose(adaptive["first_stage_error"], expected_error)
    assert torch.equal(adaptive["quantized"], expected_quantized)
    assert abs(float(adaptive["threshold"]) - 0.2) < 1e-7


def test_adaptive_decoder_accepts_stop_only_at_second_depth():
    quantizer = _deterministic_quantizer()
    valid = torch.tensor([[[[0, -1], [1, 0], [1, 1]]]])
    expected = torch.tensor([[[[0.0, 1.0, 2.0]]]])

    with torch.no_grad():
        decoded = quantizer.get_adaptive_quantized_features(valid)
    assert torch.equal(decoded, expected)

    invalid_streams = (
        torch.tensor([[[[-1, -1]]]]),  # stage one can never STOP
        torch.tensor([[[[0, -2]]]]),   # STOP is exactly -1
        torch.tensor([[[[0, 2]]]]),    # real codes remain in [0, K-1]
    )
    for invalid in invalid_streams:
        try:
            quantizer.get_adaptive_quantized_features(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid adaptive stream was accepted: {invalid}")


def test_ideal_stop_entropy_matches_mask_plus_conditional_index_entropy():
    # Joint refinement symbols are STOP x4, code-0 x2, code-1 x1, code-3 x1.
    # H(mask)=1 bit/token and H(active index)=1.5 bits/active-token.
    first = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    second = torch.tensor([-1, -1, -1, -1, 0, 0, 1, 3])
    indices = torch.stack([first, second], dim=-1).view(1, 2, 4, 2)

    summary = AdaptiveRQBitAccumulator([4]).update([indices]).summary(
        total_image_pixels=16
    )
    scale = summary["per_scale"][0]

    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(0.5) == 1.0
    assert binary_entropy(1.0) == 0.0
    assert entropy_from_counts([0, 0, 0]) == 0.0
    assert abs(entropy_from_counts([2, 1, 0, 1]) - 1.5) < 1e-12
    assert scale["total_tokens"] == 8
    assert scale["active_tokens"] == 4
    assert scale["stop_tokens"] == 4
    assert scale["first_stage_fixed_bits"] == 16.0
    assert scale["mask_entropy_bits"] == 8.0
    assert scale["active_index_entropy_bits"] == 6.0
    assert scale["joint_stop_index_entropy_bits"] == 14.0
    assert abs(scale["entropy_decomposition_error_bits"]) < 1e-12
    assert summary["ideal_bits"] == 30.0
    assert summary["ideal_bpp"] == 30.0 / 16.0
    assert summary["bpp"] == summary["ideal_bpp"]


def _production_shape_indices(all_stop):
    indices_list = []
    for height, width, codebook_size in ((32, 32, 4), (16, 16, 2)):
        token_count = height * width
        first = torch.arange(token_count).remainder(codebook_size)
        if all_stop:
            second = torch.full((token_count,), -1, dtype=torch.long)
        else:
            # Uniform active symbols make ideal entropy equal fixed-width rate.
            second = torch.arange(token_count).remainder(codebook_size)
        indices_list.append(
            torch.stack([first, second], dim=-1).view(1, height, width, 2)
        )
    return indices_list


def test_production_dense_all_active_and_all_stop_bpp_accounting():
    image_pixels = 256 * 256

    all_active = AdaptiveRQBitAccumulator([4, 2]).update(
        _production_shape_indices(all_stop=False)
    ).summary(image_pixels)
    assert all_active["dense_fixed_bits"] == 4608.0
    assert all_active["first_stage_fixed_bits"] == 2304.0
    assert all_active["raw_mask_bits"] == 1280.0
    assert all_active["raw_active_index_bits"] == 2304.0
    assert all_active["exact_raw_bits"] == 5888.0
    assert all_active["mask_entropy_bits"] == 0.0
    assert all_active["active_index_entropy_bits"] == 2304.0
    assert all_active["joint_stop_index_entropy_bits"] == 2304.0
    assert all_active["ideal_bits"] == 4608.0
    assert all_active["dense_fixed_bpp"] == 0.0703125
    assert all_active["ideal_bpp"] == 0.0703125

    all_stop = AdaptiveRQBitAccumulator([4, 2]).update(
        _production_shape_indices(all_stop=True)
    ).summary(image_pixels)
    assert all_stop["dense_fixed_bits"] == 4608.0
    assert all_stop["first_stage_fixed_bits"] == 2304.0
    assert all_stop["raw_mask_bits"] == 1280.0
    assert all_stop["raw_active_index_bits"] == 0.0
    assert all_stop["exact_raw_bits"] == 3584.0
    assert all_stop["mask_entropy_bits"] == 0.0
    assert all_stop["active_index_entropy_bits"] == 0.0
    assert all_stop["joint_stop_index_entropy_bits"] == 0.0
    assert all_stop["ideal_bits"] == 2304.0
    assert all_stop["ideal_bpp"] == 0.03515625
