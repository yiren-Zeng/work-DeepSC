"""CPU tests for per-image/per-scale exact Top-K adaptive selection."""

import math

import torch

from evaluation.adaptive_topk import (
    apply_adaptive_topk,
    rounded_active_count,
    select_per_image_topk_masks,
)
from utils.adaptive_transport import (
    ldpc_segment_lengths,
    pack_explicit_mask_segments,
)


torch.set_num_threads(1)


def _dense_indices(batch, height, width, num_embeddings):
    tokens = batch * height * width
    first = torch.arange(tokens).remainder(num_embeddings)
    second = (torch.arange(tokens) + 1).remainder(num_embeddings)
    return torch.stack([first, second], dim=-1).view(
        batch, height, width, 2
    )


def test_round_half_up_active_counts_match_production_grid():
    expected_scale0 = [
        0,
        102,
        205,
        307,
        410,
        512,
        614,
        717,
        819,
        922,
        1024,
    ]
    expected_scale1 = [
        0,
        26,
        51,
        77,
        102,
        128,
        154,
        179,
        205,
        230,
        256,
    ]
    targets = [step / 10 for step in range(11)]
    assert [
        rounded_active_count(1024, target) for target in targets
    ] == expected_scale0
    assert [
        rounded_active_count(256, target) for target in targets
    ] == expected_scale1
    # This distinguishes round-half-up from Python's bankers round.
    assert rounded_active_count(5, 0.5) == 3


def test_per_image_and_per_scale_selection_is_independent():
    scale0 = torch.tensor(
        [
            [[[1.0, 2.0, 3.0, 4.0]]],
            [[[10.0, 20.0, 30.0, 40.0]]],
        ]
    ).reshape(2, 1, 4)
    scale1 = torch.tensor(
        [
            [[[0.1, 0.9]]],
            [[[100.0, 200.0]]],
        ]
    ).reshape(2, 1, 2)

    masks, records = select_per_image_topk_masks(
        [scale0, scale1], [0.5, 0.5]
    )
    assert len(records) == 2
    assert sum(len(record["scales"]) for record in records) == 4
    assert masks[0].sum(dim=(1, 2)).tolist() == [2, 2]
    assert masks[1].sum(dim=(1, 2)).tolist() == [1, 1]
    assert records[0]["scales"][0]["threshold"] == 3.0
    assert records[1]["scales"][0]["threshold"] == 30.0
    assert abs(records[0]["scales"][1]["threshold"] - 0.9) < 1e-6
    assert records[1]["scales"][1]["threshold"] == 200.0


def test_boundary_ties_use_stable_raster_order_and_exact_count():
    errors = torch.tensor([[[9.0, 8.0, 8.0, 7.0]]])
    masks_a, records_a = select_per_image_topk_masks([errors], [0.5])
    masks_b, records_b = select_per_image_topk_masks([errors], [0.5])

    expected = torch.tensor([[[True, True, False, False]]])
    assert torch.equal(masks_a[0], expected)
    assert torch.equal(masks_b[0], expected)
    metadata = records_a[0]["scales"][0]
    assert metadata == records_b[0]["scales"][0]
    assert metadata["threshold"] == 8.0
    assert metadata["strict_above_threshold_count"] == 1
    assert metadata["threshold_equal_count"] == 2
    assert metadata["threshold_equal_selected_count"] == 1
    assert metadata["threshold_splits_tie"] is True


def test_zero_and_full_topk_only_change_second_stage_as_intended():
    dense = [_dense_indices(1, 2, 3, 4)]
    errors = [torch.tensor([[[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]]])]

    all_stop, stop_masks, stop_records = apply_adaptive_topk(
        dense, errors, [0.0]
    )
    assert torch.equal(all_stop[0][..., 0], dense[0][..., 0])
    assert bool((all_stop[0][..., 1] == -1).all())
    assert bool(stop_masks[0].all())
    assert stop_records[0]["scales"][0]["active_count"] == 0
    assert (
        stop_records[0]["scales"][0]["threshold"]
        > float(errors[0].max())
    )

    all_active, active_stop_masks, active_records = apply_adaptive_topk(
        dense, errors, [1.0]
    )
    assert torch.equal(all_active[0], dense[0])
    assert not bool(active_stop_masks[0].any())
    assert active_records[0]["scales"][0]["active_count"] == 6
    assert active_records[0]["scales"][0]["threshold"] == 0.0


def test_twenty_four_images_produce_exactly_forty_eight_thresholds():
    scale0 = torch.arange(24 * 4, dtype=torch.float32).view(24, 2, 2)
    scale1 = torch.arange(24 * 2, dtype=torch.float32).view(24, 1, 2)
    masks, records = select_per_image_topk_masks(
        [scale0, scale1], [0.25, 0.5]
    )

    assert len(records) == 24
    assert sum(len(record["scales"]) for record in records) == 48
    assert masks[0].sum(dim=(1, 2)).tolist() == [1] * 24
    assert masks[1].sum(dim=(1, 2)).tolist() == [1] * 24


def test_production_topk_source_and_ldpc_lengths_follow_exact_formula():
    dense = [
        _dense_indices(1, 32, 32, 4),
        _dense_indices(1, 16, 16, 2),
    ]
    errors = [
        torch.arange(1024, dtype=torch.float32).view(1, 32, 32),
        torch.arange(256, dtype=torch.float32).view(1, 16, 16),
    ]
    expected_coded_bits = [
        7168,
        7936,
        8448,
        8704,
        9216,
        9472,
        10240,
        10752,
        11008,
        11520,
        11776,
    ]

    for step, coded_bits_expected in enumerate(expected_coded_bits):
        target = step / 10
        adaptive, _, records = apply_adaptive_topk(
            dense, errors, [target, target]
        )
        active0 = rounded_active_count(1024, target)
        active1 = rounded_active_count(256, target)
        assert records[0]["scales"][0]["active_count"] == active0
        assert records[0]["scales"][1]["active_count"] == active1

        segments, metadata = pack_explicit_mask_segments(
            adaptive, [4, 2]
        )
        expected_source_bits = 3584 + 2 * active0 + active1
        assert metadata["source_bits"] == expected_source_bits
        coded_bits = sum(
            ldpc_segment_lengths(segment.bits.size)["coded_bits"]
            for segment in segments
        )
        assert coded_bits == coded_bits_expected
        assert math.isclose(
            metadata["source_bits"] / 65536,
            expected_source_bits / 65536,
            rel_tol=0.0,
            abs_tol=0.0,
        )


def test_topk_rejects_invalid_rates_shapes_and_nonfinite_errors():
    invalid_calls = (
        lambda: rounded_active_count(4, -0.1),
        lambda: rounded_active_count(4, 1.1),
        lambda: rounded_active_count(0, 0.5),
        lambda: select_per_image_topk_masks(
            [torch.ones(1, 4)], [0.5]
        ),
        lambda: select_per_image_topk_masks(
            [torch.tensor([[[float("nan")]]])], [0.5]
        ),
        lambda: select_per_image_topk_masks(
            [torch.tensor([[[float("inf")]]])], [0.5]
        ),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid Top-K input was accepted")
