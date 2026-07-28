import numpy as np
import torch

from utils.adaptive_transport import (
    ldpc_segment_lengths,
    pack_explicit_mask_segments,
    unpack_explicit_mask_segments,
)


def _production_indices(all_stop):
    indices = []
    for height, width, num_embeddings in ((32, 32, 4), (16, 16, 2)):
        tokens = height * width
        first = torch.arange(tokens).remainder(num_embeddings)
        if all_stop:
            second = torch.full((tokens,), -1, dtype=torch.long)
        else:
            second = torch.arange(tokens).remainder(num_embeddings)
        indices.append(
            torch.stack([first, second], dim=-1).view(
                1, height, width, 2
            )
        )
    return indices


def _decoded_source_segments(segments):
    return {segment.name: segment.bits.copy() for segment in segments}


def test_explicit_mask_roundtrip_all_stop_all_active_and_mixed():
    mixed = [
        torch.tensor(
            [[[[0, -1], [1, 2]], [[2, 1], [3, -1]]]], dtype=torch.long
        ),
        torch.tensor([[[[0, 1], [1, -1]]]], dtype=torch.long),
    ]
    for indices in (
        _production_indices(all_stop=True),
        _production_indices(all_stop=False),
        mixed,
    ):
        segments, metadata = pack_explicit_mask_segments(indices, [4, 2])
        recovered, stats = unpack_explicit_mask_segments(
            _decoded_source_segments(segments), metadata
        )
        assert all(
            torch.equal(expected.cpu(), actual)
            for expected, actual in zip(indices, recovered)
        )
        assert all(
            scale["active_count_mismatch"] == 0
            for scale in stats["per_scale"]
        )


def test_production_explicit_mask_source_and_ldpc_lengths():
    all_stop_segments, all_stop = pack_explicit_mask_segments(
        _production_indices(all_stop=True), [4, 2]
    )
    assert all_stop["first_stage_bits"] == 2304
    assert all_stop["mask_bits"] == 1280
    assert all_stop["active_second_bits"] == 0
    assert all_stop["source_bits"] == 3584
    assert sum(
        ldpc_segment_lengths(segment.bits.size)["coded_bits"]
        for segment in all_stop_segments
    ) == 7168

    all_active_segments, all_active = pack_explicit_mask_segments(
        _production_indices(all_stop=False), [4, 2]
    )
    assert all_active["first_stage_bits"] == 2304
    assert all_active["mask_bits"] == 1280
    assert all_active["active_second_bits"] == 2304
    assert all_active["source_bits"] == 5888
    assert sum(
        ldpc_segment_lengths(segment.bits.size)["coded_bits"]
        for segment in all_active_segments
    ) == 11776
    assert ldpc_segment_lengths(0)["coded_bits"] == 0
    assert ldpc_segment_lengths(129) == {
        "source_bits": 129,
        "information_blocks": 2,
        "padded_information_bits": 256,
        "padding_bits": 127,
        "coded_bits": 512,
    }


def test_received_mask_count_mismatch_uses_only_received_positions():
    indices = [
        torch.tensor(
            [[[[0, -1], [1, 2]], [[2, -1], [3, 1]]]], dtype=torch.long
        )
    ]
    segments, metadata = pack_explicit_mask_segments(indices, [4])
    decoded = _decoded_source_segments(segments)
    # Original mask is [0,1,0,1].  Receive [1,1,1,1], so the two extra
    # positions must be filled with legal index zero without consulting the
    # original mask.
    decoded["mask_s0"] = np.ones(4, dtype=np.uint8)
    recovered, stats = unpack_explicit_mask_segments(decoded, metadata)
    assert torch.equal(
        recovered[0][..., 1].reshape(-1),
        torch.tensor([2, 1, 0, 0]),
    )
    scale = stats["per_scale"][0]
    assert scale["tx_active_count"] == 2
    assert scale["rx_active_count"] == 4
    assert scale["zero_filled_second_indices"] == 2
    assert scale["truncated_second_indices"] == 0

    # Receive one active position: one payload index is mapped and the other
    # is discarded.  Its location follows only the received mask.
    decoded["mask_s0"] = np.array([0, 0, 1, 0], dtype=np.uint8)
    recovered, stats = unpack_explicit_mask_segments(decoded, metadata)
    assert torch.equal(
        recovered[0][..., 1].reshape(-1),
        torch.tensor([-1, -1, 2, -1]),
    )
    scale = stats["per_scale"][0]
    assert scale["rx_active_count"] == 1
    assert scale["truncated_second_indices"] == 1
    assert scale["zero_filled_second_indices"] == 0


def test_segment_boundaries_and_corrupted_indices_stay_scale_local():
    indices = [
        torch.tensor([[[[0, 1], [3, -1]]]], dtype=torch.long),
        torch.tensor([[[[1, 0], [0, 1]]]], dtype=torch.long),
    ]
    segments, metadata = pack_explicit_mask_segments(indices, [4, 2])
    decoded = _decoded_source_segments(segments)
    decoded["first_s0"] = 1 - decoded["first_s0"]
    decoded["second_active_s0"] = 1 - decoded["second_active_s0"]
    recovered, _ = unpack_explicit_mask_segments(decoded, metadata)
    assert torch.equal(recovered[1], indices[1])
    assert int(recovered[0][..., 0].min()) >= 0
    assert int(recovered[0][..., 0].max()) < 4
    active = recovered[0][..., 1] >= 0
    assert int(recovered[0][..., 1][active].min()) >= 0
    assert int(recovered[0][..., 1][active].max()) < 4
