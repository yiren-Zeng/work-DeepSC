import numpy as np
import torch

from models.channel import FiniteBlocklengthChannel
from utils.bit_utils import (
    bits_to_indices,
    count_index_bits,
    indices_to_bits,
)


def _stagewise_indices():
    scale_0 = torch.empty((1, 32, 32, 2), dtype=torch.long)
    scale_0[..., 0] = torch.randint(0, 8, (1, 32, 32))
    scale_0[..., 1] = torch.randint(0, 2, (1, 32, 32))

    scale_1 = torch.empty((1, 16, 16, 2), dtype=torch.long)
    scale_1[..., 0] = torch.randint(0, 2, (1, 16, 16))
    scale_1[..., 1] = torch.randint(0, 2, (1, 16, 16))
    return [scale_0, scale_1]


def test_stagewise_8_2_and_2_2_rate_and_roundtrip():
    torch.manual_seed(201)
    indices = _stagewise_indices()
    codebook_specs = [[8, 2], [2, 2]]

    stats = count_index_bits(indices, codebook_specs)
    bitstream, shapes, serialized_specs, serialized_stats = indices_to_bits(
        indices, codebook_specs, return_stats=True
    )
    recovered = bits_to_indices(bitstream, shapes, serialized_specs)

    assert stats == serialized_stats
    assert stats["bits_per_index"] == [[3, 1], [1, 1]]
    assert stats["per_scale_bits"] == [4096, 512]
    assert stats["total_bits"] == 4608
    assert len(bitstream) == 4608
    assert shapes == [(32, 32, 2), (16, 16, 2)]
    assert serialized_specs == codebook_specs
    assert all(
        torch.equal(actual, expected.squeeze(0))
        for actual, expected in zip(recovered, indices)
    )
    assert stats["total_bits"] / (256 * 256) == 0.0703125
    assert (
        stats["total_bits"] / (256 * 256) / (0.5 * 1 * 3)
        == 0.046875
    )


def test_stagewise_bitstream_is_token_major_with_depth_last():
    indices = [
        torch.tensor([[[[5, 1], [2, 0]]]], dtype=torch.long)
    ]

    bitstream, shapes, specs = indices_to_bits(indices, [[8, 2]])

    assert np.array_equal(
        bitstream,
        np.asarray([1, 0, 1, 1, 0, 1, 0, 0], dtype=np.uint8),
    )
    recovered = bits_to_indices(bitstream, shapes, specs)
    assert torch.equal(recovered[0], indices[0].squeeze(0))


def test_scalar_codebook_spec_keeps_legacy_metadata_and_roundtrip():
    indices = [
        torch.tensor([[[[0, 1], [2, 3]]]], dtype=torch.long)
    ]

    bitstream, shapes, specs, stats = indices_to_bits(
        indices, [4], return_stats=True
    )
    recovered = bits_to_indices(bitstream, shapes, specs)

    assert specs == [4]
    assert stats["bits_per_index"] == [2]
    assert stats["per_scale_bits"] == [8]
    assert torch.equal(recovered[0], indices[0].squeeze(0))


def test_channel_applies_each_depths_bit_width_and_range():
    channel = FiniteBlocklengthChannel(
        channel_coding_rate=0.5,
        coded_block_length_bits=256,
        device=torch.device("cpu"),
    )
    channel.compute_ber = lambda *args, **kwargs: torch.tensor(1.0)
    indices = torch.tensor(
        [[[[0, 0], [3, 1]], [[7, 0], [5, 1]]]],
        dtype=torch.long,
    )

    corrupted, ber = channel.apply_channel_noise(
        indices,
        num_embeddings=[8, 2],
        snr_db=torch.tensor(0.0),
        rc=0.5,
        mod_bits=1,
    )

    expected = torch.empty_like(indices)
    expected[..., 0] = 7 - indices[..., 0]
    expected[..., 1] = 1 - indices[..., 1]
    assert torch.equal(corrupted, expected)
    assert corrupted.shape == indices.shape
    assert corrupted.dtype == indices.dtype
    assert int(corrupted[..., 0].min()) >= 0
    assert int(corrupted[..., 0].max()) < 8
    assert int(corrupted[..., 1].min()) >= 0
    assert int(corrupted[..., 1].max()) < 2
    assert ber.item() == 1.0


def test_channel_scalar_codebook_spec_remains_supported():
    channel = FiniteBlocklengthChannel(
        channel_coding_rate=0.5,
        coded_block_length_bits=256,
        device=torch.device("cpu"),
    )
    channel.compute_ber = lambda *args, **kwargs: torch.tensor(1.0)
    indices = torch.tensor([[[[0, 1], [2, 3]]]], dtype=torch.long)

    corrupted, _ = channel.apply_channel_noise(
        indices,
        num_embeddings=4,
        snr_db=torch.tensor(0.0),
        rc=0.5,
        mod_bits=1,
    )

    assert torch.equal(corrupted, 3 - indices)
