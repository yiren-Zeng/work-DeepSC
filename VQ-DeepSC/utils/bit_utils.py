import math

import numpy as np
import torch


def _bits_per_index(num_embeddings):
    bits = int(math.log2(num_embeddings))
    if 2**bits != num_embeddings:
        raise ValueError("Codebook sizes must be powers of two for fixed-width conversion")
    return bits


def indices_to_bits(indices_list, num_embeddings_list):
    bit_stream_parts = []
    original_spatial_dims = []

    for indices, num_embeddings in zip(indices_list, num_embeddings_list):
        original_spatial_dims.append(indices.shape[1:])
        bits_per_index = _bits_per_index(num_embeddings)
        flattened = indices.flatten().cpu().numpy().astype(np.uint16)
        shifts = np.arange(bits_per_index - 1, -1, -1, dtype=np.uint16)
        bit_stream_parts.append(
            ((flattened[:, None] >> shifts) & 1).flatten().astype(np.uint8)
        )

    return (
        np.concatenate(bit_stream_parts),
        original_spatial_dims,
        list(num_embeddings_list),
    )


def bits_to_indices(bit_stream, spatial_dims, num_embeddings_list):
    indices_list = []
    current_position = 0

    for (height, width), num_embeddings in zip(spatial_dims, num_embeddings_list):
        bits_per_index = _bits_per_index(num_embeddings)
        num_indices = height * width
        num_bits = num_indices * bits_per_index
        scale_bits = bit_stream[current_position : current_position + num_bits]
        if len(scale_bits) < num_bits:
            scale_bits = np.pad(scale_bits, (0, num_bits - len(scale_bits)))
        current_position += num_bits

        scale_bits = scale_bits.reshape(num_indices, bits_per_index)
        powers = 1 << np.arange(bits_per_index - 1, -1, -1, dtype=np.int64)
        indices = np.sum(scale_bits * powers, axis=1).reshape(height, width)
        indices_list.append(torch.from_numpy(indices).long())

    return indices_list


def index_tensor_to_bits(indices, num_embeddings):
    bits_per_index = _bits_per_index(num_embeddings)
    flattened = indices.detach().reshape(-1).cpu().numpy().astype(np.uint64)
    shifts = np.arange(bits_per_index - 1, -1, -1, dtype=np.uint64)
    bits = ((flattened[:, None] >> shifts) & 1).reshape(-1).astype(np.uint8)
    return bits, tuple(indices.shape), int(num_embeddings)


def bits_to_index_tensor(bit_stream, original_shape, num_embeddings):
    shape = tuple(original_shape)
    bits_per_index = _bits_per_index(num_embeddings)
    num_indices = int(np.prod(shape))
    num_required_bits = num_indices * bits_per_index
    bits = np.asarray(bit_stream).reshape(-1)[:num_required_bits]
    if len(bits) < num_required_bits:
        bits = np.pad(bits, (0, num_required_bits - len(bits)), "constant")
    reshaped = bits.reshape(num_indices, bits_per_index)
    powers = 1 << np.arange(bits_per_index - 1, -1, -1, dtype=np.int64)
    indices = np.sum(reshaped * powers, axis=1)
    return torch.from_numpy(indices.reshape(shape)).long()
