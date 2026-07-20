import numpy as np
import torch


def _bits_per_index(num_embeddings):
    """Return the legacy fixed-width representation used by this project."""
    return int(np.log2(num_embeddings))


def index_tensor_to_bits(indices, num_embeddings):
    """Pack one index tensor while retaining its complete shape.

    The older :func:`indices_to_bits` API intentionally stores only spatial
    dimensions because the original real-channel evaluator always uses a
    batch size of one.  Test-time RAQ-RVQ transmits every scale/stage as an
    independent stream, so this small single-stream API preserves the full
    tensor shape and can reconstruct batches without losing information.
    """
    bits_per_index = _bits_per_index(num_embeddings)
    idx_np = indices.detach().reshape(-1).cpu().numpy().astype(np.uint64)
    shifts = np.arange(bits_per_index - 1, -1, -1, dtype=np.uint64)
    bits = ((idx_np[:, None] >> shifts) & 1).reshape(-1).astype(np.uint8)
    return bits, tuple(indices.shape), int(num_embeddings)


def bits_to_index_tensor(bit_stream, original_shape, num_embeddings):
    """Unpack one independently transmitted index stream."""
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


def indices_to_bits(indices_list, num_embeddings_list):
    bit_stream_parts = []
    original_spatial_dims = []

    for i, indices in enumerate(indices_list):
        original_spatial_dims.append(indices.shape[1:])
        bits_per_index = _bits_per_index(num_embeddings_list[i])

        idx_np = indices.flatten().cpu().numpy().astype(np.uint16)
        shifts = np.arange(bits_per_index - 1, -1, -1, dtype=np.uint16)
        bits = ((idx_np[:, None] >> shifts) & 1).flatten().astype(np.uint8)
        bit_stream_parts.append(bits)

    return np.concatenate(bit_stream_parts), original_spatial_dims, num_embeddings_list


def bits_to_indices(bit_stream, original_spatial_dims, original_num_embeddings_list):
    indices_list = []
    current_pos = 0

    for i, n_embed in enumerate(original_num_embeddings_list):
        dims = tuple(original_spatial_dims[i])
        bits_per_index = _bits_per_index(n_embed)
        num_indices_in_scale = int(np.prod(dims))
        num_bits_for_scale = num_indices_in_scale * bits_per_index

        scale_bits = bit_stream[current_pos: current_pos + num_bits_for_scale]
        if len(scale_bits) < num_bits_for_scale:
            scale_bits = np.pad(scale_bits, (0, num_bits_for_scale - len(scale_bits)), 'constant')
        current_pos += num_bits_for_scale

        scale_bits_reshaped = scale_bits.reshape(num_indices_in_scale, bits_per_index)
        powers = 1 << np.arange(bits_per_index - 1, -1, -1, dtype=np.int64)
        indices = np.sum(scale_bits_reshaped * powers, axis=1)

        indices_list.append(torch.from_numpy(indices.reshape(dims)).long())

    return indices_list
