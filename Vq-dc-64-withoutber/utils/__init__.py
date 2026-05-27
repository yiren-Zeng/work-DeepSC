import numpy as np
import torch


def indices_to_bits(indices_list, num_embeddings_list):
    bit_stream_parts = []
    original_spatial_dims = []

    for i, indices in enumerate(indices_list):
        original_spatial_dims.append(indices.shape[1:])
        bits_per_index = int(np.log2(num_embeddings_list[i]))

        idx_np = indices.flatten().cpu().numpy().astype(np.uint16)
        shifts = np.arange(bits_per_index - 1, -1, -1, dtype=np.uint16)
        bits = ((idx_np[:, None] >> shifts) & 1).flatten().astype(np.uint8)
        bit_stream_parts.append(bits)

    return np.concatenate(bit_stream_parts), original_spatial_dims, num_embeddings_list


def bits_to_indices(bit_stream, original_spatial_dims, original_num_embeddings_list):
    indices_list = []
    current_pos = 0

    for i, n_embed in enumerate(original_num_embeddings_list):
        h, w = original_spatial_dims[i]
        bits_per_index = int(np.log2(n_embed))
        num_indices_in_scale = h * w
        num_bits_for_scale = num_indices_in_scale * bits_per_index

        scale_bits = bit_stream[current_pos: current_pos + num_bits_for_scale]
        if len(scale_bits) < num_bits_for_scale:
            scale_bits = np.pad(scale_bits, (0, num_bits_for_scale - len(scale_bits)), 'constant')
        current_pos += num_bits_for_scale

        scale_bits_reshaped = scale_bits.reshape(num_indices_in_scale, bits_per_index)
        powers = 1 << np.arange(bits_per_index - 1, -1, -1, dtype=np.int64)
        indices = np.sum(scale_bits_reshaped * powers, axis=1)

        indices_list.append(torch.from_numpy(indices.reshape(h, w)).long())

    return indices_list