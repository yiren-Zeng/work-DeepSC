# import  numpy as np
# import torch
#
# def indices_to_bits(indices_list, num_embeddings_list):
#     bit_stream = []
#     original_spatial_dims = []
#     bits_per_index_list = [int(np.log2(n)) for n in num_embeddings_list]
#
#     for i, indices in enumerate(indices_list):
#         original_spatial_dims.append(indices.shape[1:])
#         bits_per_index = bits_per_index_list[i]
#         # 注意：这里要转为 cpu() 才能在 numpy 中处理，防止报错
#         for index in indices.flatten().cpu().numpy():
#             binary_str = format(int(index), f"0{bits_per_index}b")
#             bit_stream.extend([int(b) for b in binary_str])
#
#     return np.array(bit_stream, dtype=np.uint8), original_spatial_dims, num_embeddings_list
#
#
# def bits_to_indices(bit_stream, original_spatial_dims, original_num_embeddings_list):
#     indices_list = []
#     bits_per_index_list = [int(np.log2(n)) for n in original_num_embeddings_list]
#     current_pos = 0
#
#     for i, bits_per_index in enumerate(bits_per_index_list):
#         h, w = original_spatial_dims[i][0], original_spatial_dims[i][1]
#         num_indices_in_scale = h * w
#         num_bits_for_scale = num_indices_in_scale * bits_per_index
#
#         if current_pos + num_bits_for_scale > len(bit_stream):
#             scale_bits = bit_stream[current_pos:]
#             actual_len = len(scale_bits)
#             needed = num_bits_for_scale - actual_len
#             scale_bits = np.pad(scale_bits, (0, needed), 'constant')
#         else:
#             scale_bits = bit_stream[current_pos: current_pos + num_bits_for_scale]
#
#         current_pos += num_bits_for_scale
#
#         scale_bits_reshaped = scale_bits.reshape(num_indices_in_scale, bits_per_index)
#
#         indices = np.zeros(num_indices_in_scale, dtype=int)
#         for j in range(num_indices_in_scale):
#             indices[j] = int("".join(str(x) for x in scale_bits_reshaped[j]), 2)
#
#         indices_list.append(torch.from_numpy(indices.reshape(h, w)).long())
#
#     return indices_list

import numpy as np
import torch


def indices_to_bits(indices_list, num_embeddings_list):
    bit_stream_parts = []
    original_spatial_dims = []

    for i, indices in enumerate(indices_list):
        original_spatial_dims.append(indices.shape[1:])
        bits_per_index = int(np.log2(num_embeddings_list[i]))

        # 极速向量化：利用位移操作直接提取比特，抛弃字符串转换
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

        # 提取当前层的比特并处理 Padding
        scale_bits = bit_stream[current_pos: current_pos + num_bits_for_scale]
        if len(scale_bits) < num_bits_for_scale:
            scale_bits = np.pad(scale_bits, (0, num_bits_for_scale - len(scale_bits)), 'constant')
        current_pos += num_bits_for_scale

        # 极速向量化：利用 2 的幂次矩阵乘法直接还原整数，抛弃字符串拼接
        scale_bits_reshaped = scale_bits.reshape(num_indices_in_scale, bits_per_index)
        powers = 1 << np.arange(bits_per_index - 1, -1, -1, dtype=np.int64)
        indices = np.sum(scale_bits_reshaped * powers, axis=1)

        indices_list.append(torch.from_numpy(indices.reshape(h, w)).long())

    return indices_list