import numpy as np
import torch


def _bits_per_index(num_embeddings):
    """Return the legacy fixed-width representation used by this project."""
    return int(np.log2(num_embeddings))


def index_tensor_to_bits(indices, num_embeddings):
    """Pack one index tensor while retaining its complete shape.

    Every independent RAQ-RVQ scale/stage is transmitted as its own stream,
    so the complete tensor shape is retained explicitly.
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


