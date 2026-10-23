"""Training-free cosine/Gray labels used by joint CSCR.

For greedy position t, E_star[gray(t)] = E[greedy_order[t]]. Keep the original
codebook and residual quantizer, map indices before transmission, and invert
received labels before looking up and summing the original stage codewords.
"""

import torch


@torch.no_grad()
def build_cscr_mapping(codebook):
    """Return greedy order and inverse label tables on the codebook's device.

    Start at the largest L2 norm, then visit the most cosine-similar unused
    codeword. Compute on CPU in float64 for deterministic mapping across channel
    seeds/devices; ties choose the smallest original index. A zero vector has
    similarity zero to every vector. Neither codewords nor RNG state is changed.
    """
    if codebook.ndim != 2 or not codebook.is_floating_point():
        raise ValueError("CSCR expects a real floating-point [K, dimension] codebook")
    size, dimension = codebook.shape
    if size < 2 or size & (size - 1) or dimension < 1:
        raise ValueError("CSCR requires a power-of-two K >= 2 and a nonempty dimension")
    vectors = codebook.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(vectors).all():
        raise ValueError("CSCR codebook must contain only finite values")

    norms = torch.linalg.vector_norm(vectors, dim=1)
    normalized = vectors / norms.masked_fill(norms == 0, 1).unsqueeze(1)
    similarity = normalized @ normalized.T
    available = torch.ones(size, dtype=torch.bool)
    order = torch.empty(size, dtype=torch.long)
    current = int(norms.argmax())
    for position in range(size):
        order[position] = current
        available[current] = False
        if position + 1 < size:
            scores = similarity[current].masked_fill(~available, float("-inf"))
            current = int(scores.argmax())

    ranks = torch.arange(size, dtype=torch.long)
    gray = ranks ^ (ranks >> 1)
    index_to_label = torch.empty_like(order)
    label_to_index = torch.empty_like(order)
    index_to_label[order] = gray
    label_to_index[gray] = order
    return {
        "greedy_order": order.to(codebook.device),
        "index_to_label": index_to_label.to(codebook.device),
        "label_to_index": label_to_index.to(codebook.device),
    }
