"""Joint labels for the existing per-scale residual-quantizer indices.

Joining indices does not quantize the encoder features again. A mixed-radix
index selects the sum of the original stage codewords at the same position.
"""

import math

import torch

from utils.cscr import build_cscr_mapping


CSCR_MODES = ("off", "joint-cscr")
MAX_JOINT_CSCR_SIZE = 4096


def resolve_cscr_mode(cscr_mode="off"):
    """Validate the single control for original indices or joint CSCR."""
    if cscr_mode not in CSCR_MODES:
        raise ValueError(f"cscr_mode must be one of {CSCR_MODES}, got {cscr_mode!r}")
    return cscr_mode


def _stage_sizes(sizes):
    sizes = tuple(sizes)
    if not sizes or any(
        not isinstance(k, int) or isinstance(k, bool) or k < 2 or k & (k - 1)
        for k in sizes
    ):
        raise ValueError("Joint indices require nonempty power-of-two stage sizes >= 2")
    if math.prod(sizes) > torch.iinfo(torch.int64).max:
        raise ValueError("Joint index exceeds the int64 range")
    return sizes


def _check_indices(indices, size):
    if not isinstance(indices, torch.Tensor) or indices.dtype not in (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    ):
        raise ValueError("Joint indices must be integer tensors")
    if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= size):
        raise ValueError(f"Index outside [0, {size})")


def pack_joint_indices(stage_indices, sizes):
    """Pack q = (...(i0*K1 + i1)*K2 + i2)... without changing the inputs."""
    sizes = _stage_sizes(sizes)
    if len(stage_indices) != len(sizes):
        raise ValueError("Joint index stage count does not match stage sizes")
    first = stage_indices[0]
    _check_indices(first, sizes[0])
    joint = torch.zeros_like(first, dtype=torch.long)
    for indices, size in zip(stage_indices, sizes):
        _check_indices(indices, size)
        if indices.shape != first.shape or indices.device != first.device:
            raise ValueError("Joint stages must have the same index shape and device")
        joint = joint * size + indices.long()
    return joint


def unpack_joint_indices(joint, sizes):
    """Recover every original stage index, including single-stage prefixes."""
    sizes = _stage_sizes(sizes)
    _check_indices(joint, math.prod(sizes))
    remaining = joint.long()
    stages = []
    for size in reversed(sizes):
        stages.append(remaining.remainder(size))
        remaining = remaining.div(size, rounding_mode="floor")
    return list(reversed(stages))


@torch.no_grad()
def build_joint_codebook(stage_codebooks, sizes):
    """Enumerate codeword sums in the same order as pack_joint_indices."""
    sizes = _stage_sizes(sizes)
    if len(stage_codebooks) != len(sizes):
        raise ValueError("Joint codebook stage count does not match stage sizes")
    if math.prod(sizes) > MAX_JOINT_CSCR_SIZE:
        raise ValueError(
            f"Joint CSCR supports at most {MAX_JOINT_CSCR_SIZE} combinations per scale"
        )
    first = stage_codebooks[0]
    for codebook, size in zip(stage_codebooks, sizes):
        if (not isinstance(codebook, torch.Tensor) or codebook.ndim != 2
                or codebook.shape[0] != size or codebook.shape[1] < 1
                or not codebook.is_floating_point()):
            raise ValueError("Joint codebooks must be floating-point [stage K, dimension] tensors")
        if (codebook.shape[1] != first.shape[1] or codebook.device != first.device
                or codebook.dtype != first.dtype):
            raise ValueError("Joint stage codebooks must share dimension, dtype, and device")
        if not torch.isfinite(codebook).all():
            raise ValueError("Joint codebooks must contain only finite values")
    joint = first.detach()
    for codebook in stage_codebooks[1:]:
        joint = (joint[:, None, :] + codebook[None, :, :]).reshape(-1, first.shape[1])
    return joint


def build_joint_cscr_mappings(codebooks, k_lists):
    """Apply the unchanged cosine/Gray algorithm to each scale's codeword sums."""
    if not isinstance(codebooks, (list, tuple)) or len(codebooks) != len(k_lists):
        raise ValueError("Joint CSCR requires matching codebooks for every scale")
    return [
        build_cscr_mapping(build_joint_codebook(stages, sizes))
        for stages, sizes in zip(codebooks, k_lists)
    ]
