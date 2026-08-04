import math

import numpy as np
import torch


def bits_per_index(num_embeddings):
    """Return the fixed-width binary representation size for a codebook."""
    num_embeddings = int(num_embeddings)
    if num_embeddings < 2:
        raise ValueError("num_embeddings must be at least 2")
    return int(math.ceil(math.log2(num_embeddings)))


def _validate_scale_lists(indices_list, num_embeddings_list):
    if len(indices_list) != len(num_embeddings_list):
        raise ValueError(
            "indices_list and num_embeddings_list must have the same number of scales"
        )


def _normalize_num_embeddings_spec(spec, scale):
    """Normalize one scale's codebook specification.

    A scalar keeps the legacy contract where every entry in the scale tensor
    uses the same codebook size.  A list/tuple assigns one codebook size to
    each entry of the final (RQ-depth) dimension.
    """
    if isinstance(spec, np.ndarray):
        if spec.ndim == 0:
            spec = spec.item()
        else:
            spec = spec.tolist()

    if isinstance(spec, (list, tuple)):
        if not spec:
            raise ValueError(
                f"num_embeddings_list[{scale}] depth specification must not be empty"
            )
        normalized = [int(value) for value in spec]
        for value in normalized:
            bits_per_index(value)
        return normalized

    normalized = int(spec)
    bits_per_index(normalized)
    return normalized


def _normalize_num_embeddings_list(num_embeddings_list):
    return [
        _normalize_num_embeddings_spec(spec, scale)
        for scale, spec in enumerate(num_embeddings_list)
    ]


def _validate_depth_shape(token_shape, depth_spec, scale):
    if len(token_shape) < 1:
        raise ValueError(
            f"indices_list[{scale}] needs a final RQ-depth dimension for "
            "a per-depth codebook specification"
        )
    if token_shape[-1] != len(depth_spec):
        raise ValueError(
            f"indices_list[{scale}] has depth {token_shape[-1]}, but "
            f"num_embeddings_list[{scale}] specifies {len(depth_spec)} depths"
        )


def count_index_bits(indices_list, num_embeddings_list):
    """Return exact source-bit counts without serializing the indices.

    The transmission helpers intentionally operate on one image at a time.  A
    residual-quantizer tensor therefore has shape ``[1, H, W, D]`` and its
    serialized shape metadata is ``[H, W, D]``.  A scalar codebook size keeps
    the legacy behavior of assigning one bit width to every tensor entry.  A
    depth list assigns a potentially different width to each entry of ``D``.
    Rejecting larger batches here prevents the old implementation's silent
    loss of the batch dimension.
    """
    _validate_scale_lists(indices_list, num_embeddings_list)
    normalized_specs = _normalize_num_embeddings_list(num_embeddings_list)
    per_scale_bits = []
    per_scale_shapes = []
    per_scale_bits_per_index = []

    for scale, (indices, spec) in enumerate(zip(indices_list, normalized_specs)):
        if not isinstance(indices, torch.Tensor):
            raise TypeError(f"indices_list[{scale}] must be a torch.Tensor")
        if indices.ndim < 2:
            raise ValueError(
                f"indices_list[{scale}] must include a batch dimension and token dimensions"
            )
        if indices.shape[0] != 1:
            raise ValueError(
                "bit serialization supports batch_size=1 only; serialize each image separately"
            )
        token_shape = tuple(int(value) for value in indices.shape[1:])
        per_scale_shapes.append(token_shape)

        if isinstance(spec, list):
            _validate_depth_shape(token_shape, spec, scale)
            widths = [bits_per_index(value) for value in spec]
            token_count = int(np.prod(token_shape[:-1], dtype=np.int64))
            per_scale_bits_per_index.append(widths)
            per_scale_bits.append(token_count * sum(widths))
        else:
            width = bits_per_index(spec)
            per_scale_bits_per_index.append(width)
            per_scale_bits.append(
                int(np.prod(token_shape, dtype=np.int64)) * width
            )

    return {
        "total_bits": int(sum(per_scale_bits)),
        "per_scale_bits": per_scale_bits,
        "bits_per_index": per_scale_bits_per_index,
        "spatial_dims": per_scale_shapes,
    }


def indices_to_bits(indices_list, num_embeddings_list, return_stats=False):
    normalized_specs = _normalize_num_embeddings_list(num_embeddings_list)
    stats = count_index_bits(indices_list, normalized_specs)
    bit_stream_parts = []

    for scale, (indices, spec, width_spec) in enumerate(
        zip(indices_list, normalized_specs, stats["bits_per_index"])
    ):
        idx_np = indices.detach().cpu().numpy().astype(np.int64, copy=False)

        if isinstance(spec, list):
            depth = len(spec)
            token_indices = idx_np.reshape(-1, depth)
            token_bits = np.empty(
                (token_indices.shape[0], sum(width_spec)), dtype=np.uint8
            )
            bit_offset = 0
            for depth_index, (n_embed, width) in enumerate(
                zip(spec, width_spec)
            ):
                values = token_indices[:, depth_index]
                if values.size and (
                    values.min() < 0 or values.max() >= n_embed
                ):
                    raise ValueError(
                        f"indices_list[{scale}][..., {depth_index}] contains "
                        f"values outside [0, {n_embed - 1}]"
                    )
                shifts = np.arange(width - 1, -1, -1, dtype=np.int64)
                token_bits[:, bit_offset: bit_offset + width] = (
                    (values[:, None] >> shifts) & 1
                )
                bit_offset += width
            bit_stream_parts.append(token_bits.reshape(-1))
        else:
            flat_indices = idx_np.reshape(-1)
            if flat_indices.size and (
                flat_indices.min() < 0 or flat_indices.max() >= spec
            ):
                raise ValueError(
                    f"indices_list[{scale}] contains values outside "
                    f"[0, {spec - 1}]"
                )
            width = width_spec
            shifts = np.arange(width - 1, -1, -1, dtype=np.int64)
            bits = (
                (flat_indices[:, None] >> shifts) & 1
            ).reshape(-1).astype(np.uint8)
            bit_stream_parts.append(bits)

    if bit_stream_parts:
        bit_stream = np.concatenate(bit_stream_parts)
    else:
        bit_stream = np.empty(0, dtype=np.uint8)
    if return_stats:
        return bit_stream, stats["spatial_dims"], normalized_specs, stats
    return bit_stream, stats["spatial_dims"], normalized_specs


def bits_to_indices(bit_stream, original_spatial_dims, original_num_embeddings_list):
    if len(original_spatial_dims) != len(original_num_embeddings_list):
        raise ValueError(
            "original_spatial_dims and original_num_embeddings_list must have the same length"
        )

    normalized_specs = _normalize_num_embeddings_list(
        original_num_embeddings_list
    )
    indices_list = []
    current_pos = 0
    bit_stream = np.asarray(bit_stream).reshape(-1)

    for scale, spec in enumerate(normalized_specs):
        dims = tuple(int(value) for value in original_spatial_dims[scale])
        if not dims or any(value <= 0 for value in dims):
            raise ValueError(f"Invalid token shape for scale {scale}: {dims}")

        if isinstance(spec, list):
            _validate_depth_shape(dims, spec, scale)
            widths = [bits_per_index(value) for value in spec]
            num_tokens = int(np.prod(dims[:-1], dtype=np.int64))
            num_bits_for_scale = num_tokens * sum(widths)
        else:
            width = bits_per_index(spec)
            num_indices_in_scale = int(np.prod(dims, dtype=np.int64))
            num_bits_for_scale = num_indices_in_scale * width

        scale_bits = bit_stream[current_pos: current_pos + num_bits_for_scale]
        if len(scale_bits) < num_bits_for_scale:
            scale_bits = np.pad(
                scale_bits,
                (0, num_bits_for_scale - len(scale_bits)),
                "constant",
            )
        current_pos += num_bits_for_scale

        scale_bits = (np.asarray(scale_bits) != 0).astype(np.int64, copy=False)

        if isinstance(spec, list):
            token_bits = scale_bits.reshape(num_tokens, sum(widths))
            indices = np.empty((num_tokens, len(spec)), dtype=np.int64)
            bit_offset = 0
            for depth_index, (n_embed, width) in enumerate(zip(spec, widths)):
                depth_bits = token_bits[
                    :, bit_offset: bit_offset + width
                ]
                powers = 1 << np.arange(
                    width - 1, -1, -1, dtype=np.int64
                )
                values = np.sum(
                    depth_bits * powers, axis=1, dtype=np.int64
                )
                # A corrupted fixed-width word may exceed a non-power-of-two
                # codebook. Match the in-model channel's defensive clamping.
                indices[:, depth_index] = np.clip(
                    values, 0, n_embed - 1
                )
                bit_offset += width
        else:
            scale_bits_reshaped = scale_bits.reshape(
                num_indices_in_scale, width
            )
            powers = 1 << np.arange(
                width - 1, -1, -1, dtype=np.int64
            )
            indices = np.sum(
                scale_bits_reshaped * powers, axis=1, dtype=np.int64
            )
            # A corrupted fixed-width word may exceed a non-power-of-two
            # codebook. Match the in-model channel's defensive clamping.
            indices = np.clip(indices, 0, spec - 1)

        indices_list.append(torch.from_numpy(indices.reshape(dims)).long())

    return indices_list


def binary_entropy(probability):
    """Return Bernoulli entropy in bits for ``probability``.

    The explicit boundary handling avoids ``log2(0)`` and makes deterministic
    all-STOP/all-active masks cost zero bits under the ideal entropy model.
    """
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if probability == 0.0 or probability == 1.0:
        return 0.0
    return -(
        probability * math.log2(probability)
        + (1.0 - probability) * math.log2(1.0 - probability)
    )


def entropy_from_counts(counts):
    """Return zero-order entropy in bits/symbol for non-negative counts."""
    counts = np.asarray(counts, dtype=np.float64).reshape(-1)
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")
    total = float(counts.sum())
    if total == 0.0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


class AdaptiveRQBitAccumulator:
    """Aggregate two-depth adaptive RQ rate statistics over a dataset.

    Contract
    --------
    Each scale is a ``[B, H, W, 2]`` integer tensor.  Depth zero is always a
    valid codebook index.  At depth one, ``-1`` is the STOP symbol and values
    in ``[0, K-1]`` are active refinement indices.

    Three rates are reported:

    * ``dense_fixed``: both RQ depths sent at ``ceil(log2(K))`` bits/token;
    * ``exact_raw``: fixed-width first depth plus a one-bit activity bitmap and
      fixed-width indices only at active refinement positions;
    * ``ideal``: fixed-width first depth plus zero-order entropy coding of the
      joint ``{STOP, 0, ..., K-1}`` refinement alphabet.

    Ideal joint entropy is also decomposed exactly into Bernoulli mask entropy
    plus conditional active-index entropy.  Shape/header/model/threshold and
    entropy-table signalling overhead is deliberately not counted.
    """

    def __init__(self, num_embeddings_list):
        self.num_embeddings_list = [int(value) for value in num_embeddings_list]
        if not self.num_embeddings_list:
            raise ValueError("num_embeddings_list must not be empty")
        for value in self.num_embeddings_list:
            bits_per_index(value)
        self.total_tokens = [0] * len(self.num_embeddings_list)
        self.active_tokens = [0] * len(self.num_embeddings_list)
        self.first_counts = [
            np.zeros(value, dtype=np.int64) for value in self.num_embeddings_list
        ]
        self.refinement_counts = [
            np.zeros(value, dtype=np.int64) for value in self.num_embeddings_list
        ]
        self.num_samples = 0

    def update(self, indices_list):
        """Add a batch of adaptive indices and return ``self``."""
        _validate_scale_lists(indices_list, self.num_embeddings_list)
        batch_size = None
        for scale, (indices, n_embed) in enumerate(
            zip(indices_list, self.num_embeddings_list)
        ):
            if not isinstance(indices, torch.Tensor):
                raise TypeError(f"indices_list[{scale}] must be a torch.Tensor")
            if indices.ndim != 4 or indices.shape[-1] != 2:
                raise ValueError(
                    f"indices_list[{scale}] must have shape [B, H, W, 2]"
                )
            if batch_size is None:
                batch_size = int(indices.shape[0])
            elif int(indices.shape[0]) != batch_size:
                raise ValueError("all scales must have the same batch size")

            indices_cpu = indices.detach().to(device="cpu", dtype=torch.long)
            first = indices_cpu[..., 0].reshape(-1)
            refinement = indices_cpu[..., 1].reshape(-1)
            if first.numel() and (
                int(first.min().item()) < 0 or int(first.max().item()) >= n_embed
            ):
                raise ValueError(
                    f"scale {scale} first-depth indices must be in [0, {n_embed - 1}]"
                )
            if refinement.numel() and (
                int(refinement.min().item()) < -1
                or int(refinement.max().item()) >= n_embed
            ):
                raise ValueError(
                    f"scale {scale} refinement indices must be STOP=-1 or in "
                    f"[0, {n_embed - 1}]"
                )

            active = refinement >= 0
            self.total_tokens[scale] += int(refinement.numel())
            self.active_tokens[scale] += int(active.sum().item())
            self.first_counts[scale] += np.bincount(
                first.numpy(), minlength=n_embed
            )[:n_embed]
            if bool(active.any()):
                self.refinement_counts[scale] += np.bincount(
                    refinement[active].numpy(), minlength=n_embed
                )[:n_embed]

        self.num_samples += int(batch_size or 0)
        return self

    def summary(self, total_image_pixels):
        """Return JSON-serializable aggregate rate and entropy statistics."""
        total_image_pixels = int(total_image_pixels)
        if total_image_pixels <= 0:
            raise ValueError("total_image_pixels must be positive")

        per_scale = []
        totals = {
            "dense_fixed_bits": 0.0,
            "first_stage_fixed_bits": 0.0,
            "raw_mask_bits": 0.0,
            "raw_active_index_bits": 0.0,
            "exact_raw_bits": 0.0,
            "mask_entropy_bits": 0.0,
            "active_index_entropy_bits": 0.0,
            "joint_stop_index_entropy_bits": 0.0,
            "ideal_bits": 0.0,
        }

        for scale, n_embed in enumerate(self.num_embeddings_list):
            token_count = int(self.total_tokens[scale])
            active_count = int(self.active_tokens[scale])
            stop_count = token_count - active_count
            width = bits_per_index(n_embed)
            active_ratio = active_count / token_count if token_count else 0.0

            first_stage_fixed_bits = float(token_count * width)
            dense_fixed_bits = float(token_count * width * 2)
            raw_mask_bits = float(token_count)
            raw_active_index_bits = float(active_count * width)
            exact_raw_bits = (
                first_stage_fixed_bits + raw_mask_bits + raw_active_index_bits
            )

            mask_entropy_per_token = binary_entropy(active_ratio)
            mask_entropy_bits = float(token_count * mask_entropy_per_token)
            active_index_entropy = entropy_from_counts(
                self.refinement_counts[scale]
            )
            active_index_entropy_bits = float(active_count * active_index_entropy)
            joint_counts = np.concatenate(
                ([stop_count], self.refinement_counts[scale])
            )
            joint_entropy = entropy_from_counts(joint_counts)
            joint_stop_index_entropy_bits = float(token_count * joint_entropy)
            ideal_bits = first_stage_fixed_bits + joint_stop_index_entropy_bits

            # H(STOP/index) == H(mask) + P(active) H(index | active).
            entropy_decomposition_error = float(
                joint_stop_index_entropy_bits
                - mask_entropy_bits
                - active_index_entropy_bits
            )
            scale_summary = {
                "scale": scale,
                "num_embeddings": n_embed,
                "bits_per_index": width,
                "total_tokens": token_count,
                "active_tokens": active_count,
                "stop_tokens": stop_count,
                "active_ratio": float(active_ratio),
                "first_stage_counts": self.first_counts[scale].tolist(),
                "refinement_counts": self.refinement_counts[scale].tolist(),
                "joint_stop_index_counts": joint_counts.tolist(),
                "first_stage_fixed_bits": first_stage_fixed_bits,
                "dense_fixed_bits": dense_fixed_bits,
                "raw_mask_bits": raw_mask_bits,
                "raw_active_index_bits": raw_active_index_bits,
                "exact_raw_bits": exact_raw_bits,
                "mask_entropy_bits": mask_entropy_bits,
                "mask_entropy_per_token": float(mask_entropy_per_token),
                "active_index_entropy_bits": active_index_entropy_bits,
                "active_index_entropy_per_active_token": float(active_index_entropy),
                "joint_stop_index_entropy_bits": joint_stop_index_entropy_bits,
                "joint_stop_index_entropy_per_token": float(joint_entropy),
                "entropy_decomposition_error_bits": entropy_decomposition_error,
                "ideal_bits": ideal_bits,
            }
            for key in totals:
                totals[key] += float(scale_summary[key])
            per_scale.append(scale_summary)

        result = {
            "num_samples": int(self.num_samples),
            "total_image_pixels": total_image_pixels,
            "overhead_counted": False,
            "overhead_note": (
                "Shape/header/model/threshold/entropy-table signalling overhead is not counted."
            ),
            **totals,
            "per_scale": per_scale,
        }
        for key, value in totals.items():
            result[f"{key.removesuffix('_bits')}_bpp"] = float(
                value / total_image_pixels
            )
        # ``bpp`` is the requested adaptive ideal-entropy source rate.
        result["bpp"] = result["ideal_bpp"]
        return result
