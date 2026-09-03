"""Validation for independent RAQ-RVQ stage sizes."""

import operator


def validate_independent_rvq_k_lists(
    stage_k_lists,
    num_scales,
    rvq_depth=2,
    min_k=None,
    max_k=None,
):
    """Validate explicit per-scale, per-stage K values without rate coupling.

    Independent RAQ-RVQ samples every residual stage directly from the current
    RAQ curriculum. No product/bit-budget relationship is imposed between
    independently trained stages.
    """
    try:
        num_scales = operator.index(num_scales)
        rvq_depth = operator.index(rvq_depth)
    except TypeError as exc:
        raise TypeError("num_scales and rvq_depth must be integers") from exc
    if num_scales <= 0 or rvq_depth <= 0:
        raise ValueError("num_scales and rvq_depth must be positive")
    if stage_k_lists is None:
        raise ValueError("independent RAQ-RVQ requires explicit stage K lists")

    resolved = [list(stage_sizes) for stage_sizes in stage_k_lists]
    if len(resolved) != num_scales:
        raise ValueError(
            "independent RAQ-RVQ scale count mismatch: "
            f"expected {num_scales}, got {len(resolved)}"
        )

    def expand_bound(bound, name):
        if bound is None:
            return [None] * num_scales
        try:
            value = operator.index(bound)
        except TypeError:
            values = list(bound)
            if len(values) != num_scales:
                raise ValueError(
                    f"{name} scale count must be {num_scales}, got {len(values)}"
                )
            return [operator.index(value) for value in values]
        return [value] * num_scales

    min_bounds = expand_bound(min_k, "RAQ minimum")
    max_bounds = expand_bound(max_k, "RAQ maximum")
    for scale_index, stage_sizes in enumerate(resolved):
        if len(stage_sizes) != rvq_depth:
            raise ValueError(
                f"independent RAQ-RVQ scale {scale_index} requires "
                f"{rvq_depth} stages, got {len(stage_sizes)}"
            )
        for stage_index, stage_k in enumerate(stage_sizes):
            try:
                stage_k = operator.index(stage_k)
            except TypeError as exc:
                raise TypeError(
                    f"independent RAQ-RVQ scale {scale_index} stage "
                    f"{stage_index} K must be an integer"
                ) from exc
            if stage_k < 2 or stage_k & (stage_k - 1):
                raise ValueError(
                    f"independent RAQ-RVQ scale {scale_index} stage "
                    f"{stage_index} K must be a power of two >= 2, got {stage_k}"
                )
            if min_bounds[scale_index] is not None and stage_k < min_bounds[scale_index]:
                raise ValueError(
                    f"independent RAQ-RVQ scale {scale_index} stage "
                    f"{stage_index} K={stage_k} is below RAQ minimum "
                    f"{min_bounds[scale_index]}"
                )
            if max_bounds[scale_index] is not None and stage_k > max_bounds[scale_index]:
                raise ValueError(
                    f"independent RAQ-RVQ scale {scale_index} stage "
                    f"{stage_index} K={stage_k} exceeds RAQ maximum "
                    f"{max_bounds[scale_index]}"
                )
            stage_sizes[stage_index] = stage_k
    return resolved
