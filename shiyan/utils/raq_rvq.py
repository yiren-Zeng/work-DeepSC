"""Utilities for test-time residual RAQ experiments."""

import operator
import random


def split_total_codebook_bits(k_total, rvq_depth=2):
    """Split a power-of-two codebook's bit budget across two RVQ stages.

    The first stage receives the extra bit when the total number of bits is
    odd.  A one-bit budget therefore has only one active stage.

    Args:
        k_total: Baseline codebook size. It must be a power of two >= 2.
        rvq_depth: Residual depth. This zero-shot experiment currently defines
            only the two-stage split and therefore accepts only ``2``.

    Returns:
        A list containing one or two power-of-two stage codebook sizes.
    """
    try:
        k_total = operator.index(k_total)
    except TypeError as exc:
        raise TypeError("k_total must be an integer") from exc
    try:
        rvq_depth = operator.index(rvq_depth)
    except TypeError as exc:
        raise TypeError("rvq_depth must be an integer") from exc

    if rvq_depth != 2:
        raise ValueError("test-time RAQ-RVQ currently supports rvq_depth=2 only")
    if k_total < 2:
        raise ValueError("k_total must be at least 2")
    if k_total & (k_total - 1):
        raise ValueError(f"k_total must be a power of two, got {k_total}")

    total_bits = k_total.bit_length() - 1
    first_stage_bits = (total_bits + 1) // 2
    second_stage_bits = total_bits // 2
    stage_sizes = [1 << first_stage_bits]
    if second_stage_bits > 0:
        stage_sizes.append(1 << second_stage_bits)
    return stage_sizes


def sample_total_codebook_bit_split(k_total, rng=None):
    """Sample an ordered two-stage split without changing total index bits.

    A one-bit total budget has only one active K=2 stage.  For larger budgets,
    both stages receive at least one bit and their order is intentionally kept.
    """
    # Reuse the strict total/depth checks in the deterministic splitter.
    split_total_codebook_bits(k_total, rvq_depth=2)
    total_bits = int(k_total).bit_length() - 1
    if total_bits == 1:
        return [2]
    rng = rng or random
    first_stage_bits = rng.randint(1, total_bits - 1)
    second_stage_bits = total_bits - first_stage_bits
    return [1 << first_stage_bits, 1 << second_stage_bits]


def resolve_rvq_stage_k_lists(
    k_total_list,
    rvq_depth=2,
    stage_k_lists=None,
    min_k=None,
    max_k=None,
):
    """Resolve and validate per-scale RVQ stage codebook sizes.

    ``stage_k_lists=None`` preserves the original automatic balanced split.
    Otherwise the caller can choose any ordered split whose summed index-bit
    budget matches the corresponding single-stage ``k_total``.
    """
    totals = list(k_total_list)
    if stage_k_lists is None:
        resolved = [
            split_total_codebook_bits(k_total, rvq_depth=rvq_depth)
            for k_total in totals
        ]
    else:
        resolved = [list(stage_sizes) for stage_sizes in stage_k_lists]
        if len(resolved) != len(totals):
            raise ValueError(
                "RVQ stage K scale count must match RAQ target count: "
                f"got {len(resolved)} and {len(totals)}"
            )

    for scale_index, (k_total, stage_sizes) in enumerate(zip(totals, resolved)):
        # Reuse the baseline validation and obtain its total bit budget.
        automatic_sizes = split_total_codebook_bits(k_total, rvq_depth=rvq_depth)
        expected_stage_count = len(automatic_sizes)
        if len(stage_sizes) != expected_stage_count:
            raise ValueError(
                f"RVQ scale {scale_index} requires {expected_stage_count} active "
                f"stage(s) for K_total={k_total}, got {len(stage_sizes)}"
            )

        stage_bits = 0
        for stage_index, stage_k in enumerate(stage_sizes):
            try:
                stage_k = operator.index(stage_k)
            except TypeError as exc:
                raise TypeError(
                    f"RVQ scale {scale_index} stage {stage_index} K must be an integer"
                ) from exc
            if stage_k < 2 or stage_k & (stage_k - 1):
                raise ValueError(
                    f"RVQ scale {scale_index} stage {stage_index} K must be a "
                    f"power of two >= 2, got {stage_k}"
                )
            if min_k is not None and stage_k < min_k:
                raise ValueError(
                    f"RVQ scale {scale_index} stage {stage_index} K={stage_k} "
                    f"is below RAQ minimum {min_k}"
                )
            if max_k is not None and stage_k > max_k:
                raise ValueError(
                    f"RVQ scale {scale_index} stage {stage_index} K={stage_k} "
                    f"exceeds RAQ maximum {max_k}"
                )
            stage_sizes[stage_index] = stage_k
            stage_bits += stage_k.bit_length() - 1

        total_bits = int(k_total).bit_length() - 1
        if stage_bits != total_bits:
            raise ValueError(
                f"RVQ scale {scale_index} bit budget mismatch: K_total={k_total} "
                f"uses {total_bits} bits/token, but stages {stage_sizes} use "
                f"{stage_bits} bits/token"
            )

    return resolved
