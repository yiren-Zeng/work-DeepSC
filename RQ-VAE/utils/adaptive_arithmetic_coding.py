"""Lossless adaptive binary arithmetic coding for activity masks.

Each image and scale is coded as an independent binary sequence in raster
order.  Encoder and decoder start with Laplace counts ``zero=1, one=1`` and
update the same counts after every symbol, so no probability table is sent.
The caller must know the token count and arithmetic segment length from the
outer framing protocol; those framing fields are intentionally not included
in the returned payload length.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


_STATE_BITS = 32
_FULL_RANGE = 1 << _STATE_BITS
_MAX_CODE = _FULL_RANGE - 1
_HALF = _FULL_RANGE >> 1
_QUARTER = _HALF >> 1
_THREE_QUARTERS = _QUARTER * 3
_MAX_TOTAL = 1 << 15


def _normalize_binary(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    if flat.size == 0:
        raise ValueError("mask must not be empty")
    return (flat != 0).astype(np.uint8)


def _update_counts(counts: list[int], symbol: int) -> None:
    counts[symbol] += 1
    if sum(counts) >= _MAX_TOTAL:
        counts[:] = [max(1, (count + 1) // 2) for count in counts]


def encode_adaptive_binary_mask(
    mask: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Encode one non-empty binary mask with a 32-bit arithmetic coder."""

    flat = _normalize_binary(mask)
    low = 0
    high = _MAX_CODE
    pending_underflow = 0
    counts = [1, 1]
    output: list[int] = []

    def emit(bit: int) -> None:
        nonlocal pending_underflow
        output.append(bit)
        if pending_underflow:
            output.extend([1 - bit] * pending_underflow)
            pending_underflow = 0

    for raw_symbol in flat:
        symbol = int(raw_symbol)
        total = counts[0] + counts[1]
        interval = high - low + 1
        split = low + (interval * counts[0] // total)
        if symbol == 0:
            high = split - 1
        else:
            low = split

        while True:
            if high < _HALF:
                emit(0)
            elif low >= _HALF:
                emit(1)
                low -= _HALF
                high -= _HALF
            elif low >= _QUARTER and high < _THREE_QUARTERS:
                pending_underflow += 1
                low -= _QUARTER
                high -= _QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) + 1

        _update_counts(counts, symbol)

    pending_underflow += 1
    emit(0 if low < _QUARTER else 1)
    bits = np.asarray(output, dtype=np.uint8)
    metadata: Dict[str, object] = {
        "schema": "adaptive_binary_arithmetic_mask_v1",
        "token_count": int(flat.size),
        "zero_count": int((flat == 0).sum()),
        "one_count": int(flat.sum()),
        "source_bits": int(bits.size),
        "state_bits": _STATE_BITS,
        "initial_counts": [1, 1],
        "probability_model": "adaptive Laplace binary counts",
        "framing_overhead_counted": False,
    }
    return bits, metadata


def decode_adaptive_binary_mask(
    bits: np.ndarray,
    token_count: int,
) -> np.ndarray:
    """Decode exactly ``token_count`` symbols from an arithmetic payload."""

    token_count = int(token_count)
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    normalized = (np.asarray(bits).reshape(-1) != 0).astype(np.uint8)
    if normalized.size == 0:
        raise ValueError("arithmetic payload must not be empty")

    read_position = 0

    def read_bit() -> int:
        nonlocal read_position
        if read_position >= normalized.size:
            return 0
        bit = int(normalized[read_position])
        read_position += 1
        return bit

    low = 0
    high = _MAX_CODE
    code = 0
    for _ in range(_STATE_BITS):
        code = (code << 1) | read_bit()

    counts = [1, 1]
    output = np.empty(token_count, dtype=np.uint8)
    for index in range(token_count):
        total = counts[0] + counts[1]
        interval = high - low + 1
        split = low + (interval * counts[0] // total)
        if code < split:
            symbol = 0
            high = split - 1
        else:
            symbol = 1
            low = split
        output[index] = symbol

        while True:
            if high < _HALF:
                pass
            elif low >= _HALF:
                low -= _HALF
                high -= _HALF
                code -= _HALF
            elif low >= _QUARTER and high < _THREE_QUARTERS:
                low -= _QUARTER
                high -= _QUARTER
                code -= _QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) + 1
            code = (code << 1) | read_bit()

        _update_counts(counts, symbol)

    return output


__all__ = [
    "decode_adaptive_binary_mask",
    "encode_adaptive_binary_mask",
]
