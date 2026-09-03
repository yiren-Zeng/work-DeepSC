"""CPU contracts for lossless adaptive arithmetic mask coding."""

from pathlib import Path
import os
import subprocess

import numpy as np

from utils.adaptive_arithmetic_coding import (
    decode_adaptive_binary_mask,
    encode_adaptive_binary_mask,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/eval/"
    "test_rq_ema_k4-2_d2-2_rate047_adaptive_topk_arithmetic_mask.sh"
)


def _assert_roundtrip(mask):
    encoded, metadata = encode_adaptive_binary_mask(mask)
    decoded = decode_adaptive_binary_mask(encoded, np.asarray(mask).size)
    assert np.array_equal(decoded, np.asarray(mask, dtype=np.uint8))
    assert metadata["source_bits"] == encoded.size
    assert metadata["one_count"] == int(np.asarray(mask).sum())
    assert metadata["zero_count"] + metadata["one_count"] == len(mask)
    return encoded


def test_adaptive_arithmetic_roundtrips_representative_masks():
    generator = np.random.default_rng(7)
    masks = (
        np.zeros(1024, dtype=np.uint8),
        np.ones(1024, dtype=np.uint8),
        np.arange(1024, dtype=np.uint8) % 2,
        (generator.random(1024) < 0.1).astype(np.uint8),
        (generator.random(256) < 0.9).astype(np.uint8),
    )
    for mask in masks:
        _assert_roundtrip(mask)


def test_adaptive_arithmetic_compresses_constant_masks():
    shallow = _assert_roundtrip(np.zeros(1024, dtype=np.uint8))
    deep = _assert_roundtrip(np.ones(256, dtype=np.uint8))
    assert shallow.size < 1024
    assert deep.size < 256


def test_adaptive_arithmetic_is_invariant_to_binary_complement_length():
    mask = np.array([0, 0, 1, 0, 1, 1, 1, 0], dtype=np.uint8)
    encoded, _ = encode_adaptive_binary_mask(mask)
    complement, _ = encode_adaptive_binary_mask(1 - mask)
    assert encoded.size == complement.size


def test_adaptive_arithmetic_rejects_invalid_empty_inputs():
    for call in (
        lambda: encode_adaptive_binary_mask(np.empty(0, dtype=np.uint8)),
        lambda: decode_adaptive_binary_mask(np.empty(0, dtype=np.uint8), 4),
        lambda: decode_adaptive_binary_mask(np.array([0]), 0),
    ):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid arithmetic input was accepted")


def test_arithmetic_mask_script_is_isolated_and_executable():
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK)
    content = SCRIPT.read_text(encoding="utf-8")
    assert "cd /workspace/yi/work/RQ-VAE" in content
    assert "test_adaptive_topk_arithmetic_mask.py" in content
    assert "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0" in content
    assert "mask only" in content
    syntax = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr
