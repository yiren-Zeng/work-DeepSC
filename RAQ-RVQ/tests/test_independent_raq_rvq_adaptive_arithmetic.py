"""CPU contracts for adaptive arithmetic-mask transmission."""

import os
import re
import subprocess
from pathlib import Path

import numpy as np
import torch

import evaluation.independent_raq_rvq_adaptive_arithmetic as arithmetic_eval
from evaluation.independent_raq_rvq_adaptive import (
    PreparedAdaptivePacket,
    combined_bits_from_segments,
    combined_stream_lengths,
)
from evaluation.independent_raq_rvq_adaptive_arithmetic import (
    evaluate_arithmetic_mask_packets_over_channel,
    pack_topk_arithmetic_mask_segments,
    summarize_arithmetic_mask_packets,
    unpack_topk_arithmetic_mask_combined,
)
from test_independent_raq_rvq_adaptive_topk_arithmetic_mask import (
    _format_ms_ssim,
    _format_transmitted_bits,
    ms_ssim_to_db,
)
from utils.adaptive_arithmetic_coding import (
    decode_adaptive_binary_mask,
    encode_adaptive_binary_mask,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/eval/"
    "test_independent_raq_rvq_src64_64_d2_res6_6_mixture_per_k_"
    "adaptive_topk_arithmetic_mask_ldpc_bpsk_combined.sh"
)
DENSE_SCRIPT = (
    ROOT
    / "scripts/eval/"
    "test_independent_raq_rvq_src64_64_d2_res6_6_mixture_per_k_combined.sh"
)


def _indices_and_masks(active):
    indices = []
    masks = []
    for height, width, first_k, second_k in (
        (32, 32, 2, 2),
        (16, 16, 8, 2),
    ):
        tokens = height * width
        first = torch.arange(tokens).remainder(first_k).view(1, height, width)
        second = (
            torch.arange(tokens).remainder(second_k).view(1, height, width)
        )
        indices.append([first, second])
        masks.append(torch.full_like(first, active, dtype=torch.bool))
    return indices, masks


def _packet(active):
    indices, masks = _indices_and_masks(active)
    segments, metadata = pack_topk_arithmetic_mask_segments(
        indices, masks, [[2, 2], [8, 2]]
    )
    tx_indices = [
        [
            stages[0].clone(),
            torch.where(mask, stages[1], torch.full_like(stages[1], -1)),
        ]
        for stages, mask in zip(indices, masks)
    ]
    return PreparedAdaptivePacket(
        image=torch.zeros(1, 3, 16, 16),
        feature_shapes=[(32, 32), (16, 16)],
        tx_indices=tx_indices,
        codebooks=[
            [torch.zeros(2, 3), torch.zeros(2, 3)],
            [torch.zeros(8, 3), torch.zeros(2, 3)],
        ],
        active_masks=masks,
        segments=segments,
        metadata=metadata,
        selection={"image_name": "contract.png"},
        clean_psnr=100.0,
        clean_ms_ssim=1.0,
    )


def test_arithmetic_coder_roundtrips_representative_masks():
    rng = np.random.default_rng(20260807)
    masks = [
        np.zeros(1024, dtype=np.uint8),
        np.ones(1024, dtype=np.uint8),
        np.arange(1024, dtype=np.uint8) % 2,
        (rng.random(1024) < 0.1).astype(np.uint8),
        (rng.random(256) < 0.9).astype(np.uint8),
    ]
    for mask in masks:
        encoded, metadata = encode_adaptive_binary_mask(mask)
        decoded = decode_adaptive_binary_mask(encoded, mask.size)
        assert np.array_equal(decoded, mask)
        assert metadata["source_bits"] == encoded.size
    all_zero_1024, _ = encode_adaptive_binary_mask(masks[0])
    all_one_1024, _ = encode_adaptive_binary_mask(masks[1])
    assert all_zero_1024.size == all_one_1024.size == 12


def test_arithmetic_payloads_and_combined_lengths_are_exact():
    ldpc = {"k": 128, "n": 256, "rate": 0.5}
    for active, source_bits, coded_bits in (
        (False, 1814, 3840),
        (True, 3094, 6400),
    ):
        packet = _packet(active)
        metadata = packet.metadata
        assert metadata["raw_mask_bits_reference"] == 1280
        assert [
            scale["segment_source_bits"]["mask_arithmetic"]
            for scale in metadata["scales"]
        ] == [12, 10]
        assert metadata["source_bits"] == source_bits
        assert len(combined_bits_from_segments(packet.segments)) == source_bits
        summary = summarize_arithmetic_mask_packets(
            [packet], ldpc, "bpsk"
        )
        assert summary["arithmetic_mask_bits"] == 22
        assert summary["arithmetic_source_bits"] == source_bits
        assert summary["arithmetic_coded_bits"] == coded_bits


def test_arithmetic_mask_identity_roundtrip():
    first = torch.tensor([[[0, 1, 0, 1, 0, 1, 0, 1]]])
    second = torch.tensor([[[1, 0, 1, 0, 1, 0, 1, 0]]])
    mask = torch.tensor([[[False, True, False, False, True, True, False, True]]])
    segments, metadata = pack_topk_arithmetic_mask_segments(
        [[first, second]], [mask], [[2, 2]]
    )
    combined = combined_bits_from_segments(segments)
    recovered, masks, stats = unpack_topk_arithmetic_mask_combined(
        combined, metadata
    )
    assert torch.equal(masks[0], mask)
    assert torch.equal(recovered[0][0], first)
    expected_second = torch.where(mask, second, torch.full_like(second, -1))
    assert torch.equal(recovered[0][1], expected_second)
    assert stats["per_scale"][0]["active_count_mismatch"] == 0


class _ZeroReconstructionModel:
    encoder_device = torch.device("cpu")

    def reconstruct_from_indices(
        self, indices, feature_shapes=None, codebooks=None
    ):
        del indices, feature_shapes, codebooks
        return torch.zeros(1, 3, 16, 16)


def test_arithmetic_channel_evaluator_uses_one_combined_transmission():
    packet = _packet(True)
    ldpc = {"k": 128, "n": 256, "rate": 0.5}
    calls = []

    def identity(source, snr, code, device, modulation):
        del snr, device
        calls.append(source.copy())
        lengths = combined_stream_lengths(
            source.size, code, 1 if modulation == "bpsk" else 2
        )
        return source.copy(), {**lengths, "bit_errors": 0}

    result = evaluate_arithmetic_mask_packets_over_channel(
        _ZeroReconstructionModel(),
        [packet],
        0.0,
        ldpc,
        torch.device("cpu"),
        "bpsk",
        transmit_fn=identity,
    )
    assert len(calls) == 1
    assert result["payload_bits"] == 3094
    assert result["coded_bits"] == 6400
    assert result["transmitted_bits"] == 6400
    assert result["segment_bit_errors"] == {
        "first": 0,
        "mask_arithmetic": 0,
        "second_active": 0,
    }
    assert all(
        scale["exact_mask_frame_rate"] == 1.0
        for scale in result["per_scale"]
    )


def test_arithmetic_channel_evaluator_forwards_rician_settings():
    packet = _packet(True)
    ldpc = {"k": 128, "n": 256, "rate": 0.5}
    calls = []

    def identity(
        source, snr, code, device, modulation,
        channel_type="awgn", rician_k_factor=10.0,
    ):
        del snr, device
        calls.append((channel_type, rician_k_factor))
        lengths = combined_stream_lengths(
            source.size, code, 1 if modulation == "bpsk" else 2
        )
        return source.copy(), {**lengths, "bit_errors": 0}

    original = arithmetic_eval.transmit_combined_stream
    arithmetic_eval.transmit_combined_stream = identity
    try:
        result = evaluate_arithmetic_mask_packets_over_channel(
            _ZeroReconstructionModel(),
            [packet],
            0.0,
            ldpc,
            torch.device("cpu"),
            "bpsk",
            channel_type="rician",
            rician_k_factor=7.5,
        )
    finally:
        arithmetic_eval.transmit_combined_stream = original

    assert calls == [("rician", 7.5)]
    assert result["channel_type"] == "rician"
    assert result["rician_k_factor"] == 7.5


def test_transmitted_bit_output_reports_total_and_per_image():
    assert _format_transmitted_bits(12800, 2) == (
        "12800 bit total (6400.00 bit/image)"
    )


def test_ms_ssim_output_includes_db_in_parentheses():
    assert ms_ssim_to_db(0.9) == 10.0
    assert _format_ms_ssim(0.9) == "0.9000 (10.0000 dB)"


def test_arithmetic_mask_shell_is_executable_and_terminal_only():
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK)
    content = SCRIPT.read_text(encoding="utf-8")
    assert "test_independent_raq_rvq_adaptive_topk_arithmetic_mask.py" in content
    assert 'RVQ_K_LISTS="${RVQ_K_LISTS:-' in content
    assert "--json-output" not in content
    assert "--csv-output" not in content
    assert "--concise" in content
    assert 'CHANNEL_TYPES="${CHANNEL_TYPES:-awgn rician}"' in content
    assert "--channels" in content
    assert "--rician-k-factor" in content
    assert "tee " not in content
    completed = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_arithmetic_entrypoint_disables_no_channel_evaluation():
    content = (
        ROOT / "test_independent_raq_rvq_adaptive_topk_arithmetic_mask.py"
    ).read_text(encoding="utf-8")
    assert "compute_clean_metrics=False" in content
    assert "no-channel PSNR" not in content
    assert '"nochannel":' not in content


def test_adaptive_channel_defaults_match_dense_combined():
    adaptive = SCRIPT.read_text(encoding="utf-8")
    dense = DENSE_SCRIPT.read_text(encoding="utf-8")
    for name in ("SNRS", "MODULATION", "LDPC_N", "LDPC_RATE"):
        pattern = rf'^{name}="\$\{{{name}:-([^}}]+)\}}"$'
        adaptive_value = re.search(pattern, adaptive, re.MULTILINE)
        dense_value = re.search(pattern, dense, re.MULTILINE)
        assert adaptive_value is not None
        assert dense_value is not None
        assert adaptive_value.group(1) == dense_value.group(1)
