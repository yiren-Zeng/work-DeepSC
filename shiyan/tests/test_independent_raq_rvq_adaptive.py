"""CPU contracts for adaptive independent RAQ-RVQ combined transport."""

import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluation.independent_raq_rvq_adaptive import (
    PreparedAdaptivePacket,
    combined_bits_from_segments,
    combined_stream_lengths,
    decode_fixed_width_rle_mask,
    encode_fixed_width_rle_mask,
    evaluate_packets_over_channel,
    pack_topk_rle_segments,
    reconstruct_from_adaptive_indices,
    rounded_active_count,
    select_per_image_topk_masks,
    summarize_packets,
    unpack_topk_rle_combined,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/eval/"
    "test_independent_raq_rvq_src64_64_d2_adaptive_topk_rle_"
    "ldpc_qpsk_combined.sh"
)


def _production_indices():
    scales = []
    for height, width, first_k, second_k in (
        (32, 32, 4, 2),
        (16, 16, 8, 2),
    ):
        tokens = height * width
        first = torch.arange(tokens).remainder(first_k).view(1, height, width)
        second = (
            torch.arange(tokens).remainder(second_k).view(1, height, width)
        )
        scales.append([first, second])
    return scales


def _production_packet(active):
    indices = _production_indices()
    masks = [
        torch.full_like(stages[0], active, dtype=torch.bool)
        for stages in indices
    ]
    segments, metadata = pack_topk_rle_segments(
        indices, masks, [[4, 2], [8, 2]]
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
            [torch.zeros(4, 3), torch.zeros(2, 3)],
            [torch.zeros(8, 3), torch.zeros(2, 3)],
        ],
        active_masks=masks,
        segments=segments,
        metadata=metadata,
        selection={"image_name": "contract.png"},
        clean_psnr=100.0,
        clean_ms_ssim=1.0,
    )


def test_topk_rounding_and_stable_raster_ties():
    assert rounded_active_count(1024, 0.1) == 102
    assert rounded_active_count(256, 0.1) == 26
    assert rounded_active_count(5, 0.5) == 3
    errors = [torch.tensor([[[9.0, 8.0, 8.0, 7.0]]])]
    masks, records = select_per_image_topk_masks(errors, [0.5])
    assert torch.equal(
        masks[0], torch.tensor([[[True, True, False, False]]])
    )
    scale = records[0]["scales"][0]
    assert scale["active_count"] == 2
    assert scale["threshold"] == 8.0
    assert scale["threshold_splits_tie"] is True


def test_fixed_width_rle_known_example_and_corruption_concealment():
    mask = np.array(
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        dtype=np.uint8,
    )
    bits, metadata = encode_fixed_width_rle_mask(mask)
    decoded, stats = decode_fixed_width_rle_mask(bits, mask.size)
    assert metadata["run_lengths"] == [6, 3, 5]
    assert metadata["source_bits"] == 13
    assert np.array_equal(decoded, mask)
    assert stats["structurally_valid"] is True

    underflow = np.array([0, 0, 0, 0, 0], dtype=np.uint8)
    decoded, stats = decode_fixed_width_rle_mask(underflow, 4)
    assert np.array_equal(decoded, np.array([0, 1, 1, 1]))
    assert stats["underflow_tokens"] == 2
    assert stats["structurally_valid"] is False

    overflow = np.array([0, 1, 1, 1, 1], dtype=np.uint8)
    decoded, stats = decode_fixed_width_rle_mask(overflow, 4)
    assert np.array_equal(decoded, np.zeros(4, dtype=np.uint8))
    assert stats["overflow_tokens"] == 4

    alternating = np.arange(16, dtype=np.uint8) % 2
    bits, metadata = encode_fixed_width_rle_mask(alternating)
    assert metadata["run_count"] == alternating.size
    assert bits.size > alternating.size
    decoded, _ = decode_fixed_width_rle_mask(bits, alternating.size)
    assert np.array_equal(decoded, alternating)


def test_heterogeneous_k_combined_order_and_default_physical_budgets():
    indices = _production_indices()
    k_lists = [[4, 2], [8, 2]]
    expected_order = [
        "first_s0",
        "first_s1",
        "mask_rle_s0",
        "mask_rle_s1",
        "second_active_s0",
        "second_active_s1",
    ]
    ldpc = {"k": 128, "n": 256, "rate": 0.5}

    for active, payload, coded, symbols in (
        (False, 2836, 5888, 2944),
        (True, 4116, 8448, 4224),
    ):
        masks = [
            torch.full_like(stages[0], active, dtype=torch.bool)
            for stages in indices
        ]
        segments, metadata = pack_topk_rle_segments(
            indices, masks, k_lists
        )
        assert metadata["logical_segment_order"] == expected_order
        assert metadata["first_stage_bits"] == 2816
        assert metadata["rle_mask_bits"] == 20
        assert metadata["source_bits"] == payload
        assert metadata["dense_two_stage_source_bits_reference"] == 4096
        assert len(combined_bits_from_segments(segments)) == payload
        lengths = combined_stream_lengths(payload, ldpc, 2)
        assert lengths["ldpc_padding_bits"] == 108
        assert lengths["coded_bits"] == coded
        assert lengths["channel_symbols"] == symbols
        summary = summarize_packets(
            [_production_packet(active)], ldpc, modulation="qpsk"
        )
        assert summary["rle_source_bits"] == payload
        assert summary["rle_coded_bits"] == coded
        assert summary["rle_channel_symbols"] == symbols
        assert summary["dense_two_stage_reference"]["payload_bits"] == 4096


def test_identity_roundtrip_and_rx_mask_count_mismatch_mapping():
    first = torch.tensor([[[0, 1, 2, 3]]], dtype=torch.long)
    second = torch.tensor([[[1, 0, 0, 1]]], dtype=torch.long)
    mask = torch.tensor([[[False, True, False, True]]])
    segments, metadata = pack_topk_rle_segments(
        [[first, second]], [mask], [[4, 2]]
    )
    combined = combined_bits_from_segments(segments)
    recovered, recovered_masks, stats = unpack_topk_rle_combined(
        combined, metadata
    )
    assert torch.equal(recovered[0][0], first)
    assert torch.equal(recovered[0][1], torch.tensor([[[-1, 0, -1, 1]]]))
    assert torch.equal(recovered_masks[0], mask)
    assert stats["per_scale"][0]["active_count_mismatch"] == 0

    mask_segment = next(
        item for item in metadata["segments"] if item["kind"] == "mask_rle"
    )
    fewer_active = combined.copy()
    fewer_active[
        mask_segment["offset"] : mask_segment["end"]
    ] = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
    recovered, recovered_masks, stats = unpack_topk_rle_combined(
        fewer_active, metadata
    )
    assert not bool(recovered_masks[0].any())
    assert torch.equal(recovered[0][1], torch.full_like(second, -1))
    assert stats["per_scale"][0]["truncated_second_indices"] == 2

    # Decode as 0111: run lengths 1,3,1,1 are cropped after four tokens.
    combined[
        mask_segment["offset"] : mask_segment["end"]
    ] = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.uint8)
    recovered, recovered_masks, stats = unpack_topk_rle_combined(
        combined, metadata
    )
    assert torch.equal(
        recovered_masks[0], torch.tensor([[[False, True, True, True]]])
    )
    assert torch.equal(recovered[0][1], torch.tensor([[[-1, 0, 1, 0]]]))
    scale = stats["per_scale"][0]
    assert scale["active_count_mismatch"] == 1
    assert scale["zero_filled_second_indices"] == 1


def test_second_stage_k_greater_than_two_roundtrip():
    first = torch.tensor([[[0, 1, 2, 3]]], dtype=torch.long)
    second = torch.tensor([[[7, 6, 5, 4]]], dtype=torch.long)
    mask = torch.tensor([[[True, False, True, True]]])
    segments, metadata = pack_topk_rle_segments(
        [[first, second]], [mask], [[4, 8]]
    )
    recovered, recovered_masks, _ = unpack_topk_rle_combined(
        combined_bits_from_segments(segments), metadata
    )
    assert torch.equal(recovered_masks[0], mask)
    assert torch.equal(
        recovered[0][1], torch.tensor([[[7, -1, 5, 4]]])
    )


class _FakeReconstructionModel:
    device = torch.device("cpu")
    encoder_device = torch.device("cpu")

    def reconstruct_from_indices(
        self, indices, feature_shapes=None, codebooks=None
    ):
        del feature_shapes
        first = F.embedding(indices[0][0], codebooks[0][0])
        second = F.embedding(indices[0][1], codebooks[0][1])
        return (first + second).permute(0, 3, 1, 2).contiguous()


class _ZeroReconstructionModel:
    encoder_device = torch.device("cpu")

    def reconstruct_from_indices(
        self, indices, feature_shapes=None, codebooks=None
    ):
        del indices, feature_shapes, codebooks
        return torch.zeros(1, 3, 16, 16)


def test_inactive_second_stage_has_true_zero_contribution():
    first_indices = torch.tensor([[[0, 1]]])
    second_indices = torch.tensor([[[-1, 1]]])
    first_codebook = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    second_codebook = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    reconstructed = reconstruct_from_adaptive_indices(
        _FakeReconstructionModel(),
        [[first_indices, second_indices]],
        [(1, 2)],
        [[first_codebook, second_codebook]],
    )
    expected = torch.tensor([[[[1.0, 33.0]], [[2.0, 44.0]]]])
    assert torch.equal(reconstructed, expected)


def test_channel_evaluator_uses_one_combined_transmission_per_image():
    packet = _production_packet(True)
    ldpc = {"k": 128, "n": 256, "rate": 0.5}
    calls = []

    def identity_transmit(source, snr, code, device, modulation):
        del snr, device
        calls.append(source.copy())
        lengths = combined_stream_lengths(
            source.size, code, 2 if modulation == "qpsk" else 1
        )
        return source.copy(), {**lengths, "bit_errors": 0}

    result = evaluate_packets_over_channel(
        _ZeroReconstructionModel(),
        [packet],
        snr_db=30.0,
        ldpc_code=ldpc,
        device=torch.device("cpu"),
        modulation="qpsk",
        transmit_fn=identity_transmit,
    )
    assert len(calls) == 1
    assert np.array_equal(calls[0], combined_bits_from_segments(packet.segments))
    assert result["payload_bits"] == 4116
    assert result["coded_bits"] == 8448
    assert result["channel_symbols"] == 4224
    assert result["bit_errors"] == 0
    assert result["source_ber_after_ldpc"] == 0.0
    assert result["segment_bit_errors"] == {
        "first": 0,
        "mask_rle": 0,
        "second_active": 0,
    }
    assert all(
        scale["source_ber_after_ldpc"] == 0.0
        for scale in result["per_scale"]
    )
    assert result["per_image"][0]["source_ber_after_ldpc"] == 0.0
    assert result["per_image"][0]["ldpc_padding_bits"] == 108


def test_new_shell_is_isolated_configurable_and_executable():
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK)
    content = SCRIPT.read_text(encoding="utf-8")
    assert "test_independent_raq_rvq_adaptive_topk_rle.py" in content
    assert 'RVQ_K_LISTS="${RVQ_K_LISTS:-4,2;8,2}"' in content
    assert 'LDPC_RATE="${LDPC_RATE:-0.5}"' in content
    assert 'MODULATION="${MODULATION:-qpsk}"' in content
    assert 'TARGET_ACTIVE_RATE_PAIRS="${TARGET_ACTIVE_RATE_PAIRS:-}"' in content
    assert "test_real.py" not in content
    completed = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
