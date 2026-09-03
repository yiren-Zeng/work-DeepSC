"""CPU contracts for the explicit-mask adaptive baseline."""

import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from evaluation.independent_raq_rvq_adaptive import (
    PreparedAdaptivePacket,
    combined_bits_from_segments,
    combined_stream_lengths,
)
from evaluation.independent_raq_rvq_adaptive_raw import (
    evaluate_raw_mask_packets_over_channel,
    pack_topk_raw_mask_segments,
    summarize_raw_mask_packets,
    unpack_topk_raw_mask_combined,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/eval/"
    "test_independent_raq_rvq_src64_64_d2_adaptive_topk_raw_mask_"
    "ldpc_bpsk_combined.sh"
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
    segments, metadata = pack_topk_raw_mask_segments(
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


def test_raw_mask_payloads_and_combined_lengths_are_exact():
    ldpc = {"k": 128, "n": 256, "rate": 0.5}
    for active, source_bits, coded_bits in (
        (False, 3072, 6144),
        (True, 4352, 8704),
    ):
        packet = _packet(active)
        metadata = packet.metadata
        assert metadata["raw_mask_bits"] == 1280
        assert [
            scale["segment_source_bits"]["mask_raw"]
            for scale in metadata["scales"]
        ] == [1024, 256]
        assert metadata["source_bits"] == source_bits
        assert len(combined_bits_from_segments(packet.segments)) == source_bits
        summary = summarize_raw_mask_packets([packet], ldpc, "bpsk")
        assert summary["raw_mask_bits"] == 1280
        assert summary["raw_source_bits"] == source_bits
        assert summary["raw_coded_bits"] == coded_bits


def test_raw_mask_identity_roundtrip_and_single_mask_flip():
    first = torch.tensor([[[0, 1, 0, 1]]])
    second = torch.tensor([[[1, 0, 1, 0]]])
    mask = torch.tensor([[[False, True, False, True]]])
    segments, metadata = pack_topk_raw_mask_segments(
        [[first, second]], [mask], [[2, 2]]
    )
    combined = combined_bits_from_segments(segments)
    recovered, masks, stats = unpack_topk_raw_mask_combined(combined, metadata)
    assert torch.equal(masks[0], mask)
    assert torch.equal(recovered[0][0], first)
    assert torch.equal(recovered[0][1], torch.tensor([[[-1, 0, -1, 0]]]))
    assert stats["per_scale"][0]["active_count_mismatch"] == 0

    raw_segment = next(
        item for item in metadata["segments"] if item["kind"] == "mask_raw"
    )
    corrupted = combined.copy()
    corrupted[int(raw_segment["offset"])] ^= 1
    _, corrupted_masks, _ = unpack_topk_raw_mask_combined(
        corrupted, metadata
    )
    assert int((corrupted_masks[0] != mask).sum()) == 1


class _ZeroReconstructionModel:
    encoder_device = torch.device("cpu")

    def reconstruct_from_indices(
        self, indices, feature_shapes=None, codebooks=None
    ):
        del indices, feature_shapes, codebooks
        return torch.zeros(1, 3, 16, 16)


def test_raw_channel_evaluator_uses_one_combined_transmission():
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

    result = evaluate_raw_mask_packets_over_channel(
        _ZeroReconstructionModel(),
        [packet],
        0.0,
        ldpc,
        torch.device("cpu"),
        "bpsk",
        transmit_fn=identity,
    )
    assert len(calls) == 1
    assert result["payload_bits"] == 4352
    assert result["coded_bits"] == 8704
    assert result["segment_bit_errors"] == {
        "first": 0,
        "mask_raw": 0,
        "second_active": 0,
    }
    assert all(scale["exact_mask_frame_rate"] == 1.0 for scale in result["per_scale"])


def test_raw_mask_shell_is_explicit_bpsk_and_executable():
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK)
    content = SCRIPT.read_text(encoding="utf-8")
    assert "test_independent_raq_rvq_adaptive_topk_raw_mask.py" in content
    assert 'RVQ_K_LISTS="${RVQ_K_LISTS:-2,2;8,2}"' in content
    assert 'MODULATION="${MODULATION:-bpsk}"' in content
    assert 'SNRS="${SNRS:-0}"' in content
    assert "explicit one bit per token" in content
    assert "rle" not in SCRIPT.name
    completed = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
