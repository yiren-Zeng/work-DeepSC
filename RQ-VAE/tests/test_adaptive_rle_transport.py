"""CPU contracts for fixed-width RLE adaptive mask transport."""

import numpy as np
import torch
from pathlib import Path
import os
import subprocess

import evaluation.adaptive_rle_channel as rle_channel
from evaluation.adaptive_channel import PreparedAdaptivePacket
from evaluation.adaptive_rle_channel import (
    evaluate_rle_ldpc_bpsk,
    summarize_rle_packets,
)
from utils.adaptive_rle_transport import (
    decode_fixed_width_rle_mask,
    encode_fixed_width_rle_mask,
    pack_rle_mask_segments,
    rle_length_width,
    unpack_rle_mask_segments,
)
from utils.adaptive_transport import ldpc_segment_lengths


ROOT = Path(__file__).resolve().parents[1]
RLE_SCRIPT = (
    ROOT
    / "scripts/eval/"
    "test_rq_ema_k4-2_d2-2_rate047_adaptive_topk_rle_ldpc_bpsk_snr0.sh"
)


def _production_indices(all_active):
    indices = []
    for height, width, num_embeddings in ((32, 32, 4), (16, 16, 2)):
        tokens = height * width
        first = torch.arange(tokens).remainder(num_embeddings)
        second = (
            torch.arange(tokens).remainder(num_embeddings)
            if all_active
            else torch.full((tokens,), -1, dtype=torch.long)
        )
        indices.append(
            torch.stack([first, second], dim=-1).view(
                1, height, width, 2
            )
        )
    return indices


def test_fixed_width_rle_matches_the_six_three_five_example():
    mask = np.array(
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        dtype=np.uint8,
    )
    bits, metadata = encode_fixed_width_rle_mask(mask)
    decoded, stats = decode_fixed_width_rle_mask(bits, mask.size)

    assert rle_length_width(14) == 4
    assert metadata["run_lengths"] == [6, 3, 5]
    assert metadata["run_count"] == 3
    assert metadata["source_bits"] == 13
    assert np.array_equal(decoded, mask)
    assert stats["structurally_valid"] is True
    assert stats["length_sum_error"] == 0


def test_production_all_stop_and_all_active_lengths_and_roundtrip():
    for all_active, expected_source, expected_coded in (
        (False, 2324, 5120),
        (True, 4628, 9728),
    ):
        indices = _production_indices(all_active)
        segments, metadata = pack_rle_mask_segments(indices, [4, 2])
        decoded = {segment.name: segment.bits.copy() for segment in segments}
        recovered, stats = unpack_rle_mask_segments(decoded, metadata)

        assert metadata["first_stage_bits"] == 2304
        assert metadata["rle_mask_bits"] == 20
        assert metadata["raw_mask_bits_reference"] == 1280
        assert metadata["source_bits"] == expected_source
        assert [
            scale["rle"]["source_bits"] for scale in metadata["scales"]
        ] == [11, 9]
        assert sum(
            ldpc_segment_lengths(segment.bits.size)["coded_bits"]
            for segment in segments
        ) == expected_coded
        assert all(
            torch.equal(expected, actual)
            for expected, actual in zip(indices, recovered)
        )
        assert all(
            scale["structurally_valid"]
            for scale in stats["per_scale"]
        )


def test_alternating_mask_exposes_the_fixed_width_rle_worst_case():
    shallow = np.arange(1024, dtype=np.uint8) % 2
    deep = np.arange(256, dtype=np.uint8) % 2
    shallow_bits, shallow_metadata = encode_fixed_width_rle_mask(shallow)
    deep_bits, deep_metadata = encode_fixed_width_rle_mask(deep)

    assert shallow_metadata["run_count"] == 1024
    assert shallow_bits.size == 1 + 1024 * 10
    assert deep_metadata["run_count"] == 256
    assert deep_bits.size == 1 + 256 * 8
    assert shallow_bits.size > shallow.size
    assert deep_bits.size > deep.size


def test_corrupted_lengths_are_cropped_or_last_run_extended():
    mask = np.array([0, 0, 1, 1], dtype=np.uint8)
    bits, _ = encode_fixed_width_rle_mask(mask)
    # start=0, two 2-bit fields.  Decode lengths [1,1], sum 2: extend the
    # final emitted value (1) through the missing tail.
    underflow_bits = np.array([0, 0, 0, 0, 0], dtype=np.uint8)
    decoded, stats = decode_fixed_width_rle_mask(underflow_bits, 4)
    assert np.array_equal(decoded, np.array([0, 1, 1, 1]))
    assert stats["underflow_tokens"] == 2
    assert stats["structurally_valid"] is False

    # Lengths [4,4] overflow and are cropped to the four-token target.
    overflow_bits = np.array([0, 1, 1, 1, 1], dtype=np.uint8)
    decoded, stats = decode_fixed_width_rle_mask(overflow_bits, 4)
    assert np.array_equal(decoded, np.zeros(4, dtype=np.uint8))
    assert stats["overflow_tokens"] == 4
    assert stats["structurally_valid"] is False
    assert bits.size == underflow_bits.size == overflow_bits.size


def test_received_rle_mask_controls_payload_mapping_without_tx_mask():
    indices = [
        torch.tensor(
            [[[[0, -1], [1, 2]], [[2, -1], [3, 1]]]],
            dtype=torch.long,
        )
    ]
    segments, metadata = pack_rle_mask_segments(indices, [4])
    decoded = {segment.name: segment.bits.copy() for segment in segments}
    # Original mask 0101 has four runs.  Keep the same source length but
    # decode lengths [1,3,1,1], which fills the grid as 0111.
    decoded["mask_rle_s0"] = np.array(
        [0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.uint8
    )
    recovered, stats = unpack_rle_mask_segments(decoded, metadata)

    assert torch.equal(
        recovered[0][..., 1].reshape(-1),
        torch.tensor([-1, 2, 1, 0]),
    )
    scale = stats["per_scale"][0]
    assert scale["tx_active_count"] == 2
    assert scale["rx_active_count"] == 3
    assert scale["zero_filled_second_indices"] == 1


def test_rle_rejects_empty_and_misaligned_inputs():
    invalid_calls = (
        lambda: encode_fixed_width_rle_mask(np.empty(0, dtype=np.uint8)),
        lambda: decode_fixed_width_rle_mask(
            np.array([], dtype=np.uint8), 4
        ),
        lambda: decode_fixed_width_rle_mask(
            np.array([0, 1], dtype=np.uint8), 4
        ),
        lambda: rle_length_width(0),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid RLE transport input was accepted")


def test_rle_snr0_script_is_isolated_executable_and_locked():
    assert RLE_SCRIPT.is_file()
    assert os.access(RLE_SCRIPT, os.X_OK)
    content = RLE_SCRIPT.read_text(encoding="utf-8")
    assert "cd /workspace/yi/work/RQ-VAE" in content
    assert "/workspace/yi/work/Simvq-dc-64-Multi-pro-2ceng" not in content
    assert 'SIMVQ_QUANTIZER_TYPE="rq_ema"' in content
    assert 'SIMVQ_NUM_EMBEDDINGS_LIST="4,2"' in content
    assert 'SIMVQ_RQ_DEPTH_LIST="2,2"' in content
    assert "test_adaptive_topk_rle_ldpc.py" in content
    assert "--snr 0" in content
    assert "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0" in content
    assert "no raw-mask fallback" in content
    syntax = subprocess.run(
        ["bash", "-n", str(RLE_SCRIPT)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr


def test_rle_rate_summary_and_identity_channel_evaluation():
    indices = _production_indices(all_active=False)
    segments, metadata = pack_rle_mask_segments(indices, [4, 2])
    packet = PreparedAdaptivePacket(
        image=torch.zeros(1, 3, 256, 256),
        feature_shapes=[(32, 32), (16, 16)],
        tx_indices=indices,
        segments=segments,
        metadata=metadata,
        clean_psnr=21.0,
        clean_ms_ssim=0.8,
    )
    packets = [packet, packet]
    summary = summarize_rle_packets(packets)
    assert summary["rle_source_bits"] == 2 * 2324
    assert summary["raw_explicit_source_bits_reference"] == 2 * 3584
    assert summary["rle_coded_bits_with_ldpc_padding"] == 2 * 5120
    assert summary["per_scale"][0][
        "rle_mask_source_bits_mean_per_image"
    ] == 11
    assert summary["per_scale"][1][
        "rle_mask_source_bits_mean_per_image"
    ] == 9

    original_transmit = rle_channel._transmit_segment
    rle_channel._transmit_segment = (
        lambda bits, snr_db, ldpc_code, device: (
            bits.copy(),
            ldpc_segment_lengths(bits.size),
        )
    )
    try:
        channel = evaluate_rle_ldpc_bpsk(
            object(),
            packets,
            0.0,
            {"k": 128, "n": 256},
            torch.device("cpu"),
            seed=42,
        )
    finally:
        rle_channel._transmit_segment = original_transmit

    assert channel["psnr"] == 21.0
    assert channel["ms_ssim"] == 0.8
    assert channel["source_bit_errors"] == 0
    assert channel["segment_bit_errors"]["mask_rle"] == 0
    assert all(
        scale["structurally_valid_rle_frame_rate"] == 1.0
        and scale["exact_mask_frame_rate"] == 1.0
        and scale["semantic_mask_ber"] == 0.0
        for scale in channel["per_scale"]
    )
