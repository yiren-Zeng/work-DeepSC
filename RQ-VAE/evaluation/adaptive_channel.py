"""Real LDPC/BPSK channel evaluation for adaptive explicit-mask EMA-RQ."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch

from communications.channel import awgn_channel
from communications.modulation import bpsk_llr, bpsk_modulate
from evaluation.adaptive import (
    AdaptiveSample,
    _quality_metrics,
    apply_adaptive_thresholds,
)
from utils.adaptive_transport import (
    AdaptiveTransportSegment,
    ldpc_segment_lengths,
    pack_explicit_mask_segments,
    unpack_explicit_mask_segments,
)


@dataclass
class PreparedAdaptivePacket:
    """One image's source packet and clean reconstruction metrics."""

    image: torch.Tensor
    feature_shapes: List[tuple]
    tx_indices: List[torch.Tensor]
    segments: List[AdaptiveTransportSegment]
    metadata: Dict[str, object]
    clean_psnr: float
    clean_ms_ssim: float


def _reset_channel_seed(seed: int = 42) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _transmit_segment(
    source_bits: np.ndarray,
    snr_db: float,
    ldpc_code,
    device: torch.device,
):
    from communications.ldpc_coding import ldpc_decode, ldpc_encode

    source_bits = (
        np.asarray(source_bits).reshape(-1) != 0
    ).astype(np.uint8)
    lengths = ldpc_segment_lengths(
        source_bits.size,
        information_block_bits=int(ldpc_code["k"]),
        coded_block_bits=int(ldpc_code["n"]),
    )
    if source_bits.size == 0:
        return source_bits.copy(), lengths

    coded_bits = ldpc_encode(source_bits, code=ldpc_code)
    if int(np.asarray(coded_bits).size) != lengths["coded_bits"]:
        raise RuntimeError(
            "LDPC encoder length mismatch: "
            f"{np.asarray(coded_bits).size} vs {lengths['coded_bits']}"
        )
    coded_tensor = torch.from_numpy(
        np.asarray(coded_bits, dtype=np.float32)
    ).to(device)
    symbols = bpsk_modulate(coded_tensor)
    noisy_symbols = awgn_channel(symbols, float(snr_db))
    llrs = bpsk_llr(noisy_symbols, float(snr_db), device)
    decoded = np.asarray(
        ldpc_decode(llrs.detach().cpu().numpy(), ldpc_code),
        dtype=np.uint8,
    ).reshape(-1)
    if decoded.size < source_bits.size:
        decoded = np.pad(decoded, (0, source_bits.size - decoded.size))
    return (decoded[: source_bits.size] != 0).astype(np.uint8), lengths


@torch.no_grad()
def prepare_adaptive_packets(
    model,
    samples: Sequence[AdaptiveSample],
    thresholds: Sequence[float],
    num_embeddings_list: Sequence[int],
) -> List[PreparedAdaptivePacket]:
    """Prepare source packets once and cache their no-channel image metrics."""

    packets: List[PreparedAdaptivePacket] = []
    for sample in samples:
        tx_indices, _ = apply_adaptive_thresholds(
            sample.dense_indices,
            sample.first_stage_errors,
            thresholds,
        )
        segments, metadata = pack_explicit_mask_segments(
            tx_indices, num_embeddings_list
        )
        reconstructed = model.reconstruct_from_adaptive_indices(
            tx_indices, feature_shapes=sample.feature_shapes
        )
        clean_ms_ssim, clean_psnr = _quality_metrics(
            sample.image, reconstructed
        )
        packets.append(
            PreparedAdaptivePacket(
                image=sample.image,
                feature_shapes=list(sample.feature_shapes),
                tx_indices=tx_indices,
                segments=segments,
                metadata=metadata,
                clean_psnr=float(clean_psnr),
                clean_ms_ssim=float(clean_ms_ssim),
            )
        )
    return packets


def summarize_prepared_packets(
    packets: Sequence[PreparedAdaptivePacket],
) -> Dict[str, object]:
    if not packets:
        raise ValueError("packets must not be empty")
    total_pixels = 0
    source_bits = 0
    first_bits = 0
    mask_bits = 0
    active_second_bits = 0
    coded_bits = 0
    padding_bits = 0
    active_counts = [0] * len(packets[0].metadata["scales"])
    token_counts = [0] * len(active_counts)

    for packet in packets:
        batch, _, height, width = packet.image.shape
        total_pixels += int(batch * height * width)
        source_bits += int(packet.metadata["source_bits"])
        first_bits += int(packet.metadata["first_stage_bits"])
        mask_bits += int(packet.metadata["mask_bits"])
        active_second_bits += int(packet.metadata["active_second_bits"])
        for segment in packet.segments:
            lengths = ldpc_segment_lengths(segment.bits.size)
            coded_bits += lengths["coded_bits"]
            padding_bits += lengths["padding_bits"]
        for scale_metadata in packet.metadata["scales"]:
            scale = int(scale_metadata["scale"])
            active_counts[scale] += int(scale_metadata["tx_active_count"])
            token_counts[scale] += int(scale_metadata["token_count"])

    return {
        "num_images": len(packets),
        "total_image_pixels": total_pixels,
        "source_bits": source_bits,
        "first_stage_bits": first_bits,
        "mask_bits": mask_bits,
        "active_second_bits": active_second_bits,
        "source_bpp": source_bits / total_pixels,
        "coded_bits_with_ldpc_padding": coded_bits,
        "ldpc_padding_bits": padding_bits,
        "coded_bpp_with_ldpc_padding": coded_bits / total_pixels,
        "bpsk_channel_uses_per_rgb_value": coded_bits
        / (total_pixels * 3),
        "active_ratios": [
            active / tokens if tokens else 0.0
            for active, tokens in zip(active_counts, token_counts)
        ],
        "active_counts": active_counts,
        "token_counts": token_counts,
        "framing_overhead_counted": False,
    }


@torch.no_grad()
def evaluate_adaptive_ldpc_bpsk(
    model,
    packets: Sequence[PreparedAdaptivePacket],
    snr_db: float,
    ldpc_code,
    device: torch.device,
    seed: int = 42,
) -> Dict[str, object]:
    """Transmit every explicit-mask segment through LDPC 1/2+BPSK+AWGN."""

    if not packets:
        raise ValueError("packets must not be empty")
    _reset_channel_seed(seed)
    psnr_values = []
    ms_ssim_values = []
    total_source_bits = 0
    total_coded_bits = 0
    total_padding_bits = 0
    total_bit_errors = 0
    kind_bits: Dict[str, int] = {
        "first": 0,
        "mask": 0,
        "second_active": 0,
    }
    kind_errors: Dict[str, int] = {
        "first": 0,
        "mask": 0,
        "second_active": 0,
    }
    images_with_errors = 0
    count_mismatch_images = 0
    zero_filled = 0
    truncated = 0

    for packet in packets:
        decoded_segments: Dict[str, np.ndarray] = {}
        packet_errors = 0
        for segment in packet.segments:
            decoded, lengths = _transmit_segment(
                segment.bits, snr_db, ldpc_code, device
            )
            decoded_segments[segment.name] = decoded
            errors = int(np.count_nonzero(decoded != segment.bits))
            packet_errors += errors
            total_bit_errors += errors
            total_source_bits += lengths["source_bits"]
            total_coded_bits += lengths["coded_bits"]
            total_padding_bits += lengths["padding_bits"]
            kind_bits[segment.kind] += lengths["source_bits"]
            kind_errors[segment.kind] += errors

        if packet_errors == 0:
            psnr_values.append(packet.clean_psnr)
            ms_ssim_values.append(packet.clean_ms_ssim)
            continue

        images_with_errors += 1
        recovered_indices, decode_stats = unpack_explicit_mask_segments(
            decoded_segments, packet.metadata
        )
        for scale_stats in decode_stats["per_scale"]:
            mismatch = int(scale_stats["active_count_mismatch"])
            if mismatch:
                count_mismatch_images += 1
            zero_filled += int(scale_stats["zero_filled_second_indices"])
            truncated += int(scale_stats["truncated_second_indices"])
        recovered_indices = [
            indices.to(device, non_blocking=True)
            for indices in recovered_indices
        ]
        reconstructed = model.reconstruct_from_adaptive_indices(
            recovered_indices, feature_shapes=packet.feature_shapes
        )
        ms_ssim, psnr = _quality_metrics(packet.image, reconstructed)
        psnr_values.append(float(psnr))
        ms_ssim_values.append(float(ms_ssim))

    total_pixels = sum(
        int(packet.image.shape[0] * packet.image.shape[-2] * packet.image.shape[-1])
        for packet in packets
    )
    return {
        "snr_db": float(snr_db),
        "psnr": float(np.mean(psnr_values)),
        "ms_ssim": float(np.mean(ms_ssim_values)),
        "source_bits": total_source_bits,
        "source_bpp": total_source_bits / total_pixels,
        "coded_bits_with_ldpc_padding": total_coded_bits,
        "coded_bpp_with_ldpc_padding": total_coded_bits / total_pixels,
        "bpsk_channel_uses_per_rgb_value": total_coded_bits
        / (total_pixels * 3),
        "ldpc_padding_bits": total_padding_bits,
        "source_bit_errors": total_bit_errors,
        "source_ber_after_ldpc": (
            total_bit_errors / total_source_bits if total_source_bits else 0.0
        ),
        "segment_source_bits": kind_bits,
        "segment_bit_errors": kind_errors,
        "images_with_source_errors": images_with_errors,
        "mask_active_count_mismatch_scale_events": count_mismatch_images,
        "zero_filled_second_indices": zero_filled,
        "truncated_second_indices": truncated,
    }


__all__ = [
    "PreparedAdaptivePacket",
    "evaluate_adaptive_ldpc_bpsk",
    "prepare_adaptive_packets",
    "summarize_prepared_packets",
]
