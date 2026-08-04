"""Lossy fixed-width RLE-mask evaluation over LDPC 1/2+BPSK/AWGN."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from evaluation.adaptive import AdaptiveSample, _quality_metrics
from evaluation.adaptive_channel import (
    PreparedAdaptivePacket,
    _reset_channel_seed,
    _transmit_segment,
)
from evaluation.adaptive_topk import apply_adaptive_topk
from utils.adaptive_rle_transport import (
    pack_rle_mask_segments,
    unpack_rle_mask_segments,
)
from utils.adaptive_transport import ldpc_segment_lengths


@torch.no_grad()
def prepare_per_image_topk_rle_packets(
    model,
    samples: Sequence[AdaptiveSample],
    target_active_rates: Sequence[float],
    num_embeddings_list: Sequence[int],
    image_names: Sequence[str] | None = None,
) -> Tuple[List[PreparedAdaptivePacket], List[Dict[str, object]]]:
    """Prepare exact Top-K packets using RLE rather than raw mask segments."""

    if not samples:
        raise ValueError("samples must not be empty")
    if image_names is not None and len(image_names) != len(samples):
        raise ValueError("image_names must match the number of samples")
    packets: List[PreparedAdaptivePacket] = []
    selection_records: List[Dict[str, object]] = []

    for image_index, sample in enumerate(samples):
        if int(sample.image.shape[0]) != 1:
            raise ValueError("RLE Top-K preparation requires batch_size=1")
        tx_indices, _, batch_records = apply_adaptive_topk(
            sample.dense_indices,
            sample.first_stage_errors,
            target_active_rates,
        )
        if len(batch_records) != 1:
            raise RuntimeError("expected exactly one Top-K image record")
        segments, metadata = pack_rle_mask_segments(
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
        selection_records.append(
            {
                "image_index": image_index,
                "image_number": image_index + 1,
                "image_name": (
                    str(image_names[image_index])
                    if image_names is not None
                    else f"image_{image_index + 1:04d}"
                ),
                "scales": batch_records[0]["scales"],
            }
        )
    return packets, selection_records


def summarize_rle_packets(
    packets: Sequence[PreparedAdaptivePacket],
) -> Dict[str, object]:
    """Summarize source and padded LDPC lengths, globally and per scale."""

    if not packets:
        raise ValueError("packets must not be empty")
    num_scales = len(packets[0].metadata["scales"])
    per_scale: List[Dict[str, object]] = [
        {
            "scale": scale,
            "num_images": len(packets),
            "token_count": 0,
            "tx_active_count": 0,
            "first_stage_source_bits": 0,
            "raw_mask_source_bits_reference": 0,
            "rle_mask_source_bits": 0,
            "active_second_source_bits": 0,
            "rle_run_count": 0,
            "rle_mask_source_bits_per_image": [],
            "rle_run_count_per_image": [],
            "first_stage_coded_bits": 0,
            "raw_mask_coded_bits_reference": 0,
            "rle_mask_coded_bits": 0,
            "active_second_coded_bits": 0,
            "ldpc_padding_bits": 0,
        }
        for scale in range(num_scales)
    ]
    total_pixels = 0

    for packet in packets:
        total_pixels += int(
            packet.image.shape[0]
            * packet.image.shape[-2]
            * packet.image.shape[-1]
        )
        segments = {segment.name: segment for segment in packet.segments}
        for scale_metadata in packet.metadata["scales"]:
            scale = int(scale_metadata["scale"])
            stats = per_scale[scale]
            names = scale_metadata["segment_names"]
            first = segments[names["first"]]
            mask = segments[names["mask_rle"]]
            second = segments[names["second"]]
            first_lengths = ldpc_segment_lengths(first.bits.size)
            mask_lengths = ldpc_segment_lengths(mask.bits.size)
            second_lengths = ldpc_segment_lengths(second.bits.size)
            raw_mask_lengths = ldpc_segment_lengths(
                scale_metadata["raw_mask_source_bits_reference"]
            )

            stats["token_count"] += int(scale_metadata["token_count"])
            stats["tx_active_count"] += int(
                scale_metadata["tx_active_count"]
            )
            stats["first_stage_source_bits"] += int(first.bits.size)
            stats["raw_mask_source_bits_reference"] += int(
                scale_metadata["raw_mask_source_bits_reference"]
            )
            stats["rle_mask_source_bits"] += int(mask.bits.size)
            stats["active_second_source_bits"] += int(second.bits.size)
            stats["rle_run_count"] += int(
                scale_metadata["rle"]["run_count"]
            )
            stats["rle_mask_source_bits_per_image"].append(
                int(mask.bits.size)
            )
            stats["rle_run_count_per_image"].append(
                int(scale_metadata["rle"]["run_count"])
            )
            stats["first_stage_coded_bits"] += first_lengths["coded_bits"]
            stats["raw_mask_coded_bits_reference"] += raw_mask_lengths[
                "coded_bits"
            ]
            stats["rle_mask_coded_bits"] += mask_lengths["coded_bits"]
            stats["active_second_coded_bits"] += second_lengths["coded_bits"]
            stats["ldpc_padding_bits"] += (
                first_lengths["padding_bits"]
                + mask_lengths["padding_bits"]
                + second_lengths["padding_bits"]
            )

    for stats in per_scale:
        num_images = int(stats["num_images"])
        token_count = int(stats["token_count"])
        tx_active_count = int(stats["tx_active_count"])
        raw_mask = int(stats["raw_mask_source_bits_reference"])
        rle_mask = int(stats["rle_mask_source_bits"])
        second = int(stats["active_second_source_bits"])
        first_coded = int(stats["first_stage_coded_bits"])
        raw_mask_coded = int(stats["raw_mask_coded_bits_reference"])
        rle_mask_coded = int(stats["rle_mask_coded_bits"])
        second_coded = int(stats["active_second_coded_bits"])
        mask_bits_per_image = stats.pop("rle_mask_source_bits_per_image")
        run_counts_per_image = stats.pop("rle_run_count_per_image")
        stats.update(
            {
                "token_count_per_image": token_count // num_images,
                "tx_active_count_per_image": tx_active_count // num_images,
                "tx_active_ratio": (
                    tx_active_count / token_count if token_count else 0.0
                ),
                "rle_run_count_mean_per_image": (
                    sum(run_counts_per_image) / num_images
                ),
                "rle_run_count_min_per_image": min(run_counts_per_image),
                "rle_run_count_max_per_image": max(run_counts_per_image),
                "rle_mask_source_bits_mean_per_image": (
                    sum(mask_bits_per_image) / num_images
                ),
                "rle_mask_source_bits_min_per_image": min(
                    mask_bits_per_image
                ),
                "rle_mask_source_bits_max_per_image": max(
                    mask_bits_per_image
                ),
                "mask_source_bits_saved_vs_raw": raw_mask - rle_mask,
                "mask_source_saving_ratio": (
                    (raw_mask - rle_mask) / raw_mask if raw_mask else 0.0
                ),
                "second_layer_source_bits_raw_reference": raw_mask + second,
                "second_layer_source_bits_rle": rle_mask + second,
                "second_layer_source_bits_saved": raw_mask - rle_mask,
                "second_layer_coded_bits_raw_reference": (
                    raw_mask_coded + second_coded
                ),
                "second_layer_coded_bits_rle": (
                    rle_mask_coded + second_coded
                ),
                "second_layer_coded_bits_saved": (
                    raw_mask_coded - rle_mask_coded
                ),
                "total_coded_bits_raw_reference": (
                    first_coded + raw_mask_coded + second_coded
                ),
                "total_coded_bits_rle": (
                    first_coded + rle_mask_coded + second_coded
                ),
            }
        )

    first_source = sum(
        int(scale["first_stage_source_bits"]) for scale in per_scale
    )
    raw_mask_source = sum(
        int(scale["raw_mask_source_bits_reference"]) for scale in per_scale
    )
    rle_mask_source = sum(
        int(scale["rle_mask_source_bits"]) for scale in per_scale
    )
    second_source = sum(
        int(scale["active_second_source_bits"]) for scale in per_scale
    )
    raw_coded = sum(
        int(scale["total_coded_bits_raw_reference"]) for scale in per_scale
    )
    rle_coded = sum(
        int(scale["total_coded_bits_rle"]) for scale in per_scale
    )
    raw_source = first_source + raw_mask_source + second_source
    rle_source = first_source + rle_mask_source + second_source
    return {
        "num_images": len(packets),
        "total_image_pixels": total_pixels,
        "first_stage_bits": first_source,
        "raw_mask_bits_reference": raw_mask_source,
        "rle_mask_bits": rle_mask_source,
        "active_second_bits": second_source,
        "raw_explicit_source_bits_reference": raw_source,
        "rle_source_bits": rle_source,
        "raw_explicit_source_bpp_reference": raw_source / total_pixels,
        "rle_source_bpp": rle_source / total_pixels,
        "source_bits_saved_vs_raw": raw_source - rle_source,
        "source_saving_ratio_vs_raw": (
            (raw_source - rle_source) / raw_source if raw_source else 0.0
        ),
        "raw_explicit_coded_bits_with_ldpc_padding_reference": raw_coded,
        "rle_coded_bits_with_ldpc_padding": rle_coded,
        "coded_bits_saved_vs_raw": raw_coded - rle_coded,
        "raw_explicit_coded_bpp_reference": raw_coded / total_pixels,
        "rle_coded_bpp": rle_coded / total_pixels,
        "rle_bpsk_channel_uses_per_rgb_value": (
            rle_coded / (total_pixels * 3)
        ),
        "framing_overhead_counted": False,
        "per_scale": per_scale,
    }


@torch.no_grad()
def evaluate_rle_ldpc_bpsk(
    model,
    packets: Sequence[PreparedAdaptivePacket],
    snr_db: float,
    ldpc_code,
    device: torch.device,
    seed: int = 42,
) -> Dict[str, object]:
    """Transmit and decode all RLE packet segments through the real channel."""

    if not packets:
        raise ValueError("packets must not be empty")
    _reset_channel_seed(seed)
    num_scales = len(packets[0].metadata["scales"])
    per_scale: List[Dict[str, object]] = [
        {
            "scale": scale,
            "source_bits": 0,
            "coded_bits": 0,
            "source_bit_errors": 0,
            "rle_source_bits": 0,
            "rle_source_bit_errors": 0,
            "rle_start_bit_errors": 0,
            "rle_length_bit_errors": 0,
            "tx_active_count": 0,
            "rx_active_count": 0,
            "semantic_mask_bit_errors": 0,
            "mask_token_count": 0,
            "structurally_valid_rle_frames": 0,
            "exact_mask_frames": 0,
            "active_count_mismatch_frames": 0,
            "zero_filled_second_indices": 0,
            "truncated_second_indices": 0,
        }
        for scale in range(num_scales)
    ]
    kind_bits = {"first": 0, "mask_rle": 0, "second_active": 0}
    kind_errors = {"first": 0, "mask_rle": 0, "second_active": 0}
    psnr_values: List[float] = []
    ms_ssim_values: List[float] = []
    per_image: List[Dict[str, object]] = []
    total_source_bits = 0
    total_coded_bits = 0
    total_padding_bits = 0
    total_source_errors = 0
    images_with_source_errors = 0

    for image_index, packet in enumerate(packets):
        decoded_segments: Dict[str, np.ndarray] = {}
        packet_errors = 0
        packet_source_bits = 0
        packet_coded_bits = 0
        segment_records: Dict[str, Dict[str, int]] = {}
        for segment in packet.segments:
            decoded, lengths = _transmit_segment(
                segment.bits, snr_db, ldpc_code, device
            )
            decoded_segments[segment.name] = decoded
            errors = int(np.count_nonzero(decoded != segment.bits))
            segment_records[segment.name] = {
                "source_bits": int(lengths["source_bits"]),
                "coded_bits": int(lengths["coded_bits"]),
                "padding_bits": int(lengths["padding_bits"]),
                "source_bit_errors": errors,
            }
            packet_errors += errors
            packet_source_bits += lengths["source_bits"]
            packet_coded_bits += lengths["coded_bits"]
            total_source_bits += lengths["source_bits"]
            total_coded_bits += lengths["coded_bits"]
            total_padding_bits += lengths["padding_bits"]
            total_source_errors += errors
            kind_bits[segment.kind] += lengths["source_bits"]
            kind_errors[segment.kind] += errors
            scale_stats = per_scale[int(segment.scale)]
            scale_stats["source_bits"] += lengths["source_bits"]
            scale_stats["coded_bits"] += lengths["coded_bits"]
            scale_stats["source_bit_errors"] += errors
            if segment.kind == "mask_rle":
                scale_stats["rle_source_bits"] += lengths["source_bits"]
                scale_stats["rle_source_bit_errors"] += errors
                if segment.bits.size:
                    scale_stats["rle_start_bit_errors"] += int(
                        decoded[0] != segment.bits[0]
                    )
                    scale_stats["rle_length_bit_errors"] += int(
                        np.count_nonzero(decoded[1:] != segment.bits[1:])
                    )

        recovered_indices, decode_stats = unpack_rle_mask_segments(
            decoded_segments, packet.metadata
        )
        image_scale_records = []
        for scale, (tx_indices, rx_indices, scale_decode) in enumerate(
            zip(
                packet.tx_indices,
                recovered_indices,
                decode_stats["per_scale"],
            )
        ):
            tx_mask = (
                tx_indices[..., 1].detach().cpu().numpy().reshape(-1) >= 0
            )
            rx_mask = rx_indices[..., 1].numpy().reshape(-1) >= 0
            semantic_errors = int(np.count_nonzero(tx_mask != rx_mask))
            stats = per_scale[scale]
            stats["tx_active_count"] += int(tx_mask.sum())
            stats["rx_active_count"] += int(rx_mask.sum())
            stats["semantic_mask_bit_errors"] += semantic_errors
            stats["mask_token_count"] += int(tx_mask.size)
            stats["structurally_valid_rle_frames"] += int(
                scale_decode["structurally_valid"]
            )
            stats["exact_mask_frames"] += int(semantic_errors == 0)
            stats["active_count_mismatch_frames"] += int(
                scale_decode["active_count_mismatch"] != 0
            )
            stats["zero_filled_second_indices"] += int(
                scale_decode["zero_filled_second_indices"]
            )
            stats["truncated_second_indices"] += int(
                scale_decode["truncated_second_indices"]
            )
            names = packet.metadata["scales"][scale]["segment_names"]
            rle_segment = segment_records[names["mask_rle"]]
            image_scale_records.append(
                {
                    "scale": scale,
                    "tx_active_count": int(tx_mask.sum()),
                    "rx_active_count": int(rx_mask.sum()),
                    "semantic_mask_bit_errors": semantic_errors,
                    "semantic_mask_ber": semantic_errors / int(tx_mask.size),
                    "rle_source_bits": rle_segment["source_bits"],
                    "rle_coded_bits": rle_segment["coded_bits"],
                    "rle_source_bit_errors": rle_segment[
                        "source_bit_errors"
                    ],
                    "rle_run_count": int(
                        packet.metadata["scales"][scale]["rle"]["run_count"]
                    ),
                    "structurally_valid": bool(
                        scale_decode["structurally_valid"]
                    ),
                    "length_sum_error": int(
                        scale_decode["length_sum_error"]
                    ),
                    "zero_filled_second_indices": int(
                        scale_decode["zero_filled_second_indices"]
                    ),
                    "truncated_second_indices": int(
                        scale_decode["truncated_second_indices"]
                    ),
                }
            )

        if packet_errors:
            images_with_source_errors += 1
            recovered_device = [
                indices.to(device, non_blocking=True)
                for indices in recovered_indices
            ]
            reconstructed = model.reconstruct_from_adaptive_indices(
                recovered_device, feature_shapes=packet.feature_shapes
            )
            ms_ssim, psnr = _quality_metrics(packet.image, reconstructed)
            image_psnr = float(psnr)
            image_ms_ssim = float(ms_ssim)
        else:
            image_psnr = packet.clean_psnr
            image_ms_ssim = packet.clean_ms_ssim
        psnr_values.append(image_psnr)
        ms_ssim_values.append(image_ms_ssim)
        per_image.append(
            {
                "image_index": image_index,
                "image_number": image_index + 1,
                "source_bits": int(packet_source_bits),
                "coded_bits": int(packet_coded_bits),
                "source_bit_errors": int(packet_errors),
                "psnr": image_psnr,
                "ms_ssim": image_ms_ssim,
                "scales": image_scale_records,
            }
        )

    for stats in per_scale:
        source_bits = int(stats["source_bits"])
        rle_bits = int(stats["rle_source_bits"])
        token_count = int(stats["mask_token_count"])
        frame_count = len(packets)
        stats.update(
            {
                "source_ber_after_ldpc": (
                    stats["source_bit_errors"] / source_bits
                    if source_bits
                    else 0.0
                ),
                "rle_source_ber_after_ldpc": (
                    stats["rle_source_bit_errors"] / rle_bits
                    if rle_bits
                    else 0.0
                ),
                "semantic_mask_ber": (
                    stats["semantic_mask_bit_errors"] / token_count
                    if token_count
                    else 0.0
                ),
                "tx_active_ratio": (
                    stats["tx_active_count"] / token_count
                    if token_count
                    else 0.0
                ),
                "rx_active_ratio": (
                    stats["rx_active_count"] / token_count
                    if token_count
                    else 0.0
                ),
                "structurally_valid_rle_frame_rate": (
                    stats["structurally_valid_rle_frames"] / frame_count
                ),
                "exact_mask_frame_rate": (
                    stats["exact_mask_frames"] / frame_count
                ),
            }
        )

    total_pixels = sum(
        int(
            packet.image.shape[0]
            * packet.image.shape[-2]
            * packet.image.shape[-1]
        )
        for packet in packets
    )
    return {
        "snr_db": float(snr_db),
        "psnr": float(np.mean(psnr_values)),
        "ms_ssim": float(np.mean(ms_ssim_values)),
        "source_bits": int(total_source_bits),
        "source_bpp": total_source_bits / total_pixels,
        "coded_bits_with_ldpc_padding": int(total_coded_bits),
        "coded_bpp_with_ldpc_padding": total_coded_bits / total_pixels,
        "bpsk_channel_uses_per_rgb_value": (
            total_coded_bits / (total_pixels * 3)
        ),
        "ldpc_padding_bits": int(total_padding_bits),
        "source_bit_errors": int(total_source_errors),
        "source_ber_after_ldpc": (
            total_source_errors / total_source_bits
            if total_source_bits
            else 0.0
        ),
        "segment_source_bits": kind_bits,
        "segment_bit_errors": kind_errors,
        "images_with_source_errors": int(images_with_source_errors),
        "per_scale": per_scale,
        "per_image": per_image,
    }


__all__ = [
    "evaluate_rle_ldpc_bpsk",
    "prepare_per_image_topk_rle_packets",
    "summarize_rle_packets",
]
