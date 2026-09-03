"""Explicit one-bit-per-token masks for adaptive independent RAQ-RVQ.

This module is a baseline companion to
``evaluation.independent_raq_rvq_adaptive``.  It deliberately reuses the same
Top-K selection, compact second-stage payload, combined LDPC stream, channel,
and reconstruction contracts.  The only transport difference is that every
activity mask is sent directly as one bit per raster-order token instead of
being run-length encoded.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch

from evaluation.independent_raq_rvq_adaptive import (
    AdaptiveSegment,
    IndependentAdaptiveSample,
    PreparedAdaptivePacket,
    _bits_to_values,
    _quality_metrics,
    _reset_channel_seed,
    _split_combined_bits,
    _validate_dense_layout,
    _values_to_bits,
    bits_per_index,
    combined_bits_from_segments,
    combined_stream_lengths,
    reconstruct_from_adaptive_indices,
    select_per_image_topk_masks,
    transmit_combined_stream,
)


_MODULATION_BITS = {"bpsk": 1, "qpsk": 2, "16qam": 4}


def pack_topk_raw_mask_segments(
    indices_by_scale: Sequence[Sequence[torch.Tensor]],
    active_masks: Sequence[torch.Tensor],
    rvq_k_lists: Sequence[Sequence[int]],
) -> Tuple[List[AdaptiveSegment], Dict[str, object]]:
    """Pack Top-K indices with an explicit one-bit mask at each scale."""

    _validate_dense_layout(indices_by_scale, active_masks, rvq_k_lists)
    first_segments: List[AdaptiveSegment] = []
    mask_segments: List[AdaptiveSegment] = []
    second_segments: List[AdaptiveSegment] = []
    scales: List[Dict[str, object]] = []

    for scale, (stages, mask, stage_ks) in enumerate(
        zip(indices_by_scale, active_masks, rvq_k_lists)
    ):
        first_k, second_k = [int(value) for value in stage_ks]
        first_width = bits_per_index(first_k)
        second_width = bits_per_index(second_k)
        first = stages[0].detach().cpu().long().reshape(-1).numpy()
        second = stages[1].detach().cpu().long().reshape(-1).numpy()
        mask_flat = (
            mask.detach().cpu().bool().reshape(-1).numpy().astype(np.uint8)
        )
        active_second = second[mask_flat.astype(bool)]
        first_bits = _values_to_bits(first, first_width)
        second_bits = _values_to_bits(active_second, second_width)
        names = {
            "first": f"first_s{scale}",
            "mask_raw": f"mask_raw_s{scale}",
            "second": f"second_active_s{scale}",
        }
        first_segments.append(
            AdaptiveSegment(names["first"], "first", scale, first_bits)
        )
        mask_segments.append(
            AdaptiveSegment(names["mask_raw"], "mask_raw", scale, mask_flat)
        )
        second_segments.append(
            AdaptiveSegment(
                names["second"], "second_active", scale, second_bits
            )
        )
        token_count = int(first.size)
        scales.append(
            {
                "scale": scale,
                "shape": [int(stages[0].shape[1]), int(stages[0].shape[2])],
                "token_count": token_count,
                "first_num_embeddings": first_k,
                "second_num_embeddings": second_k,
                "first_bits_per_index": first_width,
                "second_bits_per_index": second_width,
                "tx_active_count": int(mask_flat.sum()),
                "segment_names": names,
                "segment_source_bits": {
                    "first": int(first_bits.size),
                    "mask_raw": int(mask_flat.size),
                    "second_active": int(second_bits.size),
                },
            }
        )

    segments = first_segments + mask_segments + second_segments
    offsets = []
    offset = 0
    for segment in segments:
        end = offset + int(segment.bits.size)
        offsets.append(
            {
                "name": segment.name,
                "kind": segment.kind,
                "scale": int(segment.scale),
                "offset": offset,
                "end": end,
                "source_bits": int(segment.bits.size),
            }
        )
        offset = end

    first_bits = sum(
        int(scale["segment_source_bits"]["first"]) for scale in scales
    )
    mask_bits = sum(
        int(scale["segment_source_bits"]["mask_raw"]) for scale in scales
    )
    second_bits = sum(
        int(scale["segment_source_bits"]["second_active"])
        for scale in scales
    )
    dense_second_bits = sum(
        int(scale["token_count"]) * int(scale["second_bits_per_index"])
        for scale in scales
    )
    metadata = {
        "schema": "independent_raq_rvq_topk_raw_mask_combined_v1",
        "stream_packing": "combined",
        "logical_segment_order": [segment.name for segment in segments],
        "segments": offsets,
        "scales": scales,
        "source_bits": first_bits + mask_bits + second_bits,
        "first_stage_bits": first_bits,
        "raw_mask_bits": mask_bits,
        "active_second_bits": second_bits,
        "dense_two_stage_source_bits_reference": (
            first_bits + dense_second_bits
        ),
        "framing_metadata_counted": False,
    }
    if int(metadata["source_bits"]) != offset:
        raise RuntimeError("raw-mask combined source length accounting mismatch")
    return segments, metadata


def unpack_topk_raw_mask_combined(
    decoded_bits: np.ndarray,
    metadata: Dict[str, object],
) -> Tuple[List[List[torch.Tensor]], List[torch.Tensor], Dict[str, object]]:
    """Recover indices using only the received explicit masks."""

    if metadata.get("schema") != "independent_raq_rvq_topk_raw_mask_combined_v1":
        raise ValueError("unsupported raw-mask combined transport schema")
    decoded = _split_combined_bits(decoded_bits, metadata)
    recovered: List[List[torch.Tensor]] = []
    masks: List[torch.Tensor] = []
    per_scale = []

    for scale_metadata in metadata["scales"]:
        scale = int(scale_metadata["scale"])
        height, width = [int(value) for value in scale_metadata["shape"]]
        token_count = int(scale_metadata["token_count"])
        first_k = int(scale_metadata["first_num_embeddings"])
        second_k = int(scale_metadata["second_num_embeddings"])
        first_width = int(scale_metadata["first_bits_per_index"])
        second_width = int(scale_metadata["second_bits_per_index"])
        tx_active_count = int(scale_metadata["tx_active_count"])
        names = scale_metadata["segment_names"]

        first = _bits_to_values(
            decoded[names["first"]], token_count, first_width, first_k
        )
        mask = (
            np.asarray(decoded[names["mask_raw"]]).reshape(-1) != 0
        ).astype(np.uint8)
        if int(mask.size) != token_count:
            raise RuntimeError("decoded explicit mask length mismatch")
        active_values = _bits_to_values(
            decoded[names["second"]],
            tx_active_count,
            second_width,
            second_k,
        )
        rx_positions = np.flatnonzero(mask)
        rx_active_count = int(rx_positions.size)
        mapped = min(rx_active_count, tx_active_count)
        zero_filled = max(rx_active_count - tx_active_count, 0)
        truncated = max(tx_active_count - rx_active_count, 0)
        second = np.full(token_count, -1, dtype=np.int64)
        if mapped:
            second[rx_positions[:mapped]] = active_values[:mapped]
        if zero_filled:
            second[rx_positions[mapped:]] = 0

        first_tensor = torch.from_numpy(first.reshape(1, height, width)).long()
        second_tensor = torch.from_numpy(second.reshape(1, height, width)).long()
        mask_tensor = second_tensor >= 0
        recovered.append([first_tensor, second_tensor])
        masks.append(mask_tensor)
        per_scale.append(
            {
                "scale": scale,
                "tx_active_count": tx_active_count,
                "rx_active_count": rx_active_count,
                "active_count_mismatch": rx_active_count - tx_active_count,
                "mapped_second_indices": mapped,
                "zero_filled_second_indices": zero_filled,
                "truncated_second_indices": truncated,
            }
        )
    return recovered, masks, {"per_scale": per_scale}


@torch.no_grad()
def prepare_topk_raw_mask_packets(
    model,
    samples: Sequence[IndependentAdaptiveSample],
    target_active_rates: Sequence[float],
    rvq_k_lists: Sequence[Sequence[int]],
    image_names: Sequence[str] | None = None,
) -> Tuple[List[PreparedAdaptivePacket], List[Dict[str, object]]]:
    """Apply Top-K and prepare one combined explicit-mask packet per image."""

    if not samples:
        raise ValueError("samples must not be empty")
    if image_names is not None and len(image_names) != len(samples):
        raise ValueError("image_names must align with samples")
    num_scales = len(rvq_k_lists)
    if num_scales <= 0 or len(target_active_rates) != num_scales:
        raise ValueError("active rates and K lists must have equal scale counts")
    packets = []
    selection_records = []

    for image_index, sample in enumerate(samples):
        if not (
            len(sample.first_stage_errors)
            == len(sample.dense_indices)
            == len(sample.codebooks)
            == len(sample.feature_shapes)
            == num_scales
        ):
            raise ValueError(
                f"sample {image_index} adaptive scale layouts do not align"
            )
        for scale, (stage_codebooks, stage_ks) in enumerate(
            zip(sample.codebooks, rvq_k_lists)
        ):
            if len(stage_codebooks) != 2 or len(stage_ks) != 2:
                raise ValueError(f"scale {scale} must have exactly two stages")
            for stage, (codebook, expected_k) in enumerate(
                zip(stage_codebooks, stage_ks)
            ):
                if int(codebook.shape[0]) != int(expected_k):
                    raise ValueError(
                        f"scale {scale} stage {stage} codebook K differs "
                        "from the configured K"
                    )

        masks, batch_records = select_per_image_topk_masks(
            sample.first_stage_errors, target_active_rates
        )
        if len(batch_records) != 1:
            raise RuntimeError("expected one selection record per sample")
        tx_indices = []
        for stages, mask in zip(sample.dense_indices, masks):
            second = torch.where(
                mask, stages[1], torch.full_like(stages[1], -1)
            )
            tx_indices.append([stages[0].clone(), second])
        segments, metadata = pack_topk_raw_mask_segments(
            sample.dense_indices, masks, rvq_k_lists
        )
        reconstructed = reconstruct_from_adaptive_indices(
            model, tx_indices, sample.feature_shapes, sample.codebooks
        )
        clean_ms_ssim, clean_psnr = _quality_metrics(
            sample.image, reconstructed
        )
        selection = {
            "image_index": image_index,
            "image_number": image_index + 1,
            "image_name": (
                str(image_names[image_index])
                if image_names is not None
                else f"image_{image_index + 1:04d}"
            ),
            "scales": batch_records[0]["scales"],
        }
        packets.append(
            PreparedAdaptivePacket(
                image=sample.image,
                feature_shapes=list(sample.feature_shapes),
                tx_indices=tx_indices,
                codebooks=sample.codebooks,
                active_masks=masks,
                segments=segments,
                metadata=metadata,
                selection=selection,
                clean_psnr=clean_psnr,
                clean_ms_ssim=clean_ms_ssim,
            )
        )
        selection_records.append(selection)
    return packets, selection_records


def summarize_raw_mask_packets(
    packets: Sequence[PreparedAdaptivePacket],
    ldpc_code: Dict[str, object],
    modulation: str,
) -> Dict[str, object]:
    """Summarize actual raw-mask source and combined physical lengths."""

    if not packets:
        raise ValueError("packets must not be empty")
    if modulation not in _MODULATION_BITS:
        raise ValueError(f"unsupported modulation: {modulation}")
    modulation_bits = _MODULATION_BITS[modulation]
    num_scales = len(packets[0].metadata["scales"])
    per_scale = [
        {
            "scale": scale,
            "token_count": 0,
            "tx_active_count": 0,
            "first_stage_source_bits": 0,
            "raw_mask_source_bits": 0,
            "active_second_source_bits": 0,
            "token_count_values": [],
            "active_count_values": [],
        }
        for scale in range(num_scales)
    ]
    totals = {
        "raw": {
            key: 0
            for key in combined_stream_lengths(
                0, ldpc_code, modulation_bits
            )
        },
        "dense": {
            key: 0
            for key in combined_stream_lengths(
                0, ldpc_code, modulation_bits
            )
        },
    }
    total_pixels = 0
    total_values = 0

    for packet in packets:
        if len(packet.metadata["scales"]) != num_scales:
            raise ValueError("all packets must have the same scale count")
        total_pixels += int(
            packet.image.shape[0]
            * packet.image.shape[-2]
            * packet.image.shape[-1]
        )
        total_values += int(packet.image.numel())
        references = {
            "raw": int(packet.metadata["source_bits"]),
            "dense": int(
                packet.metadata["dense_two_stage_source_bits_reference"]
            ),
        }
        for name, payload in references.items():
            lengths = combined_stream_lengths(
                payload, ldpc_code, modulation_bits
            )
            for key, value in lengths.items():
                totals[name][key] += int(value)
        for scale_metadata in packet.metadata["scales"]:
            stats = per_scale[int(scale_metadata["scale"])]
            segment_bits = scale_metadata["segment_source_bits"]
            stats["token_count"] += int(scale_metadata["token_count"])
            stats["tx_active_count"] += int(
                scale_metadata["tx_active_count"]
            )
            stats["first_stage_source_bits"] += int(segment_bits["first"])
            stats["raw_mask_source_bits"] += int(segment_bits["mask_raw"])
            stats["active_second_source_bits"] += int(
                segment_bits["second_active"]
            )
            stats["token_count_values"].append(
                int(scale_metadata["token_count"])
            )
            stats["active_count_values"].append(
                int(scale_metadata["tx_active_count"])
            )

    for stats in per_scale:
        token_count = int(stats["token_count"])
        active_count = int(stats["tx_active_count"])
        token_values = stats.pop("token_count_values")
        active_values = stats.pop("active_count_values")
        stats.update(
            {
                "token_count_per_image": token_count / len(packets),
                "token_count_min_per_image": min(token_values),
                "token_count_max_per_image": max(token_values),
                "tx_active_count_per_image": active_count / len(packets),
                "tx_active_count_min_per_image": min(active_values),
                "tx_active_count_max_per_image": max(active_values),
                "tx_active_ratio": active_count / token_count,
                "raw_mask_source_bits_mean_per_image": (
                    int(stats["raw_mask_source_bits"]) / len(packets)
                ),
            }
        )

    raw = totals["raw"]
    dense = totals["dense"]
    return {
        "num_images": len(packets),
        "total_image_pixels": total_pixels,
        "total_rgb_values": total_values,
        "first_stage_bits": sum(
            int(scale["first_stage_source_bits"]) for scale in per_scale
        ),
        "raw_mask_bits": sum(
            int(scale["raw_mask_source_bits"]) for scale in per_scale
        ),
        "active_second_bits": sum(
            int(scale["active_second_source_bits"]) for scale in per_scale
        ),
        "raw_source_bits": raw["payload_bits"],
        "raw_source_bpp": raw["payload_bits"] / total_pixels,
        "raw_coded_bits": raw["coded_bits"],
        "raw_coded_bpp": raw["coded_bits"] / total_pixels,
        "raw_channel_symbols": raw["channel_symbols"],
        "raw_channel_uses_per_pixel": raw["channel_symbols"] / total_pixels,
        "raw_transmission_ratio_per_rgb_value": (
            raw["channel_symbols"] / total_values
        ),
        "raw_ldpc_padding_bits": raw["ldpc_padding_bits"],
        "raw_modulation_padding_bits": raw["modulation_padding_bits"],
        "dense_two_stage_reference": dense,
        "source_bits_delta_vs_dense": (
            raw["payload_bits"] - dense["payload_bits"]
        ),
        "coded_bits_delta_vs_dense": raw["coded_bits"] - dense["coded_bits"],
        "per_scale": per_scale,
        "combined_stream": True,
        "framing_metadata_counted": False,
    }


@torch.no_grad()
def evaluate_raw_mask_packets_over_channel(
    model,
    packets: Sequence[PreparedAdaptivePacket],
    snr_db: float,
    ldpc_code: Dict[str, object],
    device: torch.device,
    modulation: str,
    seed: int = 42,
    transmit_fn: Callable[..., Tuple[np.ndarray, Dict[str, int]]] | None = None,
) -> Dict[str, object]:
    """Evaluate explicit-mask packets through one combined stream per image."""

    if not packets:
        raise ValueError("packets must not be empty")
    _reset_channel_seed(seed)
    transmit_fn = transmit_fn or transmit_combined_stream
    num_scales = len(packets[0].metadata["scales"])
    kind_bits = {"first": 0, "mask_raw": 0, "second_active": 0}
    kind_errors = {"first": 0, "mask_raw": 0, "second_active": 0}
    per_scale = [
        {
            "scale": scale,
            "tx_active_count": 0,
            "rx_active_count": 0,
            "mask_token_count": 0,
            "semantic_mask_bit_errors": 0,
            "exact_mask_frames": 0,
            "active_count_mismatch_frames": 0,
            "zero_filled_second_indices": 0,
            "truncated_second_indices": 0,
            "first_source_bits": 0,
            "first_source_bit_errors": 0,
            "raw_mask_source_bits": 0,
            "raw_mask_source_bit_errors": 0,
            "second_source_bits": 0,
            "second_source_bit_errors": 0,
        }
        for scale in range(num_scales)
    ]
    totals = {
        "payload_bits": 0,
        "ldpc_input_bits": 0,
        "ldpc_padding_bits": 0,
        "coded_bits": 0,
        "modulation_padding_bits": 0,
        "transmitted_bits": 0,
        "channel_symbols": 0,
        "bit_errors": 0,
    }
    psnr_values = []
    ms_ssim_values = []
    per_image = []
    images_with_errors = 0

    for image_index, packet in enumerate(packets):
        source = combined_bits_from_segments(packet.segments)
        decoded, channel_stats = transmit_fn(
            source, float(snr_db), ldpc_code, device, modulation
        )
        for key in totals:
            totals[key] += int(channel_stats[key])
        packet_errors = int(np.count_nonzero(decoded != source))
        if int(channel_stats["bit_errors"]) != packet_errors:
            raise RuntimeError("channel bit-error accounting mismatch")
        images_with_errors += int(packet_errors > 0)

        decoded_segments = _split_combined_bits(decoded, packet.metadata)
        source_segments = {segment.name: segment for segment in packet.segments}
        segment_records = {}
        for segment_metadata in packet.metadata["segments"]:
            name = str(segment_metadata["name"])
            kind = str(segment_metadata["kind"])
            scale = int(segment_metadata["scale"])
            sent = source_segments[name].bits
            received = decoded_segments[name]
            errors = int(np.count_nonzero(sent != received))
            bits = int(sent.size)
            kind_bits[kind] += bits
            kind_errors[kind] += errors
            segment_records[name] = {
                "source_bits": bits,
                "source_bit_errors": errors,
            }
            if kind == "first":
                per_scale[scale]["first_source_bits"] += bits
                per_scale[scale]["first_source_bit_errors"] += errors
            elif kind == "mask_raw":
                per_scale[scale]["raw_mask_source_bits"] += bits
                per_scale[scale]["raw_mask_source_bit_errors"] += errors
            else:
                per_scale[scale]["second_source_bits"] += bits
                per_scale[scale]["second_source_bit_errors"] += errors

        recovered, rx_masks, decode_stats = unpack_topk_raw_mask_combined(
            decoded, packet.metadata
        )
        image_scale_records = []
        for scale, (tx_mask, rx_mask, scale_decode) in enumerate(
            zip(packet.active_masks, rx_masks, decode_stats["per_scale"])
        ):
            tx_flat = tx_mask.detach().cpu().numpy().reshape(-1)
            rx_flat = rx_mask.detach().cpu().numpy().reshape(-1)
            semantic_errors = int(np.count_nonzero(tx_flat != rx_flat))
            stats = per_scale[scale]
            stats["tx_active_count"] += int(tx_flat.sum())
            stats["rx_active_count"] += int(rx_flat.sum())
            stats["mask_token_count"] += int(tx_flat.size)
            stats["semantic_mask_bit_errors"] += semantic_errors
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
            first_record = segment_records[names["first"]]
            mask_record = segment_records[names["mask_raw"]]
            second_record = segment_records[names["second"]]
            records = (first_record, mask_record, second_record)
            scale_source_bits = sum(int(item["source_bits"]) for item in records)
            scale_source_errors = sum(
                int(item["source_bit_errors"]) for item in records
            )
            image_scale_records.append(
                {
                    "scale": scale,
                    "tx_active_count": int(tx_flat.sum()),
                    "rx_active_count": int(rx_flat.sum()),
                    "semantic_mask_bit_errors": semantic_errors,
                    "semantic_mask_ber": semantic_errors / int(tx_flat.size),
                    "first_source_bits": first_record["source_bits"],
                    "first_source_bit_errors": first_record[
                        "source_bit_errors"
                    ],
                    "raw_mask_source_bits": mask_record["source_bits"],
                    "raw_mask_source_bit_errors": mask_record[
                        "source_bit_errors"
                    ],
                    "second_source_bits": second_record["source_bits"],
                    "second_source_bit_errors": second_record[
                        "source_bit_errors"
                    ],
                    "source_bits": scale_source_bits,
                    "source_bit_errors": scale_source_errors,
                    "source_ber_after_ldpc": (
                        scale_source_errors / scale_source_bits
                    ),
                    "zero_filled_second_indices": int(
                        scale_decode["zero_filled_second_indices"]
                    ),
                    "truncated_second_indices": int(
                        scale_decode["truncated_second_indices"]
                    ),
                }
            )

        reconstructed = reconstruct_from_adaptive_indices(
            model, recovered, packet.feature_shapes, packet.codebooks
        )
        ms_ssim, psnr = _quality_metrics(packet.image, reconstructed)
        psnr_values.append(psnr)
        ms_ssim_values.append(ms_ssim)
        per_image.append(
            {
                "image_index": image_index,
                "image_number": image_index + 1,
                "image_name": packet.selection["image_name"],
                "source_bits": int(source.size),
                "coded_bits": int(channel_stats["coded_bits"]),
                "channel_symbols": int(channel_stats["channel_symbols"]),
                "source_bit_errors": packet_errors,
                "source_ber_after_ldpc": packet_errors / int(source.size),
                "ldpc_input_bits": int(channel_stats["ldpc_input_bits"]),
                "ldpc_padding_bits": int(channel_stats["ldpc_padding_bits"]),
                "modulation_padding_bits": int(
                    channel_stats["modulation_padding_bits"]
                ),
                "transmitted_bits": int(channel_stats["transmitted_bits"]),
                "psnr": psnr,
                "ms_ssim": ms_ssim,
                "scales": image_scale_records,
            }
        )

    for stats in per_scale:
        token_count = int(stats["mask_token_count"])
        first_bits = int(stats["first_source_bits"])
        mask_bits = int(stats["raw_mask_source_bits"])
        second_bits = int(stats["second_source_bits"])
        source_bits = first_bits + mask_bits + second_bits
        source_errors = (
            int(stats["first_source_bit_errors"])
            + int(stats["raw_mask_source_bit_errors"])
            + int(stats["second_source_bit_errors"])
        )
        stats.update(
            {
                "tx_active_ratio": stats["tx_active_count"] / token_count,
                "rx_active_ratio": stats["rx_active_count"] / token_count,
                "semantic_mask_ber": (
                    stats["semantic_mask_bit_errors"] / token_count
                ),
                "raw_mask_source_ber_after_ldpc": (
                    stats["raw_mask_source_bit_errors"] / mask_bits
                ),
                "first_source_ber_after_ldpc": (
                    stats["first_source_bit_errors"] / first_bits
                ),
                "second_source_ber_after_ldpc": (
                    stats["second_source_bit_errors"] / second_bits
                    if second_bits
                    else 0.0
                ),
                "source_bits": source_bits,
                "source_bit_errors": source_errors,
                "source_ber_after_ldpc": source_errors / source_bits,
                "exact_mask_frame_rate": (
                    stats["exact_mask_frames"] / len(packets)
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
    total_values = sum(int(packet.image.numel()) for packet in packets)
    return {
        "snr_db": float(snr_db),
        "modulation": modulation,
        "psnr": float(np.mean(psnr_values)),
        "ms_ssim": float(np.mean(ms_ssim_values)),
        **{key: int(value) for key, value in totals.items()},
        "source_bpp": totals["payload_bits"] / total_pixels,
        "coded_bpp": totals["coded_bits"] / total_pixels,
        "channel_uses_per_pixel": totals["channel_symbols"] / total_pixels,
        "transmission_ratio_per_rgb_value": (
            totals["channel_symbols"] / total_values
        ),
        "source_ber_after_ldpc": (
            totals["bit_errors"] / totals["payload_bits"]
        ),
        "segment_source_bits": kind_bits,
        "segment_bit_errors": kind_errors,
        "images_with_source_errors": images_with_errors,
        "per_scale": per_scale,
        "per_image": per_image,
    }


__all__ = [
    "evaluate_raw_mask_packets_over_channel",
    "pack_topk_raw_mask_segments",
    "prepare_topk_raw_mask_packets",
    "summarize_raw_mask_packets",
    "unpack_topk_raw_mask_combined",
]
