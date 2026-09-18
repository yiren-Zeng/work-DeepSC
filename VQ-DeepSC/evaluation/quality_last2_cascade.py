import numpy as np
import torch

from evaluation.quality import (
    _MODULATIONS,
    _empty_scale_record,
    _finalize_scale_records,
    _image_quality,
    _reset_eval_seed,
    _transmit_ldpc_stream,
    _update_range,
)
from utils.bit_utils import bits_to_index_tensor, index_tensor_to_bits


ACTIVE_SCALES = (2, 3)
INACTIVE_SCALES = (0, 1)


def _validate_last2_model(model, num_embeddings_list):
    num_scales = len(num_embeddings_list)
    if num_scales != 2 or len(model.vector_quantizers) != 2:
        raise ValueError(
            "Last-two cascade evaluation requires a four-stage model with two VQs"
        )
    if tuple(model.ACTIVE_SCALES) != ACTIVE_SCALES:
        raise ValueError(f"Model active scales must be {ACTIVE_SCALES}")
    if list(model.num_embeddings_list) != list(num_embeddings_list):
        raise ValueError("Test codebooks do not match the model's two active codebooks")


def _encode_last_two(model, images):
    output = model.forward_test(images)
    if tuple(output["active_scales"]) != ACTIVE_SCALES:
        raise ValueError("Model returned unexpected active VQ scales")
    return {
        scale: indices
        for scale, indices in zip(ACTIVE_SCALES, output["indices"])
    }


def _reconstruct_last_two(model, active_indices):
    ordered_indices = [active_indices[scale] for scale in ACTIVE_SCALES]
    return model.reconstruct_from_indices(ordered_indices)


def _compression_summary(source_values, payload_bits, transmitted_bits=None):
    """Report compression against uncompressed 8-bit RGB source values."""
    source_bits = int(source_values) * 8

    def summarize(compressed_bits):
        fraction = compressed_bits / source_bits
        return {
            "bits": int(compressed_bits),
            "fraction_of_source": float(fraction),
            "percent_of_source": float(fraction * 100.0),
            "saving_percent": float((1.0 - fraction) * 100.0),
            "compression_ratio": float(source_bits / compressed_bits),
        }

    result = {
        "reference": "uncompressed 8-bit RGB",
        "source_bits": source_bits,
        "payload": summarize(int(payload_bits)),
    }
    if transmitted_bits is not None:
        result["end_to_end"] = summarize(int(transmitted_bits))
    return result


@torch.no_grad()
def evaluate_no_channel_last2_cascade(
    model,
    loader,
    num_embeddings_list,
    device,
    return_diagnostics=False,
):
    _validate_last2_model(model, num_embeddings_list)
    _reset_eval_seed()
    model.eval()
    codebook_sizes = dict(zip(ACTIVE_SCALES, num_embeddings_list))
    records = {
        scale: _empty_scale_record(scale, codebook_sizes[scale])
        for scale in ACTIVE_SCALES
    }
    ms_ssim_scores = []
    psnr_scores = []
    total_images = 0
    total_pixels = 0
    total_values = 0

    for real_images in loader:
        real_images = real_images.to(device)
        active_indices = _encode_last_two(model, real_images)
        reconstructed = _reconstruct_last_two(model, active_indices)
        ms_ssim, psnr = _image_quality(real_images, reconstructed)
        ms_ssim_scores.append(ms_ssim)
        psnr_scores.append(psnr)

        batch_size = int(real_images.shape[0])
        total_images += batch_size
        total_pixels += (
            batch_size * int(real_images.shape[-2]) * int(real_images.shape[-1])
        )
        total_values += int(real_images.numel())
        for scale in ACTIVE_SCALES:
            indices = active_indices[scale]
            bits, _, _ = index_tensor_to_bits(
                indices, codebook_sizes[scale]
            )
            record = records[scale]
            record["num_indices"] += int(indices.numel())
            record["payload_bits"] += int(len(bits))
            _update_range(record, "sent", indices.detach().cpu())

    per_scale = _finalize_scale_records(
        [records[scale] for scale in ACTIVE_SCALES], total_pixels, False
    )
    payload_bits = sum(item["payload_bits"] for item in per_scale)
    compression = _compression_summary(total_values, payload_bits)
    diagnostics = {
        "mode": "no_channel_last2_cascade",
        "decoder_mode": "full_decoder_with_copied_deep_features",
        "stream_packing": "combined",
        "active_scales": list(ACTIVE_SCALES),
        "inactive_scales": list(INACTIVE_SCALES),
        "num_images": total_images,
        "source_pixels": total_pixels,
        "source_values": total_values,
        "per_scale": per_scale,
        "total": {
            "payload_bits": payload_bits,
            "payload_bpp": payload_bits / total_pixels,
        },
        "compression": compression,
    }
    result = (float(np.mean(ms_ssim_scores)), float(np.mean(psnr_scores)))
    return (*result, diagnostics) if return_diagnostics else result


@torch.no_grad()
def evaluate_ldpc_channel_last2_cascade(
    model,
    loader,
    num_embeddings_list,
    target_snr,
    ldpc_code,
    device,
    modulation="qpsk",
    return_diagnostics=False,
    stream_packing="combined",
):
    _validate_last2_model(model, num_embeddings_list)
    if modulation not in _MODULATIONS:
        raise ValueError(f"Unsupported modulation: {modulation}")
    if stream_packing != "combined":
        raise ValueError("Last-two cascade evaluation requires combined stream packing")

    _reset_eval_seed()
    model.eval()
    codebook_sizes = dict(zip(ACTIVE_SCALES, num_embeddings_list))
    records = {
        scale: _empty_scale_record(scale, codebook_sizes[scale])
        for scale in ACTIVE_SCALES
    }
    ms_ssim_scores = []
    psnr_scores = []
    total_images = 0
    total_pixels = 0
    total_values = 0
    stream_totals = {
        "payload_bits": 0,
        "ldpc_input_bits": 0,
        "ldpc_padding_bits": 0,
        "coded_bits": 0,
        "modulation_padding_bits": 0,
        "transmitted_bits": 0,
        "channel_symbols": 0,
        "bit_errors": 0,
        "index_errors": 0,
    }

    for real_images in loader:
        real_images = real_images.to(device)
        active_indices = _encode_last_two(model, real_images)
        segments = []
        for scale in ACTIVE_SCALES:
            indices = active_indices[scale]
            size = codebook_sizes[scale]
            bits, original_shape, _ = index_tensor_to_bits(indices, size)
            segments.append((scale, indices, size, original_shape, bits))

        combined_bits = np.concatenate([segment[-1] for segment in segments])
        decoded_bits, channel_stats = _transmit_ldpc_stream(
            combined_bits,
            target_snr,
            ldpc_code,
            device,
            modulation,
        )
        stream_totals["payload_bits"] += int(len(combined_bits))
        for key, value in channel_stats.items():
            stream_totals[key] += int(value)

        recovered_indices = {}
        offset = 0
        image_index_errors = 0
        for scale, sent_indices, size, original_shape, sent_bits in segments:
            end = offset + len(sent_bits)
            recovered_bits = decoded_bits[offset:end]
            offset = end
            recovered = bits_to_index_tensor(
                recovered_bits, original_shape, size
            ).to(device)
            recovered_indices[scale] = recovered

            bit_errors = int(np.count_nonzero(recovered_bits != sent_bits))
            index_errors = int(
                torch.count_nonzero(recovered != sent_indices).item()
            )
            image_index_errors += index_errors
            record = records[scale]
            record["num_indices"] += int(sent_indices.numel())
            record["payload_bits"] += int(len(sent_bits))
            record["bit_errors"] += bit_errors
            record["index_errors"] += index_errors
            _update_range(record, "sent", sent_indices.detach().cpu())
            _update_range(record, "recovered", recovered.detach().cpu())
        if offset != len(decoded_bits):
            raise RuntimeError("Decoded combined payload was not consumed exactly")
        stream_totals["index_errors"] += image_index_errors

        reconstructed = _reconstruct_last_two(model, recovered_indices)
        ms_ssim, psnr = _image_quality(real_images, reconstructed)
        ms_ssim_scores.append(ms_ssim)
        psnr_scores.append(psnr)
        batch_size = int(real_images.shape[0])
        total_images += batch_size
        total_pixels += (
            batch_size * int(real_images.shape[-2]) * int(real_images.shape[-1])
        )
        total_values += int(real_images.numel())

    per_scale = _finalize_scale_records(
        [records[scale] for scale in ACTIVE_SCALES], total_pixels, True
    )
    total_indices = sum(item["num_indices"] for item in per_scale)
    combined_stream = dict(stream_totals)
    combined_stream.update(
        {
            "source_pixels": total_pixels,
            "source_values": total_values,
            "payload_bpp": stream_totals["payload_bits"] / total_pixels,
            "coded_bpp": stream_totals["coded_bits"] / total_pixels,
            "transmitted_bpp": stream_totals["transmitted_bits"] / total_pixels,
            "channel_uses_per_pixel": stream_totals["channel_symbols"]
            / total_pixels,
            "transmission_ratio": stream_totals["channel_symbols"]
            / total_values,
            "ber": stream_totals["bit_errors"] / stream_totals["payload_bits"],
            "index_error_rate": stream_totals["index_errors"] / total_indices,
        }
    )
    compression = _compression_summary(
        total_values,
        stream_totals["payload_bits"],
        stream_totals["transmitted_bits"],
    )
    diagnostics = {
        "mode": "ldpc_last2_cascade",
        "decoder_mode": "full_decoder_with_copied_deep_features",
        "stream_packing": "combined",
        "active_scales": list(ACTIVE_SCALES),
        "inactive_scales": list(INACTIVE_SCALES),
        "modulation": modulation,
        "modulation_bits_per_symbol": _MODULATIONS[modulation][0],
        "snr_db": float(target_snr),
        "ldpc": {
            "k": int(ldpc_code["k"]),
            "n": int(ldpc_code["n"]),
            "rate": float(ldpc_code["rate"]),
        },
        "num_images": total_images,
        "per_scale": per_scale,
        "combined_stream": combined_stream,
        "total": combined_stream,
        "compression": compression,
        "padding_notes": {
            "ldpc_padding_bits": "zero information bits added before LDPC encoding",
            "modulation_padding_bits": "zero coded bits added to fill a modulation symbol",
        },
    }
    result = (float(np.mean(ms_ssim_scores)), float(np.mean(psnr_scores)))
    return (*result, diagnostics) if return_diagnostics else result
