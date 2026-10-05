import random

import numpy as np
import torch

from communications.channel import awgn_channel, rician_channel
from communications.modulation import (
    bpsk_llr,
    bpsk_modulate,
    qam16_llr,
    qam16_modulate,
    qpsk_llr,
    qpsk_modulate,
)
from utils.bit_utils import bits_to_index_tensor, index_tensor_to_bits
from utils.metrics import calculate_ms_ssim


_MODULATIONS = {
    "bpsk": (1, bpsk_modulate, bpsk_llr),
    "qpsk": (2, qpsk_modulate, qpsk_llr),
    "16qam": (4, qam16_modulate, qam16_llr),
}
_CHANNEL_TYPES = {"awgn", "rician"}


def _reset_eval_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _image_quality(real_images, reconstructed_images):
    normalized_real = (real_images + 1) / 2
    normalized_reconstructed = (reconstructed_images + 1) / 2
    ms_ssim = calculate_ms_ssim(normalized_real, normalized_reconstructed)
    mse = torch.mean((normalized_real - normalized_reconstructed) ** 2)
    psnr = 100.0 if mse == 0 else 10 * torch.log10(1.0 / mse).item()
    return ms_ssim, psnr


def _empty_scale_record(scale, num_embeddings):
    return {
        "scale": int(scale),
        "num_embeddings": int(num_embeddings),
        "bits_per_index": int(np.log2(num_embeddings)),
        "num_indices": 0,
        "payload_bits": 0,
        "bit_errors": 0,
        "index_errors": 0,
        "sent_index_min": None,
        "sent_index_max": None,
        "recovered_index_min": None,
        "recovered_index_max": None,
    }


def _update_range(record, prefix, tensor):
    current_min = int(tensor.min().item())
    current_max = int(tensor.max().item())
    min_key = f"{prefix}_index_min"
    max_key = f"{prefix}_index_max"
    record[min_key] = (
        current_min if record[min_key] is None else min(record[min_key], current_min)
    )
    record[max_key] = (
        current_max if record[max_key] is None else max(record[max_key], current_max)
    )


def _finalize_scale_records(records, source_pixels, channel_enabled):
    finalized = []
    for record in records:
        result = dict(record)
        result["payload_bpp"] = result["payload_bits"] / source_pixels
        if channel_enabled:
            result["ber"] = (
                result["bit_errors"] / result["payload_bits"]
                if result["payload_bits"]
                else 0.0
            )
            result["index_error_rate"] = (
                result["index_errors"] / result["num_indices"]
                if result["num_indices"]
                else 0.0
            )
        else:
            result["ber"] = None
            result["index_error_rate"] = None
        finalized.append(result)
    return finalized


def _transmit_ldpc_stream(
    flat_bits,
    target_snr,
    ldpc_code,
    device,
    modulation,
    channel_type="awgn",
    rician_k_factor=10.0,
):
    from communications.ldpc_coding import ldpc_decode, ldpc_encode

    modulation_bits, modulate, calculate_llr = _MODULATIONS[modulation]
    flat_bits = np.asarray(flat_bits, dtype=np.uint8).reshape(-1)
    payload_bits = len(flat_bits)
    information_length = int(ldpc_code["k"])
    ldpc_input_bits = (
        (payload_bits + information_length - 1) // information_length
    ) * information_length
    ldpc_padding_bits = ldpc_input_bits - payload_bits

    coded = np.asarray(ldpc_encode(flat_bits, code=ldpc_code)).reshape(-1)
    coded_bits = len(coded)
    modulation_padding_bits = (-coded_bits) % modulation_bits
    transmitted = np.pad(coded, (0, modulation_padding_bits), "constant")
    transmitted_tensor = torch.from_numpy(transmitted).float().to(device)
    symbols = modulate(transmitted_tensor)
    channel_gain = None
    noise_variance = None
    if channel_type == "awgn":
        noisy_symbols, noise_variance = awgn_channel(
            symbols, target_snr, return_noise_power=True
        )
    elif channel_type == "rician":
        noisy_symbols, channel_gain, noise_variance = rician_channel(
            symbols,
            target_snr,
            k_factor=rician_k_factor,
            return_csi=True,
        )
    else:
        raise ValueError(
            f"Unsupported channel type: {channel_type!r}; "
            f"expected one of {sorted(_CHANNEL_TYPES)}"
        )
    llrs = calculate_llr(
        noisy_symbols,
        target_snr,
        device,
        channel_gain=channel_gain,
        noise_variance=noise_variance,
    ).reshape(-1)
    decoded = np.asarray(
        ldpc_decode(llrs[:coded_bits].detach().cpu().numpy(), ldpc_code)
    ).reshape(-1)[:payload_bits]
    if len(decoded) < payload_bits:
        decoded = np.pad(decoded, (0, payload_bits - len(decoded)), "constant")
    decoded = decoded.astype(np.uint8, copy=False)

    return decoded, {
        "ldpc_input_bits": int(ldpc_input_bits),
        "ldpc_padding_bits": int(ldpc_padding_bits),
        "coded_bits": int(coded_bits),
        "modulation_padding_bits": int(modulation_padding_bits),
        "transmitted_bits": int(len(transmitted)),
        "channel_symbols": int(symbols.numel()),
        "bit_errors": int(np.count_nonzero(decoded != flat_bits)),
    }


@torch.no_grad()
def evaluate_no_channel(model, loader, num_embeddings_list, device, return_diagnostics=False):
    _reset_eval_seed()
    model.eval()
    records = [
        _empty_scale_record(scale, size)
        for scale, size in enumerate(num_embeddings_list)
    ]
    ms_ssim_scores = []
    psnr_scores = []
    total_images = 0
    total_pixels = 0
    total_values = 0

    for real_images in loader:
        real_images = real_images.to(device)
        output = model.forward_test(real_images)
        reconstructed = model.reconstruct_from_indices(output["indices"])
        ms_ssim, psnr = _image_quality(real_images, reconstructed)
        ms_ssim_scores.append(ms_ssim)
        psnr_scores.append(psnr)

        batch_size = int(real_images.shape[0])
        total_images += batch_size
        total_pixels += batch_size * int(real_images.shape[-2]) * int(real_images.shape[-1])
        total_values += int(real_images.numel())
        for scale, (indices, size) in enumerate(
            zip(output["indices"], num_embeddings_list)
        ):
            bits, _, _ = index_tensor_to_bits(indices, size)
            record = records[scale]
            record["num_indices"] += int(indices.numel())
            record["payload_bits"] += int(len(bits))
            _update_range(record, "sent", indices.detach().cpu())

    per_scale = _finalize_scale_records(records, total_pixels, False)
    diagnostics = {
        "mode": "no_channel",
        "stream_packing": "combined",
        "num_images": total_images,
        "source_pixels": total_pixels,
        "source_values": total_values,
        "per_scale": per_scale,
        "total": {
            "payload_bits": sum(item["payload_bits"] for item in per_scale),
            "payload_bpp": sum(item["payload_bits"] for item in per_scale)
            / total_pixels,
        },
    }
    result = (float(np.mean(ms_ssim_scores)), float(np.mean(psnr_scores)))
    return (*result, diagnostics) if return_diagnostics else result


@torch.no_grad()
def evaluate_ldpc_channel(
    model,
    loader,
    num_embeddings_list,
    target_snr,
    ldpc_code,
    device,
    modulation="qpsk",
    return_diagnostics=False,
    stream_packing="combined",
    channel_type="awgn",
    rician_k_factor=10.0,
):
    if modulation not in _MODULATIONS:
        raise ValueError(f"Unsupported modulation: {modulation}")
    if stream_packing != "combined":
        raise ValueError("VQ-DeepSC baseline evaluation requires combined stream packing")
    channel_type = str(channel_type).strip().lower()
    if channel_type not in _CHANNEL_TYPES:
        raise ValueError(
            f"Unsupported channel type: {channel_type!r}; "
            f"expected one of {sorted(_CHANNEL_TYPES)}"
        )
    if rician_k_factor < 0:
        raise ValueError("Rician K-factor must be non-negative")

    _reset_eval_seed()
    model.eval()
    records = [
        _empty_scale_record(scale, size)
        for scale, size in enumerate(num_embeddings_list)
    ]
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
        output = model.forward_test(real_images)
        segments = []
        for scale, (indices, size) in enumerate(
            zip(output["indices"], num_embeddings_list)
        ):
            bits, original_shape, _ = index_tensor_to_bits(indices, size)
            segments.append((scale, indices, size, original_shape, bits))

        combined_bits = np.concatenate([segment[-1] for segment in segments])
        decoded_bits, channel_stats = _transmit_ldpc_stream(
            combined_bits,
            target_snr,
            ldpc_code,
            device,
            modulation,
            channel_type=channel_type,
            rician_k_factor=rician_k_factor,
        )
        stream_totals["payload_bits"] += int(len(combined_bits))
        for key, value in channel_stats.items():
            stream_totals[key] += int(value)

        recovered_indices = []
        offset = 0
        image_index_errors = 0
        for scale, sent_indices, size, original_shape, sent_bits in segments:
            end = offset + len(sent_bits)
            recovered_bits = decoded_bits[offset:end]
            offset = end
            recovered = bits_to_index_tensor(
                recovered_bits, original_shape, size
            ).to(device)
            recovered_indices.append(recovered)

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

        reconstructed = model.reconstruct_from_indices(recovered_indices)
        ms_ssim, psnr = _image_quality(real_images, reconstructed)
        ms_ssim_scores.append(ms_ssim)
        psnr_scores.append(psnr)
        batch_size = int(real_images.shape[0])
        total_images += batch_size
        total_pixels += batch_size * int(real_images.shape[-2]) * int(real_images.shape[-1])
        total_values += int(real_images.numel())

    per_scale = _finalize_scale_records(records, total_pixels, True)
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
    diagnostics = {
        "mode": "ldpc",
        "stream_packing": "combined",
        "modulation": modulation,
        "channel_type": channel_type,
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
        "padding_notes": {
            "ldpc_padding_bits": "zero information bits added before LDPC encoding",
            "modulation_padding_bits": "zero coded bits added to fill a modulation symbol",
        },
    }
    if channel_type == "rician":
        diagnostics["rician_k_factor"] = float(rician_k_factor)
        diagnostics["receiver_csi"] = "perfect_per_symbol"
    result = (float(np.mean(ms_ssim_scores)), float(np.mean(psnr_scores)))
    return (*result, diagnostics) if return_diagnostics else result
