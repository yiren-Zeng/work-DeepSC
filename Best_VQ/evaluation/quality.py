import random

import numpy as np
import torch

from communications.channel import awgn_channel
from communications.modulation import (
    bpsk_demodulate,
    bpsk_llr,
    bpsk_modulate,
    qpsk_llr,
    qpsk_modulate,
    qam16_llr,
    qam16_modulate,
)
from utils.bit_utils import bits_to_indices, indices_to_bits
from utils.metrics import calculate_ms_ssim


_MODULATION_BITS = {"bpsk": 1, "qpsk": 2, "16qam": 4}


def _new_rate_stats(channel_enabled=False):
    stats = dict.fromkeys(("num_images", "source_pixels", "source_values", "payload_bits"), 0)
    if channel_enabled:
        stats.update(dict.fromkeys((
            "ldpc_input_bits", "ldpc_padding_bits", "coded_bits",
            "modulation_padding_bits", "transmitted_bits", "channel_symbols",
        ), 0))
    return stats


def _accumulate_rate_stats(stats, real_image, payload_bits, channel_stats=None):
    stats["num_images"] += int(real_image.shape[0])
    stats["source_pixels"] += int(
        real_image.shape[0] * real_image.shape[-2] * real_image.shape[-1]
    )
    stats["source_values"] += int(real_image.numel())
    stats["payload_bits"] += int(payload_bits)
    if channel_stats is not None:
        for key, value in channel_stats.items():
            stats[key] += int(value)


def _finalize_rate_stats(stats):
    """Measured rates; raw image size assumes 8 bits per color channel."""
    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else 0.0

    stats = dict(stats)
    raw_bits = stats["source_values"] * 8
    stats.update({
        "raw_image_bits": raw_bits,
        "payload_bpp": ratio(stats["payload_bits"], stats["source_pixels"]),
        "source_bit_ratio": ratio(stats["payload_bits"], raw_bits),
        "source_compression_factor": ratio(raw_bits, stats["payload_bits"]),
    })
    if "channel_symbols" in stats:
        stats.update({
            "coded_bpp": ratio(stats["coded_bits"], stats["source_pixels"]),
            "transmitted_bpp": ratio(stats["transmitted_bits"], stats["source_pixels"]),
            "transmitted_bit_ratio": ratio(stats["transmitted_bits"], raw_bits),
            "channel_uses_per_pixel": ratio(stats["channel_symbols"], stats["source_pixels"]),
            "transmission_ratio": ratio(stats["channel_symbols"], stats["source_values"]),
        })
    return stats


def _reset_eval_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _image_quality(real_image, reconstructed_images):
    if real_image.device != reconstructed_images.device:
        real_image = real_image.to(reconstructed_images.device, non_blocking=True)
    img1 = (real_image + 1) / 2
    img2 = (reconstructed_images + 1) / 2
    ms_ssim = calculate_ms_ssim(img1, img2)
    mse = torch.mean((img1 - img2) ** 2)
    psnr = 100.0 if mse == 0 else 10 * torch.log10(1.0 / mse).item()
    return ms_ssim, psnr


@torch.no_grad()
def evaluate_no_channel(
    model, loader, device, num_embeddings_list=None, return_rate_stats=False,
    perceptual_metrics=None,
):
    model.eval()
    ms_ssim_scores = []
    psnr_scores = []
    if perceptual_metrics is not None:
        perceptual_metrics.reset()
    rate_stats = None
    if return_rate_stats and getattr(model, "quantizer_type", None) != "none":
        if num_embeddings_list is None:
            raise ValueError("num_embeddings_list is required to report source compression.")
        rate_stats = _new_rate_stats()

    for real_image in loader:
        real_image = real_image.to(device)
        out = model.forward_test(real_image)
        if rate_stats is not None:
            payload_bits = sum(
                indices.numel() * int(np.log2(codebook_size))
                for indices, codebook_size in zip(out["indices"], num_embeddings_list)
            )
            _accumulate_rate_stats(rate_stats, real_image, payload_bits)
        reconstructed_images = model.reconstruct_from_indices(
            out["indices"], feature_shapes=out.get("feature_shapes")
        )
        ms_ssim, psnr = _image_quality(real_image, reconstructed_images)
        if perceptual_metrics is not None:
            perceptual_metrics.update(real_image, reconstructed_images)
        ms_ssim_scores.append(ms_ssim)
        psnr_scores.append(psnr)

    result = (np.mean(ms_ssim_scores), np.mean(psnr_scores))
    if return_rate_stats:
        return (*result, _finalize_rate_stats(rate_stats) if rate_stats is not None else None)
    return result


@torch.no_grad()
def evaluate_ldpc_channel(
    model, loader, num_embeddings_list, target_snr, ldpc_code, device, modulation="bpsk",
    return_rate_stats=False, perceptual_metrics=None,
):
    from communications.ldpc_coding import ldpc_decode, ldpc_encode

    modulators = {
        "bpsk": (bpsk_modulate, bpsk_llr),
        "qpsk": (qpsk_modulate, qpsk_llr),
        "16qam": (qam16_modulate, qam16_llr),
    }
    if modulation not in modulators:
        raise ValueError(f"Unsupported modulation: {modulation}")
    modulate, calculate_llr = modulators[modulation]

    _reset_eval_seed()
    model.eval()
    ms_ssim_scores = []
    psnr_scores = []
    if perceptual_metrics is not None:
        perceptual_metrics.reset()
    rate_stats = _new_rate_stats(channel_enabled=True) if return_rate_stats else None

    for real_image in loader:
        real_image = real_image.to(device)
        out = model.forward_test(real_image)
        flat_bits, original_spatial_dims, original_num_embeddings = indices_to_bits(
            out["indices"], num_embeddings_list)

        coded_bits = ldpc_encode(flat_bits, code=ldpc_code)
        # Adjustable LDPC lengths may need padding to form complete QPSK/16-QAM symbols.
        modulation_padding_bits = (-len(coded_bits)) % _MODULATION_BITS[modulation]
        transmitted_bits = np.pad(coded_bits, (0, modulation_padding_bits), "constant")
        coded_bits_tensor = torch.from_numpy(transmitted_bits).float().to(device)
        symbols = modulate(coded_bits_tensor).to(device)
        if rate_stats is not None:
            k = ldpc_code["k"] if ldpc_code is not None else len(flat_bits)
            ldpc_input_bits = ((len(flat_bits) + k - 1) // k) * k
            _accumulate_rate_stats(rate_stats, real_image, len(flat_bits), {
                "ldpc_input_bits": ldpc_input_bits,
                "ldpc_padding_bits": ldpc_input_bits - len(flat_bits),
                "coded_bits": len(coded_bits),
                "modulation_padding_bits": modulation_padding_bits,
                "transmitted_bits": len(transmitted_bits),
                "channel_symbols": symbols.numel(),
            })
        noisy_symbols = awgn_channel(symbols, target_snr)
        llrs = calculate_llr(noisy_symbols, target_snr, device)
        decoded_bits = ldpc_decode(llrs[:len(coded_bits)].cpu().numpy(), ldpc_code)

        decoded_bits = decoded_bits[:len(flat_bits)]
        recovered_indices_list = bits_to_indices(
            decoded_bits, original_spatial_dims, original_num_embeddings)
        recovered_indices_list = [idx.to(device) for idx in recovered_indices_list]
        reconstructed_images = model.reconstruct_from_indices(
            recovered_indices_list, feature_shapes=out.get("feature_shapes")
        )

        ms_ssim, psnr = _image_quality(real_image, reconstructed_images)
        if perceptual_metrics is not None:
            perceptual_metrics.update(real_image, reconstructed_images)
        ms_ssim_scores.append(ms_ssim)
        psnr_scores.append(psnr)

    result = (np.mean(ms_ssim_scores), np.mean(psnr_scores))
    return (*result, _finalize_rate_stats(rate_stats)) if return_rate_stats else result


@torch.no_grad()
def evaluate_uncoded_channel(model, loader, num_embeddings_list, target_snr, device):
    _reset_eval_seed()
    model.eval()
    ms_ssim_scores = []
    psnr_scores = []

    for real_image in loader:
        real_image = real_image.to(device)
        out = model.forward_test(real_image)
        flat_bits, original_spatial_dims, original_num_embeddings = indices_to_bits(
            out["indices"], num_embeddings_list)

        bits_tensor = torch.from_numpy(flat_bits).float().to(device)
        symbols = bpsk_modulate(bits_tensor)
        noisy_symbols = awgn_channel(symbols, target_snr)
        decoded_bits = bpsk_demodulate(noisy_symbols).cpu().numpy()

        recovered_indices_list = bits_to_indices(
            decoded_bits, original_spatial_dims, original_num_embeddings)
        recovered_indices_list = [idx.to(device) for idx in recovered_indices_list]
        reconstructed_images = model.reconstruct_from_indices(
            recovered_indices_list, feature_shapes=out.get("feature_shapes")
        )

        ms_ssim, psnr = _image_quality(real_image, reconstructed_images)
        ms_ssim_scores.append(ms_ssim)
        psnr_scores.append(psnr)

    return np.mean(ms_ssim_scores), np.mean(psnr_scores)
