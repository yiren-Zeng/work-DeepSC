import random

import numpy as np
import torch

from communications.channel import awgn_channel
from communications.modulation import (
    bpsk_demodulate,
    bpsk_llr,
    bpsk_modulate,
)
from utils.bit_utils import bits_to_indices, indices_to_bits
from utils.metrics import calculate_ms_ssim


def _reset_eval_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _image_quality(real_image, reconstructed_images):
    img1 = (real_image + 1) / 2
    img2 = (reconstructed_images + 1) / 2
    ms_ssim = calculate_ms_ssim(img1, img2)
    mse = torch.mean((img1 - img2) ** 2)
    psnr = 100.0 if mse == 0 else 10 * torch.log10(1.0 / mse).item()
    return ms_ssim, psnr


@torch.no_grad()
def evaluate_no_channel(model, loader, device):
    model.eval()
    ms_ssim_scores = []
    psnr_scores = []

    for real_image in loader:
        real_image = real_image.to(device)
        out = model.forward_test(real_image)
        reconstructed_images = model.reconstruct_from_indices(out["indices"])
        ms_ssim, psnr = _image_quality(real_image, reconstructed_images)
        ms_ssim_scores.append(ms_ssim)
        psnr_scores.append(psnr)

    return np.mean(ms_ssim_scores), np.mean(psnr_scores)


@torch.no_grad()
def evaluate_ldpc_channel(model, loader, num_embeddings_list, target_snr, ldpc_code, device):
    from communications.ldpc_coding import ldpc_decode, ldpc_encode

    _reset_eval_seed()
    model.eval()
    ms_ssim_scores = []
    psnr_scores = []

    for real_image in loader:
        real_image = real_image.to(device)
        out = model.forward_test(real_image)
        flat_bits, original_spatial_dims, original_num_embeddings = indices_to_bits(
            out["indices"], num_embeddings_list)

        coded_bits = ldpc_encode(flat_bits, code=ldpc_code)
        coded_bits_tensor = torch.from_numpy(coded_bits).float().to(device)
        symbols = bpsk_modulate(coded_bits_tensor)
        noisy_symbols = awgn_channel(symbols, target_snr)
        llrs = bpsk_llr(noisy_symbols, target_snr, device)
        decoded_bits = ldpc_decode(llrs.cpu().numpy(), ldpc_code)

        decoded_bits = decoded_bits[:len(flat_bits)]
        recovered_indices_list = bits_to_indices(
            decoded_bits, original_spatial_dims, original_num_embeddings)
        recovered_indices_list = [idx.to(device) for idx in recovered_indices_list]
        reconstructed_images = model.reconstruct_from_indices(recovered_indices_list)

        ms_ssim, psnr = _image_quality(real_image, reconstructed_images)
        ms_ssim_scores.append(ms_ssim)
        psnr_scores.append(psnr)

    return np.mean(ms_ssim_scores), np.mean(psnr_scores)


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
        reconstructed_images = model.reconstruct_from_indices(recovered_indices_list)

        ms_ssim, psnr = _image_quality(real_image, reconstructed_images)
        ms_ssim_scores.append(ms_ssim)
        psnr_scores.append(psnr)

    return np.mean(ms_ssim_scores), np.mean(psnr_scores)
