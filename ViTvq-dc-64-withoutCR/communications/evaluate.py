# MS-SSIM 和 PSNR 评价指标 (VQ-ViT 无压缩版)

import torch
import numpy as np
import random
from communications.ldpc_coding import get_ldpc_code, ldpc_encode, ldpc_decode
from communications.modulation import *
from communications.channel import awgn_channel
from utils.bit_utils import indices_to_bits, bits_to_indices
from utils.metrics import calculate_ms_ssim


@torch.no_grad()
def evaluate_metrics_with_channel(model, loader, num_embeddings_list, target_snr, ldpc_code, device):
    """包含完整物理层链路的 MS-SSIM 和 PSNR 联合评估 (VQ-ViT无压缩版)"""

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

        img1 = (real_image + 1) / 2
        img2 = (reconstructed_images + 1) / 2

        ms_ssim = calculate_ms_ssim(img1, img2)
        ms_ssim_scores.append(ms_ssim)

        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            psnr = 100.0
        else:
            psnr = 10 * torch.log10(1.0 / mse).item()
        psnr_scores.append(psnr)

    return np.mean(ms_ssim_scores), np.mean(psnr_scores)


@torch.no_grad()
def evaluate_metrics_with_channel_withoutLDPC(model, loader, num_embeddings_list, target_snr, device):
    """不含 LDPC 的 MS-SSIM 和 PSNR 联合评估 (VQ-ViT无压缩版)"""

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model.eval()
    ms_ssim_scores = []
    psnr_scores = []

    for real_image in loader:
        real_image = real_image.to(device)

        out = model.forward_test(real_image)

        flat_bits, original_spatial_dims, original_num_embeddings = indices_to_bits(
            out["indices"], num_embeddings_list)

        coded_bits_tensor = torch.from_numpy(flat_bits).float().to(device)

        symbols = bpsk_modulate(coded_bits_tensor)

        noisy_symbols = awgn_channel(symbols, target_snr)

        decoded_bits = bpsk_demodulate(noisy_symbols)

        decoded_bits = decoded_bits.cpu().numpy()

        recovered_indices_list = bits_to_indices(
            decoded_bits, original_spatial_dims, original_num_embeddings)
        recovered_indices_list = [idx.to(device) for idx in recovered_indices_list]

        reconstructed_images = model.reconstruct_from_indices(recovered_indices_list)

        img1 = (real_image + 1) / 2
        img2 = (reconstructed_images + 1) / 2

        ms_ssim = calculate_ms_ssim(img1, img2)
        ms_ssim_scores.append(ms_ssim)

        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            psnr = 100.0
        else:
            psnr = 10 * torch.log10(1.0 / mse).item()
        psnr_scores.append(psnr)

    return np.mean(ms_ssim_scores), np.mean(psnr_scores)
