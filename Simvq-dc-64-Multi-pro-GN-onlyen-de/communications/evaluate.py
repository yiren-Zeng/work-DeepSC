# 无 VQ 模型的评估模块
# 连续特征 → 直接转比特流(float16 IEEE 754) → LDPC + BPSK 完整链路评估

import torch
import numpy as np
import random
from communications.ldpc_coding import get_ldpc_code, ldpc_encode, ldpc_decode
from communications.modulation import bpsk_modulate, bpsk_llr
from communications.channel import awgn_channel
from utils.bit_utils import features_to_bits, bits_to_features, BITS_PER_VALUE
from utils.metrics import calculate_ms_ssim

CHUNK_K_BLOCKS = 2000


def transmit_bits_through_channel(flat_bits, target_snr, ldpc_code, device):
    """
    将比特流分块通过完整物理层链路：LDPC编码 → BPSK调制 → AWGN → 软解调 → LDPC译码
    """
    k = ldpc_code["k"]

    num_blocks = (len(flat_bits) + k - 1) // k
    padded_len = num_blocks * k
    padded_bits = np.pad(flat_bits, (0, padded_len - len(flat_bits)), 'constant', constant_values=0)

    decoded_parts = []

    for start_block in range(0, num_blocks, CHUNK_K_BLOCKS):
        end_block = min(start_block + CHUNK_K_BLOCKS, num_blocks)
        chunk_bits = padded_bits[start_block * k : end_block * k]

        coded_bits = ldpc_encode(chunk_bits, code=ldpc_code)
        coded_bits_tensor = torch.from_numpy(coded_bits).float().to(device)

        symbols = bpsk_modulate(coded_bits_tensor)
        noisy_symbols = awgn_channel(symbols, target_snr)
        llrs = bpsk_llr(noisy_symbols, target_snr, device)

        decoded_bits = ldpc_decode(llrs.cpu().numpy(), ldpc_code)

        num_blocks_in_chunk = end_block - start_block
        decoded_parts.append(decoded_bits[:num_blocks_in_chunk * k])

        del coded_bits_tensor, symbols, noisy_symbols, llrs
        torch.cuda.empty_cache()

    all_decoded = np.concatenate(decoded_parts)[:len(flat_bits)]
    return all_decoded


@torch.no_grad()
def evaluate_metrics_with_channel(model, loader, target_snr, ldpc_code, device):
    """包含完整物理层链路的 MS-SSIM 和 PSNR 联合评估"""

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
        encoder_features = out["features"]

        flat_bits, metadata = features_to_bits(encoder_features)

        decoded_bits = transmit_bits_through_channel(flat_bits, target_snr, ldpc_code, device)

        recovered_features = bits_to_features(decoded_bits, metadata)
        recovered_features = [feat.to(device) for feat in recovered_features]

        reconstructed_images = model.reconstruct_from_features(recovered_features)

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
