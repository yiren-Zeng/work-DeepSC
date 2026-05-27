# MS-SSIM 和 PSNR 评价指标

import torch
import numpy as np
import random # 确保导入 random
from communications.ldpc_coding import get_ldpc_code, ldpc_encode, ldpc_decode
from communications.modulation import *
from communications.channel import awgn_channel
from utils.bit_utils import indices_to_bits, bits_to_indices
from utils.metrics import calculate_ms_ssim


@torch.no_grad()
def evaluate_metrics_with_channel(model, loader, k_trg, target_snr, ldpc_code, device):
    """包含完整物理层链路的 MS-SSIM 和 PSNR 联合评估"""

    # =======================================================
    # 【新增核心代码】：强制重置内部随机种子
    # 确保无论这个函数被循环调用多少次，每一帧图像加上去的 AWGN 噪声都绝对一致
    # =======================================================
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # =======================================================

    model.eval()
    ms_ssim_scores = []  # 记录 MS-SSIM 得分
    psnr_scores = []     # 记录 PSNR 得分

    for real_image in loader:
        real_image = real_image.to(device)  # 原始图像张量

        # 1. 信源编码与量化
        out = model.forward_test_raq(real_image, k_trg)

        # 2. 索引转比特流
        flat_bits_raq, original_spatial_dims_raq, original_num_embeddings_raq = indices_to_bits(
            out["indices_raq"], k_trg)

        # 3. LDPC 编码
        coded_bits = ldpc_encode(flat_bits_raq, code=ldpc_code)
        coded_bits_tensor = torch.from_numpy(coded_bits).float().to(device)

        # 4. 调制
        symbols = bpsk_modulate(coded_bits_tensor)

        # 5. AWGN 信道加噪
        noisy_symbols = awgn_channel(symbols, target_snr)

        # 6. 软解调与 LDPC 译码
        llrs = bpsk_llr(noisy_symbols, target_snr, device)
        decoded_bits = ldpc_decode(llrs.cpu().numpy(), ldpc_code)

        # 7. 截断填充与还原索引
        decoded_bits = decoded_bits[:len(flat_bits_raq)]
        recovered_indices_list = bits_to_indices(
            decoded_bits, original_spatial_dims_raq, original_num_embeddings_raq)
        recovered_indices_list = [idx.to(device) for idx in recovered_indices_list]

        # 8. 信源解码与重建
        reconstructed_images_raq = model.reconstruct_from_indices(recovered_indices_list, out["W_trg_list"])

        # 统一将图像还原到 [0, 1] 区间
        img1 = (real_image + 1) / 2
        img2 = (reconstructed_images_raq + 1) / 2

        # === 计算 MS-SSIM ===
        ms_ssim = calculate_ms_ssim(img1, img2)
        ms_ssim_scores.append(ms_ssim)

        # === 计算 PSNR ===
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            psnr = 100.0  # 理想上限
        else:
            psnr = 10 * torch.log10(1.0 / mse).item()
        psnr_scores.append(psnr)

    # 返回一个元组，同时包含两个指标的测试集平均分
    return np.mean(ms_ssim_scores), np.mean(psnr_scores)

@torch.no_grad()
def evaluate_metrics_with_channel_withoutLDPC(model, loader, k_trg, target_snr, device):

    # =======================================================
    # 强制重置内部随机种子
    # 确保无论这个函数被循环调用多少次，每一帧图像加上去的 AWGN 噪声都绝对一致
    # =======================================================
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # =======================================================

    model.eval()
    ms_ssim_scores = []  # 记录 MS-SSIM 得分
    psnr_scores = []     # 记录 PSNR 得分

    for real_image in loader:
        real_image = real_image.to(device)  # 原始图像张量

        # 1. 信源编码与量化
        out = model.forward_test_raq(real_image, k_trg)

        # 2. 索引转比特流
        flat_bits_raq, original_spatial_dims_raq, original_num_embeddings_raq = indices_to_bits(
            out["indices_raq"], k_trg)

        coded_bits_tensor = torch.from_numpy(flat_bits_raq).float().to(device)

        # 4. 调制
        symbols = bpsk_modulate(coded_bits_tensor)

        # 5. AWGN 信道加噪
        noisy_symbols = awgn_channel(symbols, target_snr)

        # 6. 解调
        decoded_bits = bpsk_demodulate(noisy_symbols)

        # 将 GPU Tensor 转移到 CPU 并转换为 NumPy 数组
        decoded_bits = decoded_bits.cpu().numpy()

        # 7. 还原索引
        recovered_indices_list = bits_to_indices(
            decoded_bits, original_spatial_dims_raq, original_num_embeddings_raq)
        recovered_indices_list = [idx.to(device) for idx in recovered_indices_list]

        # 8. 信源解码与重建
        reconstructed_images_raq = model.reconstruct_from_indices(recovered_indices_list, out["W_trg_list"])

        # ==========================================
        # 9. 核心修改区：联合计算双指标
        # ==========================================
        # 统一将图像还原到 [0, 1] 区间
        img1 = (real_image + 1) / 2
        img2 = (reconstructed_images_raq + 1) / 2

        # === 计算 MS-SSIM ===
        ms_ssim = calculate_ms_ssim(img1, img2)
        ms_ssim_scores.append(ms_ssim)

        # === 计算 PSNR ===
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            psnr = 100.0  # 理想上限
        else:
            psnr = 10 * torch.log10(1.0 / mse).item()
        psnr_scores.append(psnr)

    # 返回一个元组，同时包含两个指标的测试集平均分
    return np.mean(ms_ssim_scores), np.mean(psnr_scores)
