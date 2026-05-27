# 测试 SimVQ 模型在不同 SNR 的 MS-SSIM 和 PSNR 值

import torch
import random
import numpy as np
import os
from tqdm import tqdm
from config import Config
from models.deepsc import DeepSC
from data.datasets import get_dataloader

# === 复用项目中的工具模块 ===
from communications.ldpc_coding import get_ldpc_code, ldpc_encode, ldpc_decode
from communications.modulation import *
from communications.channel import awgn_channel
from utils.bit_utils import indices_to_bits, bits_to_indices
from utils.metrics import calculate_ms_ssim


LDPC_N = 256  # 码字块长度，不等于信息位长度
LDPC_R = 0.5


def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def test_real():
    # 1. 配置加载
    cfg = Config()

    # === 【关键】必须先固定种子 ===
    setup_seed(42)

    device = torch.device(cfg.DEVICE)
    LDPC_K = int(LDPC_N * LDPC_R)  # k=128

    TEST_SNRS = [0, 3, 6, 9, 12]

    print("=" * 40)
    print(f"开始 SimVQ 支路真实环境测试 (Real Transmission Chain)")
    print(f"LDPC: n={LDPC_N}, k={LDPC_K}, R={LDPC_R}")
    print(f"调制: QPSK")
    print(f"码本大小: {cfg.NUM_EMBEDDINGS_LIST}")
    print(f"测试 SNR: {TEST_SNRS} dB")
    print("=" * 40)

    # 2. 初始化 LDPC
    ldpc_code = get_ldpc_code(LDPC_K, rate=LDPC_R)

    # 3. 模型加载
    deepsc_model = DeepSC(
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        num_downsample_blocks=cfg.NUM_DOWNSAMPLE_BLOCKS,
        base_channels=cfg.BASE_CHANNELS,
        num_embeddings_list=cfg.NUM_EMBEDDINGS_LIST,
        embedding_dim_list=cfg.EMBEDDING_DIM_LIST,
        commitment_cost=cfg.COMMITMENT_COST,
        device=device
    ).to(device)

    checkpoint_path = os.path.join("/workspace/yi/work/Simvq-dc-64-Multi-pro-GN/checkpoints/best_vq_deepsc.pth")
    print(f"Loading checkpoint from {checkpoint_path}")
    deepsc_model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    deepsc_model.eval()

    # 测试数据加载器
    test_dataloader = get_dataloader(
        root_dir=cfg.TEST_DATASET_PATH,
        batch_size=1,
        shuffle=False,
        mode='test',
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    results = {}
    for snr in TEST_SNRS:
        print(f"\n正在测试 SNR = {snr} dB ...")

        mean_ms_ssim, mean_psnr = evaluate_metrics_with_channel(
            deepsc_model, test_dataloader, cfg.NUM_EMBEDDINGS_LIST, snr, ldpc_code, device)

        results[snr] = {'ms_ssim': mean_ms_ssim, 'psnr': mean_psnr}
        print(f"SNR {snr} dB | SimVQ Avg MS-SSIM: {mean_ms_ssim:.4f} | Avg PSNR: {mean_psnr:.4f} dB")

    print("\n" + "=" * 40)
    print("=== SimVQ 最终测试结果 ===")
    print(f"Codebook K List: {cfg.NUM_EMBEDDINGS_LIST}")
    print("=" * 40)
    print(f"{'SNR (dB)':<10} | {'MS-SSIM':<10} | {'PSNR (dB)':<10}")
    print("-" * 11 + "|" + "-" * 12 + "|" + "-" * 11)
    for snr in TEST_SNRS:
        final_ssim = results[snr]['ms_ssim']
        final_psnr = results[snr]['psnr']
        print(f"{snr:<10} | {final_ssim:<10.4f} | {final_psnr:<10.4f}")


@torch.no_grad()
def evaluate_metrics_with_channel(model, loader, num_embeddings_list, target_snr, ldpc_code, device):
    """包含完整物理层链路的 MS-SSIM 和 PSNR 联合评估 (SimVQ/SRC版本)"""

    # =======================================================
    # 【核心代码】：强制重置内部随机种子
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

        # 1. 信源编码与量化 (SimVQ 使用 forward_test)
        out = model.forward_test(real_image)

        # 2. 索引转比特流
        flat_bits, original_spatial_dims, original_num_embeddings = indices_to_bits(
            out["indices"], num_embeddings_list)

        # 3. LDPC 编码
        coded_bits = ldpc_encode(flat_bits, code=ldpc_code)
        coded_bits_tensor = torch.from_numpy(coded_bits).float().to(device)

        # 4. 调制
        symbols = bpsk_modulate(coded_bits_tensor)

        # 5. AWGN 信道加噪
        noisy_symbols = awgn_channel(symbols, target_snr)

        # 6. 软解调与 LDPC 译码
        llrs = bpsk_llr(noisy_symbols, target_snr, device)
        decoded_bits = ldpc_decode(llrs.cpu().numpy(), ldpc_code)

        # 7. 截断填充与还原索引
        decoded_bits = decoded_bits[:len(flat_bits)]
        recovered_indices_list = bits_to_indices(
            decoded_bits, original_spatial_dims, original_num_embeddings)
        recovered_indices_list = [idx.to(device) for idx in recovered_indices_list]

        # 8. 信源解码与重建
        reconstructed_images = model.reconstruct_from_indices(recovered_indices_list)

        # 统一将图像还原到 [0, 1] 区间
        img1 = (real_image + 1) / 2
        img2 = (reconstructed_images + 1) / 2

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


if __name__ == "__main__":
    test_real()
