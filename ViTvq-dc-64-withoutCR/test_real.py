# 测试 VQ-ViT 模型在不同 SNR 的 MS-SSIM 和 PSNR 值

import torch
import random
import numpy as np
import os
from tqdm import tqdm
from config import Config
from models.deepsc import DeepSC
from data.datasets import get_dataloader

from communications.ldpc_coding import get_ldpc_code, ldpc_encode, ldpc_decode
from communications.modulation import *
from communications.channel import awgn_channel
from utils.bit_utils import indices_to_bits, bits_to_indices
from utils.metrics import calculate_ms_ssim


LDPC_N = 256
LDPC_R = 0.5


def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def test_real():
    cfg = Config()

    setup_seed(42)

    device = torch.device(cfg.DEVICE)
    LDPC_K = int(LDPC_N * LDPC_R)

    TEST_SNRS = [0, 3, 6, 9, 12]

    print("=" * 40)
    print(f"开始 VQ-ViT (无压缩版) 真实环境测试 (Real Transmission Chain)")
    print(f"LDPC: n={LDPC_N}, k={LDPC_K}, R={LDPC_R}")
    print(f"调制: QPSK")
    print(f"码本大小: {cfg.NUM_EMBEDDINGS_LIST}")
    print(f"QBridge-ViT: {cfg.QB_TYPE}")
    print(f"测试 SNR: {TEST_SNRS} dB")
    print("=" * 40)

    ldpc_code = get_ldpc_code(LDPC_K, rate=LDPC_R)

    deepsc_model = DeepSC(
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        num_downsample_blocks=cfg.NUM_DOWNSAMPLE_BLOCKS,
        base_channels=cfg.BASE_CHANNELS,
        num_embeddings_list=cfg.NUM_EMBEDDINGS_LIST,
        embedding_dim_list=cfg.EMBEDDING_DIM_LIST,
        commitment_cost=cfg.COMMITMENT_COST,
        device=device,
        QB_type=cfg.QB_TYPE,
        emb_nograd=cfg.EMB_NOGRAD,
    ).to(device)

    checkpoint_path = os.path.join("/workspace/yi/work/ViTvq-dc-64-withoutCR/checkpoints/best_vq_deepsc.pth")
    print(f"Loading checkpoint from {checkpoint_path}")
    deepsc_model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    deepsc_model.eval()

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
        print(f"SNR {snr} dB | VQ-ViT Avg MS-SSIM: {mean_ms_ssim:.4f} | Avg PSNR: {mean_psnr:.4f} dB")

    print("\n" + "=" * 40)
    print("=== VQ-ViT (无压缩版) 最终测试结果 ===")
    print(f"Codebook K List: {cfg.NUM_EMBEDDINGS_LIST}")
    print(f"QBridge-ViT: {cfg.QB_TYPE}")
    print("=" * 40)
    print(f"{'SNR (dB)':<10} | {'MS-SSIM':<10} | {'PSNR (dB)':<10}")
    print("-" * 11 + "|" + "-" * 12 + "|" + "-" * 11)
    for snr in TEST_SNRS:
        final_ssim = results[snr]['ms_ssim']
        final_psnr = results[snr]['psnr']
        print(f"{snr:<10} | {final_ssim:<10.4f} | {final_psnr:<10.4f}")


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


if __name__ == "__main__":
    test_real()
