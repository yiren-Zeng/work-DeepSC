# """
# Fixed-Raq-dc-64-withoutber: 测试真实重建质量（无 BER 训练版本）
#
# 此版本不需要真实信道测试，因为训练时没有添加信道噪声。
# 主要用于验证无噪声条件下的图像重建质量，包括 SRC 和 RAQ 双支路。
# """
# import torch
# import os
# from config import Config
# from models.deepsc import DeepSC
# from data.datasets import get_dataloader
#
#
# def main():
#     cfg = Config()
#     device = torch.device(cfg.DEVICE)
#     print(f"Testing Fixed-Raq-dc-64-withoutber on {device}")
#     print("[Info] Without BER: 无信道测试 (SRC + RAQ 双支路)")
#
#     checkpoint_path = os.path.join("/workspace/yi/work/Raq-dc-64-withoutber/checkpoints/best_vq_deepsc.pth")
#     if not os.path.exists(checkpoint_path):
#         print(f"Error: Checkpoint not found at {checkpoint_path}")
#         return
#
#     deepsc_model = DeepSC(
#         in_channels=cfg.IN_CHANNELS,
#         out_channels=cfg.OUT_CHANNELS,
#         num_downsample_blocks=cfg.NUM_DOWNSAMPLE_BLOCKS,
#         base_channels=cfg.BASE_CHANNELS,
#         num_embeddings_list=cfg.NUM_EMBEDDINGS_LIST,
#         embedding_dim_list=cfg.EMBEDDING_DIM_LIST,
#         commitment_cost=cfg.COMMITMENT_COST,
#         raq_min_trg=cfg.RAQ_MIN_TRG,
#         raq_max_trg=cfg.RAQ_MAX_TRG,
#         device=device
#     ).to(device)
#
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     deepsc_model.load_state_dict(checkpoint)
#     deepsc_model.eval()
#     print(f"Loaded checkpoint from {checkpoint_path}")
#
#     test_dataloader = get_dataloader(
#         root_dir=cfg.TEST_DATASET_PATH,
#         batch_size=1,
#         shuffle=False,
#         mode='test',
#         num_workers=0,
#         pin_memory=False
#     )
#
#     print(f"\nTesting on {len(test_dataloader)} images...")
#
#     with torch.no_grad():
#         for i, images in enumerate(test_dataloader):
#             images = images.to(device)
#             out = deepsc_model.forward_val_raq(images)
#
#             recon_src = out["reconstructed_images_src"]
#             recon_raq = out["reconstructed_images_raq"]
#
#             mse_src = torch.mean((recon_src - images) ** 2).item()
#             mse_raq = torch.mean((recon_raq - images) ** 2).item()
#             psnr_src = 10 * torch.log10(1.0 / (mse_src + 1e-10)).item() if mse_src > 0 else float('inf')
#             psnr_raq = 10 * torch.log10(1.0 / (mse_raq + 1e-10)).item() if mse_raq > 0 else float('inf')
#
#             print(f"  Image {i + 1}:")
#             print(f"    SRC: MSE={mse_src:.6f}, PSNR={psnr_src:.2f} dB")
#             print(f"    RAQ: MSE={mse_raq:.6f}, PSNR={psnr_raq:.2f} dB")
#
#             # 计算码本利用率
#             cb_stats = deepsc_model.compute_codebook_utilization(
#                 test_dataloader,
#                 max_batches=1,
#                 device=device
#             )
#             for layer_idx in range(len(cb_stats['src'])):
#                 src_stat = cb_stats['src'][layer_idx]
#                 raq_stat = cb_stats['raq'][layer_idx]
#                 print(f"    Layer {layer_idx}:")
#                 print(f"      SRC: Active={src_stat['active_ratio']:.2%}, Perplexity={src_stat['perplexity']:.1f}")
#                 print(f"      RAQ: Active={raq_stat['active_ratio']:.2%}, Perplexity={raq_stat['perplexity']:.1f}")
#
#             break  # 只测试一张图
#
#     print("\nTest complete.")
#
#
# if __name__ == "__main__":
#     main()

# 选择一个K_trg码本，测试其在不同SNR的MS-SSIM和PSNR值

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
from communications.evaluate import evaluate_metrics_with_channel, evaluate_metrics_with_channel_withoutLDPC


LDPC_N = 256 # 码字块长度，不等于信息位长度
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

    # === RAQ 目标码本大小设置 ===
    # 这里我们使用 Config 中定义的列表，或者你可以手动指定，例如 [64, 64, 64, 64]
    # 确保列表长度等于 NUM_DOWNSAMPLE_BLOCKS


    print("=" * 40)
    print(f"开始 RAQ 支路真实环境测试 (Real Transmission Chain)")
    print(f"LDPC: n={LDPC_N}, k={LDPC_K}, R={LDPC_R}")
    print(f"调制: QPSK")
    print(f"RAQ 目标码本大小 (Target K): {cfg.RAQ_TARGET_LIST}")
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
        raq_min_trg=cfg.RAQ_MIN_TRG,
        raq_max_trg=cfg.RAQ_MAX_TRG,
        device=device
    ).to(device)

    checkpoint_path = os.path.join("/workspace/yi/work/Raq-dc-64-withoutber/checkpoints/best_vq_deepsc.pth")
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

        with torch.no_grad():  # 禁用梯度计算以节省内存
            mean_ms_ssim, mean_psnr = evaluate_metrics_with_channel(
                deepsc_model, test_dataloader, cfg.RAQ_TARGET_LIST, snr, ldpc_code, device)

        results[snr] = {'ms_ssim': mean_ms_ssim, 'psnr': mean_psnr}
        print(f"SNR {snr} dB | RAQ Avg MS-SSIM: {mean_ms_ssim:.4f} | RAQ Avg PSNR: {mean_psnr:.4f} dB")

    print("\n" + "=" * 40)
    print("=== RAQ 最终测试结果 ===")
    print(f"Target K List: {cfg.RAQ_TARGET_LIST}")
    print("=" * 40)
    print(f"{'SNR (dB)':<10} | {'MS-SSIM':<10} | {'PSNR (dB)':<10}")
    print("-" * 11 + "|" + "-" * 12 + "|" + "-" * 11)
    for snr in TEST_SNRS:
        # 取出保存的两个指标
        final_ssim = results[snr]['ms_ssim']
        final_psnr = results[snr]['psnr']
        # 格式化打印：MS-SSIM 保留4位小数，PSNR 保留4位小数
        print(f"{snr:<10} | {final_ssim:<10.4f} | {final_psnr:<10.4f}")

if __name__ == "__main__":
    test_real()