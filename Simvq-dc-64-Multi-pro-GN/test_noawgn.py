# 测试 SimVQ 无噪声链路的重建质量 (PSNR & MS-SSIM)
# 链路: 编码器 → 连续特征 → 量化模块 → 解码器 (无 AWGN)

import torch
import random
import numpy as np
import os
from config import Config
from models.deepsc import DeepSC
from data.datasets import get_dataloader
from utils.metrics import calculate_ms_ssim


def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def test_noawgn():
    cfg = Config()
    setup_seed(42)

    device = torch.device(cfg.DEVICE)

    print("=" * 50)
    print("SimVQ 无噪声链路重建质量测试")
    print("链路: 编码器 → 连续特征 → 量化模块 → 解码器")
    print(f"码本大小: {cfg.NUM_EMBEDDINGS_LIST}")
    print(f"嵌入维度: {cfg.EMBEDDING_DIM_LIST}")
    print("=" * 50)

    # 模型加载
    model = DeepSC(
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        num_downsample_blocks=cfg.NUM_DOWNSAMPLE_BLOCKS,
        base_channels=cfg.BASE_CHANNELS,
        num_embeddings_list=cfg.NUM_EMBEDDINGS_LIST,
        embedding_dim_list=cfg.EMBEDDING_DIM_LIST,
        commitment_cost=cfg.COMMITMENT_COST,
        device=device
    ).to(device)

    checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, "best_vq_deepsc.pth")
    print(f"Loading checkpoint from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 测试数据
    test_dataloader = get_dataloader(
        root_dir=cfg.TEST_DATASET_PATH,
        batch_size=1,
        shuffle=False,
        mode='test',
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    ms_ssim_scores = []
    psnr_scores = []

    for real_image in test_dataloader:
        real_image = real_image.to(device)

        # === 无噪声链路: 编码器 → 量化 → 解码器 ===
        # 1. 编码器提取连续特征
        encoder_features = model.semantic_encoder(real_image)

        # 2. 量化模块: 连续特征 → 量化特征
        quantized_features = []
        for i, feat in enumerate(encoder_features):
            vq_loss, quantized, encoding_idx = model.vector_quantizers[i](feat)
            quantized_features.append(quantized)

        # 3. 解码器重建
        reconstructed = model.semantic_decoder(quantized_features)

        # 统一到 [0, 1] 区间
        img_orig = (real_image + 1) / 2
        img_recon = (reconstructed + 1) / 2

        # MS-SSIM
        ms_ssim_val = calculate_ms_ssim(img_orig, img_recon)
        ms_ssim_scores.append(ms_ssim_val)

        # PSNR
        mse = torch.mean((img_orig - img_recon) ** 2)
        if mse == 0:
            psnr = 100.0
        else:
            psnr = 10 * torch.log10(1.0 / mse).item()
        psnr_scores.append(psnr)

    # 汇总结果
    avg_ms_ssim = np.mean(ms_ssim_scores)
    avg_psnr = np.mean(psnr_scores)

    print("\n" + "=" * 50)
    print("=== 无噪声链路重建质量测试结果 ===")
    print(f"测试图像数: {len(psnr_scores)}")
    print(f"码本大小:   {cfg.NUM_EMBEDDINGS_LIST}")
    print("-" * 50)
    print(f"{'指标':<15} | {'均值':<10} | {'最大值':<10} | {'最小值':<10}")
    print("-" * 15 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12)
    print(f"{'PSNR (dB)':<15} | {avg_psnr:<10.4f} | {max(psnr_scores):<10.4f} | {min(psnr_scores):<10.4f}")
    print(f"{'MS-SSIM':<15} | {avg_ms_ssim:<10.4f} | {max(ms_ssim_scores):<10.4f} | {min(ms_ssim_scores):<10.4f}")
    print("=" * 50)


if __name__ == "__main__":
    test_noawgn()
