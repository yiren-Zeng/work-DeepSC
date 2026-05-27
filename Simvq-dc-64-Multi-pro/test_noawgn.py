# 测试 SimVQ 无噪声链路: 编码器 → 连续特征 → 量化模块 → 解码器
# 仅评估量化带来的重建损失，不经过任何信道噪声

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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def test_noawgn():
    cfg = Config()
    setup_seed(42)
    device = torch.device(cfg.DEVICE)

    print("=" * 50)
    print("SimVQ 无噪声测试 (No AWGN Channel)")
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

    checkpoint_path = os.path.join("/workspace/yi/work/Simvq-dc-64-Multi-pro/checkpoints/best_vq_deepsc.pth")
    print(f"Loading checkpoint from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 测试数据加载
    test_dataloader = get_dataloader(
        root_dir=cfg.TEST_DATASET_PATH,
        batch_size=1,
        shuffle=False,
        mode='test',
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    psnr_scores = []
    ms_ssim_scores = []

    for idx, real_image in enumerate(test_dataloader):
        real_image = real_image.to(device)

        # 1. 编码器 → 连续特征
        encoder_features = model.semantic_encoder(real_image)

        # 2. 量化模块: 连续特征 → 量化特征 (不经过信道噪声)
        quantized_features = []
        for i, feat in enumerate(encoder_features):
            _, quantized, _ = model.vector_quantizers[i](feat)
            quantized_features.append(quantized)

        # 3. 解码器 → 重建图像
        reconstructed_images = model.semantic_decoder(quantized_features)

        # 图像还原到 [0, 1] 区间
        img_orig = (real_image + 1) / 2
        img_recon = (reconstructed_images + 1) / 2

        # 计算 PSNR
        mse = torch.mean((img_orig - img_recon) ** 2)
        if mse == 0:
            psnr = 100.0
        else:
            psnr = 10 * torch.log10(1.0 / mse).item()
        psnr_scores.append(psnr)

        # 计算 MS-SSIM
        ms_ssim_val = calculate_ms_ssim(img_orig, img_recon)
        ms_ssim_scores.append(ms_ssim_val)

        print(f"  Image {idx:3d} | PSNR: {psnr:.4f} dB | MS-SSIM: {ms_ssim_val:.4f}")

    # 汇总结果
    mean_psnr = np.mean(psnr_scores)
    mean_ms_ssim = np.mean(ms_ssim_scores)

    print("\n" + "=" * 50)
    print("=== SimVQ 无噪声测试最终结果 ===")
    print(f"Codebook K List: {cfg.NUM_EMBEDDINGS_LIST}")
    print(f"测试图像数: {len(psnr_scores)}")
    print("-" * 50)
    print(f"  Avg PSNR:    {mean_psnr:.4f} dB")
    print(f"  Avg MS-SSIM: {mean_ms_ssim:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    test_noawgn()
