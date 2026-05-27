# 测试无VQ模型的重建质量
# 编码器 → 连续特征 → 直接传入解码器 → 重建图像
# 无信道、无量化，测试纯自编码器的重建上界

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
def test_real():
    cfg = Config()
    setup_seed(42)
    device = torch.device(cfg.DEVICE)

    print("=" * 50)
    print("无 VQ 模型重建质量测试")
    print("链路: 编码器 → 连续特征 → 解码器 (无信道、无量化)")
    print("=" * 50)

    deepsc_model = DeepSC(
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        num_downsample_blocks=cfg.NUM_DOWNSAMPLE_BLOCKS,
        base_channels=cfg.BASE_CHANNELS,
        embedding_dim_list=cfg.EMBEDDING_DIM_LIST,
        device=device
    ).to(device)

    checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, "best_deepsc.pth")
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

    ms_ssim_scores = []
    psnr_scores = []

    for real_image in test_dataloader:
        real_image = real_image.to(device)

        # 编码器 → 连续特征 → 解码器
        out = deepsc_model.forward_test(real_image)
        reconstructed_images = deepsc_model.reconstruct_from_features(out["features"])

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

    mean_ms_ssim = np.mean(ms_ssim_scores)
    mean_psnr = np.mean(psnr_scores)

    print("\n" + "=" * 50)
    print("=== 无 VQ 模型重建结果 ===")
    print(f"MS-SSIM: {mean_ms_ssim:.4f}")
    print(f"PSNR:    {mean_psnr:.4f} dB")
    print("=" * 50)


if __name__ == "__main__":
    test_real()
