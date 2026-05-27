# 测试 SimVQ 模型的 Bits-Per-Pixel (BPP)

import torch
import math
import os
from config import Config
from models.deepsc import DeepSC
from data.datasets import get_dataloader


@torch.no_grad()
def test_bpp():
    cfg = Config()
    device = torch.device(cfg.DEVICE)

    # 模型加载
    deepsc_model = DeepSC(
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        num_downsample_blocks=cfg.NUM_DOWNSAMPLE_BLOCKS,
        base_channels=cfg.BASE_CHANNELS,
        num_embeddings_list=cfg.NUM_EMBEDDINGS_LIST,
        embedding_dim_list=cfg.EMBEDDING_DIM_LIST,
        commitment_cost=cfg.COMMITMENT_COST,
        device=device,
        strides=cfg.DOWNSAMPLE_STRIDES
    ).to(device)

    checkpoint_path = os.path.join("/workspace/yi/work/Simvq-dc-64-Multi-pro-3ceng/checkpoints/best_vq_deepsc.pth")
    print(f"Loading checkpoint from {checkpoint_path}")
    deepsc_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    deepsc_model.eval()

    # 测试数据加载
    test_dataloader = get_dataloader(
        root_dir=cfg.TEST_DATASET_PATH,
        batch_size=1,
        shuffle=False,
        mode='test',
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    num_layers = cfg.NUM_DOWNSAMPLE_BLOCKS
    bits_per_token = [math.log2(k) for k in cfg.NUM_EMBEDDINGS_LIST]  # K=64 -> 6 bits

    print("=" * 60)
    print("  BPP (Bits-Per-Pixel) 测试")
    print(f"  码本大小: {cfg.NUM_EMBEDDINGS_LIST}")
    print(f"  每层每 token 比特数: {[f'{b:.1f}' for b in bits_per_token]}")
    print("=" * 60)

    # 全局累加器
    total_bits_accumulated = 0.0
    total_pixels_accumulated = 0.0
    # 各层独立统计
    layer_bits_accumulated = [0.0] * num_layers

    for images in test_dataloader:
        images = images.to(device)
        B, C, H, W = images.shape
        batch_pixels = B * H * W

        # 前向传播获取各层索引
        out = deepsc_model.forward_test(images)
        indices_list = out["indices"]

        batch_bits = 0.0
        for l in range(num_layers):
            idx_l = indices_list[l]  # [B, H_l, W_l]
            H_l, W_l = idx_l.shape[1], idx_l.shape[2]

            # 每个空间位置承载 log2(K_l) 比特
            layer_bits = B * H_l * W_l * bits_per_token[l]

            layer_bits_accumulated[l] += layer_bits
            batch_bits += layer_bits

        total_bits_accumulated += batch_bits
        total_pixels_accumulated += batch_pixels

    # 计算整体 BPP
    average_bpp = total_bits_accumulated / total_pixels_accumulated

    # 各层 BPP 分解
    layer_bpp = [layer_bits_accumulated[l] / total_pixels_accumulated for l in range(num_layers)]

    print("\n" + "=" * 60)
    print("  BPP 测试结果")
    print("=" * 60)
    for l in range(num_layers):
        print(f"  Layer {l}: K={cfg.NUM_EMBEDDINGS_LIST[l]}, "
              f"bits/token={bits_per_token[l]:.1f}, "
              f"BPP={layer_bpp[l]:.4f}")
    print("-" * 60)
    print(f"  总 BPP = {average_bpp:.4f} bits/pixel")
    print("=" * 60)


if __name__ == "__main__":
    test_bpp()
