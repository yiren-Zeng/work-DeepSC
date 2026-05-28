# 测试 SimVQ 模型的 Bits-Per-Pixel (BPP)

import torch
import math
import os
import argparse
from config import Config
from models.deepsc import DeepSC
from data.datasets import get_dataloader


@torch.no_grad()
def test_bpp(checkpoint_path=None):
    cfg = Config()
    device = torch.device(cfg.DEVICE)

    checkpoint_path = checkpoint_path or os.path.join(cfg.CHECKPOINT_DIR, "best_vq_deepsc.pth")
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    codebook_weights = [
        state_dict[key] for key in sorted(state_dict)
        if key.endswith("codebook.embed.weight")
    ]
    num_embeddings_list = [weight.shape[0] for weight in codebook_weights]
    embedding_dim_list = [weight.shape[1] for weight in codebook_weights]
    num_layers = len(codebook_weights)
    if num_layers != cfg.NUM_DOWNSAMPLE_BLOCKS:
        raise ValueError(
            "Checkpoint layer count differs from Config; provide a compatible "
            "DOWNSAMPLE_STRIDES setting before evaluation."
        )

    # 模型加载
    deepsc_model = DeepSC(
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        num_downsample_blocks=num_layers,
        base_channels=cfg.BASE_CHANNELS,
        num_embeddings_list=num_embeddings_list,
        embedding_dim_list=embedding_dim_list,
        commitment_cost=cfg.COMMITMENT_COST,
        device=device,
        strides=cfg.DOWNSAMPLE_STRIDES
    ).to(device)

    deepsc_model.load_state_dict(state_dict)
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

    bits_per_token = [math.log2(k) for k in num_embeddings_list]

    print("=" * 60)
    print("  BPP (Bits-Per-Pixel) 测试")
    print(f"  码本大小: {num_embeddings_list} (inferred from checkpoint)")
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
        print(f"  Layer {l}: K={num_embeddings_list[l]}, "
              f"bits/token={bits_per_token[l]:.1f}, "
              f"BPP={layer_bpp[l]:.4f}")
    print("-" * 60)
    print(f"  总 BPP = {average_bpp:.4f} bits/pixel")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate BPP for a SimVQ checkpoint.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path; defaults to the best model.")
    args = parser.parse_args()
    test_bpp(args.checkpoint)
