import argparse
import os

import torch

from config import Config
from data.datasets import get_dataloader
from evaluation.bpp import calculate_bpp
from utils.checkpoint_utils import build_model_from_checkpoint


@torch.no_grad()
def evaluate_bpp(checkpoint_path=None):
    cfg = Config()
    device = torch.device(cfg.DEVICE)
    checkpoint_path = checkpoint_path or os.path.join(cfg.CHECKPOINT_DIR, "best_vq_deepsc.pth")

    print(f"Loading checkpoint from {checkpoint_path}")
    deepsc_model, inferred = build_model_from_checkpoint(checkpoint_path, cfg, device)
    num_embeddings_list = inferred["num_embeddings_list"]

    test_dataloader = get_dataloader(
        root_dir=cfg.TEST_DATASET_PATH,
        batch_size=1,
        shuffle=False,
        mode="test",
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    result = calculate_bpp(deepsc_model, test_dataloader, num_embeddings_list, device)
    print("=" * 60)
    print("  BPP (Bits-Per-Pixel) 测试")
    print(f"  码本大小: {num_embeddings_list} (inferred from checkpoint)")
    print(f"  每层每 token 比特数: {[f'{b:.1f}' for b in result['bits_per_token']]}")
    print("=" * 60)
    print("\n" + "=" * 60)
    print("  BPP 测试结果")
    print("=" * 60)
    for i, layer_bpp in enumerate(result["layer_bpp"]):
        print(f"  Layer {i}: K={num_embeddings_list[i]}, "
              f"bits/token={result['bits_per_token'][i]:.1f}, "
              f"BPP={layer_bpp:.4f}")
    print("-" * 60)
    print(f"  总 BPP = {result['average_bpp']:.4f} bits/pixel")
    print("=" * 60)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate BPP for a SimVQ checkpoint.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path; defaults to the best model.")
    args = parser.parse_args()
    evaluate_bpp(args.checkpoint)
