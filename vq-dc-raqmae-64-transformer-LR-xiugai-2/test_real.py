# 单支路 VQ：测试不同 SNR 下的 MS-SSIM 与 PSNR（LDPC + BPSK + AWGN）

import torch
import random
import numpy as np
import os
from config import Config
from models.deepsc import DeepSC
from data.datasets import get_dataloader

from communications.ldpc_coding import get_ldpc_code
from communications.evaluate import evaluate_metrics_with_channel


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
    print("单支路 VQ 真实链路测试 (Real Transmission Chain)")
    print(f"LDPC: n={LDPC_N}, k={LDPC_K}, R={LDPC_R}")
    print(f"调制: BPSK（与 evaluate 一致）")
    print(f"各层码本大小 NUM_EMBEDDINGS_LIST: {cfg.NUM_EMBEDDINGS_LIST}")
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
    ).to(device)

    checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, "best_vq_deepsc.pth")
    print(f"Loading checkpoint from {checkpoint_path}")
    deepsc_model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)

    deepsc_model.eval()

    test_dataloader = get_dataloader(
        root_dir=cfg.TEST_DATASET_PATH,
        batch_size=1,
        shuffle=False,
        mode="test",
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    results = {}
    for snr in TEST_SNRS:
        print(f"\n正在测试 SNR = {snr} dB ...")

        mean_ms_ssim, mean_psnr = evaluate_metrics_with_channel(
            deepsc_model,
            test_dataloader,
            cfg.NUM_EMBEDDINGS_LIST,
            snr,
            ldpc_code,
            device,
        )

        results[snr] = {"ms_ssim": mean_ms_ssim, "psnr": mean_psnr}
        print(f"SNR {snr} dB | Avg MS-SSIM: {mean_ms_ssim:.4f} | Avg PSNR: {mean_psnr:.4f} dB")

    print("\n" + "=" * 40)
    print("=== VQ 最终测试结果 ===")
    print(f"NUM_EMBEDDINGS_LIST: {cfg.NUM_EMBEDDINGS_LIST}")
    print("=" * 40)
    print(f"{'SNR (dB)':<10} | {'MS-SSIM':<10} | {'PSNR (dB)':<10}")
    print("-" * 11 + "|" + "-" * 12 + "|" + "-" * 11)
    for snr in TEST_SNRS:
        final_ssim = results[snr]["ms_ssim"]
        final_psnr = results[snr]["psnr"]
        print(f"{snr:<10} | {final_ssim:<10.4f} | {final_psnr:<10.4f}")


if __name__ == "__main__":
    test_real()
