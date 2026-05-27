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

    checkpoint_path = os.path.join("/workspace/yi/work/vq-dc-raqmae-64-transformer-LR-xiugai-3/checkpoints/best_vq_deepsc.pth")
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