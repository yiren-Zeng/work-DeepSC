# 测试 SimVQ 模型在不同 SNR 的 MS-SSIM 和 PSNR 值

import torch
import random
import numpy as np
import os
import argparse
import json
from tqdm import tqdm
from config import Config
from models.deepsc import DeepSC
from data.datasets import get_dataloader

# === 复用项目中的工具模块 ===
from communications.ldpc_coding import get_ldpc_code, ldpc_encode, ldpc_decode
from communications.modulation import *
from communications.channel import awgn_channel
from utils.bit_utils import indices_to_bits, bits_to_indices
from utils.metrics import calculate_ms_ssim


LDPC_N = 256  # 码字块长度，不等于信息位长度
LDPC_R = 0.5


def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def test_real(checkpoint_path=None, test_snrs=None, json_output=None, no_channel=False):
    # 1. 配置加载
    cfg = Config()

    # === 【关键】必须先固定种子 ===
    setup_seed(42)

    device = torch.device(cfg.DEVICE)
    LDPC_K = int(LDPC_N * LDPC_R)  # k=128

    test_snrs = test_snrs or [0, 3, 6, 9, 12]

    print("=" * 40)
    print(f"开始 SimVQ 支路真实环境测试 (Real Transmission Chain)")
    if no_channel:
        print("链路: no-channel source reconstruction upper bound")
    else:
        print(f"LDPC: n={LDPC_N}, k={LDPC_K}, R={LDPC_R}")
        print(f"调制: BPSK")
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
    num_downsample_blocks = len(codebook_weights)
    if num_downsample_blocks != cfg.NUM_DOWNSAMPLE_BLOCKS:
        raise ValueError(
            "Checkpoint layer count differs from Config; provide a compatible "
            "DOWNSAMPLE_STRIDES setting before evaluation."
        )

    print(f"码本大小: {num_embeddings_list} (inferred from checkpoint)")
    if not no_channel:
        print(f"测试 SNR: {test_snrs} dB")
    print("=" * 40)

    # 2. 初始化 LDPC
    ldpc_code = None if no_channel else get_ldpc_code(LDPC_K, rate=LDPC_R)

    # 3. 模型加载
    deepsc_model = DeepSC(
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        num_downsample_blocks=num_downsample_blocks,
        base_channels=cfg.BASE_CHANNELS,
        num_embeddings_list=num_embeddings_list,
        embedding_dim_list=embedding_dim_list,
        commitment_cost=cfg.COMMITMENT_COST,
        device=device,
        strides=cfg.DOWNSAMPLE_STRIDES
    ).to(device)

    deepsc_model.load_state_dict(state_dict)

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
    if no_channel:
        mean_ms_ssim, mean_psnr = evaluate_metrics_without_channel(
            deepsc_model, test_dataloader, device)
        results["no_channel"] = {'ms_ssim': mean_ms_ssim, 'psnr': mean_psnr}
        print(f"No channel | SimVQ Avg MS-SSIM: {mean_ms_ssim:.4f} | Avg PSNR: {mean_psnr:.4f} dB")
    else:
        for snr in test_snrs:
            print(f"\n正在测试 SNR = {snr} dB ...")

            mean_ms_ssim, mean_psnr = evaluate_metrics_with_channel(
                deepsc_model, test_dataloader, num_embeddings_list, snr, ldpc_code, device)

            results[snr] = {'ms_ssim': mean_ms_ssim, 'psnr': mean_psnr}
            print(f"SNR {snr} dB | SimVQ Avg MS-SSIM: {mean_ms_ssim:.4f} | Avg PSNR: {mean_psnr:.4f} dB")

    print("\n" + "=" * 40)
    print("=== SimVQ 最终测试结果 ===")
    print(f"Codebook K List: {num_embeddings_list}")
    print("=" * 40)
    print(f"{'Condition':<10} | {'MS-SSIM':<10} | {'PSNR (dB)':<10}")
    print("-" * 11 + "|" + "-" * 12 + "|" + "-" * 11)
    for condition, metrics in results.items():
        print(f"{str(condition):<10} | {metrics['ms_ssim']:<10.4f} | {metrics['psnr']:<10.4f}")

    if json_output:
        os.makedirs(os.path.dirname(json_output) or ".", exist_ok=True)
        payload = {
            "checkpoint": checkpoint_path,
            "num_embeddings_list": num_embeddings_list,
            "results": {str(condition): metrics for condition, metrics in results.items()},
        }
        with open(json_output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"JSON results saved to {json_output}")

    return results


@torch.no_grad()
def evaluate_metrics_without_channel(model, loader, device):
    """Evaluate the source coder before channel corruption."""
    model.eval()
    ms_ssim_scores = []
    psnr_scores = []
    for real_image in loader:
        real_image = real_image.to(device)
        out = model.forward_test(real_image)
        reconstructed_images = model.reconstruct_from_indices(out["indices"])
        img1 = (real_image + 1) / 2
        img2 = (reconstructed_images + 1) / 2
        ms_ssim_scores.append(calculate_ms_ssim(img1, img2))
        mse = torch.mean((img1 - img2) ** 2)
        psnr_scores.append(100.0 if mse == 0 else 10 * torch.log10(1.0 / mse).item())
    return np.mean(ms_ssim_scores), np.mean(psnr_scores)


@torch.no_grad()
def evaluate_metrics_with_channel(model, loader, num_embeddings_list, target_snr, ldpc_code, device):
    """包含完整物理层链路的 MS-SSIM 和 PSNR 联合评估 (SimVQ/SRC版本)"""

    # =======================================================
    # 【核心代码】：强制重置内部随机种子
    # 确保无论这个函数被循环调用多少次，每一帧图像加上去的 AWGN 噪声都绝对一致
    # =======================================================
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # =======================================================

    model.eval()
    ms_ssim_scores = []  # 记录 MS-SSIM 得分
    psnr_scores = []     # 记录 PSNR 得分

    for real_image in loader:
        real_image = real_image.to(device)  # 原始图像张量

        # 1. 信源编码与量化 (SimVQ 使用 forward_test)
        out = model.forward_test(real_image)

        # 2. 索引转比特流
        flat_bits, original_spatial_dims, original_num_embeddings = indices_to_bits(
            out["indices"], num_embeddings_list)

        # 3. LDPC 编码
        coded_bits = ldpc_encode(flat_bits, code=ldpc_code)
        coded_bits_tensor = torch.from_numpy(coded_bits).float().to(device)

        # 4. 调制
        symbols = bpsk_modulate(coded_bits_tensor)

        # 5. AWGN 信道加噪
        noisy_symbols = awgn_channel(symbols, target_snr)

        # 6. 软解调与 LDPC 译码
        llrs = bpsk_llr(noisy_symbols, target_snr, device)
        decoded_bits = ldpc_decode(llrs.cpu().numpy(), ldpc_code)

        # 7. 截断填充与还原索引
        decoded_bits = decoded_bits[:len(flat_bits)]
        recovered_indices_list = bits_to_indices(
            decoded_bits, original_spatial_dims, original_num_embeddings)
        recovered_indices_list = [idx.to(device) for idx in recovered_indices_list]

        # 8. 信源解码与重建
        reconstructed_images = model.reconstruct_from_indices(recovered_indices_list)

        # 统一将图像还原到 [0, 1] 区间
        img1 = (real_image + 1) / 2
        img2 = (reconstructed_images + 1) / 2

        # === 计算 MS-SSIM ===
        ms_ssim = calculate_ms_ssim(img1, img2)
        ms_ssim_scores.append(ms_ssim)

        # === 计算 PSNR ===
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            psnr = 100.0  # 理想上限
        else:
            psnr = 10 * torch.log10(1.0 / mse).item()
        psnr_scores.append(psnr)

    # 返回一个元组，同时包含两个指标的测试集平均分
    return np.mean(ms_ssim_scores), np.mean(psnr_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a SimVQ checkpoint on Kodak.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path; defaults to the best model.")
    parser.add_argument("--snrs", type=int, nargs="+", default=[0, 3, 6, 9, 12])
    parser.add_argument("--json-output", default=None, help="Optional JSON output path.")
    parser.add_argument("--no-channel", action="store_true", help="Evaluate source reconstruction only.")
    args = parser.parse_args()
    test_real(args.checkpoint, args.snrs, args.json_output, args.no_channel)
