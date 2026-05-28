import argparse
import json
import os

import torch

from config import Config
from data.datasets import get_dataloader
from evaluation.quality import evaluate_ldpc_channel, evaluate_no_channel
from utils.checkpoint_utils import build_model_from_checkpoint
from utils.reproducibility import setup_seed


LDPC_N = 256
LDPC_R = 0.5


@torch.no_grad()
def test_real(checkpoint_path=None, test_snrs=None, json_output=None, no_channel=False):
    cfg = Config()
    setup_seed(42)

    device = torch.device(cfg.DEVICE)
    test_snrs = test_snrs or [0, 3, 6, 9, 12]
    checkpoint_path = checkpoint_path or os.path.join(cfg.CHECKPOINT_DIR, "best_vq_deepsc.pth")

    print("=" * 40)
    print("开始 SimVQ 支路真实环境测试 (Real Transmission Chain)")
    if no_channel:
        print("链路: no-channel source reconstruction upper bound")
    else:
        print(f"LDPC: n={LDPC_N}, k={int(LDPC_N * LDPC_R)}, R={LDPC_R}")
        print("调制: BPSK")
    print(f"Loading checkpoint from {checkpoint_path}")

    deepsc_model, inferred = build_model_from_checkpoint(checkpoint_path, cfg, device)
    num_embeddings_list = inferred["num_embeddings_list"]
    print(f"码本大小: {num_embeddings_list} (inferred from checkpoint)")
    if not no_channel:
        print(f"测试 SNR: {test_snrs} dB")
    print("=" * 40)

    test_dataloader = get_dataloader(
        root_dir=cfg.TEST_DATASET_PATH,
        batch_size=1,
        shuffle=False,
        mode="test",
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    results = {}
    if no_channel:
        mean_ms_ssim, mean_psnr = evaluate_no_channel(deepsc_model, test_dataloader, device)
        results["no_channel"] = {"ms_ssim": mean_ms_ssim, "psnr": mean_psnr}
        print(f"No channel | SimVQ Avg MS-SSIM: {mean_ms_ssim:.4f} | Avg PSNR: {mean_psnr:.4f} dB")
    else:
        from communications.ldpc_coding import get_ldpc_code

        ldpc_code = get_ldpc_code(int(LDPC_N * LDPC_R), rate=LDPC_R)
        for snr in test_snrs:
            print(f"\n正在测试 SNR = {snr} dB ...")
            mean_ms_ssim, mean_psnr = evaluate_ldpc_channel(
                deepsc_model, test_dataloader, num_embeddings_list, snr, ldpc_code, device)
            results[snr] = {"ms_ssim": mean_ms_ssim, "psnr": mean_psnr}
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a SimVQ checkpoint on Kodak.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path; defaults to the best model.")
    parser.add_argument("--snrs", type=int, nargs="+", default=[0, 3, 6, 9, 12])
    parser.add_argument("--json-output", default=None, help="Optional JSON output path.")
    parser.add_argument("--no-channel", action="store_true", help="Evaluate source reconstruction only.")
    args = parser.parse_args()
    test_real(args.checkpoint, args.snrs, args.json_output, args.no_channel)
