import argparse
import json
import os
from fractions import Fraction
import torch
from config import Config
from data.datasets import get_dataloader
from evaluation.perceptual import PerceptualMetrics
from evaluation.quality import evaluate_ldpc_channel, evaluate_no_channel
from utils.checkpoint_utils import build_model_from_checkpoint
from utils.reproducibility import setup_seed


LDPC_N = 256
LDPC_R = 0.5


def _format_cbr(stats):
    if not stats or "channel_symbols" not in stats or not stats.get("source_values"):
        return "N/A"
    ratio = Fraction(int(stats["channel_symbols"]), int(stats["source_values"]))
    return f"{ratio.numerator}/{ratio.denominator}"


def _print_compression_stats(stats):
    cbr = _format_cbr(stats)
    if cbr != "N/A":
        print(
            "[压缩率] 传输压缩率 CBR（信道符号数/原图元素数）: "
            f"{cbr}"
        )


def _format_metric(value):
    return "N/A" if value is None else f"{value:.4f}"


@torch.no_grad()
def test_real(
    checkpoint_path=None, test_snrs=None, json_output=None, no_channel=False,
    modulation="bpsk", ldpc_n=LDPC_N, ldpc_rate=LDPC_R,
):
    if not no_channel and (
        ldpc_n <= 0 or not 0 < ldpc_rate < 1 or not 0 < int(ldpc_n * ldpc_rate) < ldpc_n
    ):
        raise ValueError("LDPC码长必须为正整数，码率必须在0和1之间，且满足0 < k < n。")
    cfg = Config()
    setup_seed(42)

    device = torch.device(cfg.DEVICE)
    test_snrs = test_snrs if test_snrs is not None else [0, 3, 6, 9, 12]
    checkpoint_path = checkpoint_path or os.path.join(cfg.CHECKPOINT_DIR, "best_vq_deepsc.pth")
    ldpc_code = None

    print("=" * 40)
    print("开始 SimVQ 支路真实环境测试 (Real Transmission Chain)")
    if no_channel:
        print("链路: no-channel source reconstruction upper bound")
    else:
        from communications.ldpc_coding import get_ldpc_code

        ldpc_code = get_ldpc_code(
            int(ldpc_n * ldpc_rate), rate=ldpc_rate, coded_block_length=ldpc_n
        )
        print(f"LDPC: n={ldpc_code['n']}, k={ldpc_code['k']}, R={ldpc_code['rate']:.8f}")
        if abs(ldpc_code["rate"] - ldpc_rate) > 1e-12:
            print(f"请求码率: {ldpc_rate}；k取整后的实际码率: {ldpc_code['rate']:.8f}")
        print(f"调制: {modulation.upper()}")
    print(f"Loading checkpoint from {checkpoint_path}")

    deepsc_model, inferred = build_model_from_checkpoint(checkpoint_path, cfg, device)
    num_embeddings_list = inferred["num_embeddings_list"]
    if inferred["quantizer_type"] == "none" and not no_channel:
        raise ValueError("No-quantization checkpoints only support --no-channel evaluation.")
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
    perceptual_metrics = PerceptualMetrics(device)

    results = {}
    if no_channel:
        mean_ms_ssim, mean_psnr, compression = evaluate_no_channel(
            deepsc_model, test_dataloader, device,
            num_embeddings_list=num_embeddings_list, return_rate_stats=True,
            perceptual_metrics=perceptual_metrics,
        )
        results["no_channel"] = {
            "ms_ssim": float(mean_ms_ssim), "psnr": float(mean_psnr),
            "compression": compression,
            **perceptual_metrics.compute(),
        }
        print(f"No channel | SimVQ Avg MS-SSIM: {mean_ms_ssim:.4f} | Avg PSNR: {mean_psnr:.4f} dB")
        print(
            f"rFID: {_format_metric(results['no_channel']['rfid'])} "
            f"| LPIPS: {_format_metric(results['no_channel']['lpips'])}"
        )
        _print_compression_stats(compression)
    else:
        for snr in test_snrs:
            print(f"\n正在测试 SNR = {snr} dB ...")
            mean_ms_ssim, mean_psnr, compression = evaluate_ldpc_channel(
                deepsc_model, test_dataloader, num_embeddings_list, snr, ldpc_code, device,
                modulation=modulation, return_rate_stats=True,
                perceptual_metrics=perceptual_metrics,
            )
            results[snr] = {
                "ms_ssim": float(mean_ms_ssim), "psnr": float(mean_psnr),
                "compression": compression,
                **perceptual_metrics.compute(),
            }
            print(f"SNR {snr} dB | SimVQ Avg MS-SSIM: {mean_ms_ssim:.4f} | Avg PSNR: {mean_psnr:.4f} dB")
            print(
                f"rFID: {_format_metric(results[snr]['rfid'])} "
                f"| LPIPS: {_format_metric(results[snr]['lpips'])}"
            )
            _print_compression_stats(compression)

    print("\n" + "=" * 40)
    print("=== SimVQ 最终测试结果 ===")
    print(f"Codebook K List: {num_embeddings_list}")
    print("=" * 40)
    print(
        f"{'Condition':<10} | {'MS-SSIM':<10} | {'PSNR (dB)':<10} "
        f"| {'rFID':<10} | {'LPIPS':<10} | {'CBR':<12}"
    )
    print("-" * 80)
    for condition, metrics in results.items():
        cbr = _format_cbr(metrics["compression"])
        print(
            f"{str(condition):<10} | {metrics['ms_ssim']:<10.4f} | {metrics['psnr']:<10.4f} "
            f"| {_format_metric(metrics['rfid']):<10} "
            f"| {_format_metric(metrics['lpips']):<10} | {cbr:<12}"
        )

    if json_output:
        os.makedirs(os.path.dirname(json_output) or ".", exist_ok=True)
        payload = {
            "checkpoint": checkpoint_path,
            "num_embeddings_list": num_embeddings_list,
            "requested_ldpc_rate": ldpc_rate if ldpc_code else None,
            "ldpc_rate": ldpc_code["rate"] if ldpc_code else None,
            "ldpc_n": ldpc_code["n"] if ldpc_code else None,
            "ldpc_k": ldpc_code["k"] if ldpc_code else None,
            "modulation": modulation if ldpc_code else None,
            "perceptual_metric_config": {
                "rfid": {"backend": "pytorch-fid", "feature_dims": 2048},
                "lpips": {"net": "alex", "version": "0.1"},
            },
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
    parser.add_argument("--modulation", choices=["bpsk", "qpsk", "16qam"], default="bpsk")
    parser.add_argument(
        "--ldpc-n", "--ldpc_n", dest="ldpc_n", type=int, default=LDPC_N,
        help="LDPC coded block length n in bits (default: 256).",
    )
    parser.add_argument(
        "--ldpc-rate", "--ldpc_rate", "--ldpc_k", dest="ldpc_rate", type=float, default=LDPC_R,
        help="LDPC coding rate k/n, e.g. 0.5 or 0.75; k=floor(n*rate).",
    )
    args = parser.parse_args()
    test_real(
        args.checkpoint, args.snrs, args.json_output, args.no_channel, args.modulation,
        args.ldpc_n, args.ldpc_rate,
    )
