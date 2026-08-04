import argparse
import json
import os
import torch
from config import Config, _source_bpp
from data.datasets import get_dataloader
from evaluation.quality import evaluate_ldpc_channel, evaluate_no_channel
from utils.checkpoint_utils import build_model_from_checkpoint
from utils.reproducibility import setup_seed


LDPC_N = 256
LDPC_R = 0.5


@torch.no_grad()
def test_real(
    checkpoint_path=None, test_snrs=None, json_output=None, no_channel=False,
    modulation="bpsk",
):
    cfg = Config()
    setup_seed(42)

    device = torch.device(cfg.DEVICE)
    test_snrs = test_snrs
    checkpoint_path = checkpoint_path or os.path.join(cfg.CHECKPOINT_DIR, "best_vq_deepsc.pth")

    print("=" * 40)
    print("开始量化语义通信真实环境测试 (Real Transmission Chain)")
    if no_channel:
        print("链路: no-channel source reconstruction upper bound")
    else:
        print(f"LDPC: n={LDPC_N}, k={int(LDPC_N * LDPC_R)}, R={LDPC_R}")
        print(f"调制: {modulation.upper()}")
    print(f"Loading checkpoint from {checkpoint_path}")

    deepsc_model, inferred = build_model_from_checkpoint(checkpoint_path, cfg, device)
    num_embeddings_list = inferred["num_embeddings_list"]
    quantizer_type = inferred["quantizer_type"]
    rq_depth_list = inferred.get("rq_depth_list", [1] * len(num_embeddings_list))
    rq_codebook_size_lists = inferred.get("rq_codebook_size_lists")
    transport_codebook_sizes = (
        rq_codebook_size_lists
        if quantizer_type == "stagewise_residual_simvq"
        else num_embeddings_list
    )
    if inferred["quantizer_type"] == "none" and not no_channel:
        raise ValueError("No-quantization checkpoints only support --no-channel evaluation.")
    print(f"码本大小: {num_embeddings_list} (inferred from checkpoint)")
    print(f"量化器: {quantizer_type}")
    if quantizer_type == "rq_ema":
        print(
            f"RQ深度: {rq_depth_list}, EMA decay={inferred.get('rq_ema_decay')}, "
            f"shared={inferred.get('rq_shared_codebook')}, "
            f"restart={inferred.get('rq_restart_unused_codes')}"
        )
    elif quantizer_type == "residual_simvq":
        print(
            f"Residual-SimVQ深度: {rq_depth_list}, "
            f"shared={inferred.get('rq_shared_codebook')}"
        )
    elif quantizer_type == "stagewise_residual_simvq":
        print(
            "Stagewise Residual-SimVQ码本: "
            f"{rq_codebook_size_lists}, "
            f"shared={inferred.get('rq_shared_codebook')}"
        )

    checkpoint_metadata = inferred.get("checkpoint_metadata", {})
    rate_strides = checkpoint_metadata.get(
        "downsample_strides", cfg.DOWNSAMPLE_STRIDES
    )
    rate_quantizer_axes = checkpoint_metadata.get(
        "quantizer_axis_list", cfg.QUANTIZER_AXIS_LIST
    )
    source_bpp = _source_bpp(
        rate_strides,
        num_embeddings_list,
        rate_quantizer_axes,
        inferred["embedding_dim_list"],
        (256, 256),
        rq_depth_list
        if quantizer_type in {
            "rq_ema", "residual_simvq", "stagewise_residual_simvq"
        }
        else None,
        rq_codebook_size_lists
        if quantizer_type == "stagewise_residual_simvq"
        else None,
    )
    bits_per_256_image = source_bpp * 256 * 256
    modulation_bits = {"bpsk": 1, "qpsk": 2, "16qam": 4}[modulation]
    transmission_ratio = source_bpp / (LDPC_R * modulation_bits * 3)
    print(
        f"Source rate @256x256: {bits_per_256_image:.0f} bits/image, "
        f"{source_bpp:.8f} bpp"
    )
    print(
        f"Transmission ratio (LDPC1/2+{modulation.upper()}+RGB): "
        f"{transmission_ratio:.8f}"
    )
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
        print(f"No channel | {quantizer_type} Avg MS-SSIM: {mean_ms_ssim:.4f} | Avg PSNR: {mean_psnr:.4f} dB")
    else:
        from communications.ldpc_coding import get_ldpc_code

        ldpc_code = get_ldpc_code(int(LDPC_N * LDPC_R), rate=LDPC_R)
        for snr in test_snrs:
            print(f"\n正在测试 SNR = {snr} dB ...")
            mean_ms_ssim, mean_psnr = evaluate_ldpc_channel(
                deepsc_model,
                test_dataloader,
                transport_codebook_sizes,
                snr,
                ldpc_code,
                device,
                modulation=modulation,
            )
            results[snr] = {"ms_ssim": mean_ms_ssim, "psnr": mean_psnr}
            print(f"SNR {snr} dB | {quantizer_type} Avg MS-SSIM: {mean_ms_ssim:.4f} | Avg PSNR: {mean_psnr:.4f} dB")

    print("\n" + "=" * 40)
    print(f"=== {quantizer_type} 最终测试结果 ===")
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
            "quantizer_type": quantizer_type,
            "num_embeddings_list": num_embeddings_list,
            "embedding_dim_list": inferred["embedding_dim_list"],
            "rq_depth_list": rq_depth_list,
            "rq_codebook_size_lists": rq_codebook_size_lists,
            "rq_ema_decay": inferred.get("rq_ema_decay"),
            "rq_restart_unused_codes": inferred.get("rq_restart_unused_codes"),
            "rq_shared_codebook": inferred.get("rq_shared_codebook"),
            "source_bits_per_256_image": bits_per_256_image,
            "source_bpp": source_bpp,
            "transmission_ratio": transmission_ratio,
            "ldpc_rate": LDPC_R,
            "modulation": modulation,
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
    args = parser.parse_args()
    test_real(args.checkpoint, args.snrs, args.json_output, args.no_channel, args.modulation)
