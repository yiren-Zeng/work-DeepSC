import argparse
import json
import os
import torch
from config import Config
from data.datasets import get_dataloader
from evaluation.quality import evaluate_ldpc_channel, evaluate_no_channel
from utils.checkpoint_utils import build_model_from_checkpoint
from utils.reproducibility import setup_seed


# LDPC_N = 256
# LDPC_R = 0.5


def _print_raq_rvq_diagnostics(diagnostics):
    stream_packing = diagnostics.get("stream_packing")
    if stream_packing is not None:
        print(f"[Test RAQ-RVQ] payload stream packing: {stream_packing}")
    quantization = diagnostics.get("rvq_quantization", {})
    for scale in quantization.get("per_scale", []):
        print(
            f"[Test RAQ-RVQ] scale {scale['scale']} energy: "
            f"input={scale['mean_input_mse_energy']:.8f}, "
            f"residuals={scale['mean_residual_mse_energies']}"
        )
        for stage in scale.get("stage_diagnostics", []):
            print(
                f"[Test RAQ-RVQ] scale {scale['scale']} stage {stage['stage']}: "
                f"K={stage['num_embeddings']}, bits/index={stage['bits_per_index']}, "
                f"codebook_size={stage['codebook_size']}, "
                f"index_range=[{stage['sent_index_min']},{stage['sent_index_max']}], "
                f"payload_bits={stage['payload_bits']}, "
                f"residual_energy={stage['mean_residual_mse_energy']:.8f}"
            )
        print(
            f"[Test RAQ-RVQ] scale {scale['scale']} bit budget: "
            f"payload={scale['payload_bits']}, baseline={scale['baseline_payload_bits']}, "
            f"match={scale['bit_budget_matches']}"
        )

    for stage in diagnostics.get("per_stage", []):
        if stream_packing == "combined":
            message = (
                f"[Test RAQ-RVQ] TX segment scale {stage['scale']} "
                f"stage {stage['stage']}: K={stage['num_embeddings']}, "
                f"payload={stage['payload_bits']} bit, "
                f"payload_bpp={stage['payload_bpp']:.8f}"
            )
        else:
            message = (
                f"[Test RAQ-RVQ] TX scale {stage['scale']} stage {stage['stage']}: "
                f"K={stage['num_embeddings']}, payload={stage['payload_bits']} bit, "
                f"LDPC-padding={stage['ldpc_padding_bits']} bit, "
                f"coded={stage['coded_bits']} bit, transmitted={stage['transmitted_bits']} bit, "
                f"symbols={stage['channel_symbols']}, payload_bpp={stage['payload_bpp']:.8f}, "
                f"transmitted_bpp={stage['transmitted_bpp']:.8f}, "
                f"transmission_ratio={stage['transmission_ratio']:.8f}"
            )
        if stage.get("ber") is not None:
            message += (
                f", BER={stage['ber']:.8f}, "
                f"index_error_rate={stage['index_error_rate']:.8f}"
            )
        print(message)

    combined = diagnostics.get("combined_stream")
    if combined:
        print(
            f"[Test RAQ-RVQ] TX combined stream: "
            f"payload={combined['payload_bits']} bit, "
            f"LDPC-padding={combined['ldpc_padding_bits']} bit, "
            f"coded={combined['coded_bits']} bit, "
            f"transmitted={combined['transmitted_bits']} bit, "
            f"symbols={combined['channel_symbols']}, "
            f"transmission_ratio={combined['transmission_ratio']:.8f}, "
            f"BER={combined['ber']:.8f}"
        )

    total = diagnostics.get("total", {})
    if total:
        message = (
            f"[Test RAQ-RVQ] TX total: payload={total['payload_bits']} bit, "
            f"coded={total['coded_bits']} bit, transmitted={total['transmitted_bits']} bit, "
            f"payload_bpp={total['payload_bpp']:.8f}, "
            f"transmitted_bpp={total['transmitted_bpp']:.8f}, "
            f"transmission_ratio={total['transmission_ratio']:.8f}"
        )
        if "single_stream_coded_bits" in total:
            message += (
                f", baseline_coded={total['single_stream_coded_bits']} bit, "
                f"payload_match={total['payload_bits_match_single_stage_budget']}, "
                f"coded_match={total['coded_bits_match_single_stream']}, "
                f"transmitted_match={total['transmitted_bits_match_single_stream']}"
            )
        print(message)
    if quantization:
        print(
            "[Test RAQ-RVQ] all source bit budgets match baseline: "
            f"{quantization.get('all_bit_budgets_match', False)}"
        )


@torch.no_grad()
def test_real(
    checkpoint_path=None, test_snrs=None, json_output=None, no_channel=False,
    modulation="bpsk", LDPC_N=256, LDPC_R=0.5,
    stream_packing="per_stage",
):
    cfg = Config()
    cfg.validate()
    setup_seed(42)

    device = torch.device(cfg.DEVICE)
    test_snrs = test_snrs
    checkpoint_path = checkpoint_path or os.path.join(cfg.CHECKPOINT_DIR, "best_vq_deepsc.pth")

    print("=" * 40)
    branch_name = "训练式独立码本RAQ-RVQ支路"
    print(f"开始 {branch_name} 真实环境测试 (Real Transmission Chain)")
    if no_channel:
        print("链路: no-channel reconstruction upper bound")
    else:
        print(f"LDPC: n={LDPC_N}, k={int(LDPC_N * LDPC_R)}, R={LDPC_R}")
        print(f"调制: {modulation.upper()}")
        print(f"Payload 打包: {stream_packing}")
    print(f"Loading checkpoint from {checkpoint_path}")

    deepsc_model, inferred = build_model_from_checkpoint(checkpoint_path, cfg, device)
    num_embeddings_list = list(inferred["num_embeddings_list"])
    print(f"源码本大小: {num_embeddings_list} (inferred from checkpoint)")
    if not no_channel:
        print(f"测试 SNR: {test_snrs} dB")
    rvq_depth = cfg.INDEPENDENT_RAQ_RVQ_DEPTH
    rvq_k_lists = [
        list(stage_sizes)
        for stage_sizes in cfg.INDEPENDENT_RAQ_RVQ_K_LISTS
    ]
    print(f"[Test RAQ-RVQ] enabled=True, depth={rvq_depth}")
    for scale_index, stage_k_list in enumerate(rvq_k_lists):
        print(
            f"[Test RAQ-RVQ] scale {scale_index}: "
            f"independent_stage_K={stage_k_list}"
        )
    print(
        "[Test RAQ-RVQ] trained independent branch: every scale/stage "
        "pair owns a distinct RAQ generator and codebook; no reserved "
        "zero codeword."
    )
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
        mean_ms_ssim, mean_psnr, diagnostics = evaluate_no_channel(
            deepsc_model, test_dataloader, device, return_diagnostics=True
        )
        _print_raq_rvq_diagnostics(diagnostics)
        results["no_channel"] = {"ms_ssim": mean_ms_ssim, "psnr": mean_psnr}
        results["no_channel"]["diagnostics"] = diagnostics
        print(f"No channel | {branch_name} Avg MS-SSIM: {mean_ms_ssim:.4f} | Avg PSNR: {mean_psnr:.4f} dB")
    else:
        from communications.ldpc_coding import get_ldpc_code

        ldpc_code = get_ldpc_code(int(LDPC_N * LDPC_R), rate=LDPC_R)
        for snr in test_snrs:
            print(f"\n正在测试 SNR = {snr} dB ...")
            mean_ms_ssim, mean_psnr, diagnostics = evaluate_ldpc_channel(
                deepsc_model,
                test_dataloader,
                snr,
                ldpc_code,
                device,
                modulation=modulation,
                return_diagnostics=True,
                stream_packing=stream_packing,
            )
            _print_raq_rvq_diagnostics(diagnostics)
            results[snr] = {"ms_ssim": mean_ms_ssim, "psnr": mean_psnr}
            results[snr]["diagnostics"] = diagnostics
            print(f"SNR {snr} dB | {branch_name} Avg MS-SSIM: {mean_ms_ssim:.4f} | Avg PSNR: {mean_psnr:.4f} dB")

    print("\n" + "=" * 40)
    print(f"=== {branch_name} 最终测试结果 ===")
    print(f"Source K List: {num_embeddings_list}")
    print(f"RVQ Stage K Lists: {rvq_k_lists}")
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
            "ldpc_rate": LDPC_R,
            "modulation": modulation,
            "stream_packing": stream_packing,
            "results": {str(condition): metrics for condition, metrics in results.items()},
        }
        payload.update({
            "rvq_depth": int(rvq_depth),
            "rvq_k_lists": rvq_k_lists,
            "rvq_training_mode": "trained_independent_codebook_residual",
            "independent_raq_rvq_enabled": True,
        })
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
    parser.add_argument(
        "--modulation",
        choices=["bpsk", "qpsk", "16qam"],
        default="bpsk",
    )
    parser.add_argument("--ldpc_n", type=int, default=256)
    parser.add_argument("--ldpc_k", type=float, default=0.5)
    parser.add_argument(
        "--stream-packing",
        choices=["per_stage", "combined"],
        default="per_stage",
    )
    args = parser.parse_args()
    test_real(
        args.checkpoint,
        args.snrs,
        args.json_output,
        args.no_channel,
        args.modulation,
        args.ldpc_n,
        args.ldpc_k,
        args.stream_packing,
    )
