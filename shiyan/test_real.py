import argparse
import json
import os
import torch
from config import Config
from data.datasets import get_dataloader
from evaluation.quality import evaluate_ldpc_channel, evaluate_no_channel
from utils.checkpoint_utils import build_model_from_checkpoint
from utils.raq_rvq import resolve_rvq_stage_k_lists
from utils.reproducibility import setup_seed


# LDPC_N = 256
# LDPC_R = 0.5


def _print_raq_rvq_diagnostics(diagnostics):
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
    modulation="bpsk", LDPC_N = 256, LDPC_R = 0.5):
    cfg = Config()
    cfg.validate()
    setup_seed(42)

    device = torch.device(cfg.DEVICE)
    test_snrs = test_snrs
    checkpoint_path = checkpoint_path or os.path.join(cfg.CHECKPOINT_DIR, "best_vq_deepsc.pth")

    print("=" * 40)
    test_raq_rvq_enabled = bool(getattr(cfg, "TEST_USE_RAQ_RVQ", False))
    branch_name = (
        "测试期两级RAQ-RVQ支路"
        if test_raq_rvq_enabled
        else ("RAQ动态目标码本支路" if getattr(cfg, "USE_RAQ", False) else "SimVQ支路")
    )
    print(f"开始 {branch_name} 真实环境测试 (Real Transmission Chain)")
    if no_channel:
        print("链路: no-channel source reconstruction upper bound")
    else:
        print(f"LDPC: n={LDPC_N}, k={int(LDPC_N * LDPC_R)}, R={LDPC_R}")
        print(f"调制: {modulation.upper()}")
    print(f"Loading checkpoint from {checkpoint_path}")

    deepsc_model, inferred = build_model_from_checkpoint(checkpoint_path, cfg, device)
    num_embeddings_list = list(cfg.RAQ_TARGET_LIST) if getattr(cfg, "USE_RAQ", False) else inferred["num_embeddings_list"]
    if inferred["quantizer_type"] == "none" and not no_channel:
        raise ValueError("No-quantization checkpoints only support --no-channel evaluation.")
    if getattr(cfg, "USE_RAQ", False):
        print(f"源码本大小: {inferred['num_embeddings_list']} (inferred from checkpoint)")
        print(f"RAQ目标码本大小: {num_embeddings_list} (from Config.RAQ_TARGET_LIST)")
    else:
        print(f"码本大小: {num_embeddings_list} (inferred from checkpoint)")
    if not no_channel:
        print(f"测试 SNR: {test_snrs} dB")
    rvq_k_lists = None
    if test_raq_rvq_enabled:
        rvq_depth = int(getattr(cfg, "TEST_RAQ_RVQ_DEPTH", 2))
        rvq_k_lists = resolve_rvq_stage_k_lists(
            num_embeddings_list,
            rvq_depth=rvq_depth,
            stage_k_lists=getattr(cfg, "TEST_RAQ_RVQ_K_LISTS", None),
            min_k=getattr(cfg, "RAQ_MIN_TRG", None),
            max_k=getattr(cfg, "RAQ_MAX_TRG", None),
        )
        print(f"[Test RAQ-RVQ] enabled=True, depth={rvq_depth}")
        for scale_index, (k_total, stage_k_list) in enumerate(
            zip(num_embeddings_list, rvq_k_lists)
        ):
            print(
                f"[Test RAQ-RVQ] scale {scale_index}: "
                f"K_total={k_total} -> stage_K={stage_k_list}"
            )
        if getattr(cfg, "USE_DYNAMIC_RAQ_RVQ", False):
            print(
                "[Test RAQ-RVQ] trained dynamic residual branch: stage 2 uses "
                "its independent allocation-conditioned RAQ generator."
            )
        else:
            print(
                "[Test RAQ-RVQ] zero-shot limitation: stage 2 reuses the trained "
                "single-stage RAQ generator and was not trained on residual features."
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
        if test_raq_rvq_enabled:
            mean_ms_ssim, mean_psnr, diagnostics = evaluate_no_channel(
                deepsc_model, test_dataloader, device, return_diagnostics=True
            )
            _print_raq_rvq_diagnostics(diagnostics)
        else:
            mean_ms_ssim, mean_psnr = evaluate_no_channel(
                deepsc_model, test_dataloader, device
            )
            diagnostics = None
        results["no_channel"] = {"ms_ssim": mean_ms_ssim, "psnr": mean_psnr}
        if diagnostics is not None:
            results["no_channel"]["diagnostics"] = diagnostics
        print(f"No channel | {branch_name} Avg MS-SSIM: {mean_ms_ssim:.4f} | Avg PSNR: {mean_psnr:.4f} dB")
    else:
        from communications.ldpc_coding import get_ldpc_code

        ldpc_code = get_ldpc_code(int(LDPC_N * LDPC_R), rate=LDPC_R)
        for snr in test_snrs:
            print(f"\n正在测试 SNR = {snr} dB ...")
            if test_raq_rvq_enabled:
                mean_ms_ssim, mean_psnr, diagnostics = evaluate_ldpc_channel(
                    deepsc_model,
                    test_dataloader,
                    num_embeddings_list,
                    snr,
                    ldpc_code,
                    device,
                    modulation=modulation,
                    return_diagnostics=True,
                )
                _print_raq_rvq_diagnostics(diagnostics)
            else:
                mean_ms_ssim, mean_psnr = evaluate_ldpc_channel(
                    deepsc_model, test_dataloader, num_embeddings_list, snr, ldpc_code, device,
                    modulation=modulation)
                diagnostics = None
            results[snr] = {"ms_ssim": mean_ms_ssim, "psnr": mean_psnr}
            if diagnostics is not None:
                results[snr]["diagnostics"] = diagnostics
            print(f"SNR {snr} dB | {branch_name} Avg MS-SSIM: {mean_ms_ssim:.4f} | Avg PSNR: {mean_psnr:.4f} dB")

    print("\n" + "=" * 40)
    print(f"=== {branch_name} 最终测试结果 ===")
    print(f"Transmission K List: {num_embeddings_list}")
    if rvq_k_lists is not None:
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
            "results": {str(condition): metrics for condition, metrics in results.items()},
        }
        if rvq_k_lists is not None:
            payload.update({
                "test_raq_rvq_enabled": True,
                "rvq_depth": int(getattr(cfg, "TEST_RAQ_RVQ_DEPTH", 2)),
                "rvq_k_lists": rvq_k_lists,
                "rvq_training_mode": (
                    "trained_dynamic_residual"
                    if getattr(cfg, "USE_DYNAMIC_RAQ_RVQ", False)
                    else "test_time_zero_shot"
                ),
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
    parser.add_argument("--modulation", choices=["bpsk", "qpsk","16qam"], default="bpsk")
    parser.add_argument("--ldpc_n", type=int, default=256)
    parser.add_argument("--ldpc_k", type=float, default=0.5)
    args = parser.parse_args()
    test_real(args.checkpoint, args.snrs, args.json_output, args.no_channel, args.modulation, args.ldpc_n, args.ldpc_k)
