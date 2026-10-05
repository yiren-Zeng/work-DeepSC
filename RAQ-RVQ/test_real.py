import argparse
from fractions import Fraction
import json
import math
import os
import numpy as np
import torch
from config import Config
from data.datasets import get_dataloader
from evaluation.quality import evaluate_ldpc_channel, evaluate_no_channel
from utils.checkpoint_utils import build_model_from_checkpoint
from utils.reproducibility import setup_seed


# LDPC_N = 256
# LDPC_R = 0.5


def _format_cbr(channel_symbols, source_values):
    ratio = Fraction(int(channel_symbols), int(source_values))
    return f"{float(ratio):.4f} ({ratio.numerator}/{ratio.denominator})"


def ms_ssim_to_db(ms_ssim):
    """Convert MS-SSIM to the commonly reported -10*log10(1-MS-SSIM)."""
    value = float(ms_ssim)
    if math.isnan(value):
        return float("nan")
    if value >= 1.0:
        return float("inf")
    return -10.0 * math.log10(1.0 - value)


def _format_ms_ssim(ms_ssim):
    return f"{float(ms_ssim):.4f} ({ms_ssim_to_db(ms_ssim):.4f} dB)"


def _channel_label(channel_type, rician_k_factor=10.0):
    if channel_type == "rician":
        return f"Rician(K={rician_k_factor:g},perfect-CSI)"
    return "AWGN"


def _normalize_channel_types(channel_types):
    if isinstance(channel_types, str):
        channel_types = [channel_types]
    normalized = []
    for value in channel_types or ["awgn"]:
        channel_type = str(value).strip().lower()
        if channel_type not in {"awgn", "rician"}:
            raise ValueError(
                f"Unsupported channel type {channel_type!r}; "
                "expected 'awgn' or 'rician'."
            )
        if channel_type not in normalized:
            normalized.append(channel_type)
    return normalized


def _normalize_channel_seeds(channel_seeds):
    seeds = [int(seed) for seed in (channel_seeds or [42])]
    if not seeds:
        raise ValueError("At least one channel seed is required")
    if any(seed < 0 for seed in seeds):
        raise ValueError("Channel seeds must be non-negative")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Channel seeds must be unique for Monte Carlo trials")
    return seeds


def _summarize_monte_carlo_trials(trials):
    """Average raw metrics first and aggregate BER over all transmissions."""
    if not trials:
        raise ValueError("At least one Monte Carlo trial is required")
    psnr_values = np.asarray([trial["psnr"] for trial in trials], dtype=float)
    ms_ssim_values = np.asarray(
        [trial["ms_ssim"] for trial in trials], dtype=float
    )
    streams = [
        trial["diagnostics"].get("combined_stream")
        or trial["diagnostics"].get("total", {})
        for trial in trials
    ]
    total_bit_errors = sum(int(stream["bit_errors"]) for stream in streams)
    total_payload_bits = sum(int(stream["payload_bits"]) for stream in streams)
    total_channel_symbols = sum(
        int(stream["channel_symbols"]) for stream in streams
    )
    total_source_values = sum(int(stream["source_values"]) for stream in streams)
    mean_ms_ssim = float(np.mean(ms_ssim_values))
    ddof = 1 if len(trials) > 1 else 0
    return {
        "num_trials": len(trials),
        "channel_seeds": [int(trial["channel_seed"]) for trial in trials],
        "psnr_mean": float(np.mean(psnr_values)),
        "psnr_std": float(np.std(psnr_values, ddof=ddof)),
        "ms_ssim_mean": mean_ms_ssim,
        "ms_ssim_std": float(np.std(ms_ssim_values, ddof=ddof)),
        "ms_ssim_db_from_mean": ms_ssim_to_db(mean_ms_ssim),
        "total_bit_errors": total_bit_errors,
        "total_payload_bits": total_payload_bits,
        "ber": total_bit_errors / total_payload_bits,
        "total_channel_symbols": total_channel_symbols,
        "total_source_values": total_source_values,
        "transmission_ratio": total_channel_symbols / total_source_values,
    }


def _print_concise_channel_result(
    channel_type, snr, mean_psnr, mean_ms_ssim, diagnostics,
    rician_k_factor=10.0,
):
    stream = diagnostics.get("combined_stream") or diagnostics.get("total", {})
    cbr = _format_cbr(stream["channel_symbols"], stream["source_values"])
    print(
        f"{_channel_label(channel_type, rician_k_factor)} | "
        f"SNR={snr:g} dB | PSNR={mean_psnr:.4f} | "
        f"MS-SSIM={_format_ms_ssim(mean_ms_ssim)} | BER={stream['ber']:.4f} | "
        f"payload={stream['payload_bits']} bit | coded={stream['coded_bits']} bit | "
        f"symbols={stream['channel_symbols']} | CBR={cbr}"
    )


def _print_concise_monte_carlo_result(
    channel_type, snr, summary, rician_k_factor=10.0
):
    cbr = _format_cbr(
        summary["total_channel_symbols"], summary["total_source_values"]
    )
    print(
        f"{_channel_label(channel_type, rician_k_factor)} | "
        f"SNR={snr:g} dB | MC={summary['num_trials']} | "
        f"seeds={summary['channel_seeds']} | "
        f"PSNR_mean={summary['psnr_mean']:.4f} | "
        f"PSNR_std={summary['psnr_std']:.4f} | "
        f"MS-SSIM_mean_raw={summary['ms_ssim_mean']:.6f} | "
        f"MS-SSIM={summary['ms_ssim_db_from_mean']:.4f} dB | "
        f"BER={summary['ber']:.8f} | CBR={cbr}"
    )


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
    stream_packing="per_stage", concise=False, channel_types=("awgn",),
    rician_k_factor=10.0, channel_seeds=(42,),
):
    cfg = Config()
    cfg.validate()
    setup_seed(42)
    channel_types = _normalize_channel_types(channel_types)
    channel_seeds = _normalize_channel_seeds(channel_seeds)
    if rician_k_factor < 0:
        raise ValueError("Rician K-factor must be non-negative")

    device = torch.device(cfg.DEVICE)
    test_snrs = test_snrs
    checkpoint_path = checkpoint_path or os.path.join(cfg.CHECKPOINT_DIR, "best_vq_deepsc.pth")

    branch_name = "训练式独立码本RAQ-RVQ支路"
    if not concise:
        print("=" * 40)
        print(f"开始 {branch_name} 真实环境测试 (Real Transmission Chain)")
        if no_channel:
            print("链路: no-channel reconstruction upper bound")
        else:
            print(f"LDPC: n={LDPC_N}, k={int(LDPC_N * LDPC_R)}, R={LDPC_R}")
            print(f"调制: {modulation.upper()}")
            print(f"Payload 打包: {stream_packing}")
            print(
                "信道: "
                + ", ".join(
                    _channel_label(channel_type, rician_k_factor)
                    for channel_type in channel_types
                )
            )
        print(f"Loading checkpoint from {checkpoint_path}")

    deepsc_model, inferred = build_model_from_checkpoint(checkpoint_path, cfg, device)
    num_embeddings_list = list(inferred["num_embeddings_list"])
    rvq_depth = cfg.INDEPENDENT_RAQ_RVQ_DEPTH
    rvq_k_lists = [
        list(stage_sizes)
        for stage_sizes in cfg.INDEPENDENT_RAQ_RVQ_K_LISTS
    ]
    if concise:
        print(
            f"Dense combined | source K={num_embeddings_list} | "
            f"RVQ K={rvq_k_lists} | LDPC={LDPC_N}x{LDPC_R:g} | "
            f"{modulation.upper()} | channels="
            f"{','.join(_channel_label(c, rician_k_factor) for c in channel_types)} | "
            f"SNRs={test_snrs} dB | channel_seeds={channel_seeds}"
        )
    else:
        print(f"源码本大小: {num_embeddings_list} (inferred from checkpoint)")
        if not no_channel:
            print(f"测试 SNR: {test_snrs} dB")
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
        results["no_channel"] = {"ms_ssim": mean_ms_ssim, "psnr": mean_psnr}
        results["no_channel"]["ms_ssim_db"] = ms_ssim_to_db(mean_ms_ssim)
        results["no_channel"]["diagnostics"] = diagnostics
        if concise:
            print(
                f"No channel | PSNR={mean_psnr:.4f} | "
                f"MS-SSIM={_format_ms_ssim(mean_ms_ssim)} | CBR=N/A"
            )
        else:
            _print_raq_rvq_diagnostics(diagnostics)
            print(f"No channel | {branch_name} Avg MS-SSIM: {_format_ms_ssim(mean_ms_ssim)} | Avg PSNR: {mean_psnr:.4f} dB")
    else:
        from communications.ldpc_coding import get_ldpc_code

        ldpc_code = get_ldpc_code(int(LDPC_N * LDPC_R), rate=LDPC_R)
        channel_results = {}
        for channel_type in channel_types:
            per_snr_results = {}
            if not concise:
                print(
                    f"\n=== {_channel_label(channel_type, rician_k_factor)} ==="
                )
            for snr in test_snrs:
                if not concise:
                    print(f"\n正在测试 SNR = {snr} dB ...")
                trials = []
                for channel_seed in channel_seeds:
                    mean_ms_ssim, mean_psnr, diagnostics = evaluate_ldpc_channel(
                        deepsc_model,
                        test_dataloader,
                        snr,
                        ldpc_code,
                        device,
                        modulation=modulation,
                        return_diagnostics=True,
                        stream_packing=stream_packing,
                        channel_type=channel_type,
                        rician_k_factor=rician_k_factor,
                        channel_seed=channel_seed,
                    )
                    trial = {
                        "channel_seed": channel_seed,
                        "ms_ssim": float(mean_ms_ssim),
                        "ms_ssim_db": ms_ssim_to_db(mean_ms_ssim),
                        "psnr": float(mean_psnr),
                        "diagnostics": diagnostics,
                    }
                    trials.append(trial)
                    if concise and len(channel_seeds) > 1:
                        stream = diagnostics.get("combined_stream") or diagnostics["total"]
                        print(
                            f"  seed={channel_seed} | PSNR={mean_psnr:.4f} | "
                            f"MS-SSIM_raw={mean_ms_ssim:.6f} | "
                            f"BER={stream['ber']:.8f}",
                            flush=True,
                        )

                summary = _summarize_monte_carlo_trials(trials)
                per_snr_results[snr] = {
                    "ms_ssim": summary["ms_ssim_mean"],
                    "ms_ssim_db": summary["ms_ssim_db_from_mean"],
                    "ms_ssim_std": summary["ms_ssim_std"],
                    "psnr": summary["psnr_mean"],
                    "psnr_std": summary["psnr_std"],
                    "ber": summary["ber"],
                    "monte_carlo": summary,
                    "trials": trials,
                    "diagnostics": trials[0]["diagnostics"],
                }
                if concise:
                    if len(channel_seeds) == 1:
                        _print_concise_channel_result(
                            channel_type,
                            snr,
                            summary["psnr_mean"],
                            summary["ms_ssim_mean"],
                            trials[0]["diagnostics"],
                            rician_k_factor,
                        )
                    else:
                        _print_concise_monte_carlo_result(
                            channel_type, snr, summary, rician_k_factor
                        )
                else:
                    _print_raq_rvq_diagnostics(trials[0]["diagnostics"])
                    print(
                        f"{_channel_label(channel_type, rician_k_factor)} | "
                        f"SNR {snr} dB | {branch_name} Avg MS-SSIM: "
                        f"{_format_ms_ssim(summary['ms_ssim_mean'])} | "
                        f"Avg PSNR: {summary['psnr_mean']:.4f} dB | "
                        f"MC trials: {summary['num_trials']}"
                    )
            channel_results[channel_type] = per_snr_results
        results = (
            channel_results[channel_types[0]]
            if len(channel_types) == 1
            else channel_results
        )

    if not concise:
        print("\n" + "=" * 40)
        print(f"=== {branch_name} 最终测试结果 ===")
        print(f"Source K List: {num_embeddings_list}")
        print(f"RVQ Stage K Lists: {rvq_k_lists}")
        print("=" * 40)
        print(f"{'Condition':<24} | {'MS-SSIM':<24} | {'PSNR (dB)':<10}")
        print("-" * 25 + "|" + "-" * 26 + "|" + "-" * 11)
        if no_channel or len(channel_types) == 1:
            rows = [(str(condition), metrics) for condition, metrics in results.items()]
        else:
            rows = [
                (
                    f"{_channel_label(channel_type, rician_k_factor)} {snr} dB",
                    metrics,
                )
                for channel_type, per_snr in results.items()
                for snr, metrics in per_snr.items()
            ]
        for condition, metrics in rows:
            print(
                f"{condition:<24} | {_format_ms_ssim(metrics['ms_ssim']):<24} | "
                f"{metrics['psnr']:<10.4f}"
            )

    if json_output:
        os.makedirs(os.path.dirname(json_output) or ".", exist_ok=True)
        payload = {
            "checkpoint": checkpoint_path,
            "num_embeddings_list": num_embeddings_list,
            "ldpc_rate": LDPC_R,
            "modulation": modulation,
            "stream_packing": stream_packing,
            "channel_types": channel_types,
            "rician_k_factor": float(rician_k_factor),
            "rician_receiver_csi": (
                "perfect_per_symbol" if "rician" in channel_types else None
            ),
            "channel_seeds": channel_seeds,
            "results": results,
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
    parser.add_argument(
        "--snrs", type=float, nargs="+", default=[0, 3, 6, 9, 12]
    )
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
    parser.add_argument(
        "--concise",
        action="store_true",
        help="Print one compact result line per channel condition.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        choices=["awgn", "rician"],
        default=["awgn"],
        help="Channel models to evaluate in order.",
    )
    parser.add_argument(
        "--rician-k-factor",
        type=float,
        default=10.0,
        help="Rician K-factor in linear scale (default: 10).",
    )
    parser.add_argument(
        "--channel-seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Independent channel seeds used for Monte Carlo trials.",
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
        args.concise,
        args.channels,
        args.rician_k_factor,
        args.channel_seeds,
    )
