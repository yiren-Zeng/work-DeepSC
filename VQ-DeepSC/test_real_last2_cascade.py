import argparse
import json
import math
import os
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from io import StringIO

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import torch
import numpy as np

from config_last2_test import Last2CascadeTestConfig
from data.datasets import get_dataloader
from evaluation.quality_last2_cascade import (
    ACTIVE_SCALES,
    INACTIVE_SCALES,
    evaluate_ldpc_channel_last2_cascade,
    evaluate_no_channel_last2_cascade,
)
from models.deepsc_last2_cascade import DeepSCLast2Cascade
from test_real import setup_seed


def build_model(cfg, device, checkpoint_full_codebook_label):
    return DeepSCLast2Cascade(
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        num_downsample_blocks=cfg.NUM_DOWNSAMPLE_BLOCKS,
        base_channels=cfg.BASE_CHANNELS,
        full_num_embeddings_list=checkpoint_full_codebook_label,
        full_embedding_dim_list=cfg.EMBEDDING_DIM_LIST,
        commitment_cost=cfg.COMMITMENT_COST,
    ).to(device)


def load_model(checkpoint_path, cfg, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(
            "Last-two cascade evaluation requires a new-format checkpoint; "
            "old four-codebook weights are not compatible"
        )
    expected_metadata = {
        "model_type": DeepSCLast2Cascade.MODEL_TYPE,
        "active_scales": list(DeepSCLast2Cascade.ACTIVE_SCALES),
        "active_num_embeddings_list": list(cfg.NUM_EMBEDDINGS_LIST),
        "active_embedding_dim_list": list(cfg.EMBEDDING_DIM_LIST[-2:]),
    }
    for key, expected_value in expected_metadata.items():
        actual_value = payload.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Checkpoint {key}={actual_value!r} does not match "
                f"requested value {expected_value!r}"
            )

    codebook_label = payload.get("num_embeddings_list")
    if codebook_label is None:
        codebook_label = payload.get("full_num_embeddings_list")
    if (
        not isinstance(codebook_label, (list, tuple))
        or len(codebook_label) not in (2, 4)
        or list(codebook_label[-2:]) != list(cfg.NUM_EMBEDDINGS_LIST)
    ):
        raise ValueError(
            "Checkpoint codebook list is inconsistent with its two active codebooks"
        )

    state_dict = payload["model_state_dict"]
    inferred_sizes = [
        int(state_dict[f"vector_quantizers.{index}.embedding.weight"].shape[0])
        for index in range(2)
    ]
    if inferred_sizes != list(cfg.NUM_EMBEDDINGS_LIST):
        raise ValueError(
            f"Checkpoint active codebooks {inferred_sizes} do not match "
            f"configured codebooks {cfg.NUM_EMBEDDINGS_LIST}"
        )
    model = build_model(cfg, device, codebook_label)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _compression_rate_fraction(diagnostics):
    combined = diagnostics.get("combined_stream")
    if combined is not None:
        # Project convention: channel symbols divided by the number of RGB values.
        return str(
            Fraction(
                int(combined["channel_symbols"]),
                int(combined["source_values"]),
            )
        )
    compression = diagnostics["compression"]
    return str(
        Fraction(
            int(compression["payload"]["bits"]),
            int(compression["source_bits"]),
        )
    )


def ms_ssim_to_db(ms_ssim):
    """Convert MS-SSIM to -10*log10(1-MS-SSIM)."""
    value = float(ms_ssim)
    if math.isnan(value):
        return float("nan")
    if value >= 1.0:
        return float("inf")
    return -10.0 * math.log10(1.0 - value)


def _format_ms_ssim(ms_ssim):
    return f"{float(ms_ssim):.4f} ({ms_ssim_to_db(ms_ssim):.4f} dB)"


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


def _channel_label(channel_type, rician_k_factor):
    if channel_type == "rician":
        return f"Rician(K={rician_k_factor:g},perfect-CSI)"
    return "AWGN"


def _result_rows(results, no_channel, channel_types, rician_k_factor):
    if no_channel:
        return [("No-channel", "-", results["no_channel"])]
    if len(channel_types) == 1:
        label = _channel_label(channel_types[0], rician_k_factor)
        return [(label, f"{condition} dB", metrics) for condition, metrics in results.items()]
    return [
        (
            _channel_label(channel_type, rician_k_factor),
            f"{condition} dB",
            metrics,
        )
        for channel_type, per_snr in results.items()
        for condition, metrics in per_snr.items()
    ]


def _print_results(
    cfg, results, modulation, no_channel, channel_types, rician_k_factor
):
    rows = _result_rows(
        results, no_channel, channel_types, rician_k_factor
    )
    first_metrics = rows[0][2]
    compression_rate = first_metrics["compression_rate"]

    print("=" * 92)
    print("VQ-DeepSC 最后两层级联测试结果")
    print(f"码本: {cfg.NUM_EMBEDDINGS_LIST}")
    print(f"调制方式: {'无信道' if no_channel else modulation.upper()}")
    print(f"压缩率: {compression_rate}")
    print("-" * 92)
    print(f"{'信道':<30}{'SNR':<12}{'PSNR (dB)':<18}{'MS-SSIM':<28}")
    print("-" * 92)
    for channel_label, snr_label, metrics in rows:
        print(
            f"{channel_label:<30}"
            f"{snr_label:<12}"
            f"{metrics['psnr']:<18.4f}"
            f"{_format_ms_ssim(metrics['ms_ssim']):<28}"
        )
    print("=" * 92)


@torch.no_grad()
def test_real_last2_cascade(
    checkpoint_path=None,
    test_snrs=None,
    json_output=None,
    no_channel=False,
    modulation="qpsk",
    ldpc_n=256,
    ldpc_rate=0.5,
    stream_packing="combined",
    channel_types=("awgn",),
    rician_k_factor=10.0,
    channel_seeds=(42,),
):
    cfg = Last2CascadeTestConfig()
    cfg.validate()
    setup_seed(42)
    channel_types = _normalize_channel_types(channel_types)
    channel_seeds = _normalize_channel_seeds(channel_seeds)
    if rician_k_factor < 0:
        raise ValueError("Rician K-factor must be non-negative")
    device = torch.device(cfg.DEVICE)
    checkpoint_path = checkpoint_path or os.path.join(
        cfg.CHECKPOINT_DIR, "best_vq_deepsc.pth"
    )
    test_snrs = test_snrs or [6]

    model = load_model(checkpoint_path, cfg, device)
    test_loader = get_dataloader(
        root_dir=cfg.TEST_DATASET_PATH,
        batch_size=cfg.TEST_BATCH_SIZE,
        shuffle=False,
        mode="test",
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    results = {}
    if no_channel:
        ms_ssim, psnr, diagnostics = evaluate_no_channel_last2_cascade(
            model,
            test_loader,
            cfg.NUM_EMBEDDINGS_LIST,
            device,
            return_diagnostics=True,
        )
        results["no_channel"] = {
            "ms_ssim": ms_ssim,
            "ms_ssim_db": ms_ssim_to_db(ms_ssim),
            "psnr": psnr,
            "compression_rate": _compression_rate_fraction(diagnostics),
            "diagnostics": diagnostics,
        }
    else:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            from communications.ldpc_coding import get_ldpc_code

        ldpc_code = get_ldpc_code(int(ldpc_n * ldpc_rate), rate=ldpc_rate)
        channel_results = {}
        for channel_type in channel_types:
            per_snr_results = {}
            for snr in test_snrs:
                trials = []
                for channel_seed in channel_seeds:
                    ms_ssim, psnr, diagnostics = evaluate_ldpc_channel_last2_cascade(
                        model,
                        test_loader,
                        cfg.NUM_EMBEDDINGS_LIST,
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
                    trials.append(
                        {
                            "channel_seed": channel_seed,
                            "ms_ssim": float(ms_ssim),
                            "ms_ssim_db": ms_ssim_to_db(ms_ssim),
                            "psnr": float(psnr),
                            "diagnostics": diagnostics,
                        }
                    )
                    stream = diagnostics.get("combined_stream") or diagnostics["total"]
                    print(
                        f"  seed={channel_seed} | PSNR={psnr:.4f} | "
                        f"MS-SSIM_raw={ms_ssim:.6f} | BER={stream['ber']:.8f}",
                        flush=True,
                    )
                summary = _summarize_monte_carlo_trials(trials)
                per_snr_results[str(snr)] = {
                    "ms_ssim": summary["ms_ssim_mean"],
                    "ms_ssim_db": summary["ms_ssim_db_from_mean"],
                    "ms_ssim_std": summary["ms_ssim_std"],
                    "psnr": summary["psnr_mean"],
                    "psnr_std": summary["psnr_std"],
                    "ber": summary["ber"],
                    "compression_rate": _compression_rate_fraction(
                        trials[0]["diagnostics"]
                    ),
                    "monte_carlo": summary,
                    "trials": trials,
                    "diagnostics": trials[0]["diagnostics"],
                }
            channel_results[channel_type] = per_snr_results
        results = (
            channel_results[channel_types[0]]
            if len(channel_types) == 1
            else channel_results
        )

    _print_results(
        cfg, results, modulation, no_channel, channel_types, rician_k_factor
    )

    if json_output:
        os.makedirs(os.path.dirname(json_output) or ".", exist_ok=True)
        with open(json_output, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "evaluation_mode": "last2_cascade",
                    "active_scales": list(ACTIVE_SCALES),
                    "inactive_scales": list(INACTIVE_SCALES),
                    "checkpoint": checkpoint_path,
                    "num_embeddings_list": cfg.NUM_EMBEDDINGS_LIST,
                    "ldpc_n": ldpc_n,
                    "ldpc_rate": ldpc_rate,
                    "modulation": modulation,
                    "stream_packing": stream_packing,
                    "channel_types": channel_types,
                    "rician_k_factor": float(rician_k_factor),
                    "rician_receiver_csi": (
                        "perfect_per_symbol"
                        if "rician" in channel_types else None
                    ),
                    "channel_seeds": channel_seeds,
                    "results": results,
                },
                handle,
                indent=2,
            )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the two deepest VQ scales with shallow U-Net skips disabled."
        )
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--snrs", type=float, nargs="+", default=[6])
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--no-channel", action="store_true")
    parser.add_argument(
        "--modulation", choices=["bpsk", "qpsk", "16qam"], default="qpsk"
    )
    parser.add_argument("--ldpc_n", type=int, default=256)
    parser.add_argument("--ldpc_k", type=float, default=0.5)
    parser.add_argument(
        "--stream-packing", choices=["combined"], default="combined"
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
    arguments = parser.parse_args()
    test_real_last2_cascade(
        checkpoint_path=arguments.checkpoint,
        test_snrs=arguments.snrs,
        json_output=arguments.json_output,
        no_channel=arguments.no_channel,
        modulation=arguments.modulation,
        ldpc_n=arguments.ldpc_n,
        ldpc_rate=arguments.ldpc_k,
        stream_packing=arguments.stream_packing,
        channel_types=arguments.channels,
        rician_k_factor=arguments.rician_k_factor,
        channel_seeds=arguments.channel_seeds,
    )
