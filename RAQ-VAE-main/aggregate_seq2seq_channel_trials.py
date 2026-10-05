"""Validate and aggregate independent Seq2Seq channel trials."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def ms_ssim_to_db(ms_ssim):
    value = float(ms_ssim)
    if math.isnan(value):
        return float("nan")
    if value >= 1.0:
        return float("inf")
    return -10.0 * math.log10(1.0 - value)


def normalize_channel_seeds(seeds):
    normalized = [int(seed) for seed in seeds]
    if not normalized:
        raise ValueError("At least one channel seed is required")
    if any(seed < 0 for seed in normalized):
        raise ValueError("Channel seeds must be non-negative")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Channel seeds must be unique")
    return normalized


def summarize_trial_records(trials):
    """Average raw trial metrics before converting mean MS-SSIM to dB."""
    if not trials:
        raise ValueError("At least one trial is required")
    psnr_values = np.asarray([trial["psnr"] for trial in trials], dtype=float)
    ms_ssim_values = np.asarray(
        [trial["ms_ssim"] for trial in trials], dtype=float
    )
    ddof = 1 if len(trials) > 1 else 0
    total_source_errors = sum(int(trial["source_bit_errors"]) for trial in trials)
    total_source_bits = sum(int(trial["source_payload_bits"]) for trial in trials)
    total_combined_errors = sum(
        int(trial["combined_bit_errors"]) for trial in trials
    )
    total_ldpc_input_bits = sum(
        int(trial["ldpc_input_bits"]) for trial in trials
    )
    total_channel_symbols = sum(
        int(trial["channel_symbols"]) for trial in trials
    )
    total_source_values = sum(int(trial["source_values"]) for trial in trials)
    mean_ms_ssim = float(np.mean(ms_ssim_values))
    return {
        "num_trials": len(trials),
        "channel_seeds": [int(trial["channel_seed"]) for trial in trials],
        "psnr_mean": float(np.mean(psnr_values)),
        "psnr_std": float(np.std(psnr_values, ddof=ddof)),
        "ms_ssim_mean": mean_ms_ssim,
        "ms_ssim_std": float(np.std(ms_ssim_values, ddof=ddof)),
        "ms_ssim_db_from_mean": ms_ssim_to_db(mean_ms_ssim),
        "total_source_bit_errors": total_source_errors,
        "total_source_payload_bits": total_source_bits,
        "source_ber": total_source_errors / total_source_bits,
        "total_combined_bit_errors": total_combined_errors,
        "total_ldpc_input_bits": total_ldpc_input_bits,
        "combined_ber": total_combined_errors / total_ldpc_input_bits,
        "total_channel_symbols": total_channel_symbols,
        "total_source_values": total_source_values,
        "transmission_ratio": total_channel_symbols / total_source_values,
    }


def _compatibility_signature(report):
    metadata = report["metadata"]
    return {
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "target_K_bottom": metadata["target_K_bottom"],
        "target_K_top": metadata["target_K_top"],
        "dataset_sha256": metadata["dataset_sha256"],
        "snr_db": metadata["snr_db"],
        "channel": metadata["channel"],
        "rician_k_factor": metadata["rician_k_factor"],
        "modulation": metadata["modulation"],
        "ldpc": metadata["ldpc"],
        "rate_plan": metadata["rate_plan"],
    }


def aggregate_reports(paths, expected_seeds):
    expected_seeds = normalize_channel_seeds(expected_seeds)
    reports = [json.loads(path.read_text()) for path in paths]
    if len(reports) != len(expected_seeds):
        raise ValueError(
            f"Expected {len(expected_seeds)} reports, got {len(reports)}"
        )
    signature = _compatibility_signature(reports[0])
    trials = []
    for path, report in zip(paths, reports):
        if _compatibility_signature(report) != signature:
            raise ValueError(f"Incompatible trial configuration: {path}")
        metadata = report["metadata"]
        seed = int(metadata.get("channel_seed", metadata["seed"]))
        totals = report["totals"]
        trials.append(
            {
                "channel_seed": seed,
                "results_path": str(path),
                "psnr": float(report["metrics"]["psnr"]),
                "ms_ssim": float(report["metrics"]["ms_ssim"]),
                "ms_ssim_db": float(report["metrics"]["ms_ssim_db"]),
                "source_bit_errors": int(totals["source_bit_errors"]),
                "source_payload_bits": int(totals["source_payload_bits"]),
                "combined_bit_errors": int(totals["combined_bit_errors"]),
                "ldpc_input_bits": int(totals["ldpc_input_bits"]),
                "channel_symbols": int(totals["channel_symbols"]),
                "source_values": int(report["num_images"] * 3 * 256 * 256),
            }
        )
    actual_seeds = [trial["channel_seed"] for trial in trials]
    if actual_seeds != expected_seeds:
        raise ValueError(
            f"Trial seeds {actual_seeds} do not match expected {expected_seeds}"
        )
    summary = summarize_trial_records(trials)
    return {
        "aggregation": (
            "arithmetic mean of raw metrics across channel seeds; MS-SSIM dB "
            "is -10*log10(1-mean_raw_ms_ssim)"
        ),
        "configuration": signature,
        "monte_carlo": summary,
        "trials": trials,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-results", nargs="+", type=Path, required=True)
    parser.add_argument("--expected-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate = aggregate_reports(args.trial_results, args.expected_seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = aggregate["monte_carlo"]
    print(
        f"Monte Carlo complete | seeds={summary['channel_seeds']} | "
        f"PSNR_mean={summary['psnr_mean']:.4f} | "
        f"MS-SSIM_mean_raw={summary['ms_ssim_mean']:.6f} | "
        f"MS-SSIM={summary['ms_ssim_db_from_mean']:.4f} dB | "
        f"BER={summary['source_ber']:.8f} | output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
