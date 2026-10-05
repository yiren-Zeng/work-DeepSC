#!/usr/bin/env python3
"""Monte Carlo BER-SNR simulation for the repository's LDPC channel chain.

The transmitted payload is random information bits. This program deliberately
does not load a learned model, dataset, RAQ/RVQ indices, or codebooks.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Callable

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class Profile:
    name: str
    label: str
    rate: float
    modulation: str
    bits_per_symbol: int


PROFILES = (
    Profile("ldpc1over2_qpsk", "1/2 LDPC + QPSK", 0.5, "qpsk", 2),
    Profile("ldpc5over8_qpsk", "5/8 LDPC + QPSK", 0.625, "qpsk", 2),
    Profile("ldpc3over4_qpsk", "3/4 LDPC + QPSK", 0.75, "qpsk", 2),
    Profile("ldpc7over16_16qam", "7/16 LDPC + 16QAM", 0.4375, "16qam", 4),
    Profile("ldpc1over2_16qam", "1/2 LDPC + 16QAM", 0.5, "16qam", 4),
)
PROFILE_BY_NAME = {profile.name: profile for profile in PROFILES}
MARKERS = ("o", "s", "^", "D", "v")


def _load_communications() -> tuple[
    Callable[..., dict[str, Any]],
    Callable[..., np.ndarray],
    Callable[..., np.ndarray],
    dict[str, tuple[Callable[..., torch.Tensor], Callable[..., torch.Tensor]]],
    Callable[..., Any],
    Callable[..., Any],
]:
    from communications.channel import awgn_channel, rician_channel
    from communications.ldpc_coding import get_ldpc_code, ldpc_decode, ldpc_encode
    from communications.modulation import (
        qam16_llr,
        qam16_modulate,
        qpsk_llr,
        qpsk_modulate,
    )

    modulators = {
        "qpsk": (qpsk_modulate, qpsk_llr),
        "16qam": (qam16_modulate, qam16_llr),
    }
    return (
        get_ldpc_code,
        ldpc_encode,
        ldpc_decode,
        modulators,
        awgn_channel,
        rician_channel,
    )


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _condition_seed(
    base_seed: int, profile_index: int, channel_index: int, snr_index: int
) -> int:
    return base_seed + profile_index * 100_000 + channel_index * 10_000 + snr_index


def _simulate_condition(
    *,
    profile: Profile,
    profile_index: int,
    channel: str,
    channel_index: int,
    snr_db: float,
    snr_index: int,
    ldpc_n: int,
    rician_k_factor: float,
    min_errors: int,
    max_bits: int,
    batch_blocks: int,
    base_seed: int,
    device: torch.device,
    code: dict[str, Any],
    ldpc_encode: Callable[..., np.ndarray],
    ldpc_decode: Callable[..., np.ndarray],
    modulators: dict[
        str, tuple[Callable[..., torch.Tensor], Callable[..., torch.Tensor]]
    ],
    awgn_channel: Callable[..., Any],
    rician_channel: Callable[..., Any],
) -> dict[str, Any]:
    ldpc_k = int(round(ldpc_n * profile.rate))
    actual_n = int(code["n"])
    actual_k = int(code["k"])
    if actual_n != ldpc_n or actual_k != ldpc_k:
        raise RuntimeError(
            f"{profile.label}: requested LDPC ({ldpc_n}, {ldpc_k}), "
            f"constructed ({actual_n}, {actual_k})"
        )
    if actual_n % profile.bits_per_symbol:
        raise ValueError(
            f"LDPC n={actual_n} is not divisible by "
            f"{profile.bits_per_symbol} bits/symbol"
        )

    max_blocks = max_bits // actual_k
    if max_blocks < 1:
        raise ValueError(
            f"max_bits={max_bits} is smaller than one information block k={actual_k}"
        )

    condition_seed = _condition_seed(
        base_seed, profile_index, channel_index, snr_index
    )
    rng = np.random.default_rng(condition_seed)
    torch.manual_seed(condition_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(condition_seed)

    modulate, calculate_llr = modulators[profile.modulation]
    total_errors = 0
    total_information_bits = 0
    total_coded_bits = 0
    total_symbols = 0
    blocks_run = 0

    while blocks_run < max_blocks:
        current_blocks = min(batch_blocks, max_blocks - blocks_run)
        information_bits = rng.integers(
            0, 2, size=current_blocks * actual_k, dtype=np.uint8
        )
        coded_bits = np.asarray(ldpc_encode(information_bits, code=code)).reshape(-1)
        expected_coded_bits = current_blocks * actual_n
        if len(coded_bits) != expected_coded_bits:
            raise RuntimeError(
                f"LDPC encoder returned {len(coded_bits)} bits; "
                f"expected {expected_coded_bits}"
            )

        coded_tensor = torch.from_numpy(coded_bits).float().to(device)
        symbols = modulate(coded_tensor)
        if channel == "awgn":
            received, noise_variance = awgn_channel(
                symbols, snr_db, return_noise_power=True
            )
            channel_gain = None
        elif channel == "rician":
            received, channel_gain, noise_variance = rician_channel(
                symbols,
                snr_db,
                K_factor=rician_k_factor,
                return_csi=True,
            )
        else:
            raise ValueError(f"unsupported channel: {channel}")

        llrs = calculate_llr(
            received,
            snr_db,
            device,
            channel_gain=channel_gain,
            noise_variance=noise_variance,
        ).reshape(-1)
        decoded = np.asarray(
            ldpc_decode(llrs.detach().cpu().numpy(), code=code)
        ).reshape(-1)
        decoded = decoded[: len(information_bits)].astype(np.uint8, copy=False)
        if len(decoded) != len(information_bits):
            raise RuntimeError(
                f"LDPC decoder returned {len(decoded)} information bits; "
                f"expected {len(information_bits)}"
            )

        total_errors += int(np.count_nonzero(decoded != information_bits))
        total_information_bits += len(information_bits)
        total_coded_bits += len(coded_bits)
        total_symbols += int(symbols.numel())
        blocks_run += current_blocks
        if total_errors >= min_errors:
            break

    ber = total_errors / total_information_bits
    return {
        "profile": profile.name,
        "label": profile.label,
        "channel": channel,
        "snr_db": float(snr_db),
        "ldpc_n": actual_n,
        "ldpc_k": actual_k,
        "ldpc_rate": actual_k / actual_n,
        "modulation": profile.modulation,
        "bits_per_symbol": profile.bits_per_symbol,
        "information_bits": total_information_bits,
        "coded_bits": total_coded_bits,
        "channel_symbols": total_symbols,
        "blocks": blocks_run,
        "bit_errors": total_errors,
        "ber": ber,
        "plot_ber": ber if ber > 0 else 0.5 / total_information_bits,
        "zero_error_plot_floor": 0.5 / total_information_bits,
        "stop_reason": (
            "min_errors" if total_errors >= min_errors else "max_bits"
        ),
        "seed": condition_seed,
    }


def _metadata(args: argparse.Namespace, profiles: list[Profile]) -> dict[str, Any]:
    return {
        "simulation": "random_bits_ldpc_modulation_monte_carlo",
        "uses_learned_model": False,
        "uses_dataset": False,
        "uses_codebook": False,
        "ber_definition": "information-bit BER after LDPC decoding",
        "snr_definition": (
            "unit-average-symbol-power SNR used by communications/channel.py"
        ),
        "rician_receiver_csi": "perfect_per_symbol",
        "ldpc_n": args.ldpc_n,
        "rician_k_factor": args.rician_k_factor,
        "min_errors": args.min_errors,
        "max_bits": args.max_bits,
        "batch_blocks": args.batch_blocks,
        "seed": args.seed,
        "snrs_db": [float(value) for value in args.snrs],
        "channels": list(args.channels),
        "profiles": [asdict(profile) for profile in profiles],
    }


def _atomic_write_json(
    output_path: Path, metadata: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "results": rows}, handle, indent=2)
    temporary.replace(output_path)


def _write_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "profile",
        "label",
        "channel",
        "snr_db",
        "ldpc_n",
        "ldpc_k",
        "ldpc_rate",
        "modulation",
        "bits_per_symbol",
        "information_bits",
        "coded_bits",
        "channel_symbols",
        "blocks",
        "bit_errors",
        "ber",
        "stop_reason",
        "seed",
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_channel(
    channel: str,
    profiles: list[Profile],
    rows: list[dict[str, Any]],
    rician_k_factor: float,
    output_dir: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    for index, profile in enumerate(profiles):
        points = sorted(
            (
                row
                for row in rows
                if row["channel"] == channel and row["profile"] == profile.name
            ),
            key=lambda row: row["snr_db"],
        )
        if not points:
            raise RuntimeError(f"no {channel} results for {profile.label}")
        marker = MARKERS[index % len(MARKERS)]
        (line,) = axis.semilogy(
            [row["snr_db"] for row in points],
            [row["plot_ber"] for row in points],
            marker=marker,
            markersize=6,
            linewidth=1.8,
            label=profile.label,
        )
        zero_points = [row for row in points if row["bit_errors"] == 0]
        if zero_points:
            axis.scatter(
                [row["snr_db"] for row in zero_points],
                [row["plot_ber"] for row in zero_points],
                marker=marker,
                s=42,
                facecolors="white",
                edgecolors=line.get_color(),
                linewidths=1.3,
                zorder=3,
            )

    title = "AWGN" if channel == "awgn" else (
        f"Rician (K={rician_k_factor:g}, perfect CSI)"
    )
    axis.set_title(title)
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("Post-LDPC information-bit BER")
    axis.grid(True, which="both", linestyle="--", alpha=0.4)
    axis.legend(fontsize=9)
    axis.text(
        0.01,
        0.01,
        "Hollow marker: 0 observed errors, displayed at 0.5/N",
        transform=axis.transAxes,
        fontsize=8,
        color="0.35",
    )
    stem = output_dir / f"ber_snr_{channel}"
    figure.savefig(stem.with_suffix(".png"), dpi=220)
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)


def _load_resume_rows(
    output_path: Path, metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []
    with output_path.open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    if saved.get("metadata") != metadata:
        raise ValueError(
            f"cannot resume {output_path}: saved simulation settings differ"
        )
    rows = saved.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"cannot resume {output_path}: invalid results list")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate post-LDPC BER versus SNR with random information bits; "
            "no learned model or codebook is used."
        )
    )
    parser.add_argument(
        "--snrs",
        type=float,
        nargs="+",
        default=list(range(0, 21, 2)),
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        choices=("awgn", "rician"),
        default=["awgn", "rician"],
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(PROFILE_BY_NAME),
        default=None,
        help="Defaults to all five requested LDPC/modulation profiles.",
    )
    parser.add_argument("--ldpc-n", type=int, default=256)
    parser.add_argument("--rician-k-factor", type=float, default=10.0)
    parser.add_argument("--min-errors", type=int, default=200)
    parser.add_argument("--max-bits", type=int, default=3_000_000)
    parser.add_argument("--batch-blocks", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/eval/ldpc_modulation_ber_snr"),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.ldpc_n <= 0:
        raise ValueError("ldpc-n must be positive")
    if args.rician_k_factor < 0:
        raise ValueError("rician-k-factor must be non-negative")
    if args.min_errors <= 0:
        raise ValueError("min-errors must be positive")
    if args.max_bits <= 0:
        raise ValueError("max-bits must be positive")
    if args.batch_blocks <= 0:
        raise ValueError("batch-blocks must be positive")
    if len(set(args.snrs)) != len(args.snrs):
        raise ValueError("snrs must not contain duplicates")
    if len(set(args.channels)) != len(args.channels):
        raise ValueError("channels must not contain duplicates")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    profiles = (
        list(PROFILES)
        if args.profiles is None
        else [PROFILE_BY_NAME[name] for name in args.profiles]
    )
    device = _resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_output = args.output_dir / "ber_snr_results.json"
    csv_output = args.output_dir / "ber_snr_results.csv"
    metadata = _metadata(args, profiles)
    rows = _load_resume_rows(json_output, metadata) if args.resume else []
    completed = {
        (row["profile"], row["channel"], float(row["snr_db"])) for row in rows
    }

    print("Simulation payload: independent random information bits")
    print("Learned model / dataset / codebook: not used")
    print(f"Torch channel device: {device}")
    print("Loading repository LDPC, modulation, and channel implementations...")
    (
        get_ldpc_code,
        ldpc_encode,
        ldpc_decode,
        modulators,
        awgn_channel,
        rician_channel,
    ) = _load_communications()

    codes: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        ldpc_k = int(round(args.ldpc_n * profile.rate))
        codes[profile.name] = get_ldpc_code(ldpc_k, rate=profile.rate)

    total_conditions = len(profiles) * len(args.channels) * len(args.snrs)
    condition_number = 0
    for profile_index, profile in enumerate(profiles):
        for channel_index, channel in enumerate(args.channels):
            for snr_index, snr_db in enumerate(args.snrs):
                condition_number += 1
                key = (profile.name, channel, float(snr_db))
                if key in completed:
                    print(
                        f"[{condition_number}/{total_conditions}] skip completed | "
                        f"{profile.label} | {channel.upper()} | SNR={snr_db:g} dB"
                    )
                    continue
                result = _simulate_condition(
                    profile=profile,
                    profile_index=profile_index,
                    channel=channel,
                    channel_index=channel_index,
                    snr_db=snr_db,
                    snr_index=snr_index,
                    ldpc_n=args.ldpc_n,
                    rician_k_factor=args.rician_k_factor,
                    min_errors=args.min_errors,
                    max_bits=args.max_bits,
                    batch_blocks=args.batch_blocks,
                    base_seed=args.seed,
                    device=device,
                    code=codes[profile.name],
                    ldpc_encode=ldpc_encode,
                    ldpc_decode=ldpc_decode,
                    modulators=modulators,
                    awgn_channel=awgn_channel,
                    rician_channel=rician_channel,
                )
                rows.append(result)
                rows.sort(
                    key=lambda row: (
                        next(
                            index
                            for index, item in enumerate(profiles)
                            if item.name == row["profile"]
                        ),
                        args.channels.index(row["channel"]),
                        args.snrs.index(float(row["snr_db"])),
                    )
                )
                _atomic_write_json(json_output, metadata, rows)
                _write_csv(csv_output, rows)
                print(
                    f"[{condition_number}/{total_conditions}] {profile.label} | "
                    f"{channel.upper()} | SNR={snr_db:g} dB | "
                    f"BER={result['ber']:.8g} | errors={result['bit_errors']} / "
                    f"bits={result['information_bits']} | {result['stop_reason']}"
                )

    expected = {
        (profile.name, channel, float(snr))
        for profile in profiles
        for channel in args.channels
        for snr in args.snrs
    }
    actual = {
        (row["profile"], row["channel"], float(row["snr_db"])) for row in rows
    }
    if actual != expected:
        raise RuntimeError("result set is incomplete or contains unexpected conditions")

    for channel in args.channels:
        _plot_channel(
            channel, profiles, rows, args.rician_k_factor, args.output_dir
        )
    print(f"JSON: {json_output}")
    print(f"CSV:  {csv_output}")
    for channel in args.channels:
        print(f"Plot: {args.output_dir / f'ber_snr_{channel}.png'}")


if __name__ == "__main__":
    main()
