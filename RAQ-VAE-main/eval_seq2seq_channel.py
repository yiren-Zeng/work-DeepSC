"""Evaluate a snapshot of the best Seq2Seq checkpoint using shiyan's real link.

The legacy padded mode fills the requested channel budget explicitly. The nearest
mode chooses the closest naturally occurring transmission ratio and adds only the
padding required by LDPC/modulation. Model and external reference code are unchanged.
Use --target-k to select a fixed shared codebook without artificial budget padding.
"""
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

from train_seq2seq_cars import (
    ROOT, RAQVAE_TWO, load_file_module, load_shiyan_loader,
    local_path, write_json,
)

import argparse
import contextlib
import csv
import datetime
import hashlib
import json
import math
import io
import shutil
import time
from fractions import Fraction
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

MODEL_CLASS = RAQVAE_TWO
SOURCE_INDEX_COUNT = 5120
OUTPUT_TAG = "seq2seq"
SPATIAL_SCALES = {"bottom": "1/4", "top": "1/8"}
DUAL_K = False
BOTTOM_INDEX_COUNT = 4096
TOP_INDEX_COUNT = 1024


def ms_ssim_to_db(ms_ssim):
    """Convert MS-SSIM to the commonly reported -10*log10(1-MS-SSIM)."""
    value = float(ms_ssim)
    if math.isnan(value):
        return float("nan")
    if value >= 1.0:
        return float("inf")
    return -10.0 * math.log10(1.0 - value)


def format_ms_ssim(ms_ssim):
    return f"{float(ms_ssim):.6f} ({ms_ssim_to_db(ms_ssim):.4f} dB)"


def awgn_channel(symbols, snr_db, return_noise_power=False):
    """Apply complex AWGN and optionally expose its generated variance."""
    snr_linear = 10 ** (snr_db / 10.0)
    signal_power = torch.mean(torch.abs(symbols) ** 2)
    noise_power = signal_power / snr_linear
    real_dtype = symbols.real.dtype if symbols.is_complex() else symbols.dtype
    noise = torch.sqrt(noise_power / 2.0) * (
        torch.randn(symbols.shape, dtype=real_dtype, device=symbols.device)
        + 1j
        * torch.randn(symbols.shape, dtype=real_dtype, device=symbols.device)
    )
    received = symbols + noise
    if return_noise_power:
        return received, noise_power
    return received


def rician_channel(symbols, snr_db, k_factor, return_csi=False):
    """Apply unit-power fast Rician fading and complex AWGN.

    When ``return_csi`` is true, return the exact per-symbol channel gain and
    generated noise variance for coherent likelihood calculation.
    """
    if k_factor < 0:
        raise ValueError("Rician K-factor must be non-negative")

    real_dtype = symbols.real.dtype if symbols.is_complex() else symbols.dtype
    complex_dtype = (
        torch.complex64 if real_dtype == torch.float32 else torch.complex128
    )
    k = torch.as_tensor(k_factor, dtype=real_dtype, device=symbols.device)
    los = torch.sqrt(k / (k + 1.0)).to(complex_dtype)
    scatter_scale = torch.sqrt(1.0 / (2.0 * (k + 1.0)))
    scatter = scatter_scale * (
        torch.randn(symbols.shape, dtype=real_dtype, device=symbols.device)
        + 1j
        * torch.randn(symbols.shape, dtype=real_dtype, device=symbols.device)
    )
    channel_gain = los + scatter
    faded = channel_gain * symbols

    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = torch.mean(torch.abs(symbols) ** 2) / snr_linear
    noise = torch.sqrt(noise_power / 2.0) * (
        torch.randn(symbols.shape, dtype=real_dtype, device=symbols.device)
        + 1j
        * torch.randn(symbols.shape, dtype=real_dtype, device=symbols.device)
    )
    received = faded + noise
    if return_csi:
        return received, channel_gain, noise_power
    return received


def _noise_variance(reference, snr_db, noise_variance):
    real_dtype = reference.real.dtype if reference.is_complex() else reference.dtype
    if noise_variance is None:
        noise_variance = 1.0 / (10 ** (snr_db / 10.0))
    variance = torch.as_tensor(
        noise_variance, dtype=real_dtype, device=reference.device
    )
    if variance.numel() != 1 or float(variance.item()) <= 0:
        raise ValueError("noise_variance must be a positive scalar")
    return variance


def _matched_observation(received_symbols, channel_gain):
    if channel_gain is None:
        return received_symbols
    gain = torch.as_tensor(
        channel_gain,
        dtype=received_symbols.dtype,
        device=received_symbols.device,
    )
    if gain.shape != received_symbols.shape:
        raise ValueError(
            "channel_gain shape must match received symbols: "
            f"{tuple(gain.shape)} != {tuple(received_symbols.shape)}"
        )
    return gain.conj() * received_symbols


def exact_bpsk_llr(
    received_symbols,
    snr_db,
    device,
    channel_gain=None,
    noise_variance=None,
):
    """Return log P(bit=1)/P(bit=0) for coherent BPSK."""
    matched = _matched_observation(received_symbols, channel_gain)
    variance = _noise_variance(matched, snr_db, noise_variance)
    return ((4.0 / variance) * matched.real).to(device)


def exact_qpsk_llr(
    received_symbols,
    snr_db,
    device,
    channel_gain=None,
    noise_variance=None,
):
    """Return coherent QPSK bit LLRs."""
    matched = _matched_observation(received_symbols, channel_gain)
    variance = _noise_variance(matched, snr_db, noise_variance)
    factor = 4.0 / (
        variance * torch.sqrt(torch.tensor(2.0, device=matched.device))
    )
    llr = torch.zeros(
        2 * len(matched), dtype=matched.real.dtype, device=matched.device
    )
    llr[0::2] = matched.real * factor
    llr[1::2] = matched.imag * factor
    return llr.to(device)


def exact_qam16_llr(
    symbols,
    snr_db,
    device,
    channel_gain=None,
    noise_variance=None,
):
    """Return exact Log-MAP Gray-coded 16QAM LLRs with optional perfect CSI."""
    constellation = torch.tensor([
        -3 - 3j, -3 - 1j, -3 + 1j, -3 + 3j,
        -1 - 3j, -1 - 1j, -1 + 1j, -1 + 3j,
         1 - 3j,  1 - 1j,  1 + 1j,  1 + 3j,
         3 - 3j,  3 - 1j,  3 + 1j,  3 + 3j,
    ], dtype=torch.complex64, device=device) / torch.sqrt(
        torch.tensor(10.0, device=device)
    )
    bit_mapping = torch.tensor([
        [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0],
        [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 1], [0, 1, 1, 0],
        [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 1], [1, 1, 1, 0],
        [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 1], [1, 0, 1, 0],
    ], dtype=torch.float32, device=device)

    symbols = symbols.to(device)
    variance = _noise_variance(symbols, snr_db, noise_variance)
    if channel_gain is None:
        hypotheses = constellation.unsqueeze(0)
    else:
        gain = torch.as_tensor(channel_gain, dtype=symbols.dtype, device=device)
        if gain.shape != symbols.shape:
            raise ValueError(
                "channel_gain shape must match received symbols: "
                f"{tuple(gain.shape)} != {tuple(symbols.shape)}"
            )
        hypotheses = gain.unsqueeze(1) * constellation.unsqueeze(0)
    log_likelihoods = -torch.abs(symbols.unsqueeze(1) - hypotheses) ** 2 / variance
    llr = torch.zeros(4 * len(symbols), device=device)
    for bit_position in range(4):
        zero = log_likelihoods[:, bit_mapping[:, bit_position].eq(0)]
        one = log_likelihoods[:, bit_mapping[:, bit_position].eq(1)]
        llr[bit_position::4] = (
            torch.logsumexp(one, dim=1) - torch.logsumexp(zero, dim=1)
        )
    return llr


def qam16_modulate_vectorized(bits):
    """Match shiyan's Gray-coded 16QAM mapping without per-bit CUDA syncs."""
    bits_reshaped = bits.detach().to(device="cpu").view(-1, 4)
    gray_levels = torch.tensor([-3.0, -1.0, 3.0, 1.0])
    real_indices = (2 * bits_reshaped[:, 0] + bits_reshaped[:, 1]).long()
    imag_indices = (2 * bits_reshaped[:, 2] + bits_reshaped[:, 3]).long()
    real_part = gray_levels[real_indices]
    imag_part = gray_levels[imag_indices]
    return (real_part + 1j * imag_part) / math.sqrt(10.0)


def channel_label(channel, rician_k_factor):
    return (
        f"Rician(K={rician_k_factor:g},perfect-CSI)"
        if channel == "rician"
        else "AWGN"
    )


def transmit_ldpc_stream_exact(
    flat_bits,
    target_snr,
    ldpc_code,
    device,
    modulate,
    calculate_llr,
    modulation_bits,
    ldpc_encode,
    ldpc_decode,
    channel,
    rician_k_factor,
):
    """Transmit one combined stream using the same receiver as RAQ-RVQ."""
    flat_bits = np.asarray(flat_bits, dtype=np.uint8).reshape(-1)
    payload_bits = len(flat_bits)
    k = int(ldpc_code["k"])
    ldpc_input_bits = ((payload_bits + k - 1) // k) * k
    ldpc_padding_bits = ldpc_input_bits - payload_bits

    coded = np.asarray(ldpc_encode(flat_bits, code=ldpc_code)).reshape(-1)
    coded_bits = len(coded)
    modulation_padding_bits = (-coded_bits) % modulation_bits
    transmitted = np.pad(coded, (0, modulation_padding_bits), "constant")
    transmitted_tensor = torch.from_numpy(transmitted).float().to(device)
    symbols = modulate(transmitted_tensor)

    channel_gain = None
    if channel == "awgn":
        received, noise_variance = awgn_channel(
            symbols, target_snr, return_noise_power=True
        )
    elif channel == "rician":
        received, channel_gain, noise_variance = rician_channel(
            symbols,
            target_snr,
            rician_k_factor,
            return_csi=True,
        )
    else:
        raise ValueError(f"Unsupported channel: {channel!r}")

    llrs = calculate_llr(
        received,
        target_snr,
        device,
        channel_gain=channel_gain,
        noise_variance=noise_variance,
    ).reshape(-1)
    decoded = np.asarray(
        ldpc_decode(llrs[:coded_bits].detach().cpu().numpy(), ldpc_code)
    ).reshape(-1)[:payload_bits]
    if len(decoded) < payload_bits:
        decoded = np.pad(decoded, (0, payload_bits - len(decoded)), "constant")
    decoded = decoded.astype(np.uint8, copy=False)
    return decoded, {
        "ldpc_input_bits": int(ldpc_input_bits),
        "ldpc_padding_bits": int(ldpc_padding_bits),
        "coded_bits": int(coded_bits),
        "modulation_padding_bits": int(modulation_padding_bits),
        "transmitted_bits": int(len(transmitted)),
        "channel_symbols": int(symbols.numel()),
        "bit_errors": int(np.count_nonzero(decoded != flat_bits)),
    }


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transmission_plan(source_indices, source_values, ldpc_k, ldpc_n, modulation_bits,
                      denominator, policy, min_k=2, max_k=64, target_k=None):
    if denominator <= 0:
        raise ValueError("ratio denominator must be positive")
    if policy not in ("fixed", "nearest", "padded"):
        raise ValueError("Unknown rate policy")
    if (policy == "fixed") != (target_k is not None):
        raise ValueError("fixed policy requires --target-k; other policies select K automatically")
    # The reference bit serializer uses int(log2(K)); non-power-of-two K would
    # silently truncate indices. Evaluate every losslessly supported K in range.
    candidates = []
    for exponent in range(1, max_k.bit_length()):
        k = 1 << exponent
        if not min_k <= k <= max_k:
            continue
        payload = source_indices * exponent
        blocks = math.ceil(payload / ldpc_k)
        coded = blocks * ldpc_n
        transmitted = math.ceil(coded / modulation_bits) * modulation_bits
        symbols = transmitted // modulation_bits
        candidates.append({
            "K": k, "bits_per_index": exponent, "source_payload_bits": payload,
            "ldpc_input_bits": blocks * ldpc_k, "ldpc_padding_bits": blocks * ldpc_k - payload,
            "coded_bits": coded, "transmitted_bits": transmitted, "channel_symbols": symbols,
            "transmission_ratio": symbols / source_values,
            "absolute_ratio_error": abs(symbols / source_values - 1 / denominator),
        })
    if not candidates:
        raise ValueError("No supported power-of-two K in the trained range")
    if policy == "fixed":
        matches = [item for item in candidates if item["K"] == target_k]
        if not matches:
            raise ValueError(f"target K must be one of {[item['K'] for item in candidates]}; "
                             "the reference bit serializer requires a power of two")
        selected = matches[0]
        budget_padding = 0
    elif policy == "nearest":
        selected = min(candidates, key=lambda item: (item["absolute_ratio_error"], item["K"]))
        budget_padding = 0
    else:
        feasible = [item for item in candidates if item["transmission_ratio"] <= 1 / denominator]
        if not feasible:
            raise ValueError("No supported K fits this exact padded channel budget")
        selected = max(feasible, key=lambda item: item["K"])
        desired_symbols_value = source_values / denominator
        if not desired_symbols_value.is_integer():
            raise ValueError("Requested exact budget is not an integer number of symbols")
        desired_symbols = int(desired_symbols_value)
        if desired_symbols * modulation_bits % ldpc_n:
            raise ValueError("Requested exact budget is not an integer number of LDPC blocks")
        input_capacity = desired_symbols * modulation_bits // ldpc_n * ldpc_k
        budget_padding = input_capacity - selected["source_payload_bits"]
    return {"selected": selected, "budget_padding_bits": budget_padding,
            "candidates": candidates, "policy": policy,
            "candidate_rule": "all power-of-two K in trained range; lossless reference fixed-width serializer"}


def dual_k_transmission_plan(bottom_indices, top_indices, bottom_k, top_k,
                             source_values, ldpc_k, ldpc_n, modulation_bits,
                             denominator, min_k=2, max_k=64):
    if denominator <= 0:
        raise ValueError("ratio denominator must be positive")
    valid = [1 << exponent for exponent in range(1, max_k.bit_length())
             if min_k <= 1 << exponent <= max_k]
    if bottom_k not in valid or top_k not in valid:
        raise ValueError(f"bottom/top K must each be one of {valid}; "
                         "the reference bit serializer requires powers of two")
    bottom_bits = bottom_k.bit_length() - 1
    top_bits = top_k.bit_length() - 1
    payload = bottom_indices * bottom_bits + top_indices * top_bits
    blocks = math.ceil(payload / ldpc_k)
    coded = blocks * ldpc_n
    transmitted = math.ceil(coded / modulation_bits) * modulation_bits
    symbols = transmitted // modulation_bits
    selected = {
        "K_bottom": bottom_k, "K_top": top_k,
        "bits_per_bottom_index": bottom_bits, "bits_per_top_index": top_bits,
        "bottom_payload_bits": bottom_indices * bottom_bits,
        "top_payload_bits": top_indices * top_bits,
        "source_payload_bits": payload, "ldpc_input_bits": blocks * ldpc_k,
        "ldpc_padding_bits": blocks * ldpc_k - payload, "coded_bits": coded,
        "transmitted_bits": transmitted, "channel_symbols": symbols,
        "transmission_ratio": symbols / source_values,
        "absolute_ratio_error": abs(symbols / source_values - 1 / denominator),
    }
    return {
        "selected": selected, "budget_padding_bits": 0, "candidates": [],
        "policy": "fixed_dual_k",
        "candidate_rule": "explicit bottom/top power-of-two K; lossless fixed-width serializer",
    }


def parse_ldpc_rate(value):
    try:
        rate = float(Fraction(value))
    except (ValueError, ZeroDivisionError, OverflowError):
        raise argparse.ArgumentTypeError(
            "Use 7/16 (0.4375), 1/2 (0.5), 5/8 (0.625), or 3/4 (0.75)"
        ) from None
    if rate not in (0.4375, 0.5, 0.625, 0.75):
        raise argparse.ArgumentTypeError(
            "Supported LDPC rates: 7/16, 1/2, 5/8, and 3/4"
        )
    return rate


def format_console_result(report, results_path):
    meta = report["metadata"]
    metrics = report["metrics"]
    rate = Fraction(meta["ldpc"]["rate"]).limit_denominator()
    ratio = report["transmission_ratio"]
    codebook_text = (f"K_bottom={meta['target_K_bottom']} | K_top={meta['target_K_top']}"
                     if meta.get("dual_k") else f"K={meta['target_K_shared']}")
    selected_channel = channel_label(meta["channel"], meta["rician_k_factor"])
    return (
        f"\n测试完成：{report['num_images']} 张图片\n"
        f"配置：{codebook_text} | LDPC {rate} | "
        f"{meta['modulation'].upper()} | {selected_channel} | "
        f"SNR={meta['snr_db']:g} dB\n"
        f"PSNR：{metrics['psnr']:.4f} dB\n"
        f"MS-SSIM：{format_ms_ssim(metrics['ms_ssim'])}\n"
        f"实际传输比：{ratio:.8f}（约 1/{1 / ratio:.4f}）\n"
        f"结果文件：{results_path}"
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", help="Defaults to the current recorded best validation checkpoint")
    parser.add_argument("--output", help="New output directory inside this project; default: timestamped runs directory")
    parser.add_argument("--shiyan-root", default="/workspace/yi/work/shiyan")
    parser.add_argument("--dataset", default="/workspace/yi/work/Kodak-256-transform-resize")
    parser.add_argument("--snr", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-seed",
        type=int,
        default=42,
        help="Fixed model/data seed; --seed is reserved for channel noise.",
    )
    parser.add_argument(
        "--channel", choices=["awgn", "rician"], default="awgn",
        help="Physical channel model; defaults to AWGN for compatibility",
    )
    parser.add_argument(
        "--rician-k-factor", type=float, default=10.0,
        help="Linear Rician K-factor used when --channel=rician",
    )
    parser.add_argument("--ldpc-rate", type=parse_ldpc_rate, default=0.75)
    parser.add_argument(
        "--modulation",
        type=str.lower,
        choices=["bpsk", "qpsk", "16qam"],
        default="qpsk",
    )
    parser.add_argument("--target-k", type=int, help="Target codebook size shared by top/bottom; trained powers of two only")
    parser.add_argument("--bottom-k", type=int, help="Dual-K mode: fixed bottom codebook size")
    parser.add_argument("--top-k", type=int, help="Dual-K mode: fixed top codebook size")
    parser.add_argument("--ratio-denominator", type=float, default=48.0)
    parser.add_argument("--rate-policy", choices=["fixed", "padded", "nearest"],
                        help="Default: fixed when --target-k is given, otherwise legacy padded mode")
    parser.add_argument("--dry-run", action="store_true", help="Validate arguments and print rate plan without evaluating or creating output")
    parser.add_argument("--concise", action="store_true",
                        help="Only print the final channel-test summary (apart from dependency messages)")
    args = parser.parse_args()
    if DUAL_K:
        if args.bottom_k is None or args.top_k is None:
            parser.error("Dual-K model requires both --bottom-k and --top-k")
        if args.target_k is not None:
            parser.error("Dual-K model uses --bottom-k/--top-k, not --target-k")
        if args.rate_policy not in (None, "fixed"):
            parser.error("Dual-K testing currently supports fixed K only")
        args.rate_policy = "fixed"
    else:
        if args.bottom_k is not None or args.top_k is not None:
            parser.error("Single-K model uses --target-k, not --bottom-k/--top-k")
        args.rate_policy = args.rate_policy or ("fixed" if args.target_k is not None else "padded")
    if not math.isfinite(args.snr):
        parser.error("--snr must be finite")
    if not math.isfinite(args.rician_k_factor) or args.rician_k_factor < 0:
        parser.error("--rician-k-factor must be finite and non-negative")
    started = time.time()
    run_dir = local_path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    status = json.loads((run_dir / "status.json").read_text())
    source_checkpoint = local_path(args.checkpoint or status["best_model_path"])
    if not source_checkpoint.is_file():
        parser.error(f"Checkpoint not found: {source_checkpoint}")
    model_args = SimpleNamespace(**config["model_args"])
    modulation_bits = {"bpsk": 1, "qpsk": 2, "16qam": 4}[args.modulation]
    ldpc_k, ldpc_n = int(256 * args.ldpc_rate), 256
    try:
        if DUAL_K:
            plan = dual_k_transmission_plan(
                BOTTOM_INDEX_COUNT, TOP_INDEX_COUNT, args.bottom_k, args.top_k,
                3 * 256 * 256, ldpc_k, ldpc_n, modulation_bits,
                args.ratio_denominator, model_args.num_embeddings_min,
                model_args.num_embeddings_max,
            )
        else:
            plan = transmission_plan(
                SOURCE_INDEX_COUNT, 3 * 256 * 256, ldpc_k, ldpc_n,
                modulation_bits, args.ratio_denominator, args.rate_policy,
                model_args.num_embeddings_min, model_args.num_embeddings_max,
                args.target_k,
            )
    except ValueError as error:
        parser.error(str(error))
    target_k = None if DUAL_K else plan["selected"]["K"]
    target_bottom_k = args.bottom_k if DUAL_K else target_k
    target_top_k = args.top_k if DUAL_K else target_k
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = local_path(args.output or ROOT / "runs" / (
        (f"eval_{OUTPUT_TAG}_kb{target_bottom_k}_kt{target_top_k}_"
         if DUAL_K else f"eval_{OUTPUT_TAG}_k{target_k}_") +
        f"ldpc{ldpc_k}of{ldpc_n}_{args.modulation}_{args.channel}_"
        f"snr{args.snr:g}_{stamp}"))
    if not args.concise or args.dry_run:
        k_text = (f"K_bottom={target_bottom_k} | K_top={target_top_k}"
                  if DUAL_K else f"K={target_k}")
        print(f"测试配置：{k_text} | LDPC {Fraction(args.ldpc_rate).limit_denominator()} | "
              f"{args.modulation.upper()} | "
              f"{channel_label(args.channel, args.rician_k_factor)} | "
              f"SNR={args.snr:g} dB | 模式={args.rate_policy}", flush=True)
    if args.dry_run:
        selected = plan["selected"]
        symbols = math.ceil((selected["source_payload_bits"] + plan["budget_padding_bits"]) / ldpc_k) * ldpc_n // modulation_bits
        print(f"预计实际传输比：{symbols / (3 * 256 * 256):.8f}"
              f"（约 1/{3 * 256 * 256 / symbols:.4f}）", flush=True)
    if not args.concise or args.dry_run:
        print(f"CHECKPOINT {source_checkpoint}\nOUTPUT {output}", flush=True)
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "checkpoint_snapshot.ckpt"
    shutil.copy2(source_checkpoint, snapshot)
    checkpoint = torch.load(snapshot, map_location="cpu", weights_only=False)
    if not args.concise:
        print(f"SNAPSHOT {source_checkpoint} -> {snapshot}", flush=True)

    reference = Path(args.shiyan_root)
    dependency_output = (contextlib.redirect_stdout(io.StringIO())
                         if args.concise else contextlib.nullcontext())
    with dependency_output:
        get_dataloader, setup_seed = load_shiyan_loader(reference)
        load_file_module("utils.bit_utils", reference / "utils/bit_utils.py")
        load_file_module("utils.metrics", reference / "utils/metrics.py")
        # Remaining reference imports resolve without displacing RAQ's utils package.
        sys.path.append(str(reference))
        quality = load_file_module("raq_reference_quality", reference / "evaluation/quality.py")
        from utils.bit_utils import index_tensor_to_bits, bits_to_index_tensor
        from utils.reproducibility import _reset_eval_seed
        from communications.modulation import (
            bpsk_modulate,
            qpsk_modulate,
        )
        from communications.ldpc_coding import get_ldpc_code, ldpc_encode, ldpc_decode
        setup_seed(args.model_seed)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    model_args.device = device
    model = MODEL_CLASS(model_args).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.requires_grad_(False)
    modulate, demodulate, modulation_bits = {
        "bpsk": (bpsk_modulate, exact_bpsk_llr, 1),
        "qpsk": (qpsk_modulate, exact_qpsk_llr, 2),
        "16qam": (qam16_modulate_vectorized, exact_qam16_llr, 4),
    }[args.modulation]
    target_bottom = torch.arange(target_bottom_k, device=device).unsqueeze(1)
    target_top = torch.arange(target_top_k, device=device).unsqueeze(1)
    if DUAL_K:
        # Always generate twice, even when target_bottom_k == target_top_k.
        codebook_bottom = model.cbk2cbk(model.src.to(device), target_bottom).squeeze(1)
        codebook_top = model.cbk2cbk(model.src.to(device), target_top).squeeze(1)
        assert torch.isfinite(codebook_bottom).all() and torch.isfinite(codebook_top).all()
        torch.save(codebook_bottom.cpu(), output / f"target_codebook_bottom_k{target_bottom_k}.pt")
        torch.save(codebook_top.cpu(), output / f"target_codebook_top_k{target_top_k}.pt")
    else:
        target = target_bottom
        codebook = model.cbk2cbk(model.src.to(device), target).squeeze(1)
        assert torch.isfinite(codebook).all()
        codebook_bottom = codebook_top = codebook
        torch.save(codebook.cpu(), output / f"target_codebook_k{target_k}.pt")

    os.environ["SIMVQ_TEST_NO_RESIZE"] = "1"
    loader = get_dataloader(args.dataset, batch_size=1, shuffle=False, mode="test",
                            num_workers=8, pin_memory=True)
    if len(loader.dataset) != 24:
        raise ValueError(f"Expected all 24 Kodak images, found {len(loader.dataset)}")
    filenames = list(loader.dataset.image_files)
    with (contextlib.redirect_stdout(io.StringIO())
          if args.concise else contextlib.nullcontext()):
        ldpc = get_ldpc_code(ldpc_k, rate=args.ldpc_rate)
    assert ldpc["k"] == ldpc_k and ldpc["n"] == ldpc_n
    metadata = {
        "source_checkpoint": str(source_checkpoint), "snapshot": str(snapshot),
        "checkpoint_sha256": digest(snapshot),
        "checkpoint_epoch_index": checkpoint["epoch"],
        "checkpoint_epoch_number": checkpoint["epoch"] + 1,
        "checkpoint_global_step": checkpoint["global_step"],
        "selection": ("explicit checkpoint supplied by caller" if args.checkpoint else
                      "minimum saved validation MSE_src + MSE_trg, not Kodak quality"),
        "training_status_at_selection": status,
        "source_K": model_args.num_embeddings, "dual_k": DUAL_K,
        "target_K_shared": target_k,
        "target_K_bottom": target_bottom_k, "target_K_top": target_top_k,
        "requested_target_K": args.target_k,
        "source_index_count": SOURCE_INDEX_COUNT, "spatial_scales": SPATIAL_SCALES,
        "dataset": args.dataset, "dataset_order": filenames,
        "dataset_sha256": {name: digest(Path(args.dataset) / name)
                           for name in filenames},
        "loader": "shiyan.data.datasets.get_dataloader(test, batch=1, shuffle=False, workers=8)",
        "test_no_resize": True, "normalization": "(RGB tensor - 0.5) / 0.5",
        "quality": "shiyan.evaluation.quality._image_quality, arithmetic mean over images",
        "no_channel_reconstruction": "original RAQVAE_TWO.decode_latent on sent indices",
        "seed": args.seed,
        "channel_seed": args.seed,
        "model_seed": args.model_seed,
        "snr_db": args.snr, "channel": args.channel,
        "channel_label": channel_label(args.channel, args.rician_k_factor),
        "rician_k_factor": args.rician_k_factor,
        "channel_implementation": (
            "local unit-power fast Rician fading plus complex AWGN"
            if args.channel == "rician"
            else "local complex AWGN"
        ),
        "receiver_csi": (
            "perfect_per_symbol" if args.channel == "rician" else "h=1"
        ),
        "llr": "exact Log-MAP; actual generated noise variance",
        "modulation": args.modulation, "ldpc": {"k": ldpc_k, "n": ldpc_n, "rate": args.ldpc_rate},
        "stream_packing": "combined, bottom indices followed by top indices; optional explicit budget padding",
        "transmission_ratio_definition": "complex channel symbols / (3*height*width)",
        "target_transmission_ratio": 1 / args.ratio_denominator,
        "rate_plan": plan,
        "padding_policy": ("only mandatory LDPC/modulation padding, no artificial budget padding"
                           if args.rate_policy != "padded" else
                           "explicit trailing zeros before LDPC to fill exact channel budget"),
        "image_export": "PNG images clipped to [0,1] for visualization only; metrics computed before export",
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "reference_source_sha256": {
            str(path): digest(path) for path in [
                reference / "data/datasets.py", reference / "utils/reproducibility.py",
                reference / "utils/metrics.py", reference / "utils/bit_utils.py",
                reference / "evaluation/quality.py", reference / "communications/channel.py",
                reference / "communications/modulation.py", reference / "communications/ldpc_coding.py",
            ]
        },
    }
    write_json(output / "metadata.json", metadata)
    channel_suffix = (
        f"rician_k{args.rician_k_factor:g}"
        if args.channel == "rician"
        else "awgn"
    )
    channel_folder = f"channel_{channel_suffix}_{args.snr:g}db"
    for name in ("reference", "no_channel", channel_folder):
        (output / name).mkdir()
    _reset_eval_seed(args.seed)
    rows = []
    for image_index, image in enumerate(loader):
        image = image.to(device)
        if tuple(image.shape) != (1, 3, 256, 256):
            raise ValueError(f"Unexpected image size without resizing: {image.shape}")
        result = (model(image, target_bottom, target_top) if DUAL_K
                  else model(image, target))
        forward_clean = result[4]
        index_top, index_bottom = result[6], result[7]
        # Use the official discrete-index reconstruction API for the noiseless
        # control as well. The forward STE has tiny float cancellation differences
        # from embedding lookup; those are amplified slightly by TF32 convolution.
        clean = (model.decode_latent(index_top, index_bottom, target_bottom, target_top)
                 if DUAL_K else model.decode_latent(index_top, index_bottom, target))
        segments = []
        for indices, segment_k in ((index_bottom, target_bottom_k),
                                   (index_top, target_top_k)):
            bits, shape, k = index_tensor_to_bits(indices, segment_k)
            roundtrip = bits_to_index_tensor(bits, shape, k).to(device)
            assert torch.equal(indices, roundtrip), "Noiseless bit packing changed indices"
            segments.append((bits, shape, k))
        payload = np.concatenate([segment[0] for segment in segments])
        assert payload.size == plan["selected"]["source_payload_bits"]
        padding_bits = plan["budget_padding_bits"]
        combined = np.pad(payload, (0, padding_bits), constant_values=0)
        decoded, stats = transmit_ldpc_stream_exact(
            combined,
            args.snr,
            ldpc,
            device,
            modulate,
            demodulate,
            modulation_bits,
            ldpc_encode,
            ldpc_decode,
            args.channel,
            args.rician_k_factor,
        )
        expected_blocks = math.ceil(combined.size / ldpc_k)
        assert stats["coded_bits"] == expected_blocks * ldpc_n
        assert stats["channel_symbols"] == math.ceil(stats["coded_bits"] / modulation_bits)
        if args.rate_policy != "padded":
            assert stats["channel_symbols"] == plan["selected"]["channel_symbols"]
        else:
            assert stats["channel_symbols"] * args.ratio_denominator == image.numel()
        received_indices = []
        offset = 0
        for bits, shape, k in segments:
            received_indices.append(bits_to_index_tensor(
                decoded[offset:offset + len(bits)], shape, k
            ).to(device))
            offset += len(bits)
        received_bottom, received_top = received_indices
        quant_top = torch.nn.functional.embedding(received_top, codebook_top).permute(0, 3, 1, 2)
        quant_bottom = torch.nn.functional.embedding(received_bottom, codebook_bottom).permute(0, 3, 1, 2)
        reconstructed = model.decode(quant_top, quant_bottom)
        if image_index == 0:
            pristine = model.decode(
                torch.nn.functional.embedding(index_top, codebook_top).permute(0, 3, 1, 2),
                torch.nn.functional.embedding(index_bottom, codebook_bottom).permute(0, 3, 1, 2),
            )
            torch.testing.assert_close(pristine, clean, rtol=1e-5, atol=1e-6)
        assert torch.isfinite(clean).all() and torch.isfinite(reconstructed).all()
        clean_ssim, clean_psnr = quality._image_quality(image, clean)
        channel_ssim, channel_psnr = quality._image_quality(image, reconstructed)
        source_errors = int(np.count_nonzero(decoded[:payload.size] != payload))
        row = {
            "image": filenames[image_index], "no_channel_psnr": clean_psnr,
            "no_channel_ms_ssim": clean_ssim,
            "no_channel_ms_ssim_db": ms_ssim_to_db(clean_ssim),
            "psnr": channel_psnr, "ms_ssim": channel_ssim,
            "ms_ssim_db": ms_ssim_to_db(channel_ssim),
            "source_payload_bits": int(payload.size), "budget_padding_bits": int(padding_bits),
            "ldpc_input_bits": stats["ldpc_input_bits"],
            "ldpc_padding_bits": stats["ldpc_padding_bits"],
            "coded_bits": stats["coded_bits"], "transmitted_bits": stats["transmitted_bits"],
            "channel_symbols": stats["channel_symbols"],
            "transmission_ratio": stats["channel_symbols"] / image.numel(),
            "source_payload_bpp": payload.size / (256 * 256),
            "padded_payload_bpp": stats["ldpc_input_bits"] / (256 * 256),
            "source_bit_errors": source_errors, "source_ber": source_errors / payload.size,
            "combined_bit_errors": stats["bit_errors"],
            "forward_vs_index_decode_max_abs": float((forward_clean - clean).abs().max()),
            "forward_vs_index_decode_mse": float((forward_clean - clean).square().mean()),
            "bottom_code_counts": torch.bincount(
                index_bottom.flatten(), minlength=target_bottom_k
            ).tolist(),
            "top_code_counts": torch.bincount(
                index_top.flatten(), minlength=target_top_k
            ).tolist(),
        }
        rows.append(row)
        for folder, tensor in (("reference", image), ("no_channel", clean),
                               (channel_folder, reconstructed)):
            save_image(((tensor + 1) / 2).clamp(0, 1), output / folder / filenames[image_index])
        write_json(output / "progress.json", {"finished_images": len(rows), "total_images": 24,
                                               "last_image": row})
        if not args.concise:
            print(f"IMAGE {image_index + 1}/24 {filenames[image_index]} "
                  f"PSNR={channel_psnr:.4f} "
                  f"MS-SSIM={format_ms_ssim(channel_ssim)}", flush=True)
    assert len(rows) == 24
    totals = {key: sum(row[key] for row in rows) for key in (
        "source_payload_bits", "budget_padding_bits", "ldpc_input_bits", "ldpc_padding_bits",
        "coded_bits", "transmitted_bits", "channel_symbols", "source_bit_errors", "combined_bit_errors",
    )}
    metrics = {key: float(np.mean([row[key] for row in rows])) for key in (
        "no_channel_psnr", "no_channel_ms_ssim", "psnr", "ms_ssim",
    )}
    metrics["no_channel_ms_ssim_db"] = ms_ssim_to_db(
        metrics["no_channel_ms_ssim"]
    )
    metrics["ms_ssim_db"] = ms_ssim_to_db(metrics["ms_ssim"])
    report = {
        "metadata": metadata, "num_images": 24, "metrics": metrics, "totals": totals,
        "source_payload_bpp": totals["source_payload_bits"] / (24 * 256 * 256),
        "padded_payload_bpp": totals["ldpc_input_bits"] / (24 * 256 * 256),
        "transmission_ratio": totals["channel_symbols"] / (24 * 3 * 256 * 256),
        "source_ber": totals["source_bit_errors"] / totals["source_payload_bits"],
        "combined_ber": totals["combined_bit_errors"] / totals["ldpc_input_bits"],
        "elapsed_seconds": time.time() - started, "per_image": rows,
    }
    write_json(output / "results.json", report)
    with (output / "per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(format_console_result(report, output / "results.json"), flush=True)


if __name__ == "__main__":
    main()
