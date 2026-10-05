"""Evaluate the single-scale ds2 LSTM-RAQ baseline on the real PHY chain."""

import argparse
import contextlib
import csv
import datetime
import io
import json
import math
import os
import shutil
import sys
import time
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torchvision.utils import save_image

import eval_seq2seq_channel as shared
from nets.model_one_ds2_src256_k32_2048 import (
    RAQVAE_ONE_DS2_SRC256_K32_2048,
)
from train_seq2seq_cars import load_file_module, load_shiyan_loader, local_path, write_json


ROOT = Path(__file__).resolve().parent
SOURCE_INDEX_COUNT = 128 * 128
SOURCE_VALUE_COUNT = 3 * 256 * 256


def format_result(report, results_path):
    metadata = report["metadata"]
    metrics = report["metrics"]
    rate = Fraction(metadata["ldpc"]["rate"]).limit_denominator()
    ratio = report["transmission_ratio"]
    return (
        f"\n测试完成：{report['num_images']} 张图片\n"
        f"配置：single K={metadata['target_K']} | LDPC {rate} | "
        f"{metadata['modulation'].upper()} | {metadata['channel_label']} | "
        f"SNR={metadata['snr_db']:g} dB\n"
        f"PSNR：{metrics['psnr']:.4f} dB\n"
        f"MS-SSIM：{shared.format_ms_ssim(metrics['ms_ssim'])}\n"
        f"实际传输比：{ratio:.8f}（约 1/{1 / ratio:.4f}）\n"
        f"结果文件：{results_path}"
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--checkpoint", help="Defaults to the recorded best validation checkpoint"
    )
    parser.add_argument("--output")
    parser.add_argument("--shiyan-root", default="/workspace/yi/work/shiyan")
    parser.add_argument("--dataset", default="/workspace/yi/work/Kodak-256-transform-resize")
    parser.add_argument("--snr", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--channel", choices=["awgn", "rician"], default="awgn")
    parser.add_argument("--rician-k-factor", type=float, default=10.0)
    parser.add_argument(
        "--ldpc-rate", type=shared.parse_ldpc_rate, default=0.5
    )
    parser.add_argument(
        "--modulation",
        type=str.lower,
        choices=["bpsk", "qpsk", "16qam"],
        default="bpsk",
    )
    parser.add_argument(
        "--target-k",
        type=int,
        help="Fixed target K; supported values are powers of two from 32 to 2048",
    )
    parser.add_argument("--ratio-denominator", type=float, default=48.0)
    parser.add_argument(
        "--rate-policy", choices=["fixed", "padded", "nearest"]
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--concise", action="store_true")
    args = parser.parse_args()

    args.rate_policy = args.rate_policy or (
        "fixed" if args.target_k is not None else "padded"
    )
    if not math.isfinite(args.snr):
        parser.error("--snr must be finite")
    if not math.isfinite(args.rician_k_factor) or args.rician_k_factor < 0:
        parser.error("--rician-k-factor must be finite and non-negative")

    started = time.time()
    run_dir = local_path(args.run_dir)
    config_path = run_dir / "config.json"
    status_path = run_dir / "status.json"
    if not config_path.is_file() or not status_path.is_file():
        parser.error("run directory must contain config.json and status.json")
    config = json.loads(config_path.read_text())
    status = json.loads(status_path.read_text())
    if config.get("variant") != "RAQVAE_ONE_DS2_SRC256_K32_2048_ORIGINAL":
        parser.error(f"unexpected run variant: {config.get('variant')!r}")
    source_checkpoint = local_path(
        args.checkpoint or status.get("best_model_path", "")
    )
    if not source_checkpoint.is_file():
        parser.error(f"Checkpoint not found: {source_checkpoint}")

    model_args = SimpleNamespace(**config["model_args"])
    modulation_bits = {"bpsk": 1, "qpsk": 2, "16qam": 4}[args.modulation]
    ldpc_k, ldpc_n = int(256 * args.ldpc_rate), 256
    try:
        plan = shared.transmission_plan(
            SOURCE_INDEX_COUNT,
            SOURCE_VALUE_COUNT,
            ldpc_k,
            ldpc_n,
            modulation_bits,
            args.ratio_denominator,
            args.rate_policy,
            int(model_args.num_embeddings_min),
            int(model_args.num_embeddings_max),
            args.target_k,
        )
    except ValueError as error:
        parser.error(str(error))
    target_k = int(plan["selected"]["K"])

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = local_path(
        args.output
        or ROOT
        / "runs"
        / (
            f"eval_seq2seq_one_ds2_src256_k{target_k}_"
            f"ldpc{ldpc_k}of{ldpc_n}_{args.modulation}_{args.channel}_"
            f"snr{args.snr:g}_{stamp}"
        )
    )
    if not args.concise or args.dry_run:
        print(
            f"测试配置：single K={target_k} | "
            f"LDPC {Fraction(args.ldpc_rate).limit_denominator()} | "
            f"{args.modulation.upper()} | "
            f"{shared.channel_label(args.channel, args.rician_k_factor)} | "
            f"SNR={args.snr:g} dB | 模式={args.rate_policy}",
            flush=True,
        )
        selected = plan["selected"]
        symbols = (
            math.ceil(
                (selected["source_payload_bits"] + plan["budget_padding_bits"])
                / ldpc_k
            )
            * ldpc_n
            // modulation_bits
        )
        print(
            f"预计实际传输比：{symbols / SOURCE_VALUE_COUNT:.8f}"
            f"（约 1/{SOURCE_VALUE_COUNT / symbols:.4f}）",
            flush=True,
        )
        print(f"CHECKPOINT {source_checkpoint}\nOUTPUT {output}", flush=True)
    if args.dry_run:
        return

    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "checkpoint_snapshot.ckpt"
    shutil.copy2(source_checkpoint, snapshot)
    checkpoint = torch.load(snapshot, map_location="cpu", weights_only=False)

    reference = Path(args.shiyan_root)
    dependency_output = (
        contextlib.redirect_stdout(io.StringIO())
        if args.concise
        else contextlib.nullcontext()
    )
    with dependency_output:
        get_dataloader, setup_seed = load_shiyan_loader(reference)
        load_file_module("utils.bit_utils", reference / "utils/bit_utils.py")
        load_file_module("utils.metrics", reference / "utils/metrics.py")
        sys.path.append(str(reference))
        quality = load_file_module(
            "raq_one_reference_quality", reference / "evaluation/quality.py"
        )
        from utils.bit_utils import bits_to_index_tensor, index_tensor_to_bits
        from utils.reproducibility import _reset_eval_seed
        from communications.modulation import bpsk_modulate, qpsk_modulate
        from communications.ldpc_coding import (
            get_ldpc_code,
            ldpc_decode,
            ldpc_encode,
        )
        setup_seed(args.model_seed)

    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    model_args.device = device
    model = RAQVAE_ONE_DS2_SRC256_K32_2048(model_args).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.requires_grad_(False)
    modulate, calculate_llr, modulation_bits = {
        "bpsk": (bpsk_modulate, shared.exact_bpsk_llr, 1),
        "qpsk": (qpsk_modulate, shared.exact_qpsk_llr, 2),
        "16qam": (
            shared.qam16_modulate_vectorized,
            shared.exact_qam16_llr,
            4,
        ),
    }[args.modulation]
    target = torch.arange(target_k, device=device).unsqueeze(1)
    codebook = model.cbk2cbk(model.src.to(device), target).squeeze(1)
    if tuple(codebook.shape) != (target_k, int(model_args.embedding_dim)):
        raise ValueError(f"unexpected generated codebook shape: {codebook.shape}")
    if not torch.isfinite(codebook).all():
        raise ValueError("generated target codebook contains nonfinite values")
    torch.save(codebook.cpu(), output / f"target_codebook_k{target_k}.pt")

    os.environ["SIMVQ_TEST_NO_RESIZE"] = "1"
    loader = get_dataloader(
        args.dataset,
        batch_size=1,
        shuffle=False,
        mode="test",
        num_workers=8,
        pin_memory=True,
    )
    if len(loader.dataset) != 24:
        raise ValueError(f"Expected all 24 Kodak images, found {len(loader.dataset)}")
    filenames = list(loader.dataset.image_files)
    with dependency_output:
        ldpc = get_ldpc_code(ldpc_k, rate=args.ldpc_rate)
    if ldpc["k"] != ldpc_k or ldpc["n"] != ldpc_n:
        raise ValueError("LDPC construction returned an unexpected code")

    metadata = {
        "variant": config["variant"],
        "source_checkpoint": str(source_checkpoint),
        "snapshot": str(snapshot),
        "checkpoint_sha256": shared.digest(snapshot),
        "checkpoint_epoch_index": checkpoint["epoch"],
        "checkpoint_epoch_number": checkpoint["epoch"] + 1,
        "checkpoint_global_step": checkpoint["global_step"],
        "selection": (
            "explicit checkpoint supplied by caller"
            if args.checkpoint
            else "minimum saved validation MSE_src + MSE_trg"
        ),
        "source_K": int(model_args.num_embeddings),
        "target_K": target_k,
        "requested_target_K": args.target_k,
        "embedding_dim": int(model_args.embedding_dim),
        "source_index_count": SOURCE_INDEX_COUNT,
        "spatial_scales": {"single": "1/2"},
        "dataset": args.dataset,
        "dataset_order": filenames,
        "test_no_resize": True,
        "normalization": "(RGB tensor - 0.5) / 0.5",
        "quality": (
            "shiyan.evaluation.quality._image_quality; arithmetic mean over images; "
            "aggregate MS-SSIM dB is converted after averaging raw MS-SSIM"
        ),
        "seed": args.seed,
        "channel_seed": args.seed,
        "model_seed": args.model_seed,
        "snr_db": args.snr,
        "channel": args.channel,
        "channel_label": shared.channel_label(
            args.channel, args.rician_k_factor
        ),
        "rician_k_factor": args.rician_k_factor,
        "receiver_csi": (
            "perfect_per_symbol" if args.channel == "rician" else "h=1"
        ),
        "llr": "exact Log-MAP; actual generated noise variance",
        "modulation": args.modulation,
        "ldpc": {"k": ldpc_k, "n": ldpc_n, "rate": args.ldpc_rate},
        "stream_packing": "one single-scale index stream",
        "transmission_ratio_definition": (
            "complex channel symbols / (3*height*width)"
        ),
        "target_transmission_ratio": 1 / args.ratio_denominator,
        "rate_plan": plan,
        "padding_policy": (
            "only mandatory LDPC/modulation padding, no artificial budget padding"
            if args.rate_policy != "padded"
            else "explicit trailing zeros before LDPC to fill exact channel budget"
        ),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "created_at": datetime.datetime.now().astimezone().isoformat(),
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
            raise ValueError(f"Unexpected image size: {image.shape}")
        result = model(image, target)
        forward_clean = result[3]
        sent_indices = result[5]
        if tuple(sent_indices.shape) != (1, 128, 128):
            raise ValueError(f"Unexpected index shape: {sent_indices.shape}")
        clean = model.decode_latent(sent_indices, target)

        bits, shape, serialized_k = index_tensor_to_bits(sent_indices, target_k)
        roundtrip = bits_to_index_tensor(bits, shape, serialized_k).to(device)
        if not torch.equal(sent_indices, roundtrip):
            raise AssertionError("Noiseless bit packing changed indices")
        payload = np.asarray(bits, dtype=np.uint8).reshape(-1)
        if payload.size != plan["selected"]["source_payload_bits"]:
            raise AssertionError("payload length differs from the rate plan")
        padding_bits = int(plan["budget_padding_bits"])
        combined = np.pad(payload, (0, padding_bits), constant_values=0)
        decoded, stats = shared.transmit_ldpc_stream_exact(
            combined,
            args.snr,
            ldpc,
            device,
            modulate,
            calculate_llr,
            modulation_bits,
            ldpc_encode,
            ldpc_decode,
            args.channel,
            args.rician_k_factor,
        )
        received_indices = bits_to_index_tensor(
            decoded[:payload.size], shape, serialized_k
        ).to(device)
        quantized = torch.nn.functional.embedding(
            received_indices, codebook
        ).permute(0, 3, 1, 2)
        reconstructed = model.decode(quantized)
        if not torch.isfinite(clean).all() or not torch.isfinite(reconstructed).all():
            raise ValueError("reconstruction contains nonfinite values")

        clean_ssim, clean_psnr = quality._image_quality(image, clean)
        channel_ssim, channel_psnr = quality._image_quality(image, reconstructed)
        source_errors = int(np.count_nonzero(decoded[:payload.size] != payload))
        row = {
            "image": filenames[image_index],
            "no_channel_psnr": clean_psnr,
            "no_channel_ms_ssim": clean_ssim,
            "no_channel_ms_ssim_db": shared.ms_ssim_to_db(clean_ssim),
            "psnr": channel_psnr,
            "ms_ssim": channel_ssim,
            "ms_ssim_db": shared.ms_ssim_to_db(channel_ssim),
            "source_payload_bits": int(payload.size),
            "budget_padding_bits": padding_bits,
            "ldpc_input_bits": stats["ldpc_input_bits"],
            "ldpc_padding_bits": stats["ldpc_padding_bits"],
            "coded_bits": stats["coded_bits"],
            "transmitted_bits": stats["transmitted_bits"],
            "channel_symbols": stats["channel_symbols"],
            "transmission_ratio": stats["channel_symbols"] / image.numel(),
            "source_payload_bpp": payload.size / (256 * 256),
            "padded_payload_bpp": stats["ldpc_input_bits"] / (256 * 256),
            "source_bit_errors": source_errors,
            "source_ber": source_errors / payload.size,
            "combined_bit_errors": stats["bit_errors"],
            "forward_vs_index_decode_max_abs": float(
                (forward_clean - clean).abs().max()
            ),
            "forward_vs_index_decode_mse": float(
                (forward_clean - clean).square().mean()
            ),
            "code_counts": torch.bincount(
                sent_indices.flatten(), minlength=target_k
            ).tolist(),
        }
        rows.append(row)
        for folder, tensor in (
            ("reference", image),
            ("no_channel", clean),
            (channel_folder, reconstructed),
        ):
            save_image(
                ((tensor + 1) / 2).clamp(0, 1),
                output / folder / filenames[image_index],
            )
        write_json(
            output / "progress.json",
            {
                "finished_images": len(rows),
                "total_images": 24,
                "last_image": row,
            },
        )
        if not args.concise:
            print(
                f"IMAGE {image_index + 1}/24 {filenames[image_index]} "
                f"PSNR={channel_psnr:.4f} "
                f"MS-SSIM={shared.format_ms_ssim(channel_ssim)}",
                flush=True,
            )

    totals = {
        key: sum(row[key] for row in rows)
        for key in (
            "source_payload_bits",
            "budget_padding_bits",
            "ldpc_input_bits",
            "ldpc_padding_bits",
            "coded_bits",
            "transmitted_bits",
            "channel_symbols",
            "source_bit_errors",
            "combined_bit_errors",
        )
    }
    metrics = {
        key: float(np.mean([row[key] for row in rows]))
        for key in (
            "no_channel_psnr",
            "no_channel_ms_ssim",
            "psnr",
            "ms_ssim",
        )
    }
    metrics["no_channel_ms_ssim_db"] = shared.ms_ssim_to_db(
        metrics["no_channel_ms_ssim"]
    )
    metrics["ms_ssim_db"] = shared.ms_ssim_to_db(metrics["ms_ssim"])
    report = {
        "metadata": metadata,
        "num_images": len(rows),
        "metrics": metrics,
        "totals": totals,
        "source_payload_bpp": (
            totals["source_payload_bits"] / (len(rows) * 256 * 256)
        ),
        "padded_payload_bpp": (
            totals["ldpc_input_bits"] / (len(rows) * 256 * 256)
        ),
        "transmission_ratio": (
            totals["channel_symbols"] / (len(rows) * SOURCE_VALUE_COUNT)
        ),
        "source_ber": (
            totals["source_bit_errors"] / totals["source_payload_bits"]
        ),
        "combined_ber": (
            totals["combined_bit_errors"] / totals["ldpc_input_bits"]
        ),
        "elapsed_seconds": time.time() - started,
        "per_image": rows,
    }
    results_path = output / "results.json"
    write_json(results_path, report)
    with (output / "per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(format_result(report, results_path), flush=True)


if __name__ == "__main__":
    main()
