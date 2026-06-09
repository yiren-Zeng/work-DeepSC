#!/usr/bin/env python3
"""Evaluate B larger [128,16] on original Kodak by 256x256 tiling."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def configure_env() -> None:
    os.environ.setdefault("SIMVQ_EXPERIMENT_STAGE", "B")
    os.environ.setdefault("SIMVQ_EXP_FAMILY", "quality_v2_B_larger_cb128-16")
    os.environ.setdefault("SIMVQ_NUM_EMBEDDINGS_LIST", "128,16")
    os.environ.setdefault("SIMVQ_DOWNSAMPLE_STRIDES", "8,2")
    os.environ.setdefault("SIMVQ_UNET_DEPTH", "2")
    os.environ.setdefault("SIMVQ_BASE_CHANNELS", "128")
    os.environ.setdefault("SIMVQ_ENCODER_RES_BLOCKS", "4")
    os.environ.setdefault("SIMVQ_DECODER_RES_BLOCKS", "4")
    os.environ.setdefault("SIMVQ_QUANTIZER_TYPE", "simvq")


configure_env()

from communications.channel import awgn_channel  # noqa: E402
from communications.ldpc_coding import get_ldpc_code, ldpc_decode, ldpc_encode  # noqa: E402
from communications.modulation import bpsk_llr, bpsk_modulate  # noqa: E402
from config import Config  # noqa: E402
from evaluation.quality import _image_quality  # noqa: E402
from utils.bit_utils import bits_to_indices, indices_to_bits  # noqa: E402
from utils.checkpoint_utils import build_model_from_checkpoint  # noqa: E402
from utils.reproducibility import setup_seed  # noqa: E402


def image_to_tensor(path: Path, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


def iter_tiles(image: torch.Tensor, tile_size: int):
    _, _, height, width = image.shape
    if height % tile_size != 0 or width % tile_size != 0:
        raise ValueError(f"Image shape {(height, width)} is not divisible by tile_size={tile_size}")
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            yield top, left, image[:, :, top:top + tile_size, left:left + tile_size]


@torch.no_grad()
def reconstruct_tile_no_channel(model, tile: torch.Tensor) -> torch.Tensor:
    out = model.forward_test(tile)
    return model.reconstruct_from_indices(out["indices"])


@torch.no_grad()
def reconstruct_tile_bpsk_snr0(model, tile: torch.Tensor, num_embeddings_list, ldpc_code, device):
    out = model.forward_test(tile)
    flat_bits, original_spatial_dims, original_num_embeddings = indices_to_bits(
        out["indices"], num_embeddings_list
    )
    coded_bits = ldpc_encode(flat_bits, code=ldpc_code)
    coded_bits_tensor = torch.from_numpy(coded_bits).float().to(device)
    symbols = bpsk_modulate(coded_bits_tensor)
    noisy_symbols = awgn_channel(symbols, 0)
    llrs = bpsk_llr(noisy_symbols, 0, device)
    decoded_bits = ldpc_decode(llrs.cpu().numpy(), ldpc_code)[:len(flat_bits)]
    recovered_indices = bits_to_indices(decoded_bits, original_spatial_dims, original_num_embeddings)
    recovered_indices = [idx.to(device) for idx in recovered_indices]
    return model.reconstruct_from_indices(recovered_indices)


@torch.no_grad()
def evaluate_tiled(model, image_paths, tile_size: int, mode: str, num_embeddings_list, device):
    setup_seed(42)
    ldpc_code = get_ldpc_code(128, rate=0.5) if mode == "bpsk_snr0" else None
    per_image = []
    psnr_scores = []
    ms_ssim_scores = []

    for path in image_paths:
        image = image_to_tensor(path, device)
        recon = torch.empty_like(image)
        tile_count = 0
        for top, left, tile in iter_tiles(image, tile_size):
            if mode == "no_channel":
                recon_tile = reconstruct_tile_no_channel(model, tile)
            elif mode == "bpsk_snr0":
                recon_tile = reconstruct_tile_bpsk_snr0(
                    model, tile, num_embeddings_list, ldpc_code, device
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")
            recon[:, :, top:top + tile_size, left:left + tile_size] = recon_tile
            tile_count += 1

        ms_ssim, psnr = _image_quality(image, recon)
        psnr_scores.append(psnr)
        ms_ssim_scores.append(ms_ssim)
        _, _, height, width = image.shape
        per_image.append({
            "file": path.name,
            "height": height,
            "width": width,
            "tile_count": tile_count,
            "psnr": psnr,
            "ms_ssim": ms_ssim,
        })

    return {
        "psnr": float(np.mean(psnr_scores)),
        "ms_ssim": float(np.mean(ms_ssim_scores)),
        "per_image": per_image,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/quality_v2_B_larger_cb128-16_unet2_ds8x2_k128-16/best_vq_deepsc.pth")
    parser.add_argument("--kodak-dir", default="/workspace/yi/work/Kodak")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--output", default="experiments/interim_results/20260605_cb128_16_tiled_kodak/tiled_eval.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_seed(42)
    cfg = Config()
    device = torch.device(cfg.DEVICE)
    model, inferred = build_model_from_checkpoint(args.checkpoint, cfg, device)
    image_paths = sorted(Path(args.kodak_dir).glob("kodim*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No kodim*.png images found in {args.kodak_dir}")

    results = {
        "checkpoint": args.checkpoint,
        "kodak_dir": args.kodak_dir,
        "tile_size": args.tile_size,
        "num_embeddings_list": inferred["num_embeddings_list"],
        "tests": {},
    }
    for mode in ("no_channel", "bpsk_snr0"):
        print(f"===== Evaluating {mode} =====", flush=True)
        results["tests"][mode] = evaluate_tiled(
            model, image_paths, args.tile_size, mode, inferred["num_embeddings_list"], device
        )
        metrics = results["tests"][mode]
        print(f"{mode}: PSNR={metrics['psnr']:.4f} dB, MS-SSIM={metrics['ms_ssim']:.4f}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
