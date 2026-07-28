"""Evaluate a trained depth-2 Residual-SimVQ at fixed depths 1 through 4.

The checkpoint is loaded strictly at its trained depth before the runtime
``rq_depth`` integers are changed.  Residual-SimVQ already registers exactly
one shared projected codebook per scale, so this path adds no model state and
saves no migrated checkpoint.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import torch

from config import Config
from data.datasets import get_dataloader
from evaluation.quality import evaluate_ldpc_channel
from evaluation.residual_simvq_depth_extension import (
    evaluate_residual_simvq_no_channel_depth_sweep,
    extend_residual_simvq_depth_for_eval,
    set_residual_simvq_depth_for_eval,
)
from utils.checkpoint_utils import build_model_from_checkpoint
from utils.reproducibility import setup_seed


PROJECT_ROOT = Path(__file__).resolve().parent
LDPC_N = 256
LDPC_RATE = 0.5
MODULATION_BITS = {"bpsk": 1, "qpsk": 2, "16qam": 4}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_project_output(path):
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            f"output must remain inside {PROJECT_ROOT}: {resolved}"
        ) from error
    return resolved


def _projected_codebook_fingerprints(model):
    fingerprints = []
    for quantizer in model.vector_quantizers:
        digest = hashlib.sha256()
        for name in ("embed.weight", "proj.weight"):
            tensor = (
                quantizer.codebook.state_dict()[name]
                .detach()
                .to(device="cpu")
                .contiguous()
            )
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        fingerprints.append(digest.hexdigest())
    return fingerprints


def _write_csv(path, no_channel, channel):
    fieldnames = [
        "condition",
        "depth",
        "snr_db",
        "source_bits_per_image",
        "source_bpp",
        "transmission_ratio",
        "ldpc_blocks_per_image",
        "source_padding_bits_per_image",
        "coded_bits_per_image",
        "coded_bpp",
        "modulated_symbols_per_image",
        "psnr",
        "ms_ssim",
        "final_residual_rms_scale0",
        "final_residual_rms_scale1",
    ]
    rows = []
    no_channel_by_depth = {
        int(item["depth"]): item for item in no_channel["depth_results"]
    }
    for item in no_channel["depth_results"]:
        residuals = item["final_residual_rms_per_scale"]
        rows.append(
            {
                "condition": "no_channel",
                "depth": item["depth"],
                "snr_db": "",
                "source_bits_per_image": item["source_bits_per_image"],
                "source_bpp": item["source_bpp"],
                "transmission_ratio": item[
                    "ldpc_bpsk_rgb_transmission_ratio"
                ],
                "ldpc_blocks_per_image": "",
                "source_padding_bits_per_image": "",
                "coded_bits_per_image": "",
                "coded_bpp": "",
                "modulated_symbols_per_image": "",
                "psnr": item["psnr"],
                "ms_ssim": item["ms_ssim"],
                "final_residual_rms_scale0": residuals[0],
                "final_residual_rms_scale1": residuals[1],
            }
        )

    if channel is not None:
        for depth_item in channel["depth_results"]:
            depth = int(depth_item["depth"])
            residuals = no_channel_by_depth[depth][
                "final_residual_rms_per_scale"
            ]
            for snr, metrics in depth_item["results"].items():
                rows.append(
                    {
                        "condition": "ldpc_1_2_bpsk",
                        "depth": depth,
                        "snr_db": snr,
                        "source_bits_per_image": depth_item[
                            "source_bits_per_image"
                        ],
                        "source_bpp": depth_item["source_bpp"],
                        "transmission_ratio": depth_item[
                            "rgb_transmission_ratio"
                        ],
                        "ldpc_blocks_per_image": depth_item[
                            "ldpc_blocks_per_image"
                        ],
                        "source_padding_bits_per_image": depth_item[
                            "source_padding_bits_per_image"
                        ],
                        "coded_bits_per_image": depth_item[
                            "coded_bits_per_image"
                        ],
                        "coded_bpp": depth_item["coded_bpp"],
                        "modulated_symbols_per_image": depth_item[
                            "modulated_symbols_per_image"
                        ],
                        "psnr": metrics["psnr"],
                        "ms_ssim": metrics["ms_ssim"],
                        "final_residual_rms_scale0": residuals[0],
                        "final_residual_rms_scale1": residuals[1],
                    }
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def run_residual_simvq_depth_sweep(
    checkpoint,
    depths,
    channel_depths,
    snrs,
    modulation,
    json_output,
    csv_output,
    no_channel_only=False,
):
    cfg = Config()
    cfg.validate()
    setup_seed(42)
    device = torch.device(cfg.DEVICE)
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    json_output = _require_project_output(json_output)
    csv_output = _require_project_output(csv_output)

    requested_depths = sorted({int(value) for value in depths})
    if not requested_depths or any(
        value not in {1, 2, 3, 4} for value in requested_depths
    ):
        raise ValueError("depths must be selected from 1, 2, 3, and 4")
    channel_depths = sorted({int(value) for value in channel_depths})
    if any(value not in requested_depths for value in channel_depths):
        raise ValueError("channel_depths must be included in depths")

    checkpoint_sha_before = _sha256(checkpoint)
    model, inferred = build_model_from_checkpoint(str(checkpoint), cfg, device)
    checkpoint_depths = [
        int(value) for value in inferred.get("rq_depth_list", [])
    ]
    if inferred.get("quantizer_type") != "residual_simvq":
        raise ValueError(
            "this test path only accepts a Residual-SimVQ checkpoint"
        )
    if not bool(inferred.get("rq_shared_codebook", False)):
        raise ValueError("checkpoint must use shared projected codebooks")

    codebook_sha_before = _projected_codebook_fingerprints(model)
    unique_parameter_count_before = len(
        {id(value) for value in model.parameters()}
    )
    unique_buffer_count_before = len({id(value) for value in model.buffers()})
    state_keys_before = tuple(model.state_dict())
    extension = extend_residual_simvq_depth_for_eval(
        model, max(requested_depths)
    )

    print("=" * 80)
    print("Residual-SimVQ zero-training shared-codebook depth 1-4 sweep")
    print(f"Checkpoint: {checkpoint}")
    print(f"Checkpoint trained depth: {checkpoint_depths}")
    print(f"Runtime depths: {requested_depths}")
    print(
        "Extension: change rq_depth only; reuse the exact same frozen embed "
        "and learned projection at every residual depth"
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    print(f"Dataset: {cfg.TEST_DATASET_PATH}")
    print("=" * 80)

    loader = get_dataloader(
        root_dir=cfg.TEST_DATASET_PATH,
        batch_size=1,
        shuffle=False,
        mode="test",
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    no_channel = evaluate_residual_simvq_no_channel_depth_sweep(
        model,
        loader,
        device,
        depths=requested_depths,
        num_embeddings_list=inferred["num_embeddings_list"],
        ldpc_rate=LDPC_RATE,
        modulation_bits=MODULATION_BITS[modulation],
    )

    print("\nNo-channel fixed-depth results")
    print(
        f"{'Depth':>5} | {'bits/image':>10} | {'BPP':>10} | "
        f"{'PSNR':>10} | {'MS-SSIM':>10} | {'residual RMS s0/s1'}"
    )
    for item in no_channel["depth_results"]:
        residuals = item["final_residual_rms_per_scale"]
        print(
            f"{item['depth']:>5d} | {item['source_bits_per_image']:>10d} | "
            f"{item['source_bpp']:>10.8f} | {item['psnr']:>10.6f} | "
            f"{item['ms_ssim']:>10.6f} | "
            f"{residuals[0]:.8f}/{residuals[1]:.8f}"
        )

    channel = None
    if not no_channel_only:
        from communications.ldpc_coding import get_ldpc_code

        ldpc_k = int(LDPC_N * LDPC_RATE)
        ldpc_code = get_ldpc_code(ldpc_k, rate=LDPC_RATE)
        no_channel_by_depth = {
            int(item["depth"]): item
            for item in no_channel["depth_results"]
        }
        channel_results = []
        for depth in channel_depths:
            print(
                f"\nLDPC 1/2 + {modulation.upper()} at Residual-SimVQ "
                f"depth {depth}"
            )
            set_residual_simvq_depth_for_eval(model, depth)
            metrics_by_snr = {}
            for snr in snrs:
                mean_ms_ssim, mean_psnr = evaluate_ldpc_channel(
                    model,
                    loader,
                    inferred["num_embeddings_list"],
                    int(snr),
                    ldpc_code,
                    device,
                    modulation=modulation,
                )
                metrics_by_snr[str(int(snr))] = {
                    "psnr": float(mean_psnr),
                    "ms_ssim": float(mean_ms_ssim),
                }
                print(
                    f"Depth {depth} | SNR {int(snr):>2d} dB | "
                    f"PSNR {mean_psnr:.6f} dB | "
                    f"MS-SSIM {mean_ms_ssim:.6f}"
                )

            rate = no_channel_by_depth[depth]
            source_bits = int(rate["source_bits_per_image"])
            ldpc_blocks = (source_bits + ldpc_k - 1) // ldpc_k
            coded_bits = ldpc_blocks * LDPC_N
            channel_results.append(
                {
                    "depth": depth,
                    "rq_depth_list": [depth] * len(checkpoint_depths),
                    "source_bits_per_image": source_bits,
                    "source_bpp": rate["source_bpp"],
                    "ldpc_blocks_per_image": ldpc_blocks,
                    "source_padding_bits_per_image": (
                        ldpc_blocks * ldpc_k - source_bits
                    ),
                    "coded_bits_per_image": coded_bits,
                    "coded_bpp": coded_bits / float(256 * 256),
                    "modulated_symbols_per_image": coded_bits
                    / MODULATION_BITS[modulation],
                    "rgb_transmission_ratio": rate[
                        "ldpc_bpsk_rgb_transmission_ratio"
                    ],
                    "results": metrics_by_snr,
                }
            )
        channel = {
            "ldpc_n": LDPC_N,
            "ldpc_k": ldpc_k,
            "ldpc_rate": LDPC_RATE,
            "modulation": modulation,
            "snrs_db": [int(value) for value in snrs],
            "depth_results": channel_results,
        }

    set_residual_simvq_depth_for_eval(model, checkpoint_depths)
    codebook_sha_after = _projected_codebook_fingerprints(model)
    checkpoint_sha_after = _sha256(checkpoint)
    if checkpoint_sha_before != checkpoint_sha_after:
        raise RuntimeError("source checkpoint changed during evaluation")
    if codebook_sha_before != codebook_sha_after:
        raise RuntimeError(
            "projected codebook weights changed during eval-only depth sweep"
        )
    if len({id(value) for value in model.parameters()}) != (
        unique_parameter_count_before
    ):
        raise RuntimeError("depth sweep changed the unique parameter count")
    if len({id(value) for value in model.buffers()}) != (
        unique_buffer_count_before
    ):
        raise RuntimeError("depth sweep changed the unique buffer count")
    if tuple(model.state_dict()) != state_keys_before:
        raise RuntimeError("depth sweep changed state-dict keys")

    depth2_result = next(
        (item for item in no_channel["depth_results"] if item["depth"] == 2),
        None,
    )
    payload = {
        "schema_version": 1,
        "method": {
            "name": "residual_simvq_zero_training_shared_codebook_depth_sweep",
            "zero_training": True,
            "saved_migrated_checkpoint": False,
            "adaptive_stop_used": False,
            "expansion_mode": (
                "change_rq_depth_only_reuse_frozen_embed_and_learned_projection"
            ),
            "warning": (
                "Only depths 1 and 2 contributed to training. Depths 3 and 4 "
                "and their summed decoder latents are out of training "
                "distribution, so quality is not guaranteed to improve."
            ),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256_before": checkpoint_sha_before,
            "sha256_after": checkpoint_sha_after,
            "unchanged": checkpoint_sha_before == checkpoint_sha_after,
            "quantizer_type": inferred["quantizer_type"],
            "num_embeddings_list": inferred["num_embeddings_list"],
            "embedding_dim_list": inferred["embedding_dim_list"],
            "checkpoint_rq_depth_list": checkpoint_depths,
            "shared_codebook": inferred.get("rq_shared_codebook"),
        },
        "runtime_extension": {
            **extension,
            "tested_depths": requested_depths,
            "channel_tested_depths": (
                [] if no_channel_only else channel_depths
            ),
            "restored_rq_depth_list": list(model.rq_depth_list),
            "unique_parameter_count_before": unique_parameter_count_before,
            "unique_parameter_count_after": len(
                {id(value) for value in model.parameters()}
            ),
            "unique_buffer_count_before": unique_buffer_count_before,
            "unique_buffer_count_after": len(
                {id(value) for value in model.buffers()}
            ),
            "state_dict_key_count_before": len(state_keys_before),
            "state_dict_key_count_after": len(tuple(model.state_dict())),
            "projected_codebook_sha256_before": codebook_sha_before,
            "projected_codebook_sha256_after": codebook_sha_after,
            "projected_codebook_weights_unchanged": (
                codebook_sha_before == codebook_sha_after
            ),
        },
        "evaluation": {
            "seed": 42,
            "dataset": str(Path(cfg.TEST_DATASET_PATH).resolve()),
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "depth2_legacy_no_channel_reference_check": depth2_result,
            "no_channel": no_channel,
            "channel": channel,
        },
    }

    json_output.parent.mkdir(parents=True, exist_ok=True)
    with open(json_output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    _write_csv(csv_output, no_channel, channel)
    print(f"\nJSON: {json_output}")
    print(f"CSV:  {csv_output}")
    print(f"Checkpoint unchanged: {checkpoint_sha_after}")
    return payload


def _parse_args():
    default_checkpoint = (
        PROJECT_ROOT
        / "checkpoints"
        / "quality_v2_B_larger_rate047_residual_simvq_unet2_ds8x2_k4-2_d2-2"
        / "best_vq_deepsc.pth"
    )
    default_output_dir = (
        PROJECT_ROOT
        / "experiments"
        / "depth_extension_eval"
        / "quality_v2_B_larger_rate047_residual_simvq_unet2_ds8x2_k4-2_d2-2"
        / "zero_training_shared_codebook_d1to4_all_depths_ldpc_bpsk"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Strictly load the trained Residual-SimVQ depth-2 checkpoint and "
            "evaluate fixed shared-codebook residual depths 1 through 4."
        )
    )
    parser.add_argument("--checkpoint", default=str(default_checkpoint))
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument(
        "--channel-depths", type=int, nargs="+", default=[1, 2, 3, 4]
    )
    parser.add_argument("--snrs", type=int, nargs="+", default=[0, 3, 6, 9, 12])
    parser.add_argument(
        "--modulation", choices=["bpsk", "qpsk", "16qam"], default="bpsk"
    )
    parser.add_argument(
        "--no-channel-only",
        action="store_true",
        help="Skip LDPC/modulation and only run the no-channel depth sweep.",
    )
    parser.add_argument(
        "--json-output", default=str(default_output_dir / "results.json")
    )
    parser.add_argument(
        "--csv-output", default=str(default_output_dir / "results.csv")
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_residual_simvq_depth_sweep(
        checkpoint=args.checkpoint,
        depths=args.depths,
        channel_depths=args.channel_depths,
        snrs=args.snrs,
        modulation=args.modulation,
        json_output=args.json_output,
        csv_output=args.csv_output,
        no_channel_only=args.no_channel_only,
    )
