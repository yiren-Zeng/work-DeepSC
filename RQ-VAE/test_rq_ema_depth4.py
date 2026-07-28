"""Zero-training shared-codebook depth extension evaluation.

The original depth-2 checkpoint is loaded strictly through the normal project
loader.  Only after loading, the existing per-scale EMA codebook object is
referenced by additional RQ stages in memory.  Nothing is trained or saved as
a new checkpoint.
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
from evaluation.rq_depth_extension import (
    evaluate_no_channel_depth_sweep,
    extend_shared_rq_depth_for_eval,
    set_shared_rq_depth_for_eval,
)
from utils.checkpoint_utils import build_model_from_checkpoint
from utils.reproducibility import setup_seed


PROJECT_ROOT = Path(__file__).resolve().parent
LDPC_N = 256
LDPC_RATE = 0.5


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


def _codebook_fingerprints(model):
    fingerprints = []
    for quantizer in model.vector_quantizers:
        weight = (
            quantizer.transformed_weight()
            .detach()
            .to(device="cpu")
            .contiguous()
            .numpy()
        )
        fingerprints.append(hashlib.sha256(weight.tobytes()).hexdigest())
    return fingerprints


def _write_csv(path, no_channel_results, channel_results):
    fieldnames = [
        "condition",
        "depth",
        "rq_depth_list",
        "snr_db",
        "source_bits_per_image",
        "source_bpp",
        "ldpc_bpsk_rgb_transmission_ratio",
        "psnr",
        "ms_ssim",
        "final_residual_rms_scale0",
        "final_residual_rms_scale1",
    ]
    rows = []
    by_depth = {
        int(item["depth"]): item for item in no_channel_results["depth_results"]
    }
    for item in no_channel_results["depth_results"]:
        residuals = item["final_residual_rms_per_scale"]
        rows.append(
            {
                "condition": "no_channel",
                "depth": item["depth"],
                "rq_depth_list": ",".join(
                    str(value) for value in item["rq_depth_list"]
                ),
                "snr_db": "",
                "source_bits_per_image": item["source_bits_per_image"],
                "source_bpp": item["source_bpp"],
                "ldpc_bpsk_rgb_transmission_ratio": item[
                    "ldpc_bpsk_rgb_transmission_ratio"
                ],
                "psnr": item["psnr"],
                "ms_ssim": item["ms_ssim"],
                "final_residual_rms_scale0": residuals[0],
                "final_residual_rms_scale1": residuals[1],
            }
        )
    if channel_results:
        target_depth = int(channel_results["target_depth"])
        rate = by_depth[target_depth]
        residuals = rate["final_residual_rms_per_scale"]
        for snr, metrics in channel_results["results"].items():
            rows.append(
                {
                    "condition": "ldpc_1_2_bpsk",
                    "depth": target_depth,
                    "rq_depth_list": ",".join(
                        str(value) for value in rate["rq_depth_list"]
                    ),
                    "snr_db": snr,
                    "source_bits_per_image": rate["source_bits_per_image"],
                    "source_bpp": rate["source_bpp"],
                    "ldpc_bpsk_rgb_transmission_ratio": rate[
                        "ldpc_bpsk_rgb_transmission_ratio"
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
def run_depth_extension_test(
    checkpoint,
    depths,
    target_depth,
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

    checkpoint_sha_before = _sha256(checkpoint)
    model, inferred = build_model_from_checkpoint(str(checkpoint), cfg, device)
    checkpoint_depths = [
        int(value) for value in inferred.get("rq_depth_list", [])
    ]
    if inferred.get("quantizer_type") != "rq_ema":
        raise ValueError("this test path only accepts an EMA-RQ checkpoint")
    if not bool(inferred.get("rq_shared_codebook", False)):
        raise ValueError("checkpoint must use shared EMA-RQ codebooks")

    requested_depths = sorted({int(value) for value in depths})
    target_depth = int(target_depth)
    if target_depth not in requested_depths:
        raise ValueError("target_depth must be included in depths")
    maximum_depth = max(requested_depths)

    codebook_sha_before = _codebook_fingerprints(model)
    unique_parameter_count_before = len({id(value) for value in model.parameters()})
    unique_buffer_count_before = len({id(value) for value in model.buffers()})
    extension = extend_shared_rq_depth_for_eval(model, maximum_depth)
    unique_parameter_count_after = len({id(value) for value in model.parameters()})
    unique_buffer_count_after = len({id(value) for value in model.buffers()})

    print("=" * 78)
    print("EMA-RQ zero-training shared-codebook depth extension")
    print(f"Checkpoint: {checkpoint}")
    print(f"Checkpoint depth: {checkpoint_depths}")
    print(f"Runtime maximum depth: {[maximum_depth] * len(checkpoint_depths)}")
    print(
        "Extension: reuse the exact same trained codebook object at every "
        "additional depth"
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    print(f"Dataset: {cfg.TEST_DATASET_PATH}")
    print("=" * 78)

    loader = get_dataloader(
        root_dir=cfg.TEST_DATASET_PATH,
        batch_size=1,
        shuffle=False,
        mode="test",
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    no_channel = evaluate_no_channel_depth_sweep(
        model,
        loader,
        device,
        depths=requested_depths,
        num_embeddings_list=inferred["num_embeddings_list"],
        ldpc_rate=LDPC_RATE,
        modulation_bits={"bpsk": 1, "qpsk": 2, "16qam": 4}[modulation],
    )

    print("\nNo-channel fixed-depth prefix results")
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
        if modulation != "bpsk":
            print(
                f"\nRunning LDPC 1/2 + {modulation.upper()} at depth "
                f"{target_depth}"
            )
        else:
            print(f"\nRunning LDPC 1/2 + BPSK at depth {target_depth}")
        from communications.ldpc_coding import get_ldpc_code

        ldpc_code = get_ldpc_code(int(LDPC_N * LDPC_RATE), rate=LDPC_RATE)
        set_shared_rq_depth_for_eval(model, target_depth)
        channel_metrics = {}
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
            channel_metrics[str(int(snr))] = {
                "psnr": float(mean_psnr),
                "ms_ssim": float(mean_ms_ssim),
            }
            print(
                f"SNR {int(snr):>2d} dB | PSNR {mean_psnr:.6f} dB | "
                f"MS-SSIM {mean_ms_ssim:.6f}"
            )
        target_rate = next(
            item
            for item in no_channel["depth_results"]
            if int(item["depth"]) == target_depth
        )
        source_bits_per_image = int(target_rate["source_bits_per_image"])
        ldpc_k = int(LDPC_N * LDPC_RATE)
        ldpc_blocks_per_image = (
            source_bits_per_image + ldpc_k - 1
        ) // ldpc_k
        coded_bits_per_image = ldpc_blocks_per_image * LDPC_N
        channel = {
            "target_depth": target_depth,
            "rq_depth_list": [target_depth] * len(checkpoint_depths),
            "ldpc_n": LDPC_N,
            "ldpc_k": ldpc_k,
            "ldpc_rate": LDPC_RATE,
            "modulation": modulation,
            "source_bits_per_image": source_bits_per_image,
            "ldpc_blocks_per_image": ldpc_blocks_per_image,
            "source_padding_bits_per_image": (
                ldpc_blocks_per_image * ldpc_k - source_bits_per_image
            ),
            "coded_bits_per_image": coded_bits_per_image,
            "coded_bpp": coded_bits_per_image / float(256 * 256),
            "modulated_symbols_per_image": coded_bits_per_image
            / {"bpsk": 1, "qpsk": 2, "16qam": 4}[modulation],
            "rgb_transmission_ratio": target_rate[
                "ldpc_bpsk_rgb_transmission_ratio"
            ],
            "snrs_db": [int(value) for value in snrs],
            "results": channel_metrics,
        }

    codebook_sha_after = _codebook_fingerprints(model)
    checkpoint_sha_after = _sha256(checkpoint)
    if checkpoint_sha_before != checkpoint_sha_after:
        raise RuntimeError("source checkpoint changed during evaluation")
    if codebook_sha_before != codebook_sha_after:
        raise RuntimeError("EMA codebook weights changed during eval-only extension")
    if unique_parameter_count_before != unique_parameter_count_after:
        raise RuntimeError("depth extension created new unique parameters")
    if unique_buffer_count_before != unique_buffer_count_after:
        raise RuntimeError("depth extension created new unique buffers")

    depth2_result = next(
        (item for item in no_channel["depth_results"] if item["depth"] == 2),
        None,
    )
    payload = {
        "schema_version": 1,
        "method": {
            "name": "zero_training_shared_codebook_depth_extension",
            "zero_training": True,
            "saved_migrated_checkpoint": False,
            "expansion_mode": "same_codebook_object_alias_per_scale",
            "adaptive_stop_used": False,
            "warning": (
                "Depths above 2 were not trained. Their residual distribution "
                "and the resulting summed decoder latent are out of training "
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
            "unique_parameter_count_before": unique_parameter_count_before,
            "unique_parameter_count_after": unique_parameter_count_after,
            "unique_buffer_count_before": unique_buffer_count_before,
            "unique_buffer_count_after": unique_buffer_count_after,
            "codebook_sha256_before": codebook_sha_before,
            "codebook_sha256_after": codebook_sha_after,
            "codebook_weights_unchanged": codebook_sha_before
            == codebook_sha_after,
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
        / "quality_v2_B_larger_rate047_rq_ema_unet2_ds8x2_k4-2_d2-2"
        / "best_vq_deepsc.pth"
    )
    default_output_dir = (
        PROJECT_ROOT
        / "experiments"
        / "depth_extension_eval"
        / "quality_v2_B_larger_rate047_rq_ema_unet2_ds8x2_k4-2_d2-2"
        / "zero_training_shared_codebook_d1to4"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Strictly load the trained EMA-RQ depth-2 checkpoint, then test "
            "shared-codebook depth prefixes through an eval-only in-memory path."
        )
    )
    parser.add_argument("--checkpoint", default=str(default_checkpoint))
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--target-depth", type=int, default=4)
    parser.add_argument("--snrs", type=int, nargs="+", default=[0, 3, 6, 9, 12])
    parser.add_argument(
        "--modulation", choices=["bpsk", "qpsk", "16qam"], default="bpsk"
    )
    parser.add_argument(
        "--no-channel-only",
        action="store_true",
        help="Skip LDPC/modulation and only run the fixed-depth prefix sweep.",
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
    run_depth_extension_test(
        checkpoint=args.checkpoint,
        depths=args.depths,
        target_depth=args.target_depth,
        snrs=args.snrs,
        modulation=args.modulation,
        json_output=args.json_output,
        csv_output=args.csv_output,
        no_channel_only=args.no_channel_only,
    )
