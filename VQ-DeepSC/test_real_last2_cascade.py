import argparse
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from io import StringIO

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import torch

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

    full_codebook_label = payload.get("full_num_embeddings_list")
    if (
        not isinstance(full_codebook_label, (list, tuple))
        or len(full_codebook_label) != 4
        or list(full_codebook_label[-2:]) != list(cfg.NUM_EMBEDDINGS_LIST)
    ):
        raise ValueError(
            "Checkpoint historical label is inconsistent with its two active codebooks"
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
    model = build_model(cfg, device, full_codebook_label)
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


def _print_results(cfg, results, modulation, no_channel):
    first_metrics = next(iter(results.values()))
    compression_rate = first_metrics["compression_rate"]

    print("=" * 56)
    print("VQ-DeepSC 最后两层级联测试结果")
    print(f"码本: {cfg.NUM_EMBEDDINGS_LIST}")
    print(f"调制方式: {'无信道' if no_channel else modulation.upper()}")
    print(f"压缩率: {compression_rate}")
    print("-" * 56)
    print(f"{'SNR':<12}{'PSNR (dB)':<18}{'MS-SSIM':<16}")
    print("-" * 56)
    for condition, metrics in results.items():
        snr_label = "No-channel" if condition == "no_channel" else f"{condition} dB"
        print(
            f"{snr_label:<12}"
            f"{metrics['psnr']:<18.4f}"
            f"{metrics['ms_ssim']:<16.4f}"
        )
    print("=" * 56)


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
):
    cfg = Last2CascadeTestConfig()
    cfg.validate()
    setup_seed(42)
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
            "psnr": psnr,
            "compression_rate": _compression_rate_fraction(diagnostics),
            "diagnostics": diagnostics,
        }
    else:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            from communications.ldpc_coding import get_ldpc_code

        ldpc_code = get_ldpc_code(int(ldpc_n * ldpc_rate), rate=ldpc_rate)
        for snr in test_snrs:
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
            )
            results[str(snr)] = {
                "ms_ssim": ms_ssim,
                "psnr": psnr,
                "compression_rate": _compression_rate_fraction(diagnostics),
                "diagnostics": diagnostics,
            }

    _print_results(cfg, results, modulation, no_channel)

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
    parser.add_argument("--snrs", type=int, nargs="+", default=[6])
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
    )
