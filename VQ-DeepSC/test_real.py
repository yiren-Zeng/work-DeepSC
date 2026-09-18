import argparse
import json
import os
import random

import numpy as np
import torch

from config import Config
from data.datasets import get_dataloader
from evaluation.quality import evaluate_ldpc_channel, evaluate_no_channel
from models.deepsc import DeepSC


def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(cfg, device):
    return DeepSC(
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        num_downsample_blocks=cfg.NUM_DOWNSAMPLE_BLOCKS,
        base_channels=cfg.BASE_CHANNELS,
        num_embeddings_list=cfg.NUM_EMBEDDINGS_LIST,
        embedding_dim_list=cfg.EMBEDDING_DIM_LIST,
        commitment_cost=cfg.COMMITMENT_COST,
    ).to(device)


def load_model(checkpoint_path, cfg, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    inferred_sizes = [
        int(state_dict[f"vector_quantizers.{scale}.embedding.weight"].shape[0])
        for scale in range(cfg.NUM_DOWNSAMPLE_BLOCKS)
    ]
    if inferred_sizes != list(cfg.NUM_EMBEDDINGS_LIST):
        raise ValueError(
            f"Checkpoint codebooks {inferred_sizes} do not match configured "
            f"codebooks {cfg.NUM_EMBEDDINGS_LIST}"
        )
    model = build_model(cfg, device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def print_diagnostics(diagnostics):
    print(f"Stream packing: {diagnostics['stream_packing']}")
    for scale in diagnostics.get("per_scale", []):
        message = (
            f"Scale {scale['scale']}: K={scale['num_embeddings']}, "
            f"bits/index={scale['bits_per_index']}, "
            f"payload={scale['payload_bits']} bits, "
            f"payload_bpp={scale['payload_bpp']:.8f}"
        )
        if scale.get("ber") is not None:
            message += (
                f", BER={scale['ber']:.8f}, "
                f"index_error_rate={scale['index_error_rate']:.8f}"
            )
        print(message)
    combined = diagnostics.get("combined_stream")
    if combined:
        print(
            "Combined stream: "
            f"payload={combined['payload_bits']} bits, "
            f"LDPC-padding={combined['ldpc_padding_bits']} bits, "
            f"coded={combined['coded_bits']} bits, "
            f"transmitted={combined['transmitted_bits']} bits, "
            f"symbols={combined['channel_symbols']}, "
            f"transmission_ratio={combined['transmission_ratio']:.8f}, "
            f"BER={combined['ber']:.8f}, "
            f"index_error_rate={combined['index_error_rate']:.8f}"
        )


@torch.no_grad()
def test_real(
    checkpoint_path=None,
    test_snrs=None,
    json_output=None,
    no_channel=False,
    modulation="qpsk",
    ldpc_n=256,
    ldpc_rate=0.5,
    stream_packing="combined",
):
    cfg = Config()
    cfg.validate()
    setup_seed(42)
    device = torch.device(cfg.DEVICE)
    checkpoint_path = checkpoint_path or os.path.join(
        cfg.CHECKPOINT_DIR, "best_vq_deepsc.pth"
    )
    test_snrs = test_snrs or [6]

    print("=" * 60)
    print("VQ-DeepSC real transmission baseline evaluation")
    print(f"Experiment: {cfg.EXPERIMENT_NAME}")
    print(f"Codebooks: {cfg.NUM_EMBEDDINGS_LIST}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {cfg.TEST_DATASET_PATH}")
    if no_channel:
        print("Link: no-channel reconstruction upper bound")
    else:
        print(f"LDPC: n={ldpc_n}, k={int(ldpc_n * ldpc_rate)}, R={ldpc_rate}")
        print(f"Modulation: {modulation.upper()}")
        print(f"Stream packing: {stream_packing}")
        print(f"SNRs: {test_snrs} dB")
    print("=" * 60)

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
        ms_ssim, psnr, diagnostics = evaluate_no_channel(
            model,
            test_loader,
            cfg.NUM_EMBEDDINGS_LIST,
            device,
            return_diagnostics=True,
        )
        results["no_channel"] = {
            "ms_ssim": ms_ssim,
            "psnr": psnr,
            "diagnostics": diagnostics,
        }
        print_diagnostics(diagnostics)
        print(f"No channel | MS-SSIM: {ms_ssim:.4f} | PSNR: {psnr:.4f} dB")
    else:
        from communications.ldpc_coding import get_ldpc_code

        ldpc_code = get_ldpc_code(int(ldpc_n * ldpc_rate), rate=ldpc_rate)
        for snr in test_snrs:
            ms_ssim, psnr, diagnostics = evaluate_ldpc_channel(
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
                "diagnostics": diagnostics,
            }
            print_diagnostics(diagnostics)
            print(f"SNR {snr} dB | MS-SSIM: {ms_ssim:.4f} | PSNR: {psnr:.4f} dB")

    if json_output:
        os.makedirs(os.path.dirname(json_output) or ".", exist_ok=True)
        with open(json_output, "w", encoding="utf-8") as handle:
            json.dump(
                {
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
        print(f"JSON results saved to {json_output}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a VQ-DeepSC checkpoint on Kodak through LDPC + AWGN."
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
    test_real(
        checkpoint_path=arguments.checkpoint,
        test_snrs=arguments.snrs,
        json_output=arguments.json_output,
        no_channel=arguments.no_channel,
        modulation=arguments.modulation,
        ldpc_n=arguments.ldpc_n,
        ldpc_rate=arguments.ldpc_k,
        stream_packing=arguments.stream_packing,
    )
