"""Time three trained schemes on one GPU and a shared communications stack.

Run: python -B experiments/benchmark_equal_conditions.py --gpu 2
No training/model/evaluation source files are changed. Only the JSON result is
written. Target-codebook indices are checked against each native quantizer on
all images before timing. Losses and unused reference reconstructions are not
part of the inference path being timed.
"""

import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = {
    "RAQ-RVQ": ROOT / "RAQ-RVQ/checkpoints/shiyan_independent_raq_rvq_src64-64_trg2-64_d2_curriculum_rate094_A_patch_ch256-512_res6-6_mixture_per_k_unet2_ds8x2_k64/best_vq_deepsc.pth",
    "VQ-DeepSC": Path("/home/yi/vq_deepsc_runs_20260919/checkpoints/k16_16_last2_copy_200ep/best_vq_deepsc.pth"),
    "RAQ-VAE": ROOT / "RAQ-VAE-main/runs/seq2seq_ds8_dualk_src64_20260910_113736_956705/checkpoints/epoch=145-step=78840.ckpt",
}


def project_imports(project):
    # These independent repositories use overlapping top-level package names.
    # Existing classes retain their original module globals after removing the
    # names from the import cache; no model forward imports these packages.
    for name in list(sys.modules):
        if name.split(".")[0] in {"models", "utils", "data"}:
            del sys.modules[name]
    project_paths = {str(ROOT / name) for name in ("RAQ-RVQ", "VQ-DeepSC", "RAQ-VAE-main")}
    sys.path[:] = [path for path in sys.path if path not in project_paths]
    sys.path.insert(0, str(ROOT / project))


def load_models(device):
    project_imports("RAQ-RVQ")
    from models.deepsc import DeepSC
    from utils.checkpoint_utils import load_state_dict_compatible
    from models.independent_raq_rvq import quantize_independent_raq_rvq
    from utils.bit_utils import index_tensor_to_bits, bits_to_index_tensor
    ours = DeepSC(
        3, 3, 2, 128, [64, 64], [256, 512], .25, device,
        strides=[8, 2], skip_dropout_p=[.1], norm_type="group",
        norm_groups=32, activation="silu", encoder_res_blocks=6,
        decoder_res_blocks=6, upsample_mode="bilinear",
        use_cascade_downsample=False, raq_min_trg=2, raq_max_trg=64,
        independent_raq_rvq_depth=2,
        independent_raq_rvq_k_lists=[[8, 2], [8, 2]],
    )
    checkpoint = torch.load(CHECKPOINTS["RAQ-RVQ"], map_location="cpu", weights_only=False)
    load_state_dict_compatible(ours, checkpoint.get("model_state_dict", checkpoint))
    del checkpoint
    ours.to(device).eval()
    print("Loaded RAQ-RVQ", flush=True)

    project_imports("VQ-DeepSC")
    from models.deepsc_last2_cascade import DeepSCLast2Cascade
    vq = DeepSCLast2Cascade(3, 3, 4, 64, [16, 16], [128, 256, 512, 1024], .25)
    checkpoint = torch.load(CHECKPOINTS["VQ-DeepSC"], map_location="cpu", weights_only=False)
    vq.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    del checkpoint
    vq.to(device).eval()
    print("Loaded VQ-DeepSC", flush=True)

    project_imports("RAQ-VAE-main")
    sys.path.insert(0, str(ROOT / "RAQ-VAE-main/.runtime/deps"))
    np.Inf = np.inf  # Compatibility for this repository's Lightning 1.9.
    from nets.model_ds8_dualk import RAQVAE_TWO_DS8_DUALK
    vae = RAQVAE_TWO_DS8_DUALK(SimpleNamespace(
        n_hid=64, embedding_dim=64, num_embeddings=64,
        num_embeddings_min=2, num_embeddings_max=64, num_embeddings_test=16,
        lr=.0005, device=str(device), raq_type="dd", model_type="vqvae2_ds8_dualk",
    ))
    checkpoint = torch.load(CHECKPOINTS["RAQ-VAE"], map_location="cpu", weights_only=False)
    vae.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    vae.to(device).eval()
    print("Loaded RAQ-VAE", flush=True)
    gc.collect()
    return ({"RAQ-RVQ": ours, "VQ-DeepSC": vq, "RAQ-VAE": vae},
            quantize_independent_raq_rvq, index_tensor_to_bits, bits_to_index_tensor)


class Scheme:
    def __init__(self, name, model):
        self.name, self.model = name, model
        self.ks = [[8, 2], [8, 2]] if name == "RAQ-RVQ" else [[16], [16]]
        self.books = self.generate()
        self.features = []
        self.validation = {}

    def generate(self):
        m = self.model
        if self.name == "RAQ-RVQ":
            books = []
            for i in range(2):
                src = m.vector_quantizers[i].transformed_weight()
                books.append([
                    generator.generate_codebook_transformer(k, src)
                    for generator, k in zip((m.raqs[i], m.raqs_rvq_stage2[i]), self.ks[i])
                ])
            return books
        if self.name == "VQ-DeepSC":
            return [[q.embedding.weight] for q in m.vector_quantizers]
        target = torch.arange(16, device=device).unsqueeze(1)
        source = m.src.to(device)
        # Keep both independent calls specified by the dual-K implementation.
        return [[m.cbk2cbk(source, target).squeeze(1)] for _ in range(2)]

    def encode_features(self, x):
        m = self.model
        if self.name == "RAQ-RVQ":
            return m.semantic_encoder(x)
        if self.name == "VQ-DeepSC":
            return m.semantic_encoder(x)[-2:]
        bottom = m.encoder_b(x)
        top = m.encoder_t(bottom)
        return [m.quantize_conv_b(bottom), m.quantize_conv_t(top)]

    def indices(self, features, books):
        outputs = []
        for feature, weights in zip(features, books):
            residual = feature.permute(0, 2, 3, 1).contiguous()
            batch, height, width, dim = residual.shape
            stage_indices = []
            for stage, weight in enumerate(weights):
                flat = residual.reshape(-1, dim)
                if self.name == "RAQ-VAE":
                    # Preserve the original arithmetic order and index rule.
                    dist = (flat.pow(2).sum(1, keepdim=True)
                            - 2 * flat @ weight.t()
                            + weight.pow(2).sum(1, keepdim=True).t())
                    _, index = (-dist).max(1)
                else:
                    dist = (flat.pow(2).sum(1, keepdim=True)
                            + weight.pow(2).sum(1)
                            - 2 * torch.einsum("bd,nd->bn", flat, weight))
                    index = dist.argmin(1)
                stage_indices.append(index.view(batch, height, width))
                if stage + 1 < len(weights):
                    residual = residual - F.embedding(index, weight).view_as(residual)
            outputs.append(stage_indices)
        return outputs

    def reconstruct(self, indices, books):
        features = []
        for stage_indices, stage_books in zip(indices, books):
            feature = F.embedding(stage_indices[0], stage_books[0])
            for index, weight in zip(stage_indices[1:], stage_books[1:]):
                feature = feature + F.embedding(index, weight)
            features.append(feature.permute(0, 3, 1, 2).contiguous())
        if self.name == "RAQ-VAE":
            return self.model.decode(features[1], features[0])
        return self.model.semantic_decoder(features)

    def native_indices(self, features):
        if self.name == "RAQ-RVQ":
            return [native_rvq(q, feature, weights)["indices"] for q, feature, weights in
                    zip(self.model.vector_quantizers, features, self.books)]
        if self.name == "VQ-DeepSC":
            return [[q(feature)[2]] for q, feature in zip(self.model.vector_quantizers, features)]
        return [[self.model.quantize_cbk(feature.permute(0, 2, 3, 1), weights[0])[2]]
                for feature, weights in zip(features, self.books)]

    def roundtrip(self, image, regenerate=False, channel=None):
        features = self.encode_features(image)
        books = self.generate() if regenerate else self.books
        indices = self.indices(features, books)
        if channel:
            streams, layout = [], []
            for stage_indices, stage_ks in zip(indices, self.ks):
                for index, k in zip(stage_indices, stage_ks):
                    bits, shape, _ = pack_bits(index, k)
                    streams.append(bits)
                    layout.append((len(bits), shape, k))
            payload = np.concatenate(streams)
            assert len(payload) == 5120
            decoded = transmit(payload, channel)
            recovered, offset, entry = [], 0, 0
            for stage_ks in self.ks:
                branch = []
                for _ in stage_ks:
                    length, shape, k = layout[entry]
                    branch.append(unpack_bits(decoded[offset:offset + length], shape, k).to(device))
                    offset += length
                    entry += 1
                recovered.append(branch)
            indices = recovered
        return self.reconstruct(indices, books)


def transmit(bits, channel):
    encoded = ldpc_encode(bits, code=ldpc_code)
    symbols = qpsk_modulate(torch.from_numpy(encoded).float().to(device))
    gain = None
    if channel == "awgn":
        received, variance = awgn_channel(symbols, 4.5, return_noise_power=True)
    else:
        received, gain, variance = rician_channel(symbols, 4.5, 10., return_csi=True)
    llr = qpsk_llr(received, 4.5, device, channel_gain=gain, noise_variance=variance)
    return ldpc_decode(llr.detach().cpu().numpy(), code=ldpc_code).astype(np.uint8)


def measure(fn):
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    output = fn()
    torch.cuda.synchronize()
    return (time.perf_counter_ns() - start) / 1e6, output


def statistics(samples):
    values = np.array([sample["ms"] for sample in samples])
    return {"mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)), "std_ms": float(values.std()),
            "n": len(samples), "samples": samples}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("latency_equal_conditions_snr4p5.json"))
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:0")
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    models, native_rvq, pack_bits, unpack_bits = load_models(device)
    # Reset the global settings after importing all repository dependencies.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    sys.path.append(str(ROOT / "RAQ-RVQ"))
    from communications.ldpc_coding import get_ldpc_code, ldpc_encode, ldpc_decode
    from communications.modulation import qpsk_modulate, qpsk_llr
    from communications.channel import awgn_channel, rician_channel
    import tensorflow as tf
    import sionna
    ldpc_code = get_ldpc_code(160, .625)
    images, image_names = [], []
    for path in sorted((ROOT / "Kodak-256-transform-resize").glob("*.png")):
        image = Image.open(path).convert("RGB")
        assert image.size == (256, 256), (str(path), image.size)
        pixels = np.asarray(image).copy()
        images.append((torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.
                       * 2. - 1.).unsqueeze(0).to(device))
        image_names.append(path.name)
    assert len(images) == 24
    with torch.inference_mode():
        schemes = [Scheme(name, model) for name, model in models.items()]
        for scheme in schemes:
            for i, image in enumerate(images):
                features = scheme.encode_features(image)
                indices = scheme.indices(features, scheme.books)
                native = scheme.native_indices(features)
                for branch, native_branch in zip(indices, native):
                    for index, native_index in zip(branch, native_branch):
                        assert torch.equal(index, native_index), (scheme.name, image_names[i])
                reconstructed = scheme.reconstruct(indices, scheme.books)
                assert reconstructed.shape == image.shape and torch.isfinite(reconstructed).all()
                streams = []
                for branch, ks in zip(indices, scheme.ks):
                    for index, k in zip(branch, ks):
                        bits, shape, _ = pack_bits(index, k)
                        assert torch.equal(unpack_bits(bits, shape, k).to(device), index)
                        streams.append(bits)
                assert sum(len(bits) for bits in streams) == 5120
                scheme.features.append(features)
            scheme.validation = {"native_index_match_images": 24, "bit_roundtrip_images": 24,
                                 "output_shape": [1, 3, 256, 256], "finite_outputs": True}
            print(f"Validated {scheme.name}: 24 exact native-index and bit roundtrips", flush=True)

        results = {
            "conditions": {"gpu_index": args.gpu, "gpu_name": torch.cuda.get_device_name(0),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(), "tensorflow": tf.__version__,
                "sionna": sionna.__version__, "dtype": "float32", "tf32": False,
                "amp": False, "compile": False, "cudnn_benchmark": True,
                "torch_threads": 1, "tf_intraop_threads": 1, "tf_interop_threads": 1,
                "images": image_names, "image_shape": [1, 3, 256, 256],
                "normalization": "RGB [0,255] -> [-1,1]",
                "warmup_per_metric_and_scheme": args.warmup, "repeats": args.repeats,
                "snr_db": 4.5, "rician_k_linear": 10, "modulation": "QPSK",
                "ldpc_k": 160, "ldpc_n": 256, "ldpc_rate": .625,
                "ldpc_iterations": int(ldpc_code["decoder"].num_iter.numpy()),
                "ldpc_device": "CPU", "payload_bits": 5120, "ldpc_blocks": 32,
                "coded_bits": 8192, "channel_symbols": 4096,
                "timing": "perf_counter_ns; CUDA synchronize before and after each sample",
                "input_resident_on_gpu": True, "model_order": "rotated each image and repeat",
                "excluded": ["checkpoint load", "disk I/O", "quality metrics", "saving images",
                             "training losses", "source-codebook reference branches"],
                "cache": "both adaptive schemes use fixed pre-generated target codebooks; per-image generation also measured",
                "receiver_codebooks": "shared fixed codebooks; generation included once per image in regenerate metrics",
            },
            "schemes": {s.name: {"checkpoint": str(CHECKPOINTS[s.name]), "target_ks": s.ks,
                         "validation": s.validation, "metrics": {}} for s in schemes},
        }
        metrics = {
            "codebook_generation": lambda s, i: s.generate(),
            "target_quantization": lambda s, i: s.indices(s.features[i], s.books),
            "codec_cached": lambda s, i: s.roundtrip(images[i]),
            "codec_regenerate": lambda s, i: s.roundtrip(images[i], regenerate=True),
            "e2e_awgn_cached": lambda s, i: s.roundtrip(images[i], channel="awgn"),
            "e2e_rician_cached": lambda s, i: s.roundtrip(images[i], channel="rician"),
            "e2e_awgn_regenerate": lambda s, i: s.roundtrip(images[i], regenerate=True, channel="awgn"),
        }
        for metric_index, (metric, operation) in enumerate(metrics.items()):
            print(f"Warming {metric}...", flush=True)
            for i in range(args.warmup):
                for scheme in schemes:
                    output = operation(scheme, i % len(images))
            torch.cuda.synchronize()
            samples = {s.name: [] for s in schemes}
            for repeat in range(args.repeats):
                for i in range(len(images)):
                    # Each scheme sees the same channel RNG seed for this image.
                    order = schemes[(i + repeat + metric_index) % 3:] + schemes[:(i + repeat + metric_index) % 3]
                    for scheme in order:
                        torch.manual_seed(10000 + 100 * repeat + i)
                        ms, output = measure(lambda: operation(scheme, i))
                        samples[scheme.name].append({"image": image_names[i], "repeat": repeat, "ms": ms})
                print(f"Measured {metric} repeat {repeat + 1}/{args.repeats}", flush=True)
            for scheme in schemes:
                stats = statistics(samples[scheme.name])
                results["schemes"][scheme.name]["metrics"][metric] = stats
                print(f"  {scheme.name}: mean={stats['mean_ms']:.3f} ms median={stats['median_ms']:.3f} ms p95={stats['p95_ms']:.3f} ms", flush=True)
            args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print(f"Saved {args.output}", flush=True)
