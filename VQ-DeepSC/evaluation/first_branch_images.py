"""Image export for the first-branch experiment, using the existing evaluators."""

from fractions import Fraction
from pathlib import Path

import torch
from PIL import Image

from data.datasets import get_dataloader
from evaluation.quality import evaluate_ldpc_channel, evaluate_no_channel
from test_real import load_model, setup_seed


def reconstruction_directory(root, modulation="bpsk", snr=None, ldpc_code=None):
    root = Path(root)
    if ldpc_code is None:
        return root / "no_channel"
    rate = Fraction(int(ldpc_code["k"]), int(ldpc_code["n"]))
    block_suffix = "" if ldpc_code["n"] == 256 else f"_n{ldpc_code['n']}"
    return root / (
        f"{modulation}_ldpc{rate.numerator}{rate.denominator}"
        f"{block_suffix}_snr{snr}"
    )


class ReconstructionWriter(torch.nn.Module):
    """Save the exact reconstructions passed back to the quality evaluator."""

    def __init__(self, model, filenames, output_dir):
        super().__init__()
        self.model = model
        self.filenames = [Path(name).stem + ".png" for name in filenames]
        if len(set(self.filenames)) != len(self.filenames):
            raise ValueError("Reconstruction filenames must have unique stems")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.saved_count = 0

    def forward_test(self, images):
        return self.model.forward_test(images)

    def reconstruct_from_indices(self, indices):
        reconstructed = self.model.reconstruct_from_indices(indices)
        if self.saved_count + len(reconstructed) > len(self.filenames):
            raise RuntimeError("More reconstructions than source filenames")
        for image in reconstructed:
            pixels = (
                ((image.detach().cpu() + 1) * 127.5)
                .clamp(0, 255)
                .round()
                .to(torch.uint8)
                .permute(1, 2, 0)
                .contiguous()
                .numpy()
            )
            Image.fromarray(pixels).save(
                self.output_dir / self.filenames[self.saved_count], format="PNG"
            )
            self.saved_count += 1
        return reconstructed

    def assert_complete(self):
        if self.saved_count != len(self.filenames):
            raise RuntimeError(
                f"Saved {self.saved_count} of {len(self.filenames)} reconstructions"
            )


@torch.no_grad()
def evaluate_with_images(
    *,
    cfg,
    image_output_dir,
    checkpoint_path=None,
    test_snrs=None,
    no_channel=False,
    modulation="bpsk",
    ldpc_n=256,
    ldpc_rate=0.5,
    stream_packing="combined",
):
    cfg.validate()
    setup_seed(42)
    device = torch.device(cfg.DEVICE)
    checkpoint_path = checkpoint_path or str(
        Path(cfg.CHECKPOINT_DIR) / "best_vq_deepsc.pth"
    )
    model = load_model(checkpoint_path, cfg, device)
    loader = get_dataloader(
        root_dir=cfg.TEST_DATASET_PATH,
        batch_size=cfg.TEST_BATCH_SIZE,
        shuffle=False,
        mode="test",
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    filenames = list(loader.dataset.image_files)
    output_root = Path(image_output_dir).resolve()
    ldpc_code = None
    if not no_channel:
        from communications.ldpc_coding import get_ldpc_code

        ldpc_code = get_ldpc_code(int(ldpc_n * ldpc_rate), rate=ldpc_rate)

    results = {}
    for snr in ([None] if no_channel else (test_snrs or [6])):
        output_dir = reconstruction_directory(output_root, modulation, snr, ldpc_code)
        writer = ReconstructionWriter(model, filenames, output_dir)
        if no_channel:
            ms_ssim, psnr, diagnostics = evaluate_no_channel(
                writer, loader, cfg.NUM_EMBEDDINGS_LIST, device,
                return_diagnostics=True,
            )
        else:
            ms_ssim, psnr, diagnostics = evaluate_ldpc_channel(
                writer, loader, cfg.NUM_EMBEDDINGS_LIST, snr, ldpc_code, device,
                modulation=modulation,
                return_diagnostics=True,
                stream_packing=stream_packing,
            )
        writer.assert_complete()
        results["no_channel" if no_channel else str(snr)] = {
            "ms_ssim": ms_ssim,
            "psnr": psnr,
            "diagnostics": diagnostics,
            "image_output_dir": str(output_dir),
            "num_saved_images": writer.saved_count,
        }
    return results
