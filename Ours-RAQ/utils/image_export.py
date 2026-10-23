"""Save evaluation reconstructions without changing the tensors used for metrics."""

from pathlib import Path

import torch
from PIL import Image


class ReconstructionImageWriter:
    """Write batches in evaluation order, using source filenames when available."""

    def __init__(self, output_dir, image_names=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.image_names = list(image_names) if image_names is not None else None
        self.saved_count = 0
        self._used_names = set()

    def __call__(self, reconstructed_images):
        batch_size = reconstructed_images.shape[0]
        if self.image_names is not None and self.saved_count + batch_size > len(self.image_names):
            raise ValueError("More reconstructions than source image filenames.")
        # Model output is in [-1, 1]; PNG stores clamped, rounded 8-bit RGB.
        pixels = (
            ((reconstructed_images.detach().float().cpu() + 1) / 2)
            .clamp(0, 1)
            .mul(255)
            .round()
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .numpy()
        )
        for image in pixels:
            index = self.saved_count
            stem = (
                Path(self.image_names[index]).stem
                if self.image_names is not None
                else f"image_{index + 1:06d}"
            )
            filename = f"{stem}.png"
            suffix = 1
            while filename in self._used_names:
                filename = f"{stem}_{suffix}.png"
                suffix += 1
            Image.fromarray(image).save(self.output_dir / filename)
            self._used_names.add(filename)
            self.saved_count += 1
