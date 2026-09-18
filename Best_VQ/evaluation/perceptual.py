"""Reconstruction FID (Inception pool3/2048) and LPIPS (AlexNet v0.1)."""

import numpy as np
import torch


def reconstruction_fid(real_features, reconstructed_features):
    """FID between paired image sets, with unbiased sample covariances."""
    real = np.asarray(real_features, dtype=np.float64)
    reconstructed = np.asarray(reconstructed_features, dtype=np.float64)
    if real.ndim != 2 or real.shape != reconstructed.shape:
        raise ValueError("rFID requires matching [num_images, feature_dim] arrays.")
    if len(real) < 2:
        return None

    real_mean = real.mean(axis=0)
    reconstructed_mean = reconstructed.mean(axis=0)
    if len(real) < real.shape[1]:
        # For small datasets such as Kodak, covariances are rank deficient.
        # Their square-root trace equals the nuclear norm of X @ Y.T / (N-1).
        # This computes the full 2048-dimensional FID without a singular sqrtm.
        x = real - real_mean
        y = reconstructed - reconstructed_mean
        covariance_trace = (np.sum(x * x) + np.sum(y * y)) / (len(real) - 1)
        cross = (x @ y.T) / (len(real) - 1)
        sqrt_trace = np.linalg.svd(cross, compute_uv=False).sum()
        mean_diff = real_mean - reconstructed_mean
        score = mean_diff.dot(mean_diff) + covariance_trace - 2 * sqrt_trace
    else:
        from pytorch_fid.fid_score import calculate_frechet_distance

        score = calculate_frechet_distance(
            real_mean, np.cov(real, rowvar=False),
            reconstructed_mean, np.cov(reconstructed, rowvar=False),
        )
    if not np.isfinite(score):
        raise ValueError("rFID produced a non-finite value.")
    return max(0.0, float(score))


class PerceptualMetrics:
    """Reuse pretrained networks; reset accumulated samples per SNR."""

    def __init__(self, device):
        try:
            import lpips
            from pytorch_fid.inception import InceptionV3
        except ImportError as error:
            raise ImportError(
                "rFID/LPIPS需要额外依赖，请运行: python -m pip install -r requirements-eval.txt"
            ) from error

        self.device = torch.device(device)
        self.inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]])
        self.inception.to(self.device).eval().requires_grad_(False)
        self.lpips = lpips.LPIPS(net="alex", version="0.1", verbose=False)
        self.lpips.to(self.device).eval().requires_grad_(False)
        self.reset()

    def reset(self):
        self.real_features = []
        self.reconstructed_features = []
        self.lpips_sum = 0.0
        self.num_images = 0

    @torch.no_grad()
    def update(self, real_image, reconstructed_image):
        if real_image.shape != reconstructed_image.shape:
            raise ValueError("rFID/LPIPS requires matching original and reconstructed image shapes.")
        real = real_image.to(self.device, dtype=torch.float32).clamp(-1, 1)
        reconstructed = reconstructed_image.to(self.device, dtype=torch.float32).clamp(-1, 1)

        # LPIPS expects RGB in [-1,1]; pytorch-fid expects RGB in [0,1].
        distances = self.lpips(real, reconstructed).flatten()
        self.lpips_sum += float(distances.sum().item())
        self.num_images += int(real.shape[0])
        self.real_features.append(
            self.inception((real + 1) / 2)[0].flatten(1).cpu().numpy()
        )
        self.reconstructed_features.append(
            self.inception((reconstructed + 1) / 2)[0].flatten(1).cpu().numpy()
        )

    def compute(self):
        if not self.num_images:
            raise ValueError("Cannot compute rFID/LPIPS on an empty dataset.")
        return {
            "rfid": reconstruction_fid(
                np.concatenate(self.real_features), np.concatenate(self.reconstructed_features)
            ),
            "lpips": self.lpips_sum / self.num_images,
        }
