import cv2
import numpy as np
import torch
from scipy.signal import convolve2d


def _gaussian_kernel(size, sigma):
    axis = np.arange(-size // 2 + 1.0, size // 2 + 1.0)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)


def ssim(image1, image2, k1=0.01, k2=0.03, win_size=11, data_range=255):
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    kernel = _gaussian_kernel(win_size, 1.5)
    ssim_maps = []
    contrast_maps = []

    for channel in range(image1.shape[2]):
        channel1 = image1[:, :, channel]
        channel2 = image2[:, :, channel]
        mean1 = convolve2d(channel1, kernel, mode="same", boundary="symm")
        mean2 = convolve2d(channel2, kernel, mode="same", boundary="symm")
        mean1_squared = mean1**2
        mean2_squared = mean2**2
        mean_product = mean1 * mean2
        variance1 = (
            convolve2d(channel1 * channel1, kernel, mode="same", boundary="symm")
            - mean1_squared
        )
        variance2 = (
            convolve2d(channel2 * channel2, kernel, mode="same", boundary="symm")
            - mean2_squared
        )
        covariance = (
            convolve2d(channel1 * channel2, kernel, mode="same", boundary="symm")
            - mean_product
        )
        variance1 = np.maximum(variance1, 1e-10)
        variance2 = np.maximum(variance2, 1e-10)

        numerator = (2 * mean_product + c1) * (2 * covariance + c2)
        denominator = (mean1_squared + mean2_squared + c1) * (
            variance1 + variance2 + c2
        )
        ssim_map = np.ones_like(numerator)
        valid = denominator > 1e-10
        ssim_map[valid] = numerator[valid] / denominator[valid]
        ssim_maps.append(np.nan_to_num(ssim_map, nan=1.0))

        contrast_denominator = variance1 + variance2 + c2
        contrast_map = np.ones_like(covariance)
        valid_contrast = contrast_denominator > 1e-10
        contrast_map[valid_contrast] = (2 * covariance + c2)[valid_contrast] / (
            contrast_denominator[valid_contrast]
        )
        contrast_maps.append(np.clip(np.nan_to_num(contrast_map, nan=1.0), 0, 1))

    return np.mean(ssim_maps, axis=0), np.mean(contrast_maps, axis=0)


def ms_ssim(image1, image2, max_level=5):
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)
    if not np.isfinite(image1).all() or not np.isfinite(image2).all():
        return 0.0

    data_range = 255 if image1.max() > 1.0 else 1.0
    ssim_values = []
    contrast_values = []
    for level in range(min(max_level, len(weights))):
        if min(image1.shape[0], image1.shape[1]) < 11:
            break
        ssim_map, contrast_map = ssim(image1, image2, data_range=data_range)
        ssim_values.append(float(np.nan_to_num(np.mean(ssim_map), nan=1.0)))
        contrast_values.append(float(np.nan_to_num(np.mean(contrast_map), nan=1.0)))
        if level < max_level - 1:
            image1 = cv2.pyrDown(image1.astype(np.float32)).astype(np.float64)
            image2 = cv2.pyrDown(image2.astype(np.float32)).astype(np.float64)

    if not contrast_values:
        return 0.0
    actual_weights = weights[: len(contrast_values)]
    actual_weights = actual_weights / actual_weights.sum()
    if len(contrast_values) == 1:
        value = ssim_values[0]
    else:
        value = np.prod(
            np.asarray(contrast_values[:-1]) ** actual_weights[:-1]
        )
        value *= ssim_values[-1] ** actual_weights[-1]
    return float(np.clip(np.nan_to_num(value, nan=0.0), 0.0, 1.0))


def calculate_ms_ssim(image1, image2):
    try:
        if image1 is None or image2 is None:
            return 0.0
        if image1.dim() == 3:
            image1 = image1.unsqueeze(0)
        if image2.dim() == 3:
            image2 = image2.unsqueeze(0)
        if image1.shape != image2.shape:
            return 0.0
        if not torch.isfinite(image1).all() or not torch.isfinite(image2).all():
            return 0.0
        array1 = image1[0].permute(1, 2, 0).cpu().numpy()
        array2 = image2[0].permute(1, 2, 0).cpu().numpy()
        array1 = np.clip(array1, 0, 1)
        array2 = np.clip(array2, 0, 1)
        return ms_ssim(array1, array2)
    except Exception as error:
        print(f"Error in calculate_ms_ssim: {error}")
        return 0.0
