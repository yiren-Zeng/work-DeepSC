import numpy as np
import cv2
import torch
from scipy.signal import convolve2d


def _gaussian_kernel(size, sigma):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return kernel / np.sum(kernel)


def ssim(img1, img2, k1=0.01, k2=0.03, win_size=11, L=255):
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    kernel = _gaussian_kernel(win_size, 1.5)

    ssim_maps = []
    cs_maps = []

    for c in range(img1.shape[2]):
        ch1 = img1[:, :, c]
        ch2 = img2[:, :, c]

        mu1 = convolve2d(ch1, kernel, mode='same', boundary='symm')
        mu2 = convolve2d(ch2, kernel, mode='same', boundary='symm')

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = convolve2d(ch1 * ch1, kernel, mode='same', boundary='symm') - mu1_sq
        sigma2_sq = convolve2d(ch2 * ch2, kernel, mode='same', boundary='symm') - mu2_sq
        sigma12 = convolve2d(ch1 * ch2, kernel, mode='same', boundary='symm') - mu1_mu2

        sigma1_sq = np.maximum(sigma1_sq, 1e-10)
        sigma2_sq = np.maximum(sigma2_sq, 1e-10)

        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_map = np.ones_like(numerator)
        valid = denominator > 1e-10
        ssim_map[valid] = numerator[valid] / denominator[valid]
        ssim_map = np.nan_to_num(ssim_map, nan=1.0)

        cs_map = np.ones_like(sigma12)
        cs_denom = sigma1_sq + sigma2_sq + C2
        valid_cs = cs_denom > 1e-10
        cs_map[valid_cs] = (2 * sigma12 + C2)[valid_cs] / cs_denom[valid_cs]
        cs_map = np.nan_to_num(cs_map, nan=1.0)
        cs_map = np.clip(cs_map, 0, 1)

        ssim_maps.append(ssim_map)
        cs_maps.append(cs_map)

    ssim_avg = np.mean(ssim_maps, axis=0)
    cs_avg = np.mean(cs_maps, axis=0)

    ssim_avg = np.nan_to_num(ssim_avg, nan=1.0)
    cs_avg = np.nan_to_num(cs_avg, nan=1.0)

    return ssim_avg, cs_avg


def ms_ssim(img1, img2, max_level=5):
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = min(max_level, len(weights))

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if np.any(np.isnan(img1)) or np.any(np.isnan(img2)) or \
            np.any(np.isinf(img1)) or np.any(np.isinf(img2)):
        return 0.0

    L = 255 if img1.max() > 1.0 else 1.0

    mssim_vals = []
    cs_vals = []

    for i in range(levels):
        if min(img1.shape[0], img1.shape[1]) < 11:
            break

        ssim_map, cs_map = ssim(img1, img2, L=L)

        if np.any(np.isnan(ssim_map)) or np.any(np.isnan(cs_map)):
            ssim_val = 1.0
            cs_val = 1.0
        else:
            ssim_val = np.mean(ssim_map)
            cs_val = np.mean(cs_map)

        ssim_val = np.nan_to_num(ssim_val, nan=1.0)
        cs_val = np.nan_to_num(cs_val, nan=1.0)

        mssim_vals.append(ssim_val)
        cs_vals.append(cs_val)

        if i < levels - 1:
            try:
                img1 = cv2.pyrDown(img1.astype(np.float32)).astype(np.float64)
                img2 = cv2.pyrDown(img2.astype(np.float32)).astype(np.float64)
            except cv2.error as e:
                print(f"Error in pyramid downsampling at level {i}: {e}")
                break

    if len(cs_vals) == 0:
        return 0.0

    cs_vals = np.array(cs_vals)
    mssim_vals = np.array(mssim_vals)
    cs_vals = np.nan_to_num(cs_vals, nan=1.0)
    mssim_vals = np.nan_to_num(mssim_vals, nan=1.0)

    actual_levels = len(cs_vals)
    if actual_levels == 0:
        return 0.0

    adjusted_weights = weights[:actual_levels]
    adjusted_weights = adjusted_weights / np.sum(adjusted_weights)

    if actual_levels == 1:
        overall = mssim_vals[0]
    else:
        overall = np.prod(cs_vals[:-1] ** adjusted_weights[:-1])
        overall *= mssim_vals[-1] ** adjusted_weights[-1]

    overall = np.nan_to_num(overall, nan=0.0)
    return float(np.clip(overall, 0.0, 1.0))


def calculate_ms_ssim(img1, img2):
    try:
        if img1 is None or img2 is None:
            return 0.0

        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)

        if img1.shape != img2.shape:
            print(f"Warning: Image shapes don't match: {img1.shape} vs {img2.shape}")
            return 0.0

        if torch.any(torch.isnan(img1)) or torch.any(torch.isnan(img2)) or \
                torch.any(torch.isinf(img1)) or torch.any(torch.isinf(img2)):
            print("Warning: Input images contain NaN or Inf values")
            return 0.0

        img1 = img1[0]
        img2 = img2[0]
        img1 = img1.permute(1, 2, 0).cpu().numpy()
        img2 = img2.permute(1, 2, 0).cpu().numpy()

        if np.any(np.isnan(img1)) or np.any(np.isnan(img2)) or \
                np.any(np.isinf(img1)) or np.any(np.isinf(img2)):
            print("Warning: Converted numpy arrays contain NaN or Inf values")
            return 0.0

        img1 = np.clip(img1, 0, 1)
        img2 = np.clip(img2, 0, 1)

        ms_ssim_score = ms_ssim(img1, img2)

        if np.isnan(ms_ssim_score) or np.isinf(ms_ssim_score):
            print("Warning: MS-SSIM result is NaN or Inf, returning 0")
            return 0.0

        return ms_ssim_score

    except Exception as e:
        print(f"Error in calculate_ms_ssim: {e}")
        return 0.0
