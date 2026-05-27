# MS-SSIM指标所借用函数

import numpy as np
import cv2
import torch
from scipy.signal import convolve2d


def _gaussian_kernel(size, sigma):
    """生成归一化的二维高斯核"""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return kernel / np.sum(kernel)


def ssim(img1, img2, k1=0.01, k2=0.03, win_size=11, L=255):
    """计算单尺度 SSIM 和 CS"""
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

        # 处理NaN值
        ssim_map = np.nan_to_num(ssim_map, nan=1.0)

        cs_map = np.ones_like(sigma12)
        cs_denom = sigma1_sq + sigma2_sq + C2
        valid_cs = cs_denom > 1e-10
        cs_map[valid_cs] = (2 * sigma12 + C2)[valid_cs] / cs_denom[valid_cs]

        # 处理NaN值
        cs_map = np.nan_to_num(cs_map, nan=1.0)
        cs_map = np.clip(cs_map, 0, 1)

        ssim_maps.append(ssim_map)
        cs_maps.append(cs_map)

    ssim_avg = np.mean(ssim_maps, axis=0)
    cs_avg = np.mean(cs_maps, axis=0)

    # 处理NaN值
    ssim_avg = np.nan_to_num(ssim_avg, nan=1.0)
    cs_avg = np.nan_to_num(cs_avg, nan=1.0)

    return ssim_avg, cs_avg


def ms_ssim(img1, img2, max_level=5):
    """计算 MS-SSIM 值"""
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = min(max_level, len(weights))

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # 检查输入是否有NaN或无穷大
    if np.any(np.isnan(img1)) or np.any(np.isnan(img2)) or \
            np.any(np.isinf(img1)) or np.any(np.isinf(img2)):
        return 0.0

    # 自动判定 L（图像动态范围）
    L = 255 if img1.max() > 1.0 else 1.0

    mssim_vals = []
    cs_vals = []

    for i in range(levels):
        if min(img1.shape[0], img1.shape[1]) < 11:
            break

        ssim_map, cs_map = ssim(img1, img2, L=L)

        # 检查SSIM和CS映射是否有NaN值
        if np.any(np.isnan(ssim_map)) or np.any(np.isnan(cs_map)):
            ssim_val = 1.0
            cs_val = 1.0
        else:
            ssim_val = np.mean(ssim_map)
            cs_val = np.mean(cs_map)

        # 处理NaN值
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

    # 处理NaN值
    cs_vals = np.nan_to_num(cs_vals, nan=1.0)
    mssim_vals = np.nan_to_num(mssim_vals, nan=1.0)

    # 调整权重以匹配实际层数
    actual_levels = len(cs_vals)
    if actual_levels == 0:
        return 0.0

    adjusted_weights = weights[:actual_levels]
    adjusted_weights = adjusted_weights / np.sum(adjusted_weights)  # 重新归一化

    # 计算多尺度SSIM
    if actual_levels == 1:
        overall = mssim_vals[0]
    else:
        overall = np.prod(cs_vals[:-1] ** adjusted_weights[:-1])
        overall *= mssim_vals[-1] ** adjusted_weights[-1]

    # 处理最终结果的NaN值
    overall = np.nan_to_num(overall, nan=0.0)
    return float(np.clip(overall, 0.0, 1.0))


def calculate_ms_ssim(img1, img2):
    """计算两个PyTorch张量之间的MS-SSIM，处理NaN值"""
    try:
        # 检查输入是否为有效张量
        if img1 is None or img2 is None:
            return 0.0

        # 确保输入是4D张量 (B, C, H, W)
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)

        # 检查张量形状是否匹配
        if img1.shape != img2.shape:
            print(f"Warning: Image shapes don't match: {img1.shape} vs {img2.shape}")
            return 0.0

        # 检查张量是否包含NaN或无穷大值
        if torch.any(torch.isnan(img1)) or torch.any(torch.isnan(img2)) or \
                torch.any(torch.isinf(img1)) or torch.any(torch.isinf(img2)):
            print("Warning: Input images contain NaN or Inf values")
            return 0.0

        # 计算MS-SSIM
        img1 = img1[0]  # img1和img2都是PyTorch张量，将BCHW变成CHW
        img2 = img2[0]
        img1 = img1.permute(1, 2, 0).cpu().numpy()  # 将CHW变成HWC，并且转换为numpy数组,通道应该为RGB
        img2 = img2.permute(1, 2, 0).cpu().numpy()

        # 检查转换后的numpy数组
        if np.any(np.isnan(img1)) or np.any(np.isnan(img2)) or \
                np.any(np.isinf(img1)) or np.any(np.isinf(img2)):
            print("Warning: Converted numpy arrays contain NaN or Inf values")
            return 0.0

        # 确保图像在有效范围内
        img1 = np.clip(img1, 0, 1)
        img2 = np.clip(img2, 0, 1)

        ms_ssim_score = ms_ssim(img1, img2)

        # 处理最终结果的NaN值
        if np.isnan(ms_ssim_score) or np.isinf(ms_ssim_score):
            print("Warning: MS-SSIM result is NaN or Inf, returning 0")
            return 0.0

        return ms_ssim_score

    except Exception as e:
        print(f"Error in calculate_ms_ssim: {e}")
        return 0.0