import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm
import random
import sys
from pathlib import Path

# 导入你自己的模块
from utils.metrics import calculate_ms_ssim

# 直接复用 Best_VQ 的指标实现，保证权重、预处理和统计方式完全一致。
sys.path.append(str(Path(__file__).resolve().parents[1]))
from Best_VQ.evaluation.perceptual import PerceptualMetrics

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main(ori_path=None, rx_path=None):
    setup_seed(42)

    # ================= 参数配置 =================
    # 原始未压缩图片所在的文件夹路径
    ori_path = ori_path or '/workspace/yi/work/Kodak-256-transform-resize'

    # 已经解压完毕的图片文件夹路径
    rx_path = rx_path or '/workspace/yi/work/JPEG/output_jpeg'

    psnr_scores = [] # 用于保存 24 张图的 PSNR 得分
    msssim_scores = [] # 用于保存 24 张图的 MS-SSIM 得分
    num_decode_failures = 0

    # 【补全1】必须实例化图像转换器
    to_tensor = ToTensor()
    perceptual_metrics = PerceptualMetrics(device)

    print(f"=== 开始评估文件夹质量 ===")
    print(f"对比路径: {rx_path}")

    # 内层循环：遍历 24 张图片
    for k in tqdm(range(1, 25), desc="Evaluating"):
        # 【补全2】拼接出当前循环的具体文件名
        img_name = f'kodim{k:02d}.png'
        img_name2 = f'val_0000{k:02d}.jp2'
        ori_file = os.path.join(ori_path, img_name)
        rx_jpeg_file = os.path.join(rx_path, img_name2)

        with Image.open(ori_file) as image:
            img_ori = to_tensor(image.convert('RGB')).unsqueeze(0).to(device)

        # 【补全3】判断解压图片是否存在（对应后面的 else 悬崖效应）
        if os.path.exists(rx_jpeg_file):
            # 读取原图和恢复图，并强制转换为 RGB 色彩模式
            # 转换为 Tensor 格式，增加批次维度 (unsqueeze(0) 变成 1xCxHxW)
            with Image.open(rx_jpeg_file) as image:
                img_dec = to_tensor(image.convert('RGB')).unsqueeze(0).to(device)

            # PSNR
            # 使用均方误差 (MSE) 推导
            mse = torch.mean((img_ori - img_dec) ** 2)
            psnr = 100.0 if mse == 0 else 10 * torch.log10(1.0 / mse).item()

            # MS-SSIM
            with torch.no_grad():
                # 直接把两张张量图片传进你的函数里
                m_val = calculate_ms_ssim(img_dec, img_ori)

                # 提取标量数值：处理返回结果是 Tensor 或是标量的情况
                if isinstance(m_val, torch.Tensor):
                    msssim = m_val.mean().item()
                else:
                    msssim = float(m_val)
        else:
            # 【悬崖效应】如果解码失败 (找不到文件)，得分为 0
            psnr = 0.0
            msssim = 0.0
            num_decode_failures += 1
            # 与 BPG 一致，新指标按黑图计入失败样本。
            img_dec = torch.zeros_like(img_ori)

        # JPEG 张量为 [0,1]；共享实现接收与 Best_VQ 相同的 [-1,1] RGB 输入。
        perceptual_metrics.update(img_ori * 2 - 1, img_dec * 2 - 1)

        psnr_scores.append(psnr)
        msssim_scores.append(msssim)

    # 汇总最终结果
    if psnr_scores:
        results = {
            'psnr': float(np.mean(psnr_scores)),
            'ms_ssim': float(np.mean(msssim_scores)),
            **perceptual_metrics.compute(),
            'num_images': len(psnr_scores),
            'num_decode_failures': num_decode_failures,
        }
        print("\n" + "="*50)
        # 【补全4】去掉了原来未定义的 snr 变量
        print(f"综合评估结果:")
        print(f"平均 PSNR    = {results['psnr']:.2f} dB")
        print(f"平均 MS-SSIM = {results['ms_ssim']:.4f}")
        rfid = 'N/A' if results['rfid'] is None else f"{results['rfid']:.4f}"
        print(f"rFID         = {rfid}")
        print(f"平均 LPIPS   = {results['lpips']:.4f}")
        if num_decode_failures:
            print(f"解码失败 {num_decode_failures} 张；rFID/LPIPS 按黑图计入。")
        print("="*50)
        return results

if __name__ == "__main__":
    main()
