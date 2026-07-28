import torch
from torchvision.utils import save_image
import os
import numpy as np
import pandas as pd
from datetime import datetime
from config import Config
from models.deepsc import VQDeepSC
from data.datasets import get_dataloader
from utils.channel import awgn_channel, rician_channel
from utils.modulation import bpsk_modulate, bpsk_demodulate, qpsk_modulate, qpsk_demodulate, qam16_modulate, \
    qam16_demodulate
from utils.ldpc_coding import indices_to_bits, bits_to_indices, ldpc_encode, ldpc_decode, get_ldpc_code
from utils.metrics import calculate_ms_ssim
from collections import OrderedDict


def setup_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def test():

    # 加载配置
    cfg = Config()

    # === 【关键】必须先固定种子 ===
    setup_seed(42)

    # 设备设置
    device = torch.device(cfg.DEVICE)

    # 创建模型实例
    # Create model instances
    vq_deepsc_model = VQDeepSC(
        in_channels=cfg.IN_CHANNELS,  # 输入通道数
        out_channels=cfg.OUT_CHANNELS,  # 输出通道数
        num_downsample_blocks=cfg.NUM_DOWNSAMPLE_BLOCKS,  # 下采样块数量
        base_channels=cfg.BASE_CHANNELS,  # 基础通道数
        num_embeddings_list=cfg.NUM_EMBEDDINGS_LIST,  # 向量量化字典大小列表
        embedding_dim_list=cfg.EMBEDDING_DIM_LIST,  # 向量量化维度列表
        commitment_cost=cfg.COMMITMENT_COST,  # 向量量化承诺成本系数
        raq_min_trg=cfg.RAQ_MIN_TRG,  # RAQ目标最小值
        raq_max_trg=cfg.RAQ_MAX_TRG,  # RAQ目标最大值
        device=cfg.DEVICE  # 设备
    ).to(device)  # 将模型移动到指定设备

    checkpoint_path = os.path.join("/home/yi/wk-1/vq-dc-raqmae-64-transformer-last/checkpoints/vq_deepsc_epoch_80.pth")
    print(f"Loading checkpoint from {checkpoint_path}")

    # === 【修改开始】 ===
    # 1. 加载原始的 state_dict
    state_dict = torch.load(checkpoint_path, map_location=device)

    # 2. 创建一个新的字典，去掉 key 中的 'module.' 前缀
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        # 如果 key 是以 'module.' 开头，就去掉前 7 个字符
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    # 3. 使用处理后的字典加载权重
    vq_deepsc_model.load_state_dict(new_state_dict)
    # === 【修改结束】 ===

    vq_deepsc_model.eval()

    # 测试数据加载器
    test_dataloader = get_dataloader(
        root_dir=cfg.TEST_DATASET_PATH,
        batch_size=1,
        shuffle=False,
        mode='test',
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    print(f"Starting testing on {device}...")

    # 信噪比范围（dB）
    snr_range_db = np.arange(cfg.SNR_RANGE_DB[0], cfg.SNR_RANGE_DB[1] + 1, 3)


    # 遍历不同的信噪比进行测试
    for snr_db in snr_range_db:
        print(f"Testing with SNR: {snr_db} dB")
        current_snr_ms_ssim_scores_src = []
        current_snr_ms_ssim_scores_raq = []
        current_snr_ms_ssim_scores_raq_best = []
        current_snr_ms_ssim_scores_src_db = []
        current_snr_ms_ssim_scores_raq_db = []
        current_snr_ms_ssim_scores_raq_best_db = []


        with torch.no_grad(): # 禁用梯度计算以节省内存
            for i, real_image in enumerate(test_dataloader):

                print(f"Processing image {i + 1} for SNR {snr_db} dB.")
                real_image = real_image.to(device)

                out = vq_deepsc_model.forward_test_raq(real_image, cfg.RAQ_TARGET_LIST)
                out_best = vq_deepsc_model.forward_test_raq(real_image, cfg.RAQ_TARGET_LIST_BEST)

                # ========== 1. 将索引转换为比特流 ==========
                flat_bits_src, original_spatial_dims_src, original_num_embeddings_src = indices_to_bits(
                    out["indices_src"], cfg.NUM_EMBEDDINGS_LIST) # 得到的比特流，是一个由比特0、1构成的numpy数组
                flat_bits_raq, original_spatial_dims_raq, original_num_embeddings_raq = indices_to_bits(
                    out["indices_raq"], cfg.RAQ_TARGET_LIST) # 得到的比特流，是一个由比特0、1构成的numpy数组
                flat_bits_raq_best, original_spatial_dims_raq_best, original_num_embeddings_raq_best = indices_to_bits(
                    out_best["indices_raq"], cfg.RAQ_TARGET_LIST_BEST)

                # ========== 编码 ==========
                encoded_bits_src = flat_bits_src
                encoded_bits_raq = flat_bits_raq
                encoded_bits_raq_best = flat_bits_raq_best
                print(f"SRC-不使用LDPC，直接使用原始比特流: {len(encoded_bits_src)}")
                print(f"RAQ-不使用LDPC，直接使用原始比特流: {len(encoded_bits_raq)}")
                print(f"RAQ-Best 不使用LDPC，直接使用原始比特流: {len(encoded_bits_raq_best)}")
                # 将编码后的比特转换为PyTorch张量
                bits_tensor_src = torch.from_numpy(encoded_bits_src).float().to(device)
                bits_tensor_raq = torch.from_numpy(encoded_bits_raq).float().to(device)
                bits_tensor_raq_best = torch.from_numpy(encoded_bits_raq_best).float().to(device)

                # ========== 3. 调制(BPSK) ==========
                modulated_signal_src = bpsk_modulate(bits_tensor_src)
                modulated_signal_raq = bpsk_modulate(bits_tensor_raq)
                modulated_signal_raq_best = bpsk_modulate(bits_tensor_raq_best)

                # ========== 4. 信道仿真 ==========
                if cfg.CHANNEL_TYPE == "AWGN":
                    received_signal_src = awgn_channel(modulated_signal_src, snr_db)
                    received_signal_raq = awgn_channel(modulated_signal_raq, snr_db)
                    received_signal_raq_best = awgn_channel(modulated_signal_raq_best, snr_db)
                else:
                    raise ValueError(f"Unknown channel type: {cfg.CHANNEL_TYPE}")

                # ========== 5.硬判决(就是没有LDPC编码的情况下才硬判决） ==========

                # 不使用LDPC，直接进行硬判决
                # 使用BPSK解调函数进行硬判决
                demodulated_bits_src = bpsk_demodulate(received_signal_src)
                demodulated_bits_raq = bpsk_demodulate(received_signal_raq)
                demodulated_bits_raq_best = bpsk_demodulate(received_signal_raq_best)

                decoded_bits_src = demodulated_bits_src.cpu().numpy()
                decoded_bits_raq = demodulated_bits_raq.cpu().numpy()
                decoded_bits_raq_best = demodulated_bits_raq_best.cpu().numpy()

                decoded_bits_src = decoded_bits_src.astype(np.uint8)
                decoded_bits_raq = decoded_bits_raq.astype(np.uint8)
                decoded_bits_raq_best = decoded_bits_raq_best.astype(np.uint8)

                # ========== 7. 将比特转换回索引 ==========
                reconstructed_indices_src = bits_to_indices(decoded_bits_src, original_spatial_dims_src,
                                                            original_num_embeddings_src)
                reconstructed_indices_raq = bits_to_indices(decoded_bits_raq, original_spatial_dims_raq,
                                                            original_num_embeddings_raq)
                reconstructed_indices_raq_best = bits_to_indices(decoded_bits_raq_best, original_spatial_dims_raq_best,
                                                                original_num_embeddings_raq_best)

                # 将索引张量移动到模型所在的设备
                reconstructed_indices_src = [tensor.to(device) for tensor in reconstructed_indices_src]
                reconstructed_indices_raq = [tensor.to(device) for tensor in reconstructed_indices_raq]
                reconstructed_indices_raq_best = [tensor.to(device) for tensor in reconstructed_indices_raq_best]

                # ========== 8. 从索引重建图像 ==========
                reconstructed_images_src = vq_deepsc_model.reconstruct_from_indices_src(reconstructed_indices_src)
                reconstructed_images_raq = vq_deepsc_model.reconstruct_from_indices_raq(reconstructed_indices_raq,
                                                                                        out["codebooks"])
                reconstructed_images_raq_best = vq_deepsc_model.reconstruct_from_indices_raq(
                    reconstructed_indices_raq_best, out_best["codebooks"])

                print("源码本重建图像的形状是 : ", reconstructed_images_src.shape)
                print("RAQ重建图像的形状是 : ", reconstructed_images_raq.shape)
                print("RAQ-Best重建图像的形状是 : ", reconstructed_images_raq_best.shape)

                # ========== 9. 计算MS-SSIM ==========
                ms_ssim_src = calculate_ms_ssim((real_image + 1) / 2, (reconstructed_images_src + 1) / 2)
                ms_ssim_raq = calculate_ms_ssim((real_image + 1) / 2, (reconstructed_images_raq + 1) / 2)
                ms_ssim_raq_best = calculate_ms_ssim((real_image + 1) / 2, (reconstructed_images_raq_best + 1) / 2)

                ms_ssim_src_db = -10 * np.log10(max(1e-10, 1 - ms_ssim_src))
                ms_ssim_raq_db = -10 * np.log10(max(1e-10, 1 - ms_ssim_raq))
                ms_ssim_raq_best_db = -10 * np.log10(max(1e-10, 1 - ms_ssim_raq_best))

                current_snr_ms_ssim_scores_src.append(ms_ssim_src)
                current_snr_ms_ssim_scores_raq.append(ms_ssim_raq)
                current_snr_ms_ssim_scores_raq_best.append(ms_ssim_raq_best)
                current_snr_ms_ssim_scores_src_db.append(ms_ssim_src_db)
                current_snr_ms_ssim_scores_raq_db.append(ms_ssim_raq_db)
                current_snr_ms_ssim_scores_raq_best_db.append(ms_ssim_raq_best_db)



        # 计算当前SNR的平均值并保存到结果中
        avg_ms_ssim_src = np.mean(current_snr_ms_ssim_scores_src)
        avg_ms_ssim_raq = np.mean(current_snr_ms_ssim_scores_raq)
        avg_ms_ssim_raq_best = np.mean(current_snr_ms_ssim_scores_raq_best)
        avg_ms_ssim_src_db = np.mean(current_snr_ms_ssim_scores_src_db)
        avg_ms_ssim_raq_db = np.mean(current_snr_ms_ssim_scores_raq_db)
        avg_ms_ssim_raq_best_db = np.mean(current_snr_ms_ssim_scores_raq_best_db)

        print(f"With SNR: {snr_db} dB,current_snr_ms_ssim_scores_src:{current_snr_ms_ssim_scores_src}")
        print(f"With SNR: {snr_db} dB,current_snr_ms_ssim_scores_raq:{current_snr_ms_ssim_scores_raq}")
        print(f"With SNR: {snr_db} dB,current_snr_ms_ssim_scores_raq_best:{current_snr_ms_ssim_scores_raq_best}")
        print(
            f"With SNR: {snr_db} dB,current_snr_ms_ssim_scores_src_db:{[float(x) for x in current_snr_ms_ssim_scores_src_db]}")
        print(
            f"With SNR: {snr_db} dB,current_snr_ms_ssim_scores_raq_db:{[float(x) for x in current_snr_ms_ssim_scores_raq_db]}")
        print(
            f"With SNR: {snr_db} dB,current_snr_ms_ssim_scores_raq_best_db:{[float(x) for x in current_snr_ms_ssim_scores_raq_best_db]}")

        print(f"Average MS-SSIM-SRC for SNR {snr_db} dB: {avg_ms_ssim_src:.4f}\n"
              f"Average MS-SSIM-RAQ for SNR {snr_db} dB: {avg_ms_ssim_raq:.4f}\n"
              f"Average MS-SSIM-RAQ-BEST for SNR {snr_db} dB: {avg_ms_ssim_raq_best:.4f}\n"
              f"Average MS-SSIM-SRC-DB for SNR {snr_db} dB: {avg_ms_ssim_src_db:.4f}\n"
              f"Average MS-SSIM-RAQ-DB for SNR {snr_db} dB: {avg_ms_ssim_raq_db:.4f}\n"
              f"Average MS-SSIM-RAQ-BEST-DB for SNR {snr_db} dB: {avg_ms_ssim_raq_best_db:.4f}\n")


if __name__ == "__main__":
    test()



