import torch
from torchvision.utils import save_image
import os
import numpy as np
from config import Config
from models.deepsc import VQDeepSC
from data.datasets import get_dataloader
# from communications.channel import awgn_channel, rician_channel
# from communications.modulation import bpsk_modulate, bpsk_demodulate, qpsk_modulate, qpsk_demodulate, qam16_modulate, qam16_demodulate
# from communications.ldpc_coding import indices_to_bits, bits_to_indices, ldpc_encode, ldpc_decode, get_ldpc_code
from utils.metrics import calculate_ms_ssim


def test():
    # 加载配置
    cfg = Config()

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

    checkpoint_path = os.path.join("/home/yi/wk-1/vq-dc-raqmae-64-transformer-last/checkpoints/best_vq_deepsc.pth")
    print(f"Loading checkpoint from {checkpoint_path}")
    vq_deepsc_model.load_state_dict(torch.load(checkpoint_path, map_location=device))


    # 设置模型为评估模式
    vq_deepsc_model.eval()

    # 测试数据加载器
    test_dataloader = get_dataloader(
        root_dir=cfg.TEST_DATASET_PATH,
        batch_size=1,  # 测试时批处理大小为1
        shuffle=False,
        mode='test'
    )

    # 重建图像输出目录
    output_dir = "./reconstructed_images"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting testing on {device}...")

    # 信噪比范围（dB）
    snr_range_db = np.arange(cfg.SNR_RANGE_DB[0], cfg.SNR_RANGE_DB[1] + 1, 5)
    # ms_ssim_scores = []  # 初始化MS-SSIM分数列表


    # 遍历不同的信噪比进行测试
    for snr_db in snr_range_db:
        print(f"Testing with SNR: {snr_db} dB")  # 当前测试的信噪比（dB）
        current_snr_ms_ssim_scores_src = []
        current_snr_ms_ssim_scores_raq = []
        current_snr_ms_ssim_scores_src_db = []
        current_snr_ms_ssim_scores_raq_db = []

        with torch.no_grad():  # 禁用梯度计算以节省内存
            for i, real_image in enumerate(test_dataloader):
                real_image = real_image.to(device)

                out = vq_deepsc_model.forward_test_raq(real_image, cfg.RAQ_TARGET_LIST)

                # ========== 10. 计算MS-SSIM ==========
                ms_ssim_src = calculate_ms_ssim(((real_image+1)/2), (out["reconstructed_images_src"]+1)/ 2)
                ms_ssim_raq = calculate_ms_ssim(((real_image+1)/2), (out["reconstructed_images_raq"]+1)/ 2)

                ms_ssim_src_db = -10 * np.log10(1 - ms_ssim_src)
                ms_ssim_raq_db = -10 * np.log10(1 - ms_ssim_raq)

                current_snr_ms_ssim_scores_src.append(ms_ssim_src)
                current_snr_ms_ssim_scores_raq.append(ms_ssim_raq)
                current_snr_ms_ssim_scores_src_db.append(ms_ssim_src_db)
                current_snr_ms_ssim_scores_raq_db.append(ms_ssim_raq_db)

                # 保存重建图像
                # 将图像从[-1, 1]范围反归一化到[0, 1]范围以便保存
                save_image((out["reconstructed_images_raq"]+1)/ 2, # 这一步不可漏
                           os.path.join(output_dir, f"reconstructed_snr_{snr_db}_img_{i + 1}.png"))

        # 计算当前信噪比下的平均MS-SSIM
        avg_ms_ssim_src = np.mean(current_snr_ms_ssim_scores_src)
        avg_ms_ssim_raq = np.mean(current_snr_ms_ssim_scores_raq)
        avg_ms_ssim_src_db = np.mean(current_snr_ms_ssim_scores_src_db)
        avg_ms_ssim_raq_db = np.mean(current_snr_ms_ssim_scores_raq_db)
        print(f"Average MS-SSIM-SRC for SNR {snr_db} dB: {avg_ms_ssim_src:.4f}\n"
              f"Average MS-SSIM-RAQ for SNR {snr_db} dB: {avg_ms_ssim_raq:.4f}\n"
              f"Average MS-SSIM-SRC-DB for SNR {snr_db} dB: {avg_ms_ssim_src_db:.4f}\n"
              f"Average MS-SSIM-RAQ-DB for SNR {snr_db} dB: {avg_ms_ssim_raq_db:.4f}\n")
        # ms_ssim_scores.append(avg_ms_ssim)

    # # 打印最终结果
    # print("\n--- Test Results ---")
    # for snr_db, score in zip(snr_range_db, ms_ssim_scores):
    #     print(f"SNR: {snr_db} dB, Average MS-SSIM: {score:.4f}")


if __name__ == "__main__":
    test()





