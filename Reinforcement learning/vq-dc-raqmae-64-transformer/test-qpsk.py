import torch
from torchvision.utils import save_image
import os
import numpy as np
import pandas as pd
from datetime import datetime
from config import Config
from data.datasets import get_dataloader
from communications.channel import awgn_channel, rician_channel
from communications.modulation import qpsk_modulate, qpsk_demodulate
from communications.ldpc_coding import indices_to_bits, bits_to_indices, ldpc_encode, ldpc_decode, get_ldpc_code
from communications.metrics import calculate_ms_ssim


def test(use_ldpc=True):
    """
    测试函数
    Args:
        use_ldpc: 是否使用LDPC编码
    """
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

    checkpoint_path = os.path.join("/home/yi/wk-1/vq-dc-raqmae-64/checkpoints/vq_deepsc_epoch_180.pth")
    print(f"Loading checkpoint from {checkpoint_path}")
    vq_deepsc_model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    vq_deepsc_model.eval()

    # 测试数据加载器
    test_dataloader = get_dataloader(
        root_dir=cfg.TEST_DATASET_PATH,
        batch_size=1,
        shuffle=False,
        mode='test'
    )

    # 重建图像输出目录
    output_dir_suffix = "_ldpc" if use_ldpc else "_noldpc"
    output_dir_src = f"./reconstructed_images_src{output_dir_suffix}"
    output_dir_raq = f"./reconstructed_images_raq{output_dir_suffix}"
    output_dir_raq_best = f"./reconstructed_images_raq_best{output_dir_suffix}"

    os.makedirs(output_dir_src, exist_ok=True)
    os.makedirs(output_dir_raq, exist_ok=True)
    os.makedirs(output_dir_raq_best, exist_ok=True)

    print(f"Starting testing on {device}...")
    print(f"LDPC编码: {'启用' if use_ldpc else '禁用'}")

    # 信噪比范围（dB）
    snr_range_db = np.arange(cfg.SNR_RANGE_DB[0], cfg.SNR_RANGE_DB[1] + 1, 2)

    if use_ldpc:
        ldpc_code = get_ldpc_code(block_length=648)
        print(f"LDPC code initialized with k={ldpc_code['k']}")

    # 创建结果存储结构
    results = {
        'SNR_dB': [],
        'VQ_MS_SSIM': [],
        'RAQ_MS_SSIM': [],
        'RAQ_Best_MS_SSIM': [],
        'VQ_MS_SSIM_dB': [],
        'RAQ_MS_SSIM_dB': [],
        'RAQ_Best_MS_SSIM_dB': [],
        'VQ_BER': [],
        'RAQ_BER': [],
        'RAQ_Best_BER': [],
        'VQ_Bit_Length': [],
        'RAQ_Bit_Length': [],
        'RAQ_Best_Bit_Length': [],
        'Image_Count': []
    }

    # 存储每张图像的详细结果
    detailed_results = []

    # 遍历不同的信噪比进行测试
    for snr_db in snr_range_db:
        print(f"Testing with SNR: {snr_db} dB")
        current_snr_ms_ssim_scores_src = []
        current_snr_ms_ssim_scores_raq = []
        current_snr_ms_ssim_scores_raq_best = []
        current_snr_ms_ssim_scores_src_db = []
        current_snr_ms_ssim_scores_raq_db = []
        current_snr_ms_ssim_scores_raq_best_db = []
        current_snr_ber_src = []
        current_snr_ber_raq = []
        current_snr_ber_raq_best = []

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

                print(f"SRC-Actual bit length: {len(flat_bits_src)}")
                print(f"RAQ-Actual bit length: {len(flat_bits_raq)}")

                # ========== 2. LDPC编码 ==========
                if use_ldpc:
                    # 确保flat_bits_*是一个由0和1组成的一维numpy数组，用于ldpc_encode
                    encoded_bits_src = ldpc_encode(flat_bits_src, code=ldpc_code)
                    encoded_bits_raq = ldpc_encode(flat_bits_raq, code=ldpc_code)
                    encoded_bits_raq_best = ldpc_encode(flat_bits_raq_best, code=ldpc_code)
                    print(f"SRC-Actual encoded length: {len(encoded_bits_src)}")
                    print(f"RAQ-Actual encoded length: {len(encoded_bits_raq)}")
                    print(f"RAQ-Actual-Best encoded length: {len(encoded_bits_raq_best)}")

                    # 将编码后的比特转换为PyTorch张量
                    bits_tensor_src = torch.from_numpy(encoded_bits_src).float().to(device)
                    bits_tensor_raq = torch.from_numpy(encoded_bits_raq).float().to(device)
                    bits_tensor_raq_best = torch.from_numpy(encoded_bits_raq_best).float().to(device)
                else:
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

                # ========== 3. 调制(QPSK) ==========
                modulated_signal_src = qpsk_modulate(bits_tensor_src)
                modulated_signal_raq = qpsk_modulate(bits_tensor_raq)
                modulated_signal_raq_best = qpsk_modulate(bits_tensor_raq_best)

                # ========== 4. 信道仿真 ==========
                if cfg.CHANNEL_TYPE == "AWGN":
                    received_signal_src = awgn_channel(modulated_signal_src, snr_db)
                    received_signal_raq = awgn_channel(modulated_signal_raq, snr_db)
                    received_signal_raq_best = awgn_channel(modulated_signal_raq_best, snr_db)
                elif cfg.CHANNEL_TYPE == "Rician":
                    received_signal_src = rician_channel(modulated_signal_src, snr_db, k_factor=cfg.RICIAN_K_FACTOR)
                    received_signal_raq = rician_channel(modulated_signal_raq, snr_db, k_factor=cfg.RICIAN_K_FACTOR)
                    received_signal_raq_best = rician_channel(modulated_signal_raq_best, snr_db,
                                                             k_factor=cfg.RICIAN_K_FACTOR)
                else:
                    raise ValueError(f"Unknown channel type: {cfg.CHANNEL_TYPE}")

                # ========== 5. 解调 ==========
                # 正确的QPSK软解调，计算LLR
                snr_linear = 10 ** (snr_db / 10.0)

                # 对于QPSK，分别计算实部和虚部的LLR
                # 噪声方差 = 1 / (2 * snr_linear) 因为QPSK符号功率为1
                noise_variance = 1.0 / (2.0 * snr_linear)

                # 计算实部和虚部的LLR
                llr_real_src = (2.0 / noise_variance) * received_signal_src.real
                llr_imag_src = (2.0 / noise_variance) * received_signal_src.imag
                llr_real_raq = (2.0 / noise_variance) * received_signal_raq.real
                llr_imag_raq = (2.0 / noise_variance) * received_signal_raq.imag
                llr_real_raq_best = (2.0 / noise_variance) * received_signal_raq_best.real
                llr_imag_raq_best = (2.0 / noise_variance) * received_signal_raq_best.imag

                # 将实部和虚部的LLR交错排列
                llr_src = torch.zeros(2 * len(received_signal_src), device=device)
                llr_src[0::2] = llr_real_src  # 实部对应的比特
                llr_src[1::2] = llr_imag_src  # 虚部对应的比特

                llr_raq = torch.zeros(2 * len(received_signal_raq), device=device)
                llr_raq[0::2] = llr_real_raq
                llr_raq[1::2] = llr_imag_raq

                llr_raq_best = torch.zeros(2 * len(received_signal_raq_best), device=device)
                llr_raq_best[0::2] = llr_real_raq_best
                llr_raq_best[1::2] = llr_imag_raq_best

                print("SRC-LLR length is : ", len(llr_src))
                print("RAQ-LLR length is : ", len(llr_raq))
                print("RAQ-Best-LLR length is : ", len(llr_raq_best))

                # ========== 6. LDPC解码或硬判决(就是没有LDPC编码的情况下蔡硬判决） ==========
                if use_ldpc:
                    # ldpc_decode期望numpy数组作为输入
                    decoded_bits_src = ldpc_decode(llr_src.cpu().numpy(), code=ldpc_code)
                    decoded_bits_raq = ldpc_decode(llr_raq.cpu().numpy(), code=ldpc_code)
                    decoded_bits_raq_best = ldpc_decode(llr_raq_best.cpu().numpy(), code=ldpc_code)
                    print("SRC-Decoded bits length is : ", len(decoded_bits_src))
                    print("RAQ-Decoded bits length is : ", len(decoded_bits_raq))
                    print("RAQ-Best-Decoded bits length is : ", len(decoded_bits_raq_best))

                    # 截断到原始比特流长度，因为LDPC编码可能会增加比特数
                    decoded_bits_src = decoded_bits_src[:len(flat_bits_src)]
                    decoded_bits_raq = decoded_bits_raq[:len(flat_bits_raq)]
                    decoded_bits_raq_best = decoded_bits_raq_best[:len(flat_bits_raq_best)]
                else:
                    # 不使用LDPC，直接进行硬判决
                    # 使用QPSK解调函数进行硬判决
                    demodulated_bits_src = qpsk_demodulate(received_signal_src)
                    demodulated_bits_raq = qpsk_demodulate(received_signal_raq)
                    demodulated_bits_raq_best = qpsk_demodulate(received_signal_raq_best)

                    decoded_bits_src = demodulated_bits_src.cpu().numpy()[:len(flat_bits_src)]
                    decoded_bits_raq = demodulated_bits_raq.cpu().numpy()[:len(flat_bits_raq)]
                    decoded_bits_raq_best = demodulated_bits_raq_best.cpu().numpy()[:len(flat_bits_raq_best)]

                    print("SRC-硬判决比特长度: ", len(decoded_bits_src))
                    print("RAQ-硬判决比特长度: ", len(decoded_bits_raq))
                    print("RAQ-Best-硬判决比特长度: ", len(decoded_bits_raq_best))

                print("截断后的比特流长度是 : ", len(decoded_bits_src), len(decoded_bits_raq), len(decoded_bits_raq_best))

                # 计算误比特率
                ber_src = np.mean(decoded_bits_src != flat_bits_src)
                ber_raq = np.mean(decoded_bits_raq != flat_bits_raq)
                ber_raq_best = np.mean(decoded_bits_raq_best != flat_bits_raq_best)
                print(f"VQ-误比特率 (BER): {ber_src:.6f}")
                print(f"RAQ-误比特率 (BER): {ber_raq:.6f}")
                print(f"RAQ-Best-误比特率 (BER): {ber_raq_best:.6f}")

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

                current_snr_ber_src.append(ber_src)
                current_snr_ber_raq.append(ber_raq)
                current_snr_ber_raq_best.append(ber_raq_best)

                # 保存每张图像的详细结果
                detailed_results.append({
                    'SNR_dB': snr_db,
                    'Image_Index': i + 1,
                    'VQ_MS_SSIM': ms_ssim_src,
                    'RAQ_MS_SSIM': ms_ssim_raq,
                    'RAQ_Best_MS_SSIM': ms_ssim_raq_best,
                    'VQ_MS_SSIM_dB': ms_ssim_src_db,
                    'RAQ_MS_SSIM_dB': ms_ssim_raq_db,
                    'RAQ_Best_MS_SSIM_dB': ms_ssim_raq_best_db,
                    'VQ_BER': ber_src,
                    'RAQ_BER': ber_raq,
                    'RAQ_Best_BER': ber_raq_best,
                    'VQ_Bit_Length': len(flat_bits_src),
                    'RAQ_Bit_Length': len(flat_bits_raq),
                    'RAQ_Best_Bit_Length': len(flat_bits_raq_best),
                    'LDPC_Enabled': use_ldpc
                })

                # 保存重建图像
                save_image((reconstructed_images_src + 1) / 2,
                           os.path.join(output_dir_src, f"reconstructed_snr_{snr_db}_img_{i + 1}.png"))
                save_image((reconstructed_images_raq + 1) / 2,
                           os.path.join(output_dir_raq, f"reconstructed_snr_{snr_db}_img_{i + 1}.png"))
                save_image((reconstructed_images_raq_best + 1) / 2,
                           os.path.join(output_dir_raq_best, f"reconstructed_snr_{snr_db}_img_{i + 1}.png"))

        # 计算当前SNR的平均值并保存到结果中
        avg_ms_ssim_src = np.mean(current_snr_ms_ssim_scores_src)
        avg_ms_ssim_raq = np.mean(current_snr_ms_ssim_scores_raq)
        avg_ms_ssim_raq_best = np.mean(current_snr_ms_ssim_scores_raq_best)
        avg_ms_ssim_src_db = np.mean(current_snr_ms_ssim_scores_src_db)
        avg_ms_ssim_raq_db = np.mean(current_snr_ms_ssim_scores_raq_db)
        avg_ms_ssim_raq_best_db = np.mean(current_snr_ms_ssim_scores_raq_best_db)
        avg_ber_src = np.mean(current_snr_ber_src)
        avg_ber_raq = np.mean(current_snr_ber_raq)
        avg_ber_raq_best = np.mean(current_snr_ber_raq_best)

        results['SNR_dB'].append(snr_db)
        results['VQ_MS_SSIM'].append(avg_ms_ssim_src)
        results['RAQ_MS_SSIM'].append(avg_ms_ssim_raq)
        results['RAQ_Best_MS_SSIM'].append(avg_ms_ssim_raq_best)
        results['VQ_MS_SSIM_dB'].append(avg_ms_ssim_src_db)
        results['RAQ_MS_SSIM_dB'].append(avg_ms_ssim_raq_db)
        results['RAQ_Best_MS_SSIM_dB'].append(avg_ms_ssim_raq_best_db)
        results['VQ_BER'].append(avg_ber_src)
        results['RAQ_BER'].append(avg_ber_raq)
        results['RAQ_Best_BER'].append(avg_ber_raq_best)
        results['VQ_Bit_Length'].append(len(flat_bits_src) if 'flat_bits_src' in locals() else 0)
        results['RAQ_Bit_Length'].append(len(flat_bits_raq) if 'flat_bits_raq' in locals() else 0)
        results['RAQ_Best_Bit_Length'].append(len(flat_bits_raq_best) if 'flat_bits_raq_best' in locals() else 0)
        results['Image_Count'].append(len(current_snr_ms_ssim_scores_src))

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

    # ========== 10. 保存结果到Excel文件 ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ldpc_suffix = "with_ldpc" if use_ldpc else "without_ldpc"

    # 创建结果目录
    results_dir = "./test_results_qpsk"
    os.makedirs(results_dir, exist_ok=True)

    # 保存汇总结果
    summary_df = pd.DataFrame(results)
    summary_filename = f"{results_dir}/ms_ssim_summary_{ldpc_suffix}_{timestamp}.xlsx"
    summary_df.to_excel(summary_filename, index=False, sheet_name='Summary')
    print(f"汇总结果已保存到: {summary_filename}")

    # 保存详细结果
    detailed_df = pd.DataFrame(detailed_results)
    detailed_filename = f"{results_dir}/ms_ssim_detailed_{ldpc_suffix}_{timestamp}.xlsx"
    detailed_df.to_excel(detailed_filename, index=False, sheet_name='Detailed')
    print(f"详细结果已保存到: {detailed_filename}")

    # # 可选：创建一个包含多个工作表的Excel文件
    # combined_filename = f"{results_dir}/ms_ssim_results_{ldpc_suffix}_{timestamp}.xlsx"
    # with pd.ExcelWriter(combined_filename, engine='openpyxl') as writer:
    #     summary_df.to_excel(writer, sheet_name='Summary', index=False)
    #     detailed_df.to_excel(writer, sheet_name='Detailed', index=False)
    #
    #     # 添加一个对比表格
    #     comparison_data = {
    #         'Metric': ['VQ MS-SSIM', 'RAQ MS-SSIM', 'VQ MS-SSIM (dB)', 'RAQ MS-SSIM (dB)', 'VQ BER', 'RAQ BER'],
    #         'Best_SNR': [
    #             results['SNR_dB'][np.argmax(results['VQ_MS_SSIM'])],
    #             results['SNR_dB'][np.argmax(results['RAQ_MS_SSIM'])],
    #             results['SNR_dB'][np.argmax(results['VQ_MS_SSIM_dB'])],
    #             results['SNR_dB'][np.argmax(results['RAQ_MS_SSIM_dB'])],
    #             results['SNR_dB'][np.argmin(results['VQ_BER'])],
    #             results['SNR_dB'][np.argmin(results['RAQ_BER'])]
    #         ],
    #         'Best_Value': [
    #             np.max(results['VQ_MS_SSIM']),
    #             np.max(results['RAQ_MS_SSIM']),
    #             np.max(results['VQ_MS_SSIM_dB']),
    #             np.max(results['RAQ_MS_SSIM_dB']),
    #             np.min(results['VQ_BER']),
    #             np.min(results['RAQ_BER'])
    #         ]
    #     }
    #     comparison_df = pd.DataFrame(comparison_data)
    #     comparison_df.to_excel(writer, sheet_name='Comparison', index=False)

    # print(f"完整结果已保存到: {combined_filename}")

    return summary_df, detailed_df


if __name__ == "__main__":
    # 可以选择是否使用LDPC编码
    use_ldpc = False  # 设置为False则不使用LDPC编码
    summary, detailed = test(use_ldpc=use_ldpc)

    # 打印最终汇总结果
    print("\n" + "=" * 80)
    print("测试完成！结果汇总:")
    print("=" * 80)
    print(f"测试配置: LDPC编码 = {'启用' if use_ldpc else '禁用'}")
    print(f"SNR范围: {np.min(summary['SNR_dB'])} dB 到 {np.max(summary['SNR_dB'])} dB")
    print(f"测试图像数量: {summary['Image_Count'].iloc[0]}")
    print(
        f"\n最佳VQ MS-SSIM: {np.max(summary['VQ_MS_SSIM']):.4f} (在 {summary['SNR_dB'][np.argmax(summary['VQ_MS_SSIM'])]} dB)")
    print(
        f"最佳RAQ MS-SSIM: {np.max(summary['RAQ_MS_SSIM']):.4f} (在 {summary['SNR_dB'][np.argmax(summary['RAQ_MS_SSIM'])]} dB)\n")
    print(
        f"最佳RAQ-Best MS-SSIM: {np.max(summary['RAQ_Best_MS_SSIM']):.4f} (在 {summary['SNR_dB'][np.argmax(summary['RAQ_Best_MS_SSIM'])]} dB)")
