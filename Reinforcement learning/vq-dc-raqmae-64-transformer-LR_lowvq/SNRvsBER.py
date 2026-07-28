import torch
import numpy as np
import matplotlib.pyplot as plt

# 导入你自己的物理层模块
from communications.ldpc_coding import get_ldpc_code, ldpc_encode, ldpc_decode
from communications.channel import awgn_channel
from communications.modulation import (
    bpsk_modulate, bpsk_llr,
    qpsk_modulate, qpsk_llr,
    qam16_modulate, qam16_llr
)

# =========================================================================
# 第一部分：系统与核心加速参数设置
# =========================================================================
LDPC_K = 128  # LDPC 信息比特长度
MAX_ERR = 150  # 【150错误原则】：收集满 150 个错即停止当前 SNR 测试
MAX_BITS = 1e7  # 【兜底机制】：最高测试 1000 万比特

# 🚀 核心加速引擎：批量处理
BATCH_BLOCKS = 500  # 每次直接并行处理 500 个数据块
FRAME_SIZE = BATCH_BLOCKS * LDPC_K  # 每次喂给 GPU 51200 个比特

# 测试的 SNR 范围：从 -3 dB 扫到 12 dB
SNR_ARRAY = np.arange(-3, 13, 1)

# 自动选择计算设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    print(f"=== 开始物理层 BER 瀑布曲线极速仿真 ===")
    print(f"使用设备: {device} | 每次并行发送 {FRAME_SIZE} 比特")

    # 1. 初始化 Sionna LDPC 编解码器实例
    ldpc_code = get_ldpc_code(LDPC_K)

    # 2. 定义不同的调制方案
    mod_funcs = {
        'BPSK': {'mod_func': bpsk_modulate, 'llr_func': bpsk_llr},
        'QPSK': {'mod_func': qpsk_modulate, 'llr_func': qpsk_llr},
        '16-QAM': {'mod_func': qam16_modulate, 'llr_func': qam16_llr}
    }

    # =========================================================================
    # 🌟 核心修改区：定义你要测试的 4 种系统配置方案
    # =========================================================================
    configs = [
        {'name': 'LDPC 1/2 + BPSK', 'mod_type': 'BPSK', 'rate': 0.5, 'color': 'b', 'marker': 'o'},
        {'name': 'LDPC 1/2 + QPSK', 'mod_type': 'QPSK', 'rate': 0.5, 'color': 'g', 'marker': 's'},
        {'name': 'LDPC 3/4 + QPSK', 'mod_type': 'QPSK', 'rate': 0.75, 'color': 'm', 'marker': 'D'},  # 洋红色菱形
        {'name': 'LDPC 1/2 + 16-QAM', 'mod_type': '16-QAM', 'rate': 0.5, 'color': 'r', 'marker': '^'}
    ]

    # 初始化结果字典，此时以配置名称 ('name') 作为键
    results_snr = {cfg['name']: [] for cfg in configs}
    results_ber = {cfg['name']: [] for cfg in configs}

    # =========================================================================
    # 第二部分：外层循环 - 遍历不同的调制方式
    # =========================================================================
    for cfg in configs:
        target_name = cfg['name']
        rate = cfg['rate']
        mod_type = cfg['mod_type']
        scheme = mod_funcs[mod_type]

        print(f"\n>>> 正在测试配置: {target_name}")

        # 🌟 根据当前配置的码率，动态初始化相应的 LDPC 编解码器
        ldpc_code = get_ldpc_code(LDPC_K, rate=rate)

        for snr in SNR_ARRAY:
            ttl_err = 0
            ttl_bits = 0

            while ttl_err < MAX_ERR and ttl_bits < MAX_BITS:
                # 步骤 A：生成巨大的批量比特流
                tx_bits = np.random.randint(0, 2, FRAME_SIZE).astype(np.uint8)

                # 步骤 B：LDPC 批量编码
                coded_bits = ldpc_encode(tx_bits, code=ldpc_code)

                with torch.no_grad():
                    coded_bits_tensor = torch.from_numpy(coded_bits).float().to(device)

                    # 步骤 C、D、E：调制 -> 加噪 -> 算 LLR
                    symbols = scheme['mod_func'](coded_bits_tensor)
                    rx_symbols = awgn_channel(symbols, snr)
                    llrs = scheme['llr_func'](rx_symbols, snr, device)

                    llrs_np = llrs.cpu().numpy()

                # 步骤 F：LDPC 批量解码
                rx_bits = ldpc_decode(llrs_np, code=ldpc_code)

                # 对齐截断
                rx_bits = rx_bits[:FRAME_SIZE]

                # 步骤 G：极速统计误码数
                errs = np.sum(tx_bits != rx_bits)
                ttl_err += errs
                ttl_bits += FRAME_SIZE

            # 计算最终 BER 并记录
            ber = ttl_err / ttl_bits
            results_snr[target_name].append(snr)
            results_ber[target_name].append(ber)

            print(f"SNR: {snr:2d} dB | 传输比特: {ttl_bits:8d} | 误码数: {ttl_err:4d} | BER: {ber:.2e}")

            if ber == 0.0:
                print(f"[*] {target_name} 在 {snr}dB 达到无误码传输，跳过更高 SNR。")
                break

    # =========================================================================
    # 第三部分：绘制并保存 BER 瀑布曲线图
    # =========================================================================
    print("\n=== 仿真结束，正在生成对比图表 ===")
    plt.figure(figsize=(10, 7))

    for cfg in configs:
        target_name = cfg['name']

        snrs = np.array(results_snr[target_name])
        bers = np.array(results_ber[target_name])

        # 巧妙过滤：只画存在误码的点，避免 log(0) 报错导致曲线断裂
        valid_idx = bers > 0

        plt.semilogy(snrs[valid_idx], bers[valid_idx],
                     marker=cfg['marker'], color=cfg['color'],
                     linewidth=2.5, markersize=8, label=target_name)

    # 图表学术级美化
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.title(f'Physical Layer BER Performance (LDPC K={LDPC_K}, AWGN)', fontsize=14)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)

    plt.xlim([min(SNR_ARRAY), max(SNR_ARRAY)])
    plt.ylim([1e-7, 1])
    plt.legend(fontsize=12)

    save_path = 'modulation_rates_ber_waterfall.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🎉 成功！图表已保存为: {save_path}")


if __name__ == "__main__":
    main()