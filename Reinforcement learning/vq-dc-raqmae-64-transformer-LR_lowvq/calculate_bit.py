import math


def calculate_numbits_budget(H, W, compression_ratio_str, R, M):
    """
    根据导师的公式计算允许的最大传输比特数 (NumBits)

    参数:
    H: 图像高度 (Kodak通常为 512)
    W: 图像宽度 (Kodak通常为 768)
    compression_ratio_str: 压缩率字符串，如 "1/12" 或 "1/24"
    R: 信道编码率 (如 1/2, 3/4)
    M: 调制阶数 (BPSK=2, QPSK=4, 16QAM=16)
    """
    # 解析压缩率 (例如把 "1/12" 解析为 BCR = 1/12, 即 n/k = 12)
    numerator, denominator = map(int, compression_ratio_str.split('/'))
    # 物理意义：1个符号传 denominator 个像素信息
    compression_factor = denominator / numerator

    # 1. 计算原始图像总亚像素数 (n)
    total_subpixels = 3 * H * W

    # 2. 计算物理层允许传输的信道符号数 (k)
    # 根据公式: 压缩率因子 = n / k  =>  k = n / 压缩率因子
    num_symbols = total_subpixels / compression_factor

    # 3. 反推 NumBits
    # 因为 k = NumBits / R / log2(M)
    # 所以 NumBits = k * R * log2(M)
    bits_per_symbol = math.log2(M)
    numbits = num_symbols * R * bits_per_symbol

    return int(numbits), int(num_symbols)


if __name__ == "__main__":
    # Kodak 数据集标准分辨率
    H = 768
    W = 512

    # 导师要求测试的压缩率
    compression_ratios = ["1/12", "1/24"]

    # 导师要求测试的 编码和调制格式组合: (名字, 码率 R, 调制阶数 M)
    mcs_configs = [
        ("LDPC 1/2 with BPSK", 1 / 2, 2),
        ("LDPC 1/2 with QPSK", 1 / 2, 4),
        ("LDPC 3/4 with QPSK", 3 / 4, 4),
        ("LDPC 1/2 with 16QAM", 1 / 2, 16)
    ]

    print("=" * 60)
    print(f"基准图像分辨率: {W} x {H} (Kodak)")
    print(f"原始图像总像素数据量: 3 x {H} x {W} = {3 * H * W} pixels")
    print("=" * 60)

    for ratio in compression_ratios:
        print(f"\n【目标压缩率 (Bandwidth Compression Ratio): {ratio} 】")
        print("-" * 60)
        print(f"{'配置方案 (MCS)':<25} | {'信道符号数 (k)':<15} | {'最大比特预算 (NumBits)':<15}")
        print("-" * 60)

        for name, R, M in mcs_configs:
            numbits, num_symbols = calculate_numbits_budget(H, W, ratio, R, M)
            print(f"{name:<25} | {num_symbols:<15} | {numbits:<15}")
        print("-" * 60)