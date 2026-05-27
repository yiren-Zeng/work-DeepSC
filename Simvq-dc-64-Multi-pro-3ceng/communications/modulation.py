import torch


def bpsk_modulate(bits):
    # BPSK modulation: 0 -> -1, 1 -> 1
    return 2 * bits - 1


def bpsk_demodulate(symbols):  # 硬解调
    # BPSK demodulation: values > 0 -> 1, values <= 0 -> 0
    # 确保处理复数输入，只使用实部
    if symbols.is_complex():
        return (symbols.real > 0).float()
    else:
        return (symbols > 0).float()


def bpsk_llr(received_symbols, snr_db, device):
    """
    计算 BPSK 的 Log-Likelihood Ratios (LLR)
    """
    snr_linear = 10 ** (snr_db / 10.0)
    # 对于归一化符号能量为 1 的 BPSK，单边噪声功率谱密度 N0 = 1 / snr_linear
    n0 = 1.0 / snr_linear

    # 提取实部 (如果经过衰落信道后变成了复数，BPSK的信息依然只承载在实部和相位的投影上)
    if received_symbols.is_complex():
        y_real = received_symbols.real
    else:
        y_real = received_symbols

    # BPSK 映射为: bit 0 -> -1, bit 1 -> +1
    # 标准 AWGN 下的精确 LLR 公式为: 4 * y_real / N0
    llr = (4.0 / n0) * y_real

    return llr.to(device)


def qpsk_modulate(bits):
    # QPSK modulation: takes 2 bits at a time
    # (0,0) -> -1-1j, (0,1) -> -1+1j, (1,0) -> 1-1j, (1,1) -> 1+1j
    # Normalize by sqrt(2) for unit average power
    bits_reshaped = bits.view(-1, 2)
    real_part = 2 * bits_reshaped[:, 0] - 1
    imag_part = 2 * bits_reshaped[:, 1] - 1
    return (real_part + 1j * imag_part) / torch.sqrt(torch.tensor(2.0))


def qpsk_demodulate(symbols):  # 硬解调
    # QPSK demodulation
    real_part = (symbols.real > 0).float()
    imag_part = (symbols.imag > 0).float()
    return torch.stack((real_part, imag_part), dim=-1).view(-1)


def qpsk_llr(received_symbols, snr_db, device):
    """QPSK 极速软解调：纯数学运算，无距离搜索"""
    snr_linear = 10 ** (snr_db / 10.0)
    n0 = 1.0 / snr_linear  # 假设符号能量为 1

    # 根据 QPSK 映射 (real/imag), LLR 近似为 4 * y / N0 (若按 sqrt(2) 归一化需乘以 sqrt(2))
    factor = 4.0 / (n0 * torch.sqrt(torch.tensor(2.0, device=device)))

    y_real = received_symbols.real * factor
    y_imag = received_symbols.imag * factor

    llr = torch.zeros(2 * len(received_symbols), device=device)
    llr[0::2] = y_real
    llr[1::2] = y_imag
    return llr  # 这是张量


def qam16_modulate(bits):
    # 16-QAM modulation: takes 4 bits at a time
    bits_reshaped = bits.view(-1, 4)

    real_bits = bits_reshaped[:, :2]
    imag_bits = bits_reshaped[:, 2:]

    def gray_map(b0, b1):
        val = 2 * b0 + b1
        if val == 0: return -3
        if val == 1: return -1
        if val == 3: return 1  # 11
        if val == 2: return 3  # 10

    real_part = torch.tensor([gray_map(b[0], b[1]) for b in real_bits], dtype=torch.float32)
    imag_part = torch.tensor([gray_map(b[0], b[1]) for b in imag_bits], dtype=torch.float32)

    # Normalize by sqrt(10) for unit average power for standard 16-QAM
    return (real_part + 1j * imag_part) / torch.sqrt(torch.tensor(10.0))


def qam16_demodulate(symbols):
    # 16-QAM demodulation
    # Denormalize first
    symbols_denorm = symbols * torch.sqrt(torch.tensor(10.0))

    real_part = symbols_denorm.real
    imag_part = symbols_denorm.imag

    def gray_demap(val):
        if val < -2: return torch.tensor([0, 0])  # -3
        if val < 0: return torch.tensor([0, 1])  # -1
        if val < 2: return torch.tensor([1, 1])  # 1
        return torch.tensor([1, 0])  # 3

    # Apply gray_demap to each element
    real_bits = torch.stack([gray_demap(r) for r in real_part])
    imag_bits = torch.stack([gray_demap(i) for i in imag_part])

    return torch.cat((real_bits, imag_bits), dim=-1).view(-1).float()


def qam16_llr(symbols, snr_db, device='cuda'):
    """
    16-QAM软解调 - 优化的LLR计算
    已修复：星座图比特映射与调制器严格对齐
    """
    snr_linear = 10 ** (snr_db / 10.0)
    # 对于复数 AWGN 信道，这里计算 N0
    noise_variance = 1.0 / snr_linear

    # 【核心修复】：星座点排列必须与 qam16_modulate 完全一致！
    # qam16_modulate 采用的映射：00->-3, 01->-1, 11->1, 10->3
    constellation = torch.tensor([
        -3 - 3j, -3 - 1j, -3 + 1j, -3 + 3j,  # 对应 0000 到 0010
        -1 - 3j, -1 - 1j, -1 + 1j, -1 + 3j,  # 对应 0100 到 0110
         1 - 3j,  1 - 1j,  1 + 1j,  1 + 3j,  # 对应 1100 到 1110
         3 - 3j,  3 - 1j,  3 + 1j,  3 + 3j   # 对应 1000 到 1010
    ], dtype=torch.complex64, device=device) / torch.sqrt(torch.tensor(10.0, device=device))

    # 对应的比特映射（Gray编码），与上面的星座点绝对一一对应
    bit_mapping = torch.tensor([
        [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0],
        [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 1], [0, 1, 1, 0],
        [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 1], [1, 1, 1, 0],
        [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 1], [1, 0, 1, 0]
    ], dtype=torch.float32, device=device)

    symbols = symbols.to(device)
    num_symbols = len(symbols)

    # 扩展符号和星座点以进行向量化计算
    symbols_expanded = symbols.unsqueeze(1).expand(-1, 16)
    constellation_expanded = constellation.unsqueeze(0).expand(num_symbols, -1)

    # 计算所有符号到所有星座点的距离
    distances = torch.abs(symbols_expanded - constellation_expanded) ** 2

    # 初始化LLR数组
    llr = torch.zeros(4 * num_symbols, device=device)

    # 对每个比特位置计算LLR
    for bit_pos in range(4):
        # 找出该比特为0和1的星座点索引
        zero_mask = bit_mapping[:, bit_pos] == 0
        one_mask = bit_mapping[:, bit_pos] == 1

        # 扩展掩码以匹配符号数量
        zero_mask_expanded = zero_mask.unsqueeze(0).expand(num_symbols, -1)
        one_mask_expanded = one_mask.unsqueeze(0).expand(num_symbols, -1)

        # 获取对应距离
        zero_distances = torch.where(zero_mask_expanded, distances, torch.tensor(float('inf'), device=device))
        one_distances = torch.where(one_mask_expanded, distances, torch.tensor(float('inf'), device=device))

        # 计算最小距离
        min_zero_dist = torch.min(zero_distances, dim=1)[0]
        min_one_dist = torch.min(one_distances, dim=1)[0]

        # 【算法修复】：在标准的 LLR 距离推导中，分母应该是噪声的方差 (N0)
        llr[bit_pos::4] = (min_zero_dist - min_one_dist) / noise_variance

    return llr
