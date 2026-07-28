
import torch

def bpsk_modulate(bits):
    # BPSK modulation: 0 -> -1, 1 -> 1
    return 2 * bits - 1

def bpsk_demodulate(symbols):
    # BPSK demodulation: values > 0 -> 1, values <= 0 -> 0
    # 确保处理复数输入，只使用实部
    if symbols.is_complex():
        return (symbols.real > 0).float()
    else:
        return (symbols > 0).float()

def qpsk_modulate(bits):
    # QPSK modulation: takes 2 bits at a time
    # (0,0) -> -1-1j, (0,1) -> -1+1j, (1,0) -> 1-1j, (1,1) -> 1+1j
    # Normalize by sqrt(2) for unit average power
    bits_reshaped = bits.view(-1, 2)
    real_part = 2 * bits_reshaped[:, 0] - 1
    imag_part = 2 * bits_reshaped[:, 1] - 1
    return (real_part + 1j * imag_part) / torch.sqrt(torch.tensor(2.0))

def qpsk_demodulate(symbols):
    # QPSK demodulation
    real_part = (symbols.real > 0).float()
    imag_part = (symbols.imag > 0).float()
    return torch.stack((real_part, imag_part), dim=-1).view(-1)

def qam16_modulate(bits):
    # 16-QAM modulation: takes 4 bits at a time
    # Map 4 bits to 16-QAM constellation points
    # Example mapping (Gray coding is preferred in practice):
    # 0000 -> -3-3j, 0001 -> -3-1j, 0010 -> -3+3j, 0011 -> -3+1j
    # 0100 -> -1-3j, 0101 -> -1-1j, 0110 -> -1+3j, 0111 -> -1+1j
    # 1000 ->  3-3j, 1001 ->  3-1j, 1010 ->  3+3j, 1011 ->  3+1j
    # 1100 ->  1-3j, 1101 ->  1-1j, 1110 ->  1+3j, 1111 ->  1+1j
    # Normalize by sqrt(10) for unit average power (for standard 16-QAM)
    
    bits_reshaped = bits.view(-1, 4)
    
    # Convert 2 bits to integer for real and imag parts
    # Example: 00 -> 0, 01 -> 1, 11 -> 2, 10 -> 3
    # Then map to constellation points: (2*val - 3) for val in [0,1,2,3]
    
    # Gray coding for 16-QAM
    # I: 00->-3, 01->-1, 11->1, 10->3
    # Q: 00->-3, 01->-1, 11->1, 10->3
    
    # Map bits to constellation points using Gray coding
    # First two bits for real part, last two for imaginary part
    real_bits = bits_reshaped[:, :2]
    imag_bits = bits_reshaped[:, 2:]
    
    def gray_map(b0, b1):
        val = 2 * b0 + b1
        if val == 0: return -3
        if val == 1: return -1
        if val == 3: return 1 # 11
        if val == 2: return 3 # 10

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
        if val < -2: return torch.tensor([0,0]) # -3
        if val < 0: return torch.tensor([0,1])  # -1
        if val < 2: return torch.tensor([1,1])  # 1
        return torch.tensor([1,0]) # 3

    # Apply gray_demap to each element
    real_bits = torch.stack([gray_demap(r) for r in real_part])
    imag_bits = torch.stack([gray_demap(i) for i in imag_part])
    
    return torch.cat((real_bits, imag_bits), dim=-1).view(-1).float()


def qam16_llr(symbols, snr_db, device='cuda'):
    """
    16-QAM软解调 - 优化的LLR计算（向量化实现，更快）
    Args:
        symbols: 接收到的复数符号
        snr_db: 信噪比(dB)
        device: 设备
    Returns:
        LLR值，形状为(4*len(symbols),)
    """
    snr_linear = 10 ** (snr_db / 10.0)
    noise_variance = 1.0 / snr_linear

    # 16-QAM星座点（归一化后的值）
    constellation = torch.tensor([
        -3 - 3j, -3 - 1j, -3 + 3j, -3 + 1j,
        -1 - 3j, -1 - 1j, -1 + 3j, -1 + 1j,
        3 - 3j, 3 - 1j, 3 + 3j, 3 + 1j,
        1 - 3j, 1 - 1j, 1 + 3j, 1 + 1j
    ], dtype=torch.complex64, device=device) / torch.sqrt(torch.tensor(10.0, device=device))

    # 对应的比特映射（Gray编码）
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

        # 计算LLR
        llr[bit_pos::4] = (min_one_dist - min_zero_dist) / (2 * noise_variance)

    return llr

