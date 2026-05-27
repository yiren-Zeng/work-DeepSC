# import torch
# import torch.nn as nn
# import math
#
#
# class FiniteBlocklengthChannel(nn.Module):
#     def __init__(self, channel_coding_rate, modulation_bits, coded_block_length_bits, device):
#         """
#         Args:
#             channel_coding_rate (float): 信道编码率 R_c (e.g., 0.5)
#             modulation_bits (int): 调制阶数 M (e.g., QPSK=2 bits/symbol)
#             coded_block_length_bits (int): 编码后的码块长度(bits) (e.g., 256)
#                                            注意：这是指 LDPC 输出的比特数 n
#             device: 设备
#         """
#         super().__init__()
#         self.R_c = channel_coding_rate
#         self.M_bits = modulation_bits
#
#         # === 关键修正 ===
#         # 论文公式 (8) 中的 L 指的是 Channel Uses (符号长度)。
#         # L = 编码比特数 / 调制阶数
#         # 例如：256 bits / 2 (QPSK) = 128 Symbols
#         self.L_channel_uses = coded_block_length_bits // modulation_bits
#
#         self.device = device
#
#     def q_function(self, x):
#         """
#         Q函数近似计算: Q(x) = 0.5 * erfc(x / sqrt(2))
#         """
#         return 0.5 * torch.erfc(x / math.sqrt(2))
#
#     def compute_ber(self, snr_db, rc=None):
#         """
#         基于有限码长理论计算 BER。
#         """
#         real_rc = rc if rc is not None else self.R_c
#
#         # === 传输速率 R ===
#         # 论文公式 (8) 中的 R 也是指 bits per channel use (bpcu)。
#         # R = 编码率(bits/coded_bit) * 调制阶数(coded_bits/symbol)
#         # 例如: 0.5 * 2 = 1.0 bit/symbol
#         R_transport = real_rc * self.M_bits
#
#         # SINR (线性值)
#         gamma = 10 ** (snr_db / 10.0)
#
#         # AWGN 信道容量 C = log2(1 + SNR)
#         C = torch.log2(1 + gamma)
#
#         # 信道色散 V
#         # V = (1 - (1 + gamma)^(-2)) * (log2(e))^2
#         log2_e = math.log2(math.e)
#         V = (1 - (1 + gamma).pow(-2)) * (log2_e ** 2)
#         V = torch.clamp(V, min=1e-9)
#
#         # === 关键修正 ===
#         # 使用计算出的符号长度 L (Channel Uses) 代入公式
#         L_tensor = torch.tensor(self.L_channel_uses, device=self.device).float()
#
#         # FBL 误差概率公式 (论文 Eq. 8): Q( sqrt(L) * (C - R) / sqrt(V) )
#         q_arg = torch.sqrt(L_tensor) * (C - R_transport) / torch.sqrt(V)
#
#         # 包错误率 (Block Error Rate)
#         rho = self.q_function(q_arg)
#
#         # 近似 BER: BER ≈ rho / (R_transport * L)
#         # 即 rho / (有效信息比特数 k)
#         # k = R_transport * L = (real_rc * M) * (n / M) = real_rc * n
#         rho = torch.clamp(rho, min=1e-12, max=1.0)
#
#         k_info_bits = R_transport * L_tensor
#         ber = rho / k_info_bits
#
#         # 物理限制：BER 最大 0.5
#         ber = torch.clamp(ber, max=0.5)
#
#         return ber
#
#     def apply_channel_noise(self, indices, num_embeddings, snr_db, rc=None):
#         """
#         模拟信道传输：计算 BER 并翻转比特。
#         """
#         # 1. 计算当前条件下的 BER
#         ber = self.compute_ber(snr_db, rc=rc)
#
#         if isinstance(ber, torch.Tensor):
#             if ber.item() < 1e-9:
#                 return indices, ber
#         elif ber < 1e-9:
#             return indices, ber
#
#         # 2. 整数索引 -> 二进制
#         bits_per_token = int(math.ceil(math.log2(num_embeddings)))
#
#         bits = torch.zeros((*indices.shape, bits_per_token), device=self.device, dtype=torch.float)
#
#         for i in range(bits_per_token):
#             bits[..., i] = ((indices >> i) & 1).float()
#
#         # 3. 生成错误掩码
#         mask = torch.bernoulli(torch.full_like(bits, ber))
#
#         # 4. 异或翻转
#         corrupted_bits = torch.abs(bits - mask)
#
#         # 5. 二进制 -> 整数
#         corrupted_indices = torch.zeros_like(indices)
#         for i in range(bits_per_token):
#             corrupted_indices += corrupted_bits[..., i].long() * (2 ** i)
#
#         # 6. 截断保护
#         corrupted_indices = torch.clamp(corrupted_indices, 0, num_embeddings - 1)
#
#         return corrupted_indices, ber


import torch
import torch.nn as nn
import math


class FiniteBlocklengthChannel(nn.Module):
    def __init__(self, channel_coding_rate, coded_block_length_bits, device):
        """
        Args:
            channel_coding_rate (float): 信道编码率 R_c (e.g., 0.5)
            coded_block_length_bits (int): 编码后的码块长度(bits) (e.g., 256)
                                           注意：这是指 LDPC 输出的比特数 n
            device: 设备
        """
        super().__init__()
        self.R_c = channel_coding_rate

        # === 关键修正 ===
        # 论文公式 (8) 中的 L 指的是 Channel Uses (符号长度)。
        # L = 编码比特数 / 调制阶数
        # 例如：256 bits / 2 (QPSK) = 128 Symbols
        self.coded_block_length_bits = coded_block_length_bits
        self.device = device

    def q_function(self, x):
        """
        Q函数近似计算: Q(x) = 0.5 * erfc(x / sqrt(2))
        """
        return 0.5 * torch.erfc(x / math.sqrt(2))

    def compute_ber(self, snr_db, rc=None, mod_bits=2):
        """
        基于有限码长理论计算 BER。
        """
        real_rc = rc if rc is not None else self.R_c

        # 动态计算符号长度L
        L_uses = self.coded_block_length_bits // mod_bits
        L_tensor = torch.tensor(L_uses, device=self.device).float()

        # === 动态计算传输速率 R ===
        # 论文公式 (8) 中的 R 也是指 bits per channel use (bpcu)。
        # R = 编码率(bits/coded_bit) * 调制阶数(coded_bits/symbol)
        # 例如: 0.5 * 2 = 1.0 bit/symbol
        R_transport = real_rc * mod_bits

        # SINR (线性值)
        gamma = 10 ** (snr_db / 10.0)

        # AWGN 信道容量 C = log2(1 + SNR)
        C = torch.log2(1 + gamma)

        # 信道色散 V
        # V = (1 - (1 + gamma)^(-2)) * (log2(e))^2
        log2_e = math.log2(math.e)
        V = (1 - (1 + gamma).pow(-2)) * (log2_e ** 2)
        V = torch.clamp(V, min=1e-9)

        # FBL 误差概率公式 (论文 Eq. 8): Q( sqrt(L) * (C - R) / sqrt(V) )
        q_arg = torch.sqrt(L_tensor) * (C - R_transport) / torch.sqrt(V)

        # 包错误率 (Block Error Rate)
        rho = self.q_function(q_arg)

        # 近似 BER: BER ≈ rho / (R_transport * L)
        # 即 rho / (有效信息比特数 k)
        # k = R_transport * L = (real_rc * M) * (n / M) = real_rc * n
        rho = torch.clamp(rho, min=1e-12, max=1.0)

        k_info_bits = R_transport * L_tensor
        ber = rho / k_info_bits

        # 物理限制：BER 最大 0.5
        ber = torch.clamp(ber, max=0.5)

        return ber

    def apply_channel_noise(self, indices, num_embeddings, snr_db, rc=None, mod_bits=2):
        # 【核心修改 3】将 mod_bits 传递给 compute_ber
        ber = self.compute_ber(snr_db, rc=rc, mod_bits=mod_bits)

        if isinstance(ber, torch.Tensor):
            if ber.item() < 1e-9:
                return indices, ber
        elif ber < 1e-9:
            return indices, ber

        bits_per_token = int(math.ceil(math.log2(num_embeddings)))
        bits = torch.zeros((*indices.shape, bits_per_token), device=self.device, dtype=torch.float)

        for i in range(bits_per_token):
            bits[..., i] = ((indices >> i) & 1).float()

        mask = torch.bernoulli(torch.full_like(bits, ber))
        corrupted_bits = torch.abs(bits - mask)

        corrupted_indices = torch.zeros_like(indices)
        for i in range(bits_per_token):
            corrupted_indices += corrupted_bits[..., i].long() * (2 ** i)

        corrupted_indices = torch.clamp(corrupted_indices, 0, num_embeddings - 1)

        return corrupted_indices, ber

