import torch
import torch.nn as nn
import math


class FiniteBlocklengthChannel(nn.Module):
    def __init__(self, channel_coding_rate, coded_block_length_bits, device):
        super().__init__()
        self.R_c = channel_coding_rate
        self.coded_block_length_bits = coded_block_length_bits
        self.device = device

    def q_function(self, x):
        return 0.5 * torch.erfc(x / math.sqrt(2))

    def compute_ber(self, snr_db, rc=None, mod_bits=2):
        real_rc = rc if rc is not None else self.R_c

        L_uses = self.coded_block_length_bits // mod_bits
        L_tensor = torch.tensor(L_uses, device=self.device).float()

        R_transport = real_rc * mod_bits
        gamma = 10 ** (snr_db / 10.0)

        C = torch.log2(1 + gamma)

        log2_e = math.log2(math.e)
        V = (1 - (1 + gamma).pow(-2)) * (log2_e ** 2)
        V = torch.clamp(V, min=1e-9)

        q_arg = torch.sqrt(L_tensor) * (C - R_transport) / torch.sqrt(V)
        rho = self.q_function(q_arg)
        rho = torch.clamp(rho, min=1e-12, max=1.0)

        k_info_bits = R_transport * L_tensor
        ber = rho / k_info_bits
        ber = torch.clamp(ber, max=0.5)

        return ber

    def apply_channel_noise(self, indices, num_embeddings, snr_db, rc=None, mod_bits=2):
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
