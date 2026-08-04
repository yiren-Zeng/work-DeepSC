import torch
import torch.nn as nn
import math

from utils.bit_utils import bits_per_index


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

    @staticmethod
    def _normalize_num_embeddings(num_embeddings):
        if isinstance(num_embeddings, (list, tuple)):
            if not num_embeddings:
                raise ValueError(
                    "per-depth num_embeddings must not be empty"
                )
            normalized = [int(value) for value in num_embeddings]
            for value in normalized:
                bits_per_index(value)
            return normalized

        normalized = int(num_embeddings)
        bits_per_index(normalized)
        return normalized

    @staticmethod
    def _apply_fixed_width_noise(indices, num_embeddings, ber):
        width = bits_per_index(num_embeddings)
        bits = torch.zeros(
            (*indices.shape, width),
            device=indices.device,
            dtype=torch.float,
        )

        for bit_index in range(width):
            bits[..., bit_index] = ((indices >> bit_index) & 1).float()

        if isinstance(ber, torch.Tensor):
            probability = ber.to(device=indices.device, dtype=bits.dtype)
        else:
            probability = float(ber)
        mask = torch.bernoulli(torch.ones_like(bits) * probability)
        corrupted_bits = torch.abs(bits - mask)

        corrupted_indices = torch.zeros_like(indices)
        for bit_index in range(width):
            corrupted_indices += (
                corrupted_bits[..., bit_index].to(dtype=indices.dtype)
                * (2 ** bit_index)
            )

        return torch.clamp(
            corrupted_indices, 0, num_embeddings - 1
        )

    def apply_channel_noise(
        self,
        indices,
        num_embeddings,
        snr_db,
        rc=None,
        mod_bits=2,
    ):
        num_embeddings = self._normalize_num_embeddings(num_embeddings)
        if isinstance(num_embeddings, list):
            if indices.ndim < 1 or indices.shape[-1] != len(num_embeddings):
                actual_depth = indices.shape[-1] if indices.ndim >= 1 else None
                raise ValueError(
                    f"indices has depth {actual_depth}, but num_embeddings "
                    f"specifies {len(num_embeddings)} depths"
                )

        ber = self.compute_ber(snr_db, rc=rc, mod_bits=mod_bits)

        if isinstance(ber, torch.Tensor):
            if ber.item() < 1e-9:
                return indices, ber
        elif ber < 1e-9:
            return indices, ber

        if isinstance(num_embeddings, list):
            corrupted_indices = torch.empty_like(indices)
            for depth_index, depth_num_embeddings in enumerate(num_embeddings):
                corrupted_indices[..., depth_index] = (
                    self._apply_fixed_width_noise(
                        indices[..., depth_index],
                        depth_num_embeddings,
                        ber,
                    )
                )
        else:
            corrupted_indices = self._apply_fixed_width_noise(
                indices, num_embeddings, ber
            )

        return corrupted_indices, ber
