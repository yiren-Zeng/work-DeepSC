import torch


def bpsk_modulate(bits):
    return 2 * bits - 1


def bpsk_llr(received_symbols, snr_db, device):
    snr_linear = 10 ** (snr_db / 10.0)
    noise_variance = 1.0 / snr_linear
    values = received_symbols.real if received_symbols.is_complex() else received_symbols
    return ((4.0 / noise_variance) * values).to(device)


def qpsk_modulate(bits):
    bits = bits.view(-1, 2)
    real_part = 2 * bits[:, 0] - 1
    imaginary_part = 2 * bits[:, 1] - 1
    return (real_part + 1j * imaginary_part) / torch.sqrt(
        torch.tensor(2.0, device=bits.device)
    )


def qpsk_llr(received_symbols, snr_db, device):
    snr_linear = 10 ** (snr_db / 10.0)
    noise_variance = 1.0 / snr_linear
    factor = 4.0 / (
        noise_variance * torch.sqrt(torch.tensor(2.0, device=device))
    )
    llr = torch.zeros(2 * len(received_symbols), device=device)
    llr[0::2] = received_symbols.real * factor
    llr[1::2] = received_symbols.imag * factor
    return llr


def qam16_modulate(bits):
    bits = bits.view(-1, 4)
    mapping = torch.tensor([-3.0, -1.0, 3.0, 1.0], device=bits.device)
    real_index = (2 * bits[:, 0] + bits[:, 1]).long()
    imaginary_index = (2 * bits[:, 2] + bits[:, 3]).long()
    real_part = mapping[real_index]
    imaginary_part = mapping[imaginary_index]
    return (real_part + 1j * imaginary_part) / torch.sqrt(
        torch.tensor(10.0, device=bits.device)
    )


def qam16_llr(symbols, snr_db, device):
    snr_linear = 10 ** (snr_db / 10.0)
    noise_variance = 1.0 / snr_linear
    constellation = torch.tensor(
        [
            -3 - 3j, -3 - 1j, -3 + 1j, -3 + 3j,
            -1 - 3j, -1 - 1j, -1 + 1j, -1 + 3j,
            1 - 3j, 1 - 1j, 1 + 1j, 1 + 3j,
            3 - 3j, 3 - 1j, 3 + 1j, 3 + 3j,
        ],
        dtype=torch.complex64,
        device=device,
    ) / torch.sqrt(torch.tensor(10.0, device=device))
    bit_mapping = torch.tensor(
        [
            [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0],
            [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 1], [0, 1, 1, 0],
            [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 1], [1, 1, 1, 0],
            [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 1], [1, 0, 1, 0],
        ],
        dtype=torch.float32,
        device=device,
    )
    distances = torch.abs(symbols.to(device).unsqueeze(1) - constellation.unsqueeze(0)) ** 2
    llr = torch.zeros(4 * len(symbols), device=device)
    infinity = torch.tensor(float("inf"), device=device)
    for bit_position in range(4):
        zero_distances = torch.where(
            bit_mapping[:, bit_position].eq(0).unsqueeze(0), distances, infinity
        )
        one_distances = torch.where(
            bit_mapping[:, bit_position].eq(1).unsqueeze(0), distances, infinity
        )
        llr[bit_position::4] = (
            zero_distances.min(dim=1).values - one_distances.min(dim=1).values
        ) / noise_variance
    return llr
