import torch


def bpsk_modulate(bits):
    """Map bits to unit-power BPSK symbols."""
    return 2 * bits - 1


def bpsk_demodulate(symbols):
    """Hard-decision BPSK demodulation."""
    values = symbols.real if symbols.is_complex() else symbols
    return (values > 0).float()


def bpsk_llr(received_symbols, snr_db, device):
    """Return BPSK log-likelihood ratios."""
    snr_linear = 10 ** (snr_db / 10.0)
    noise_variance = 1.0 / snr_linear
    values = (
        received_symbols.real
        if received_symbols.is_complex()
        else received_symbols
    )
    return ((4.0 / noise_variance) * values).to(device)


def qpsk_modulate(bits):
    """Map groups of two bits to unit-power QPSK symbols."""
    bits_reshaped = bits.view(-1, 2)
    real_part = 2 * bits_reshaped[:, 0] - 1
    imag_part = 2 * bits_reshaped[:, 1] - 1
    return (real_part + 1j * imag_part) / torch.sqrt(
        torch.tensor(2.0, device=bits.device)
    )


def qpsk_demodulate(symbols):
    """Hard-decision QPSK demodulation."""
    real_part = (symbols.real > 0).float()
    imag_part = (symbols.imag > 0).float()
    return torch.stack((real_part, imag_part), dim=-1).view(-1)


def qpsk_llr(received_symbols, snr_db, device):
    """Return QPSK log-likelihood ratios."""
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
    """Map groups of four bits to unit-power Gray-coded 16QAM symbols."""
    bits_reshaped = bits.view(-1, 4)

    def gray_map(b0, b1):
        value = 2 * b0 + b1
        if value == 0:
            return -3
        if value == 1:
            return -1
        if value == 3:
            return 1
        return 3

    real_part = torch.tensor(
        [gray_map(bits[0], bits[1]) for bits in bits_reshaped[:, :2]],
        dtype=torch.float32,
    )
    imag_part = torch.tensor(
        [gray_map(bits[0], bits[1]) for bits in bits_reshaped[:, 2:]],
        dtype=torch.float32,
    )
    return (real_part + 1j * imag_part) / torch.sqrt(torch.tensor(10.0))


def qam16_demodulate(symbols):
    """Hard-decision Gray-coded 16QAM demodulation."""
    symbols_denormalized = symbols * torch.sqrt(
        torch.tensor(10.0, device=symbols.device)
    )

    def gray_demap(value):
        if value < -2:
            return torch.tensor([0, 0], device=symbols.device)
        if value < 0:
            return torch.tensor([0, 1], device=symbols.device)
        if value < 2:
            return torch.tensor([1, 1], device=symbols.device)
        return torch.tensor([1, 0], device=symbols.device)

    real_bits = torch.stack([
        gray_demap(value) for value in symbols_denormalized.real
    ])
    imag_bits = torch.stack([
        gray_demap(value) for value in symbols_denormalized.imag
    ])
    return torch.cat((real_bits, imag_bits), dim=-1).view(-1).float()


def qam16_llr(symbols, snr_db, device):
    """Return max-log LLRs matching the Gray mapping in qam16_modulate."""
    snr_linear = 10 ** (snr_db / 10.0)
    noise_variance = 1.0 / snr_linear
    constellation = torch.tensor([
        -3 - 3j, -3 - 1j, -3 + 1j, -3 + 3j,
        -1 - 3j, -1 - 1j, -1 + 1j, -1 + 3j,
         1 - 3j,  1 - 1j,  1 + 1j,  1 + 3j,
         3 - 3j,  3 - 1j,  3 + 1j,  3 + 3j,
    ], dtype=torch.complex64, device=device) / torch.sqrt(
        torch.tensor(10.0, device=device)
    )
    bit_mapping = torch.tensor([
        [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0],
        [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 1], [0, 1, 1, 0],
        [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 1], [1, 1, 1, 0],
        [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 1], [1, 0, 1, 0],
    ], dtype=torch.float32, device=device)

    symbols = symbols.to(device)
    distances = torch.abs(symbols.unsqueeze(1) - constellation.unsqueeze(0)) ** 2
    llr = torch.zeros(4 * len(symbols), device=device)
    infinity = torch.tensor(float("inf"), device=device)
    for bit_position in range(4):
        zero_distances = torch.where(
            bit_mapping[:, bit_position].eq(0).unsqueeze(0),
            distances,
            infinity,
        )
        one_distances = torch.where(
            bit_mapping[:, bit_position].eq(1).unsqueeze(0),
            distances,
            infinity,
        )
        llr[bit_position::4] = (
            zero_distances.min(dim=1).values
            - one_distances.min(dim=1).values
        ) / noise_variance
    return llr
