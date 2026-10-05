import torch


def _noise_variance(reference, snr_db, noise_variance):
    real_dtype = reference.real.dtype if reference.is_complex() else reference.dtype
    if noise_variance is None:
        value = 1.0 / (10 ** (snr_db / 10.0))
        variance = torch.as_tensor(
            value, dtype=real_dtype, device=reference.device
        )
    else:
        variance = torch.as_tensor(
            noise_variance, dtype=real_dtype, device=reference.device
        )
    if variance.numel() != 1 or float(variance.item()) <= 0:
        raise ValueError("noise_variance must be a positive scalar")
    return variance


def _matched_observation(received_symbols, channel_gain):
    if channel_gain is None:
        return received_symbols
    gain = torch.as_tensor(
        channel_gain,
        dtype=received_symbols.dtype,
        device=received_symbols.device,
    )
    if gain.shape != received_symbols.shape:
        raise ValueError(
            "channel_gain shape must match received symbols: "
            f"{tuple(gain.shape)} != {tuple(received_symbols.shape)}"
        )
    return gain.conj() * received_symbols


def bpsk_modulate(bits):
    return 2 * bits - 1


def bpsk_llr(
    received_symbols,
    snr_db,
    device,
    channel_gain=None,
    noise_variance=None,
):
    """Return log P(bit=1)/P(bit=0), optionally with perfect CSI."""
    matched = _matched_observation(received_symbols, channel_gain)
    variance = _noise_variance(matched, snr_db, noise_variance)
    return ((4.0 / variance) * matched.real).to(device)


def qpsk_modulate(bits):
    bits = bits.view(-1, 2)
    real_part = 2 * bits[:, 0] - 1
    imaginary_part = 2 * bits[:, 1] - 1
    return (real_part + 1j * imaginary_part) / torch.sqrt(
        torch.tensor(2.0, device=bits.device)
    )


def qpsk_llr(
    received_symbols,
    snr_db,
    device,
    channel_gain=None,
    noise_variance=None,
):
    """Return QPSK LLRs, optionally with a known complex channel gain."""
    matched = _matched_observation(received_symbols, channel_gain)
    variance = _noise_variance(matched, snr_db, noise_variance)
    factor = 4.0 / (
        variance * torch.sqrt(torch.tensor(2.0, device=matched.device))
    )
    llr = torch.zeros(
        2 * len(matched), dtype=matched.real.dtype, device=matched.device
    )
    llr[0::2] = matched.real * factor
    llr[1::2] = matched.imag * factor
    return llr.to(device)


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


def qam16_llr(
    symbols,
    snr_db,
    device,
    channel_gain=None,
    noise_variance=None,
):
    """Return exact Gray-coded 16QAM LLRs with optional perfect CSI."""
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
    symbols = symbols.to(device)
    variance = _noise_variance(symbols, snr_db, noise_variance)
    if channel_gain is None:
        hypotheses = constellation.unsqueeze(0)
    else:
        gain = torch.as_tensor(
            channel_gain, dtype=symbols.dtype, device=device
        )
        if gain.shape != symbols.shape:
            raise ValueError(
                "channel_gain shape must match received symbols: "
                f"{tuple(gain.shape)} != {tuple(symbols.shape)}"
            )
        hypotheses = gain.unsqueeze(1) * constellation.unsqueeze(0)
    distances = torch.abs(symbols.unsqueeze(1) - hypotheses) ** 2
    log_likelihoods = -distances / variance
    llr = torch.zeros(4 * len(symbols), device=device)
    for bit_position in range(4):
        zero_log_likelihoods = log_likelihoods[
            :, bit_mapping[:, bit_position].eq(0)
        ]
        one_log_likelihoods = log_likelihoods[
            :, bit_mapping[:, bit_position].eq(1)
        ]
        llr[bit_position::4] = (
            torch.logsumexp(one_log_likelihoods, dim=1)
            - torch.logsumexp(zero_log_likelihoods, dim=1)
        )
    return llr
