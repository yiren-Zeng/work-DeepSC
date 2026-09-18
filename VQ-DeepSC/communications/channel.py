import torch


def awgn_channel(symbols, snr_db):
    snr_linear = 10 ** (snr_db / 10.0)
    signal_power = torch.mean(torch.abs(symbols) ** 2)
    noise_power = signal_power / snr_linear
    real_dtype = symbols.real.dtype if symbols.is_complex() else symbols.dtype
    noise_real = torch.randn(symbols.shape, dtype=real_dtype, device=symbols.device)
    noise_imag = torch.randn(symbols.shape, dtype=real_dtype, device=symbols.device)
    noise = torch.sqrt(noise_power / 2) * (noise_real + 1j * noise_imag)
    return symbols + noise
