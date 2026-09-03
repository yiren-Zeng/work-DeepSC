import torch


def awgn_channel(x, snr):
    # x: input signal
    # snr: Signal-to-Noise Ratio in dB

    snr_linear = 10 ** (snr / 10.0)
    power_x = torch.mean(torch.abs(x) ** 2)
    noise_power = power_x / snr_linear

    real_dtype = x.real.dtype if x.is_complex() else x.dtype
    noise_real = torch.randn(x.shape, dtype=real_dtype, device=x.device)
    noise_imag = torch.randn(x.shape, dtype=real_dtype, device=x.device)
    noise = torch.sqrt(noise_power / 2) * (noise_real + 1j * noise_imag)

    return x + noise


def rician_channel(x, snr, K_factor):
    # x: input signal
    # snr: Signal-to-Noise Ratio in dB
    # K_factor: Rician K-Factor in linear scale

    snr_linear = 10 ** (snr / 10.0)
    power_x = torch.mean(torch.abs(x) ** 2)
    noise_power = power_x / snr_linear

    # Rician fading coefficients
    sigma = torch.sqrt(1 / (2 * (K_factor + 1)))
    s = torch.sqrt(K_factor / (K_factor + 1))

    h_los = s * (torch.randn(1) + 1j * torch.randn(1))
    h_nlos = sigma * (torch.randn_like(x) + 1j * torch.randn_like(x))

    h = h_los + h_nlos

    real_dtype = x.real.dtype if x.is_complex() else x.dtype
    noise_real = torch.randn(x.shape, dtype=real_dtype, device=x.device)
    noise_imag = torch.randn(x.shape, dtype=real_dtype, device=x.device)
    noise = torch.sqrt(noise_power / 2) * (noise_real + 1j * noise_imag)

    return h * x + noise
