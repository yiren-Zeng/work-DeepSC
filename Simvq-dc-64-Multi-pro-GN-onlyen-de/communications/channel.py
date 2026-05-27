import torch


def awgn_channel(x, snr):
    # x: input signal
    # snr: Signal-to-Noise Ratio in dB

    snr_linear = 10 ** (snr / 10.0)
    power_x = torch.mean(torch.abs(x) ** 2)
    noise_power = power_x / snr_linear

    noise = torch.sqrt(noise_power / 2) * (torch.randn_like(x) + 1j * torch.randn_like(x))

    return x + noise


def rician_channel(x, snr, K_factor):
    # x: input signal
    # snr: Signal-to-Noise Ratio in dB
    # K_factor: Rician K-factor in linear scale

    snr_linear = 10 ** (snr / 10.0)
    power_x = torch.mean(torch.abs(x) ** 2)
    noise_power = power_x / snr_linear

    # Rician fading coefficients
    sigma = torch.sqrt(1 / (2 * (K_factor + 1)))
    s = torch.sqrt(K_factor / (K_factor + 1))

    h_los = s * (torch.randn(1) + 1j * torch.randn(1))
    h_nlos = sigma * (torch.randn_like(x) + 1j * torch.randn_like(x))

    h = h_los + h_nlos

    noise = torch.sqrt(noise_power / 2) * (torch.randn_like(x) + 1j * torch.randn_like(x))

    return h * x + noise
