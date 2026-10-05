import torch


def awgn_channel(x, snr, return_noise_power=False):
    # x: input signal
    # snr: Signal-to-Noise Ratio in dB

    snr_linear = 10 ** (snr / 10.0)
    power_x = torch.mean(torch.abs(x) ** 2)
    noise_power = power_x / snr_linear

    real_dtype = x.real.dtype if x.is_complex() else x.dtype
    noise_real = torch.randn(x.shape, dtype=real_dtype, device=x.device)
    noise_imag = torch.randn(x.shape, dtype=real_dtype, device=x.device)
    noise = torch.sqrt(noise_power / 2) * (noise_real + 1j * noise_imag)

    received = x + noise
    if return_noise_power:
        return received, noise_power
    return received


def rician_channel(x, snr, K_factor, return_csi=False):
    """Apply unit-power fast Rician fading and complex AWGN.

    ``K_factor`` is linear, not dB. When ``return_csi`` is true, return
    ``(received, h, noise_power)`` so a coherent receiver can evaluate
    likelihoods conditioned on the exact per-symbol channel coefficient.
    """

    if K_factor < 0:
        raise ValueError("Rician K-factor must be non-negative")

    snr_linear = 10 ** (snr / 10.0)
    power_x = torch.mean(torch.abs(x) ** 2)
    noise_power = power_x / snr_linear

    # Unit-power Rician coefficient: deterministic LOS plus complex Gaussian
    # scatter. Construct every tensor on x.device so this also works on CUDA.
    real_dtype = x.real.dtype if x.is_complex() else x.dtype
    k = torch.as_tensor(K_factor, dtype=real_dtype, device=x.device)
    sigma = torch.sqrt(1.0 / (2.0 * (k + 1.0)))
    s = torch.sqrt(k / (k + 1.0))

    h_los = s.to(dtype=torch.complex64 if real_dtype == torch.float32 else torch.complex128)
    fading_real = torch.randn(x.shape, dtype=real_dtype, device=x.device)
    fading_imag = torch.randn(x.shape, dtype=real_dtype, device=x.device)
    h_nlos = sigma * (fading_real + 1j * fading_imag)

    h = h_los + h_nlos

    noise_real = torch.randn(x.shape, dtype=real_dtype, device=x.device)
    noise_imag = torch.randn(x.shape, dtype=real_dtype, device=x.device)
    noise = torch.sqrt(noise_power / 2) * (noise_real + 1j * noise_imag)

    received = h * x + noise
    if return_csi:
        return received, h, noise_power
    return received
