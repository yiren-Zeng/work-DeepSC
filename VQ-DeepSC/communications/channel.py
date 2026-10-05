import torch


def awgn_channel(symbols, snr_db, return_noise_power=False):
    snr_linear = 10 ** (snr_db / 10.0)
    signal_power = torch.mean(torch.abs(symbols) ** 2)
    noise_power = signal_power / snr_linear
    real_dtype = symbols.real.dtype if symbols.is_complex() else symbols.dtype
    noise_real = torch.randn(symbols.shape, dtype=real_dtype, device=symbols.device)
    noise_imag = torch.randn(symbols.shape, dtype=real_dtype, device=symbols.device)
    noise = torch.sqrt(noise_power / 2) * (noise_real + 1j * noise_imag)
    received = symbols + noise
    if return_noise_power:
        return received, noise_power
    return received


def rician_channel(symbols, snr_db, k_factor, return_csi=False):
    """Apply unit-power Rician fading followed by complex AWGN.

    ``k_factor`` is the linear LOS-to-scatter power ratio. When
    ``return_csi`` is true, return ``(received, fading, noise_power)`` for a
    coherent receiver with exact per-symbol CSI.
    """
    if k_factor < 0:
        raise ValueError("Rician K-factor must be non-negative")

    snr_linear = 10 ** (snr_db / 10.0)
    signal_power = torch.mean(torch.abs(symbols) ** 2)
    noise_power = signal_power / snr_linear
    real_dtype = symbols.real.dtype if symbols.is_complex() else symbols.dtype
    complex_dtype = (
        torch.complex64 if real_dtype == torch.float32 else torch.complex128
    )

    k = torch.as_tensor(k_factor, dtype=real_dtype, device=symbols.device)
    los_amplitude = torch.sqrt(k / (k + 1.0))
    scatter_sigma = torch.sqrt(1.0 / (2.0 * (k + 1.0)))
    scatter_real = torch.randn(
        symbols.shape, dtype=real_dtype, device=symbols.device
    )
    scatter_imag = torch.randn(
        symbols.shape, dtype=real_dtype, device=symbols.device
    )
    fading = los_amplitude.to(complex_dtype) + scatter_sigma * (
        scatter_real + 1j * scatter_imag
    )

    noise_real = torch.randn(
        symbols.shape, dtype=real_dtype, device=symbols.device
    )
    noise_imag = torch.randn(
        symbols.shape, dtype=real_dtype, device=symbols.device
    )
    noise = torch.sqrt(noise_power / 2) * (noise_real + 1j * noise_imag)
    received = fading * symbols + noise
    if return_csi:
        return received, fading, noise_power
    return received
