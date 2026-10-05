import itertools
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from communications.channel import rician_channel
from communications.modulation import (
    bpsk_llr,
    bpsk_modulate,
    qam16_llr,
    qam16_modulate,
    qpsk_llr,
    qpsk_modulate,
)
from evaluation.quality import _transmit_ldpc_stream
from test_real import (
    _format_ms_ssim,
    _normalize_channel_seeds,
    _summarize_monte_carlo_trials,
    ms_ssim_to_db,
)


class ChannelReportingTest(unittest.TestCase):
    def test_qam16_modulator_preserves_gray_mapping(self):
        bits = torch.tensor(list(itertools.product([0.0, 1.0], repeat=4)))
        actual = qam16_modulate(bits.reshape(-1))
        levels = {
            (0, 0): -3,
            (0, 1): -1,
            (1, 1): 1,
            (1, 0): 3,
        }
        expected = torch.tensor(
            [
                complex(
                    levels[tuple(map(int, row[:2]))],
                    levels[tuple(map(int, row[2:]))],
                )
                / math.sqrt(10.0)
                for row in bits.tolist()
            ],
            dtype=torch.complex64,
        )

        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.device.type, "cpu")

    def test_ms_ssim_db_conversion_and_format(self):
        self.assertAlmostEqual(ms_ssim_to_db(0.9), 10.0)
        self.assertEqual(_format_ms_ssim(0.9), "0.9000 (10.0000 dB)")
        self.assertTrue(math.isinf(ms_ssim_to_db(1.0)))

    def test_monte_carlo_averages_raw_ms_ssim_before_db_conversion(self):
        trials = [
            {
                "channel_seed": 42,
                "psnr": 20.0,
                "ms_ssim": 0.8,
                "diagnostics": {
                    "combined_stream": {
                        "bit_errors": 1,
                        "payload_bits": 100,
                        "channel_symbols": 50,
                        "source_values": 200,
                    }
                },
            },
            {
                "channel_seed": 43,
                "psnr": 22.0,
                "ms_ssim": 0.9,
                "diagnostics": {
                    "combined_stream": {
                        "bit_errors": 3,
                        "payload_bits": 100,
                        "channel_symbols": 50,
                        "source_values": 200,
                    }
                },
            },
        ]
        summary = _summarize_monte_carlo_trials(trials)
        self.assertAlmostEqual(summary["psnr_mean"], 21.0)
        self.assertAlmostEqual(summary["ms_ssim_mean"], 0.85)
        self.assertAlmostEqual(
            summary["ms_ssim_db_from_mean"], ms_ssim_to_db(0.85)
        )
        self.assertAlmostEqual(summary["ber"], 4 / 200)
        self.assertEqual(summary["channel_seeds"], [42, 43])

    def test_monte_carlo_channel_seeds_must_be_unique(self):
        self.assertEqual(_normalize_channel_seeds([42, 43]), [42, 43])
        with self.assertRaises(ValueError):
            _normalize_channel_seeds([42, 42])

    def test_rician_channel_accepts_linear_float_k_on_input_device(self):
        torch.manual_seed(42)
        symbols = torch.ones(32, dtype=torch.complex64)
        received = rician_channel(symbols, snr=10.0, K_factor=10.0)
        self.assertEqual(received.shape, symbols.shape)
        self.assertEqual(received.device, symbols.device)
        self.assertTrue(received.is_complex())

    def test_rician_channel_can_return_per_symbol_csi(self):
        torch.manual_seed(42)
        symbols = torch.ones(32, dtype=torch.complex64)
        received, gain, noise_variance = rician_channel(
            symbols,
            snr=10.0,
            K_factor=10.0,
            return_csi=True,
        )

        self.assertEqual(received.shape, symbols.shape)
        self.assertEqual(gain.shape, symbols.shape)
        self.assertEqual(gain.device, symbols.device)
        self.assertEqual(noise_variance.ndim, 0)
        self.assertGreater(float(noise_variance), 0.0)

    def test_modulation_llrs_remove_known_complex_gain(self):
        cases = (
            (
                torch.tensor([0.0, 1.0]),
                bpsk_modulate,
                bpsk_llr,
            ),
            (
                torch.tensor(list(itertools.product([0.0, 1.0], repeat=2))),
                qpsk_modulate,
                qpsk_llr,
            ),
            (
                torch.tensor(list(itertools.product([0.0, 1.0], repeat=4))),
                qam16_modulate,
                qam16_llr,
            ),
        )
        complex_gain = 1.3 * torch.exp(torch.tensor(0.7j))
        for bits, modulate, calculate_llr in cases:
            flat_bits = bits.reshape(-1)
            symbols = modulate(flat_bits)
            gain = torch.ones(
                symbols.shape, dtype=torch.complex64, device=symbols.device
            ) * complex_gain
            received = gain * symbols
            llrs = calculate_llr(
                received,
                snr_db=20.0,
                device=torch.device("cpu"),
                channel_gain=gain,
                noise_variance=torch.tensor(0.01),
            )
            torch.testing.assert_close(
                llrs.gt(0).to(flat_bits.dtype), flat_bits
            )

    def test_ldpc_stream_dispatches_to_rician(self):
        bits = np.array([0, 1, 1, 0], dtype=np.uint8)

        def modulate(values):
            return values.to(torch.complex64)

        llr_kwargs = {}

        def llr(values, _snr, _device, **kwargs):
            llr_kwargs.update(kwargs)
            return values.real

        def encode(values, code=None):
            return values

        def decode(values, code=None):
            return (values > 0.5).astype(np.uint8)

        with patch(
            "evaluation.quality.rician_channel",
            side_effect=lambda values, snr, K_factor, return_csi: (
                values,
                torch.ones_like(values),
                torch.tensor(0.5, device=values.device),
            ),
        ) as channel:
            decoded, _ = _transmit_ldpc_stream(
                bits,
                target_snr=3.0,
                ldpc_code=None,
                device=torch.device("cpu"),
                modulate=modulate,
                calculate_llr=llr,
                modulation_bits=1,
                ldpc_encode=encode,
                ldpc_decode=decode,
                channel_type="rician",
                rician_k_factor=10.0,
            )

        np.testing.assert_array_equal(decoded, bits)
        channel.assert_called_once()
        self.assertEqual(channel.call_args.kwargs["K_factor"], 10.0)
        self.assertTrue(channel.call_args.kwargs["return_csi"])
        self.assertIn("channel_gain", llr_kwargs)
        self.assertIn("noise_variance", llr_kwargs)


if __name__ == "__main__":
    unittest.main()
