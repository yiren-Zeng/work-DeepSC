"""CPU-only checks for fixed/automatic rates and LDPC input parsing."""
import argparse
import itertools
import math
from pathlib import Path
import sys
import unittest

import torch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_seq2seq_channel import (
    awgn_channel,
    dual_k_transmission_plan,
    exact_bpsk_llr,
    exact_qam16_llr,
    exact_qpsk_llr,
    format_ms_ssim,
    ms_ssim_to_db,
    parse_ldpc_rate,
    qam16_modulate_vectorized,
    rician_channel,
    transmission_plan,
)
from aggregate_seq2seq_channel_trials import (
    normalize_channel_seeds,
    summarize_trial_records,
)


class RatePlanTests(unittest.TestCase):
    def test_monte_carlo_averages_raw_ms_ssim_before_db_conversion(self):
        trials = [
            {
                "channel_seed": 42,
                "psnr": 20.0,
                "ms_ssim": 0.8,
                "source_bit_errors": 1,
                "source_payload_bits": 100,
                "combined_bit_errors": 1,
                "ldpc_input_bits": 100,
                "channel_symbols": 50,
                "source_values": 200,
            },
            {
                "channel_seed": 43,
                "psnr": 22.0,
                "ms_ssim": 0.9,
                "source_bit_errors": 3,
                "source_payload_bits": 100,
                "combined_bit_errors": 3,
                "ldpc_input_bits": 100,
                "channel_symbols": 50,
                "source_values": 200,
            },
        ]
        summary = summarize_trial_records(trials)
        self.assertAlmostEqual(summary["psnr_mean"], 21.0)
        self.assertAlmostEqual(summary["ms_ssim_mean"], 0.85)
        self.assertAlmostEqual(
            summary["ms_ssim_db_from_mean"], -10 * math.log10(0.15)
        )
        self.assertAlmostEqual(summary["source_ber"], 4 / 200)

    def test_monte_carlo_seeds_must_be_unique(self):
        self.assertEqual(normalize_channel_seeds([42, 43]), [42, 43])
        with self.assertRaises(ValueError):
            normalize_channel_seeds([42, 42])

    def test_ms_ssim_db_format(self):
        self.assertAlmostEqual(ms_ssim_to_db(0.9), 10.0)
        self.assertEqual(format_ms_ssim(0.9), "0.900000 (10.0000 dB)")

    def test_rician_channel_runs_on_input_device(self):
        torch.manual_seed(42)
        symbols = torch.ones(32, dtype=torch.complex64)
        received = rician_channel(symbols, snr_db=10.0, k_factor=10.0)
        self.assertEqual(received.shape, symbols.shape)
        self.assertEqual(received.device, symbols.device)
        self.assertTrue(received.is_complex())

    def test_channels_expose_receiver_parameters(self):
        torch.manual_seed(42)
        symbols = torch.ones(32, dtype=torch.complex64)
        awgn_received, awgn_variance = awgn_channel(
            symbols, snr_db=10.0, return_noise_power=True
        )
        rician_received, gain, rician_variance = rician_channel(
            symbols, snr_db=10.0, k_factor=10.0, return_csi=True
        )
        self.assertEqual(awgn_received.shape, symbols.shape)
        self.assertEqual(rician_received.shape, symbols.shape)
        self.assertEqual(gain.shape, symbols.shape)
        torch.testing.assert_close(awgn_variance, torch.tensor(0.1))
        torch.testing.assert_close(rician_variance, torch.tensor(0.1))

    def test_exact_llrs_use_known_complex_gain(self):
        cases = (
            (
                torch.tensor([0.0, 1.0]),
                lambda bits: 2 * bits - 1,
                exact_bpsk_llr,
            ),
            (
                torch.tensor(list(itertools.product([0.0, 1.0], repeat=2))),
                lambda bits: (
                    (2 * bits.view(-1, 2)[:, 0] - 1)
                    + 1j * (2 * bits.view(-1, 2)[:, 1] - 1)
                ) / math.sqrt(2.0),
                exact_qpsk_llr,
            ),
            (
                torch.tensor(list(itertools.product([0.0, 1.0], repeat=4))),
                qam16_modulate_vectorized,
                exact_qam16_llr,
            ),
        )
        complex_gain = 1.3 * torch.exp(torch.tensor(0.7j))
        for bits, modulate, calculate_llr in cases:
            flat_bits = bits.reshape(-1)
            symbols = modulate(flat_bits)
            gain = torch.ones_like(symbols, dtype=torch.complex64) * complex_gain
            llrs = calculate_llr(
                gain * symbols,
                snr_db=20.0,
                device=torch.device("cpu"),
                channel_gain=gain,
                noise_variance=torch.tensor(0.01),
            )
            torch.testing.assert_close(
                llrs.gt(0).to(flat_bits.dtype), flat_bits
            )

    def test_vectorized_qam16_preserves_gray_mapping(self):
        bits = torch.tensor(list(itertools.product([0.0, 1.0], repeat=4)))
        actual = qam16_modulate_vectorized(bits.reshape(-1))
        levels = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}
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

    def plan(self, policy, target_k=None, ldpc_k=128, modulation_bits=2, denominator=48):
        return transmission_plan(5120, 196608, ldpc_k, 256, modulation_bits,
                                 denominator, policy, target_k=target_k)

    def test_fixed_cases(self):
        for k, ldpc_k, bits, symbols, padding in [(2, 128, 2, 5120, 0),
                                                  (2, 192, 2, 3456, 64),
                                                  (4, 128, 4, 5120, 0)]:
            with self.subTest(k=k, ldpc_k=ldpc_k, bits=bits):
                plan = self.plan('fixed', k, ldpc_k, bits)
                self.assertEqual(plan['selected']['channel_symbols'], symbols)
                self.assertEqual(plan['selected']['ldpc_padding_bits'], padding)
                self.assertEqual(plan['budget_padding_bits'], 0)

    def test_all_trained_fixed_sizes(self):
        for k in (2, 4, 8, 16, 32, 64):
            self.assertEqual(self.plan('fixed', k)['selected']['K'], k)

    def test_invalid_fixed_sizes(self):
        for k in (0, 1, 3, 6, 65, 128):
            with self.assertRaises(ValueError):
                self.plan('fixed', k)

    def test_policy_conflicts(self):
        for policy, k in [('fixed', None), ('nearest', 2), ('padded', 2), ('unknown', None)]:
            with self.assertRaises(ValueError):
                self.plan(policy, k)
        with self.assertRaises(ValueError):
            self.plan('nearest', denominator=0)

    def test_legacy_nearest(self):
        for ldpc_k, bits, expected_k in [(128, 2, 2), (192, 2, 2), (128, 4, 4)]:
            self.assertEqual(self.plan('nearest', ldpc_k=ldpc_k, modulation_bits=bits)['selected']['K'], expected_k)

    def test_legacy_padding(self):
        plan = self.plan('padded', ldpc_k=192)
        self.assertEqual(plan['selected']['K'], 2)
        self.assertEqual(plan['budget_padding_bits'], 1024)
        with self.assertRaises(ValueError):
            self.plan('padded', ldpc_k=192, denominator=64)

    def test_ldpc_parse(self):
        for text, expected in [
            ('7/16', .4375), ('0.4375', .4375),
            ('1/2', .5), ('0.5', .5),
            ('5/8', .625), ('0.625', .625),
            ('3/4', .75), ('0.75', .75),
        ]:
            self.assertEqual(parse_ldpc_rate(text), expected)
        for text in ('1/0', '2/3', '3/5', 'nan', 'wrong'):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_ldpc_rate(text)

    def test_dual_k_exact_one_over_64(self):
        cases = [
            (4, 16, 128, 2, 3072),
            (16, 4, 192, 2, 4608),
            (8, 64, 192, 2, 4608),
            (32, 16, 128, 4, 6144),
        ]
        for bottom_k, top_k, ldpc_k, modulation_bits, payload in cases:
            with self.subTest(bottom_k=bottom_k, top_k=top_k):
                plan = dual_k_transmission_plan(
                    1024, 256, bottom_k, top_k, 196608,
                    ldpc_k, 256, modulation_bits, 64,
                )
                self.assertEqual(plan['selected']['source_payload_bits'], payload)
                self.assertEqual(plan['selected']['channel_symbols'], 3072)
                self.assertEqual(plan['selected']['transmission_ratio'], 1 / 64)

    def test_dual_k_rejects_non_power_of_two(self):
        for bottom_k, top_k in ((3, 4), (4, 6), (1, 2), (64, 128)):
            with self.assertRaises(ValueError):
                dual_k_transmission_plan(
                    1024, 256, bottom_k, top_k, 196608,
                    128, 256, 2, 64,
                )

    def test_dual_k_supports_fractional_ratio_label(self):
        plan = dual_k_transmission_plan(
            1024, 256, 64, 64, 196608,
            128, 256, 4, 51.2,
        )
        self.assertEqual(plan['selected']['channel_symbols'], 3840)
        self.assertEqual(plan['selected']['absolute_ratio_error'], 0.0)


if __name__ == '__main__':
    unittest.main()
