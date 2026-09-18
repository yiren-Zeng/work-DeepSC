"""CPU-only checks for fixed/automatic rates and LDPC input parsing."""
import argparse
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_seq2seq_channel import (
    dual_k_transmission_plan, parse_ldpc_rate, transmission_plan,
)


class RatePlanTests(unittest.TestCase):
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
        for text, expected in [('1/2', .5), ('0.5', .5), ('3/4', .75), ('0.75', .75)]:
            self.assertEqual(parse_ldpc_rate(text), expected)
        for text in ('1/0', '2/3', 'nan', 'wrong'):
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


if __name__ == '__main__':
    unittest.main()
