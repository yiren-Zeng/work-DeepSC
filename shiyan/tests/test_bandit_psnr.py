import csv
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import torch

from bandit_psnr_search import (
    CHANNEL_PROFILES,
    EvaluationMetrics,
    _write_csv,
    _write_json,
    _validate_seed_partitions,
    _vectorized_qam16_modulate,
    build_parser,
    calculate_physical_lengths,
    enumerate_exact_actions,
    run_epsilon_greedy,
    summarize_evaluations,
)
from communications.modulation import qam16_modulate


INDEX_COUNTS = (1024, 256)
SOURCE_VALUES = 3 * 256 * 256
TARGET_RATIO = Fraction(1, 32)


class ExactPhysicalBudgetTest(unittest.TestCase):
    def test_default_profile_and_four_exact_action_sets(self):
        self.assertEqual(
            build_parser().parse_args([]).channel_profile, "ldpc12_bpsk"
        )
        expected = {
            "ldpc12_bpsk": {(2, 256), (4, 16)},
            "ldpc12_qpsk": {(16, 256), (32, 16)},
            "ldpc34_qpsk": {(128, 256), (256, 16)},
            "ldpc12_16qam": {(1024, 256), (2048, 16)},
        }
        for profile_key, expected_actions in expected.items():
            with self.subTest(profile=profile_key):
                actions, ledger = enumerate_exact_actions(
                    INDEX_COUNTS,
                    SOURCE_VALUES,
                    CHANNEL_PROFILES[profile_key],
                    TARGET_RATIO,
                    min_k=2,
                    max_k=2048,
                    ldpc_n=256,
                )
                self.assertEqual(set(actions), expected_actions)
                self.assertEqual(set(ledger), expected_actions)
                for action in actions:
                    self.assertEqual(ledger[action].channel_symbols, 6144)
                    self.assertEqual(ledger[action].transmission_ratio, TARGET_RATIO)

    def test_bpsk_8_2_is_not_one_over_32(self):
        lengths = calculate_physical_lengths(
            (8, 2),
            INDEX_COUNTS,
            SOURCE_VALUES,
            CHANNEL_PROFILES["ldpc12_bpsk"],
            ldpc_n=256,
        )
        self.assertEqual(lengths.payload_bits, 3328)
        self.assertEqual(lengths.coded_bits, 6656)
        self.assertEqual(lengths.channel_symbols, 6656)
        self.assertEqual(lengths.transmission_ratio, Fraction(13, 384))
        self.assertNotEqual(lengths.transmission_ratio, TARGET_RATIO)

    def test_exact_calculation_includes_padding(self):
        lengths = calculate_physical_lengths(
            (2, 2),
            (1, 1),
            source_values=12,
            profile=CHANNEL_PROFILES["ldpc12_16qam"],
            ldpc_n=256,
        )
        self.assertEqual(lengths.payload_bits, 2)
        self.assertEqual(lengths.ldpc_input_bits, 128)
        self.assertEqual(lengths.ldpc_padding_bits, 126)
        self.assertEqual(lengths.coded_bits, 256)
        self.assertEqual(lengths.modulation_padding_bits, 0)
        self.assertEqual(lengths.channel_symbols, 64)


class BanditPsnrRewardTest(unittest.TestCase):
    def test_psnr_not_ms_ssim_drives_q_and_common_seed_warmup(self):
        actions = [(2, 256), (4, 16)]
        calls = []

        def evaluator(action, seed):
            calls.append((action, seed))
            # Rankings deliberately disagree: action (4,16) wins only on PSNR.
            psnr = 20.0 if action == (4, 16) else 10.0
            ms_ssim = 0.1 if action == (4, 16) else 0.9
            return EvaluationMetrics(
                action=action,
                snr=0.0,
                seed=seed,
                ms_ssim=ms_ssim,
                psnr=psnr,
                total_diagnostics={},
            )

        agent, trace = run_epsilon_greedy(
            actions=actions,
            evaluator=evaluator,
            episodes=4,
            warmup_pulls=1,
            search_seed_base=42000,
            eps_start=0.0,
            eps_end=0.0,
            eps_decay=30.0,
            agent_seed=42,
        )

        self.assertEqual(calls[0], ((2, 256), 42000))
        self.assertEqual(calls[1], ((4, 16), 42000))
        self.assertEqual(agent.q[(2, 256)], 10.0)
        self.assertEqual(agent.q[(4, 16)], 20.0)
        self.assertGreater(agent.n[(4, 16)], agent.n[(2, 256)])
        self.assertTrue(all("reward_psnr" in record for record in trace))
        self.assertEqual(trace[-1]["action"], [4, 16])
        json.dumps(trace)

    def test_fixed_channel_seed_is_used_for_every_pull(self):
        actions = [(2, 256), (4, 16)]
        calls = []

        def evaluator(action, seed):
            calls.append((action, seed))
            return EvaluationMetrics(
                action=action,
                snr=0.0,
                seed=seed,
                ms_ssim=0.8,
                psnr=20.0 if action == (4, 16) else 10.0,
                total_diagnostics={},
            )

        _, trace = run_epsilon_greedy(
            actions=actions,
            evaluator=evaluator,
            episodes=5,
            warmup_pulls=1,
            search_seed_base=42000,
            eps_start=0.0,
            eps_end=0.0,
            eps_decay=30.0,
            agent_seed=42,
            fixed_channel_seed=42,
        )

        self.assertEqual(len(calls), 5)
        self.assertTrue(all(seed == 42 for _, seed in calls))
        self.assertTrue(all(record["seed"] == 42 for record in trace))
        self.assertEqual(calls[:2], [((2, 256), 42), ((4, 16), 42)])

    def test_fixed_seed_mode_allows_intentional_stage_overlap(self):
        args = build_parser().parse_args(
            [
                "--fixed-channel-seed",
                "42",
                "--confirm-seeds",
                "42",
                "--report-seeds",
                "42",
            ]
        )
        _validate_seed_partitions(args)

    def test_summary_reports_seed_mean_std_and_ci(self):
        records = [
            EvaluationMetrics((4, 16), 0.0, 1, 0.8, 20.0, {}),
            EvaluationMetrics((4, 16), 0.0, 2, 0.9, 22.0, {}),
        ]
        summary = summarize_evaluations(records)
        self.assertEqual(summary["psnr_mean"], 21.0)
        self.assertGreater(summary["psnr_std"], 0.0)
        self.assertGreater(summary["psnr_ci95"], 0.0)
        self.assertTrue(summary["uncertainty_estimated"])
        self.assertEqual(summary["seeds"], [1, 2])
        json.dumps(summary)

    def test_single_fixed_seed_does_not_claim_uncertainty(self):
        summary = summarize_evaluations(
            [EvaluationMetrics((4, 16), 0.0, 42, 0.8, 21.95, {})]
        )
        self.assertEqual(summary["num_seeds"], 1)
        self.assertFalse(summary["uncertainty_estimated"])
        self.assertIn("not estimable", summary["psnr_ci95_method"])


class FastQam16Test(unittest.TestCase):
    def test_vectorized_modulator_matches_legacy_mapping(self):
        patterns = []
        for value in range(16):
            patterns.extend([(value >> shift) & 1 for shift in (3, 2, 1, 0)])
        bits = torch.tensor(patterns, dtype=torch.float32)
        expected = qam16_modulate(bits)
        actual = _vectorized_qam16_modulate(bits)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.device, bits.device)


class OutputSerializationTest(unittest.TestCase):
    def test_json_and_csv_outputs_preserve_selected_psnr_result(self):
        summary = {
            "action": [4, 16],
            "psnr_mean": 21.0,
            "psnr_std": 0.2,
            "psnr_ci95": 0.25,
            "ms_ssim_mean": 0.8,
        }
        payload = {
            "target_ratio": "1/32",
            "profiles": {
                "ldpc12_bpsk": {
                    "ldpc_rate": 0.5,
                    "modulation": "bpsk",
                    "action_ledger": {
                        "4,16": {
                            "channel_symbols": 6144,
                            "transmission_ratio": "1/32",
                        }
                    },
                    "snr_results": {
                        "0": {
                            "best_action": [4, 16],
                            "confirmation": [summary],
                            "report": summary,
                        }
                    },
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "result.json"
            csv_path = Path(directory) / "result.csv"
            _write_json(str(json_path), payload)
            _write_csv(str(csv_path), payload)
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(
                loaded["profiles"]["ldpc12_bpsk"]["snr_results"]["0"][
                    "best_action"
                ],
                [4, 16],
            )
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["selected"], "True")
            self.assertEqual(rows[0]["channel_symbols_per_image"], "6144")


if __name__ == "__main__":
    unittest.main()
