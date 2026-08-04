"""Contracts for PSNR-bandit search over four independent RAQ-RVQ K values."""

import csv
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

from bandit_independent_raq_rvq_psnr_search import (
    CHANNEL_PROFILES,
    EvaluationMetrics,
    _evaluate_model_action,
    _write_csv,
    _write_json,
    build_parser,
    calculate_physical_lengths,
    enumerate_exact_actions,
    run_epsilon_greedy,
)


INDEX_COUNTS = (1024, 256)
SOURCE_VALUES = 3 * 256 * 256
TARGET_RATIO = Fraction(3, 32)
REFERENCE_ACTION = ((16, 16), (4, 4))


class FourStreamPhysicalBudgetTest(unittest.TestCase):
    def test_default_ratio_is_one_over_32(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.target_ratio, Fraction(1, 32))
        self.assertEqual(args.episodes, 20)
        self.assertEqual(args.confirm_top_k, 2)
        self.assertEqual(args.stream_packing, "per_stage")

    def test_each_scale_stage_stream_is_padded_independently(self):
        lengths = calculate_physical_lengths(
            ((2, 2), (2, 2)),
            index_counts=(1, 1),
            source_values=12,
            profile=CHANNEL_PROFILES["ldpc12_bpsk"],
            ldpc_n=256,
        )

        self.assertEqual(len(lengths.streams), 4)
        self.assertEqual(
            [stream.payload_bits for stream in lengths.streams],
            [1, 1, 1, 1],
        )
        self.assertEqual(
            [stream.ldpc_input_bits for stream in lengths.streams],
            [128, 128, 128, 128],
        )
        self.assertEqual(
            [stream.ldpc_padding_bits for stream in lengths.streams],
            [127, 127, 127, 127],
        )
        self.assertEqual(
            [stream.coded_bits for stream in lengths.streams],
            [256, 256, 256, 256],
        )
        self.assertEqual(
            [stream.modulation_padding_bits for stream in lengths.streams],
            [0, 0, 0, 0],
        )
        self.assertEqual(lengths.payload_bits, 4)
        self.assertEqual(lengths.ldpc_input_bits, 512)
        self.assertEqual(lengths.ldpc_padding_bits, 508)
        self.assertEqual(lengths.coded_bits, 1024)
        self.assertEqual(lengths.channel_symbols, 1024)

    def test_kodak_ldpc12_bpsk_three_over_32_has_exactly_50_arms(self):
        actions, ledger = enumerate_exact_actions(
            index_counts=INDEX_COUNTS,
            source_values=SOURCE_VALUES,
            profile=CHANNEL_PROFILES["ldpc12_bpsk"],
            target_ratio=TARGET_RATIO,
            min_k=2,
            max_k=64,
            ldpc_n=256,
        )

        self.assertEqual(len(actions), 50)
        self.assertEqual(len(set(actions)), 50)
        self.assertEqual(set(actions), set(ledger))
        self.assertIn(REFERENCE_ACTION, actions)
        for action in actions:
            self.assertEqual(len(action), 2)
            self.assertTrue(all(len(scale) == 2 for scale in action))
            self.assertEqual(len(ledger[action].streams), 4)
            self.assertEqual(ledger[action].channel_symbols, 18432)
            self.assertEqual(ledger[action].transmission_ratio, TARGET_RATIO)

    def test_kodak_ldpc12_bpsk_one_over_32_has_three_arms(self):
        actions, ledger = enumerate_exact_actions(
            index_counts=INDEX_COUNTS,
            source_values=SOURCE_VALUES,
            profile=CHANNEL_PROFILES["ldpc12_bpsk"],
            target_ratio=Fraction(1, 32),
            min_k=2,
            max_k=64,
            ldpc_n=256,
        )

        expected = {
            ((2, 2), (2, 8)),
            ((2, 2), (4, 4)),
            ((2, 2), (8, 2)),
        }
        self.assertEqual(set(actions), expected)
        self.assertEqual(set(ledger), expected)
        for action in actions:
            self.assertEqual(ledger[action].payload_bits, 3072)
            self.assertEqual(ledger[action].channel_symbols, 6144)
            self.assertEqual(
                ledger[action].transmission_ratio, Fraction(1, 32)
            )

    def test_four_k_are_ordered_and_have_no_product_or_broadcast_constraint(self):
        asymmetric_action = ((64, 32), (2, 4))
        lengths = calculate_physical_lengths(
            asymmetric_action,
            INDEX_COUNTS,
            SOURCE_VALUES,
            CHANNEL_PROFILES["ldpc12_bpsk"],
            ldpc_n=256,
        )

        self.assertEqual(lengths.action, asymmetric_action)
        self.assertEqual(lengths.bits_per_index, ((6, 5), (1, 2)))
        self.assertEqual(lengths.to_dict()["action"], [[64, 32], [2, 4]])
        self.assertNotEqual(64, 32)
        self.assertNotEqual(2, 4)
        self.assertNotEqual(64 * 32, 2 * 4)

        actions, _ = enumerate_exact_actions(
            INDEX_COUNTS,
            SOURCE_VALUES,
            CHANNEL_PROFILES["ldpc12_bpsk"],
            TARGET_RATIO,
            min_k=2,
            max_k=64,
            ldpc_n=256,
        )
        # This exact-rate arm is asymmetric at both scales and has unequal
        # per-scale products. It must not be rejected or silently broadcast.
        self.assertIn(((2, 64), (4, 64)), actions)

    def test_combined_ldpc34_qpsk_has_one_over_32_without_four_stream_padding(self):
        action = ((4, 64), (8, 2))
        profile = CHANNEL_PROFILES["ldpc34_qpsk"]
        combined = calculate_physical_lengths(
            action,
            INDEX_COUNTS,
            SOURCE_VALUES,
            profile,
            ldpc_n=256,
            stream_packing="combined",
        )
        per_stage = calculate_physical_lengths(
            action,
            INDEX_COUNTS,
            SOURCE_VALUES,
            profile,
            ldpc_n=256,
            stream_packing="per_stage",
        )

        self.assertEqual(combined.payload_bits, 9216)
        self.assertEqual(len(combined.streams), 1)
        self.assertEqual(combined.ldpc_input_bits, 9216)
        self.assertEqual(combined.ldpc_padding_bits, 0)
        self.assertEqual(combined.coded_bits, 12288)
        self.assertEqual(combined.channel_symbols, 6144)
        self.assertEqual(combined.transmission_ratio, Fraction(1, 32))
        self.assertEqual(per_stage.channel_symbols, 6272)
        self.assertEqual(len(per_stage.streams), 4)
        self.assertEqual(per_stage.transmission_ratio, Fraction(49, 1536))

        actions, ledger = enumerate_exact_actions(
            INDEX_COUNTS,
            SOURCE_VALUES,
            profile,
            Fraction(1, 32),
            min_k=2,
            max_k=64,
            ldpc_n=256,
            stream_packing="combined",
        )
        self.assertEqual(len(actions), 50)
        self.assertIn(action, actions)
        self.assertEqual(ledger[action].transmission_ratio, Fraction(1, 32))

        per_stage_actions, _ = enumerate_exact_actions(
            INDEX_COUNTS,
            SOURCE_VALUES,
            profile,
            Fraction(1, 32),
            min_k=2,
            max_k=64,
            ldpc_n=256,
            stream_packing="per_stage",
        )
        self.assertEqual(len(per_stage_actions), 54)
        self.assertNotIn(action, per_stage_actions)


class IndependentRaqRvqBanditRewardTest(unittest.TestCase):
    def test_psnr_not_ms_ssim_drives_q_with_nested_actions(self):
        actions = [((16, 16), (4, 4)), ((64, 4), (4, 4))]
        calls = []

        def evaluator(action, seed):
            calls.append((action, seed))
            psnr = 25.0 if action == actions[1] else 20.0
            ms_ssim = 0.1 if action == actions[1] else 0.9
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

        self.assertEqual(calls[:2], [(actions[0], 42000), (actions[1], 42000)])
        self.assertEqual(agent.q[actions[0]], 20.0)
        self.assertEqual(agent.q[actions[1]], 25.0)
        self.assertGreater(agent.n[actions[1]], agent.n[actions[0]])
        self.assertTrue(all("reward_psnr" in record for record in trace))
        self.assertEqual(trace[-1]["action"], [[64, 4], [4, 4]])
        json.dumps(trace)

    def test_temporary_model_k_layout_is_restored_after_exception(self):
        original = [[16, 16], [4, 4]]
        observed_packing = []
        model = SimpleNamespace(
            independent_raq_rvq_k_lists=[list(scale) for scale in original]
        )

        class RaisingQuality:
            def __init__(self):
                self.qam16_modulate = object()

            @staticmethod
            def _reset_eval_seed(seed):
                del seed

            @staticmethod
            def evaluate_ldpc_channel(*args, **kwargs):
                del args
                observed_packing.append(kwargs["stream_packing"])
                raise RuntimeError("synthetic evaluator failure")

        with self.assertRaisesRegex(RuntimeError, "synthetic evaluator failure"):
            _evaluate_model_action(
                model=model,
                loader=[],
                action=((64, 4), (4, 4)),
                snr=0.0,
                seed=42,
                ldpc_code={},
                device="cpu",
                profile=CHANNEL_PROFILES["ldpc12_bpsk"],
                target_ratio=TARGET_RATIO,
                quality_module=RaisingQuality(),
                stream_packing="combined",
            )

        self.assertEqual(model.independent_raq_rvq_k_lists, original)
        self.assertEqual(observed_packing, ["combined"])


class IndependentRaqRvqBanditOutputTest(unittest.TestCase):
    def test_json_and_csv_preserve_nested_selected_action(self):
        lengths = calculate_physical_lengths(
            REFERENCE_ACTION,
            INDEX_COUNTS,
            SOURCE_VALUES,
            CHANNEL_PROFILES["ldpc12_bpsk"],
            ldpc_n=256,
        )
        summary = {
            "action": [[16, 16], [4, 4]],
            "psnr_mean": 24.0,
            "psnr_std": 0.2,
            "psnr_ci95": 0.25,
            "ms_ssim_mean": 0.8,
        }
        payload = {
            "target_ratio": "3/32",
            "profiles": {
                "ldpc12_bpsk": {
                    "ldpc_rate": 0.5,
                    "modulation": "bpsk",
                    "action_ledger": {
                        "16,16;4,4": lengths.to_dict(),
                    },
                    "snr_results": {
                        "0": {
                            "best_action": [[16, 16], [4, 4]],
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
                [[16, 16], [4, 4]],
            )
            self.assertEqual(
                loaded["profiles"]["ldpc12_bpsk"]["action_ledger"][
                    "16,16;4,4"
                ]["action"],
                [[16, 16], [4, 4]],
            )

            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["selected"], "True")
            self.assertEqual(row["channel_symbols_per_image"], "18432")
            self.assertEqual(row["actual_ratio"], "3/32")
            self.assertEqual(row["k_scale0_stage0"], "16")
            self.assertEqual(row["k_scale0_stage1"], "16")
            self.assertEqual(row["k_scale1_stage0"], "4")
            self.assertEqual(row["k_scale1_stage1"], "4")


if __name__ == "__main__":
    unittest.main()
