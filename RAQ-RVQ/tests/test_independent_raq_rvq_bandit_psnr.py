"""Checks for the randomval independent RAQ-RVQ rate-constrained search."""

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
    enumerate_exact_actions,
    resolve_channel_profiles,
    run_epsilon_greedy,
)


class RandomvalBanditSearchTest(unittest.TestCase):
    def test_custom_ldpc_rate_and_modulation(self):
        args = build_parser().parse_args(
            ["--ldpc-n", "256", "--ldpc-rate", "5/8", "--modulation", "qpsk"]
        )
        profiles = resolve_channel_profiles(args)
        self.assertEqual(len(profiles), 1)
        key, profile = profiles[0]
        self.assertEqual(key, "ldpc5over8_qpsk")
        self.assertEqual(profile.information_block_length(args.ldpc_n), 160)
        self.assertEqual(profile.bits_per_symbol, 2)

    def test_custom_ldpc_rate_must_fit_n_exactly(self):
        args = build_parser().parse_args(["--ldpc-n", "256", "--ldpc-rate", "2/3"])
        _, profile = resolve_channel_profiles(args)[0]
        with self.assertRaisesRegex(ValueError, "incompatible with rate 2/3"):
            profile.information_block_length(args.ldpc_n)

    def test_default_combined_rate_has_three_exact_actions(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.target_ratio, Fraction(1, 64))
        self.assertEqual(args.channel_profile, "ldpc12_qpsk")
        self.assertEqual(args.stream_packing, "combined")

        actions, ledger = enumerate_exact_actions(
            index_counts=(1024, 256),
            source_values=3 * 256 * 256,
            profile=CHANNEL_PROFILES["ldpc12_qpsk"],
            target_ratio=args.target_ratio,
            min_k=2,
            max_k=64,
            ldpc_n=256,
            stream_packing="combined",
        )
        self.assertEqual(
            set(actions),
            {
                ((2, 2), (2, 8)),
                ((2, 2), (4, 4)),
                ((2, 2), (8, 2)),
            },
        )
        for action in actions:
            self.assertEqual(ledger[action].channel_symbols, 3072)
            self.assertEqual(ledger[action].transmission_ratio, Fraction(1, 64))

    def test_reward_uses_psnr_and_common_seed_warmup(self):
        actions = [((2, 2), (2, 8)), ((2, 2), (8, 2))]
        calls = []

        def evaluator(action, seed):
            calls.append((action, seed))
            return EvaluationMetrics(
                action=action,
                snr=3.0,
                seed=seed,
                ms_ssim=0.1 if action == actions[1] else 0.9,
                psnr=25.0 if action == actions[1] else 20.0,
                total_diagnostics={},
            )

        agent, _ = run_epsilon_greedy(
            actions=actions,
            evaluator=evaluator,
            episodes=4,
            warmup_pulls=1,
            search_seed_base=42000,
            eps_start=0,
            eps_end=0,
            eps_decay=30,
            agent_seed=42,
        )
        self.assertEqual(calls[:2], [(actions[0], 42000), (actions[1], 42000)])
        self.assertEqual(agent.q[actions[1]], 25.0)
        self.assertGreater(agent.n[actions[1]], agent.n[actions[0]])

    def test_target_repository_evaluator_signature_and_layout_restore(self):
        original_layout = [[4, 2], [8, 2]]
        model = SimpleNamespace(independent_raq_rvq_k_lists=[row[:] for row in original_layout])
        seen = []

        class Quality:
            qam16_modulate = object()

            @staticmethod
            def _reset_eval_seed(seed):
                seen.append(("seed", seed))

            @staticmethod
            def evaluate_ldpc_channel(
                model_arg, loader, snr, ldpc_code, device,
                modulation="bpsk", return_diagnostics=False,
                stream_packing="per_stage",
            ):
                self.assertIs(model_arg, model)
                self.assertEqual(model_arg.independent_raq_rvq_k_lists, [[2, 2], [4, 4]])
                self.assertEqual((loader, snr, ldpc_code, device), ([], 3.0, {}, "cpu"))
                self.assertEqual((modulation, return_diagnostics, stream_packing),
                                 ("qpsk", True, "combined"))
                quality._reset_eval_seed()
                return 0.8, 28.0, {
                    "rvq_enabled": True,
                    "total": {"channel_symbols": 3072, "source_values": 196608},
                }

        quality = Quality()
        result = _evaluate_model_action(
            model=model,
            loader=[],
            action=((2, 2), (4, 4)),
            snr=3.0,
            seed=52000,
            ldpc_code={},
            device="cpu",
            profile=CHANNEL_PROFILES["ldpc12_qpsk"],
            target_ratio=Fraction(1, 64),
            quality_module=quality,
            stream_packing="combined",
        )
        self.assertEqual(result.psnr, 28.0)
        self.assertEqual(model.independent_raq_rvq_k_lists, original_layout)
        self.assertEqual(seen, [("seed", 52000)])

    def test_json_and_csv_save_selected_four_k_layout(self):
        action = [[2, 2], [4, 4]]
        summary = {
            "action": action,
            "psnr_mean": 28.0,
            "psnr_std": 0.1,
            "psnr_ci95": 0.2,
            "ms_ssim_mean": 0.8,
        }
        payload = {
            "num_scales": 2,
            "target_ratio": "1/64",
            "stream_packing": "combined",
            "profiles": {
                "ldpc12_qpsk": {
                    "ldpc_rate": 0.5,
                    "modulation": "qpsk",
                    "action_ledger": {
                        "2,2;4,4": {
                            "channel_symbols": 3072,
                            "transmission_ratio": "1/64",
                        }
                    },
                    "snr_results": {
                        "3": {
                            "best_action": action,
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
            self.assertEqual(json.loads(json_path.read_text())["target_ratio"], "1/64")
            with csv_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                [rows[0][f"k_scale{s}_stage{t}"] for s in range(2) for t in range(2)],
                ["2", "2", "4", "4"],
            )
            self.assertEqual(rows[0]["actual_ratio"], "1/64")


if __name__ == "__main__":
    unittest.main()
