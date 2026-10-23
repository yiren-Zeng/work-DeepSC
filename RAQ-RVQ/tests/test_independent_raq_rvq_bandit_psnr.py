"""Checks for the randomval independent RAQ-RVQ rate-constrained search."""

import csv
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bandit_independent_raq_rvq_psnr_search import (
    CHANNEL_PROFILES,
    EpsilonGreedyAgent,
    EvaluationMetrics,
    _evaluate_model_action,
    _write_csv,
    _write_json,
    build_parser,
    enumerate_exact_actions,
    resolve_channel_profiles,
    resolve_evaluation_seeds,
    run,
    run_epsilon_greedy,
    summarize_search,
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

    def test_exploitation_before_full_coverage_does_not_resample_or_update_q(self):
        actions = [((2, 2), (2, 8)), ((2, 2), (4, 4)), ((2, 2), (8, 2))]
        calls = []

        def evaluator(action, seed):
            calls.append((action, seed))
            return EvaluationMetrics(
                action=action,
                snr=3.0,
                seed=seed,
                ms_ssim=0.9 if action == actions[0] else 0.1,
                psnr=20.0 if action == actions[0] else 25.0,
                total_diagnostics={},
            )

        with patch.object(EpsilonGreedyAgent, "epsilon", side_effect=[1, 0, 0]):
            agent, trace = run_epsilon_greedy(
                actions, evaluator, episodes=4, initial_action=actions[0],
            )
        self.assertEqual([row["decision"] for row in trace],
                         ["initialize", "explore", "exploit", "exploit"])
        self.assertEqual(len(calls), 2)
        self.assertEqual([seed for _, seed in calls], [42000, 42000])
        self.assertNotEqual(calls[0][0], calls[1][0])
        best = calls[1][0]
        self.assertEqual(agent.q[best], 25.0)
        self.assertEqual(agent.n[best], 1)
        self.assertEqual(agent.cache_hits, 2)
        self.assertEqual(trace[-1]["action"], [list(row) for row in best])
        self.assertEqual(trace[-1]["source"], "cache")
        self.assertEqual(trace[-1]["evaluation_calls"], 2)
        self.assertEqual(agent.stop_reason, "decision_budget_exhausted")
        self.assertEqual(sum(row["q"] is None for row in agent.snapshot()), 1)

    @staticmethod
    def _metrics(action, seed, psnr=20.0):
        return EvaluationMetrics(action, 3.0, seed, 0.8, psnr, {})

    def test_one_decision_can_search_44_actions_without_full_initialization(self):
        args = build_parser().parse_args(["--ldpc-rate", "3/4", "--modulation", "qpsk"])
        _, profile = resolve_channel_profiles(args)[0]
        actions, _ = enumerate_exact_actions(
            (1024, 256), 3 * 256 * 256, profile, Fraction(1, 48),
            2, 256, 256, stream_packing="combined",
        )
        self.assertEqual(len(actions), 44)
        initial = ((16, 2), (8, 2))
        with patch.object(self, "_metrics", wraps=self._metrics) as evaluator:
            agent, trace = run_epsilon_greedy(actions, evaluator, 1, initial_action=initial)
        evaluator.assert_called_once_with(initial, 42000)
        summary = summarize_search(agent, trace)
        self.assertEqual(summary["evaluated_actions"], 1)
        self.assertEqual(summary["total_actions"], 44)
        self.assertEqual(summary["decision_count"], 1)

    def test_unmeasured_actions_do_not_outrank_negative_psnr(self):
        actions = [((2, 2), (2, 8)), ((2, 2), (8, 2))]
        calls = []

        def evaluator(action, seed):
            calls.append((action, seed))
            return self._metrics(action, seed, -5.0)

        agent, trace = run_epsilon_greedy(
            actions, evaluator, 5, initial_action=actions[1], eps_start=0, eps_end=0,
        )
        self.assertEqual(calls, [(actions[1], 42000)])
        self.assertEqual(agent.ranked_actions(), [actions[1]])
        self.assertIsNone(agent.q[actions[0]])
        self.assertEqual(agent.q[actions[1]], -5.0)
        self.assertEqual(agent.n[actions[1]], 1)
        self.assertEqual(agent.cache_hits, 4)
        self.assertEqual(len(trace), 5)

    def test_exploration_samples_without_replacement_and_stops_when_exhausted(self):
        actions = [((2, 2), (2, 8)), ((2, 2), (4, 4)), ((2, 2), (8, 2))]
        calls = []

        def evaluator(action, seed):
            calls.append((action, seed))
            return self._metrics(action, seed)

        agent, trace = run_epsilon_greedy(
            actions, evaluator, 100, initial_action=actions[0], eps_start=1, eps_end=1,
        )
        self.assertEqual(len(trace), 3)
        self.assertEqual({action for action, _ in calls}, set(actions))
        self.assertEqual(len(calls), 3)
        self.assertEqual({seed for _, seed in calls}, {42000})
        self.assertEqual(agent.stop_reason, "all_actions_evaluated")

    def test_every_new_allocation_uses_same_complete_seed_set_and_mean_reward(self):
        actions = [((2, 2), (2, 8)), ((2, 2), (8, 2))]
        calls = []

        def evaluator(action, seed):
            calls.append((action, seed))
            psnr = (40.0 if seed == 17 else 0.0) if action == actions[0] else 25.0
            return self._metrics(action, seed, psnr)

        agent, trace = run_epsilon_greedy(
            actions, evaluator, 10, initial_action=actions[0], search_seeds=[17, 18],
            eps_start=1, eps_end=1,
        )
        self.assertEqual(calls, [(actions[0], 17), (actions[0], 18),
                                 (actions[1], 17), (actions[1], 18)])
        self.assertEqual(agent.q[actions[0]], 20.0)
        self.assertEqual(agent.q[actions[1]], 25.0)
        self.assertEqual(agent.ranked_actions()[0], actions[1])
        self.assertEqual(agent.n[actions[0]], 1)
        self.assertEqual(trace[-1]["evaluation_calls"], 4)
        self.assertIsNone(trace[-1]["seed"])
        self.assertEqual(trace[-1]["seeds"], [17, 18])

    def test_initialization_is_a_small_distinct_subset_and_falls_back_to_legal_action(self):
        actions = [((2, 2), (2, 8)), ((2, 2), (4, 4)), ((2, 2), (8, 2))]
        agent, trace = run_epsilon_greedy(
            actions, self._metrics, 2, initial_actions=2, initial_action=((64, 64), (64, 64)),
        )
        self.assertEqual(len(agent.evaluations), 2)
        self.assertTrue(set(agent.evaluations) <= set(actions))
        self.assertEqual([row["decision"] for row in trace], ["initialize", "initialize"])

    def test_single_legal_action_stops_after_initialization(self):
        agent, trace = run_epsilon_greedy([((2, 2), (4, 4))], self._metrics, 100)
        self.assertEqual(len(trace), 1)
        self.assertEqual(agent.stop_reason, "all_actions_evaluated")

    def test_invalid_initialization_budget_and_seed_settings_fail_early(self):
        actions = [((2, 2), (4, 4))]
        for settings in (
            {"episodes": 0}, {"episodes": 1, "initial_actions": 0},
            {"episodes": 1, "initial_actions": 2},
            {"episodes": 2, "initial_actions": 2},
            {"episodes": 1, "search_seeds": []},
            {"episodes": 1, "search_seeds": [1, 1]},
            {"episodes": 1, "search_seeds": [-1]},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                run_epsilon_greedy(actions, self._metrics, **settings)

    def test_seed_defaults_and_optional_independent_reporting(self):
        self.assertEqual(resolve_evaluation_seeds(build_parser().parse_args([])), ((42000,), ()))
        args = build_parser().parse_args(
            ["--search-seeds", "17", "18", "--report-seeds", "27", "28"]
        )
        self.assertEqual(resolve_evaluation_seeds(args), ((17, 18), (27, 28)))
        for flags in (
            ["--fixed-channel-seed", "-1"],
            ["--search-seeds", "17", "17"],
            ["--report-seeds", "42000"],
            ["--report-seeds", "27", "27"],
            ["--report-seeds", "-1"],
        ):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                resolve_evaluation_seeds(build_parser().parse_args(flags))

    def test_run_only_exports_measured_candidates_and_report_does_not_reselect(self):
        initial = ((2, 2), (4, 4))

        class Dataset:
            image_files = []

            def __len__(self):
                return 24

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pth"
            checkpoint.touch()
            cfg = SimpleNamespace(
                validate=lambda: None, DEVICE="cpu", TEST_DATASET_PATH=directory,
                NUM_WORKERS=0, PIN_MEMORY=False,
            )
            model = SimpleNamespace(
                independent_raq_rvq_k_lists=[[2, 2], [4, 4]], independent_raq_rvq_depth=2,
            )
            loader = SimpleNamespace(dataset=Dataset())
            modules = {
                "torch": SimpleNamespace(device=lambda value: value),
                "communications.ldpc_coding": SimpleNamespace(
                    get_ldpc_code=lambda k, rate: {"k": k, "n": int(k / rate)},
                ),
                "config": SimpleNamespace(Config=lambda: cfg),
                "data.datasets": SimpleNamespace(get_dataloader=lambda **kwargs: loader),
                "evaluation": SimpleNamespace(quality=SimpleNamespace()),
                "utils.checkpoint_utils": SimpleNamespace(
                    build_model_from_checkpoint=lambda *args: (model, {"num_embeddings_list": [64, 64]}),
                ),
                "utils.reproducibility": SimpleNamespace(setup_seed=lambda seed: None),
            }
            for report_seeds in ([], [62000, 62001]):
                calls = []

                def evaluator(action, seed):
                    calls.append((action, seed))
                    return self._metrics(action, seed, 20.0 if seed == 42000 else 5.0)

                flags = ["--checkpoint", str(checkpoint), "--episodes", "3",
                         "--eps-start", "0", "--eps-end", "0",
                         "--json-output", str(Path(directory) / "result.json"),
                         "--csv-output", str(Path(directory) / "result.csv")]
                if report_seeds:
                    flags += ["--report-seeds", *map(str, report_seeds)]
                with self.subTest(report_seeds=report_seeds), patch.dict("sys.modules", modules), \
                        patch("bandit_independent_raq_rvq_psnr_search._probe_layout",
                              return_value=((1024, 256), 196608, [1, 3, 256, 256])), \
                        patch("bandit_independent_raq_rvq_psnr_search._make_runtime_evaluator",
                              return_value=evaluator), patch("builtins.print"):
                    payload = run(build_parser().parse_args(flags))
                result = payload["profiles"]["ldpc12_qpsk"]["snr_results"]["0"]
                # The parser's default SNR list has five independent searches.
                self.assertEqual(len(calls), 5 * (1 + len(report_seeds)))
                self.assertTrue(all(action == initial for action, _ in calls))
                self.assertEqual(result["best_action"], [[2, 2], [4, 4]])
                self.assertEqual(result["best_psnr"], 20.0)
                self.assertEqual(result["evaluated_actions"], 1)
                self.assertEqual(result["decision_count"], 3)
                self.assertEqual(result["cache_hits"], 2)
                self.assertEqual(len(result["confirmation"]), 1)
                self.assertEqual(result["report"]["psnr_mean"], 5.0 if report_seeds else 20.0)
                self.assertEqual(result["report_evaluation_calls"], len(report_seeds))
                self.assertEqual(result["report_basis"],
                                 "independent_channel_seeds" if report_seeds else "search_cache")
                saved = json.loads((Path(directory) / "result.json").read_text())
                self.assertEqual(saved["bandit"]["search_seeds"], [42000])
                with (Path(directory) / "result.csv").open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(rows[0]["evaluated_actions"], "1")
                self.assertEqual(rows[0]["cache_hits"], "2")

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
                stream_packing="per_stage", channel_seed=42,
            ):
                self.assertIs(model_arg, model)
                self.assertEqual(model_arg.independent_raq_rvq_k_lists, [[2, 2], [4, 4]])
                self.assertEqual((loader, snr, ldpc_code, device), ([], 3.0, {}, "cpu"))
                self.assertEqual((modulation, return_diagnostics, stream_packing),
                                 ("qpsk", True, "combined"))
                self.assertEqual(channel_seed, 52000)
                quality._reset_eval_seed(channel_seed)
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
