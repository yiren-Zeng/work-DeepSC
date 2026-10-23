"""Contracts for the global-uniform, always-on-channel training ablation."""

import random
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import train_uniform_channel1 as training
from training.schedules import compute_schedule as original_schedule


def config():
    return SimpleNamespace(
        RAQ_MIN_TRG_LIST=[2, 2],
        RAQ_MAX_TRG_LIST=[64, 64],
        INDEPENDENT_RAQ_RVQ_DEPTH=2,
        NUM_EPOCHS=200,
        PHASE1_END=0.1,
        PHASE2_END=0.4,
        SKIP_DROPOUT_P_INIT=[0.1],
        SKIP_DROPOUT_P_FINAL=[0.0],
        LAYER_LOSS_WEIGHTS_INIT=[0.25, 0.5],
        LAYER_LOSS_WEIGHTS_FINAL=[0.25, 0.25],
        CHANNEL_PROB_START_EPOCH=80,
        CHANNEL_PROB_END_EPOCH=120,
    )


class UniformChannelOneTest(unittest.TestCase):
    def test_sampling_has_no_epoch_dependent_restriction(self):
        cfg = config()
        reference = None
        for epoch in (0, 19, 20, 79, 80, 119, 120, 199):
            rng = random.Random(42)
            with patch.object(training.random, "choice", side_effect=rng.choice):
                samples = [
                    training.sample_independent_raq_rvq_for_epoch(epoch, cfg)
                    for _ in range(100)
                ]
            self.assertTrue(all(phase == "uniform" for _, phase in samples))
            if reference is None:
                reference = samples
            self.assertEqual(samples, reference)
            for scale in range(2):
                for stage in range(2):
                    self.assertEqual(
                        {layout[scale][stage] for layout, _ in samples},
                        {2, 4, 8, 16, 32, 64},
                    )

    def test_four_k_draws_are_uniform_and_independent(self):
        cfg = config()
        rng = random.Random(123)
        with patch.object(training.random, "choice", side_effect=rng.choice):
            layouts = [
                tuple(k for scale in training.sample_independent_raq_rvq_for_epoch(0, cfg)[0] for k in scale)
                for _ in range(12000)
            ]
        for position in range(4):
            counts = Counter(layout[position] for layout in layouts)
            self.assertEqual(set(counts), {2, 4, 8, 16, 32, 64})
            for count in counts.values():
                self.assertLess(abs(count / len(layouts) - 1 / 6), 0.02)
        # All 6^4 configurations must be reachable, including unequal K values.
        self.assertEqual(len(set(layouts)), 6 ** 4)
        for first in range(4):
            for second in range(first + 1, 4):
                pairs = Counter((layout[first], layout[second]) for layout in layouts)
                self.assertEqual(len(pairs), 36)
                for count in pairs.values():
                    self.assertLess(abs(count / len(layouts) - 1 / 36), 0.01)

    def test_channel_is_always_on_and_other_schedules_match_baseline(self):
        cfg = config()
        for epoch in range(cfg.NUM_EPOCHS):
            actual = training.compute_schedule(epoch, cfg.NUM_EPOCHS, cfg)
            original = original_schedule(epoch, cfg.NUM_EPOCHS, cfg)
            self.assertEqual(actual[2], 1.0)
            self.assertEqual(actual[:2], original[:2])
            self.assertEqual(actual[3], original[3])


if __name__ == "__main__":
    unittest.main()
