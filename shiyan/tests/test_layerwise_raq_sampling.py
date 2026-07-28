import unittest
from types import SimpleNamespace
from unittest import mock

import torch.nn as nn

from models.deepsc import DeepSC
from train import sample_raq_target_list_for_epoch
from utils.raq_rvq import resolve_rvq_stage_k_lists


def _sampling_config(**overrides):
    values = {
        "NUM_DOWNSAMPLE_BLOCKS": 2,
        "NUM_EPOCHS": 100,
        "PHASE1_END": 0.1,
        "PHASE2_END": 0.4,
        "RAQ_USE_CURRICULUM": False,
        "RAQ_MIN_TRG": 2,
        "RAQ_MAX_TRG": 16,
        "RAQ_MIN_TRG_LIST": None,
        "RAQ_MAX_TRG_LIST": None,
        "RAQ_CURRICULUM_EARLY_LIST": [8, 16],
        "RAQ_CURRICULUM_MIDDLE_LIST": [4, 8, 16],
        "RAQ_CURRICULUM_LATE_LIST": [2, 4, 8, 16],
        "RAQ_CURRICULUM_EARLY_LISTS": None,
        "RAQ_CURRICULUM_MIDDLE_LISTS": None,
        "RAQ_CURRICULUM_LATE_LISTS": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class LayerwiseRaqTrainingSamplingTest(unittest.TestCase):
    def test_legacy_scalar_range_is_reused_for_every_layer(self):
        cfg = _sampling_config()

        with mock.patch("train.sample_trg", side_effect=lambda low, high: high) as sample:
            targets, phase = sample_raq_target_list_for_epoch(0, cfg)

        self.assertEqual(targets, [16, 16])
        self.assertEqual(phase, "uniform")
        self.assertEqual(
            sample.call_args_list,
            [mock.call(2, 16), mock.call(2, 16)],
        )

    def test_layerwise_ranges_are_sampled_independently(self):
        cfg = _sampling_config(
            RAQ_MIN_TRG_LIST=[2, 2],
            RAQ_MAX_TRG_LIST=[16, 4],
        )

        with mock.patch("train.sample_trg", side_effect=lambda low, high: high) as sample:
            targets, phase = sample_raq_target_list_for_epoch(0, cfg)

        self.assertEqual(targets, [16, 4])
        self.assertEqual(phase, "uniform")
        self.assertEqual(
            sample.call_args_list,
            [mock.call(2, 16), mock.call(2, 4)],
        )

    def test_legacy_curriculum_list_is_reused_for_every_layer(self):
        cfg = _sampling_config(RAQ_USE_CURRICULUM=True)

        with mock.patch(
            "train.random.choice", side_effect=lambda choices: choices[0]
        ) as choice:
            targets, phase = sample_raq_target_list_for_epoch(0, cfg)

        self.assertEqual(targets, [8, 8])
        self.assertEqual(phase, "early")
        self.assertEqual(
            choice.call_args_list,
            [mock.call([8, 16]), mock.call([8, 16])],
        )

    def test_layerwise_curriculum_changes_each_layers_candidates_by_phase(self):
        cfg = _sampling_config(
            RAQ_USE_CURRICULUM=True,
            RAQ_CURRICULUM_EARLY_LISTS=[[8, 16], [4]],
            RAQ_CURRICULUM_MIDDLE_LISTS=[[4, 8, 16], [4]],
            RAQ_CURRICULUM_LATE_LISTS=[[2, 4, 8, 16], [2, 4]],
        )
        cases = (
            (0, "early", [8, 4], [[8, 16], [4]]),
            (10, "middle", [4, 4], [[4, 8, 16], [4]]),
            (40, "late", [2, 2], [[2, 4, 8, 16], [2, 4]]),
        )

        for epoch, expected_phase, expected_targets, expected_candidates in cases:
            with self.subTest(epoch=epoch), mock.patch(
                "train.random.choice", side_effect=lambda choices: choices[0]
            ) as choice:
                targets, phase = sample_raq_target_list_for_epoch(epoch, cfg)

            self.assertEqual(targets, expected_targets)
            self.assertEqual(phase, expected_phase)
            self.assertEqual(
                choice.call_args_list,
                [mock.call(values) for values in expected_candidates],
            )


class LayerwiseRvqBoundsTest(unittest.TestCase):
    def test_each_scale_uses_its_own_bounds(self):
        self.assertEqual(
            resolve_rvq_stage_k_lists(
                [16, 4],
                stage_k_lists=[[4, 4], [2, 2]],
                min_k=[2, 2],
                max_k=[16, 4],
            ),
            [[4, 4], [2, 2]],
        )

    def test_second_scale_rejects_stage_k_above_four(self):
        with self.assertRaisesRegex(
            ValueError,
            r"scale 1 stage 0 K=8 exceeds RAQ maximum 4",
        ):
            resolve_rvq_stage_k_lists(
                [16, 4],
                stage_k_lists=[[4, 4], [8, 2]],
                min_k=[2, 2],
                max_k=[16, 4],
            )


class _RecordingRaq(nn.Module):
    def __init__(
        self,
        embedding_dim,
        n_embed_src,
        n_embed_min_trg,
        n_embed_max_trg,
        **kwargs,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_embed_src = n_embed_src
        self.n_embed_min_trg = n_embed_min_trg
        self.n_embed_max_trg = n_embed_max_trg


class LayerwiseRaqModelSamplingTest(unittest.TestCase):
    @staticmethod
    def _build_model(**range_overrides):
        kwargs = {
            "in_channels": 3,
            "out_channels": 3,
            "num_downsample_blocks": 2,
            "base_channels": 4,
            "num_embeddings_list": [16, 4],
            "embedding_dim_list": [8, 16],
            "commitment_cost": 0.25,
            "device": "cpu",
            "strides": [2, 2],
            "norm_groups": 1,
            "use_raq": True,
            "raq_target_list": [16, 4],
            "raq_min_trg": 2,
            "raq_max_trg": 16,
        }
        kwargs.update(range_overrides)
        with mock.patch("models.deepsc.RAQ", _RecordingRaq):
            return DeepSC(**kwargs)

    def test_legacy_scalar_constructor_range_remains_supported(self):
        model = self._build_model()

        self.assertEqual(
            [
                (raq.n_embed_min_trg, raq.n_embed_max_trg)
                for raq in model.raqs
            ],
            [(2, 16), (2, 16)],
        )
        with mock.patch(
            "models.deepsc.sample_trg", side_effect=lambda low, high: high
        ) as sample:
            self.assertEqual(model._sample_raq_target_list(), [16, 16])
        self.assertEqual(
            sample.call_args_list,
            [mock.call(2, 16), mock.call(2, 16)],
        )

    def test_layerwise_constructor_ranges_drive_modules_and_sampling(self):
        model = self._build_model(
            raq_min_trg_list=[2, 2],
            raq_max_trg_list=[16, 4],
        )

        self.assertEqual(
            [
                (raq.n_embed_min_trg, raq.n_embed_max_trg)
                for raq in model.raqs
            ],
            [(2, 16), (2, 4)],
        )
        with mock.patch(
            "models.deepsc.sample_trg", side_effect=lambda low, high: high
        ) as sample:
            self.assertEqual(model._sample_raq_target_list(), [16, 4])
        self.assertEqual(
            sample.call_args_list,
            [mock.call(2, 16), mock.call(2, 4)],
        )

    def test_layerwise_bounds_are_preserved_as_model_attributes(self):
        model = self._build_model(
            raq_min_trg_list=[2, 2],
            raq_max_trg_list=[16, 4],
        )

        self.assertEqual(model.raq_min_trg_list, [2, 2])
        self.assertEqual(model.raq_max_trg_list, [16, 4])
        self.assertEqual(model.raq_min_trg, 2)
        self.assertEqual(model.raq_max_trg, 16)

    def test_target_is_rejected_against_its_own_layer_bound(self):
        with self.assertRaisesRegex(
            ValueError,
            r"RAQ target layer 1 K=8 is outside \[2,4\]",
        ):
            self._build_model(
                raq_target_list=[16, 8],
                raq_min_trg_list=[2, 2],
                raq_max_trg_list=[16, 4],
            )


if __name__ == "__main__":
    unittest.main()
