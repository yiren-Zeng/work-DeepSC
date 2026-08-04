"""Contracts for the trained four-codebook independent RAQ-RVQ mode."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from models.deepsc import DeepSC
from models.independent_raq_rvq import (
    quantize_independent_raq_rvq,
)
from models.vector_quantizer import VectorQuantizer
from monitoring.codebook import compute_codebook_utilization
from train import sample_independent_raq_rvq_for_epoch
from utils.raq_rvq import validate_independent_rvq_k_lists


torch.set_num_threads(1)


def _build_model():
    return DeepSC(
        in_channels=3,
        out_channels=3,
        num_downsample_blocks=2,
        base_channels=4,
        num_embeddings_list=[4, 4],
        embedding_dim_list=[8, 16],
        commitment_cost=0.25,
        device="cpu",
        strides=[2, 2],
        norm_groups=1,
        encoder_res_blocks=1,
        decoder_res_blocks=1,
        use_raq=True,
        raq_target_list=[4, 2],
        raq_min_trg=2,
        raq_max_trg=4,
        raq_recon_grad_mode="ste",
        use_independent_raq_rvq=True,
        independent_raq_rvq_depth=2,
        independent_raq_rvq_k_lists=[[4, 4], [2, 2]],
    )


def _has_finite_nonzero_grad(parameters):
    gradients = [
        parameter.grad
        for parameter in parameters
        if parameter.requires_grad and parameter.grad is not None
    ]
    return bool(gradients) and all(
        torch.isfinite(gradient).all() for gradient in gradients
    ) and any(
        int(torch.count_nonzero(gradient)) > 0
        for gradient in gradients
    )


class IndependentRaqRvqTest(unittest.TestCase):
    def test_validator_has_no_cross_stage_rate_constraint(self):
        self.assertEqual(
            validate_independent_rvq_k_lists(
                [[64, 32], [2, 4]],
                num_scales=2,
                rvq_depth=2,
                min_k=[2, 2],
                max_k=[64, 64],
            ),
            [[64, 32], [2, 4]],
        )
        with self.assertRaisesRegex(ValueError, "power of two"):
            validate_independent_rvq_k_lists(
                [[3, 2], [2, 2]],
                num_scales=2,
            )

    def test_curriculum_samples_all_four_k_independently(self):
        cfg = SimpleNamespace(
            NUM_DOWNSAMPLE_BLOCKS=2,
            NUM_EPOCHS=100,
            PHASE1_END=0.1,
            PHASE2_END=0.4,
            RAQ_USE_CURRICULUM=True,
            RAQ_CURRICULUM_EARLY_LIST=[32, 64],
            RAQ_CURRICULUM_MIDDLE_LIST=[8, 16, 32, 64],
            RAQ_CURRICULUM_LATE_LIST=[2, 4, 8, 16, 32, 64],
            RAQ_CURRICULUM_EARLY_LISTS=[
                [32, 64], [2, 4]
            ],
            RAQ_CURRICULUM_MIDDLE_LISTS=[
                [8, 16, 32, 64], [2, 4]
            ],
            RAQ_CURRICULUM_LATE_LISTS=[
                [2, 4, 8, 16, 32, 64], [2, 4]
            ],
            INDEPENDENT_RAQ_RVQ_DEPTH=2,
        )
        with mock.patch(
            "train.random.choice",
            side_effect=[64, 32, 2, 4],
        ) as choice:
            sampled, phase = sample_independent_raq_rvq_for_epoch(
                0, cfg
            )
        self.assertEqual(sampled, [[64, 32], [2, 4]])
        self.assertEqual(phase, "early")
        self.assertEqual(choice.call_count, 4)

    def test_stateless_quantizer_uses_distinct_stage_codebooks(self):
        quantizer = VectorQuantizer(2, 1, commitment_cost=0.25)
        inputs = torch.tensor(
            [[[[0.0, 0.4, 1.0, 1.8]]]], requires_grad=True
        )
        first = torch.tensor([[0.0], [1.0]], requires_grad=True)
        second = torch.tensor([[-0.5], [0.5]], requires_grad=True)
        result = quantize_independent_raq_rvq(
            quantizer, inputs, [first, second]
        )

        self.assertEqual(len(result["indices"]), 2)
        self.assertEqual(
            tuple(result["quantized"].shape), tuple(inputs.shape)
        )
        result["quantized"].sum().backward()
        torch.testing.assert_close(
            inputs.grad, torch.ones_like(inputs)
        )
        self.assertIsNone(first.grad)
        self.assertIsNone(second.grad)

        inputs.grad = None
        result = quantize_independent_raq_rvq(
            quantizer, inputs, [first, second]
        )
        result["loss"].backward()
        self.assertIsNotNone(first.grad)
        self.assertIsNotNone(second.grad)

    def test_architecture_training_and_four_generator_gradients(self):
        torch.manual_seed(7)
        model = _build_model().train()
        model.set_channel_prob(0.0)
        self.assertEqual(len(model.raqs), 2)
        self.assertEqual(len(model.raqs_rvq_stage2), 2)
        self.assertFalse(
            any(
                key.startswith(
                    "raqs_rvq_stage2.0.allocation_condition"
                )
                for key in model.state_dict()
            )
        )

        out = model.forward_train(
            torch.randn(1, 3, 16, 16),
            raq_rvq_k_lists=[[4, 2], [2, 4]],
        )
        self.assertEqual(out["rvq_k_lists"], [[4, 2], [2, 4]])
        self.assertEqual(
            [
                [codebook.shape[0] for codebook in scale_codebooks]
                for scale_codebooks in out["rvq_codebooks_list"]
            ],
            [[4, 2], [2, 4]],
        )
        loss = out["reconstructed_images_raq"].pow(2).mean()
        loss = loss + sum(out["vq_losses_raq"])
        loss.backward()
        for scale_index in range(2):
            self.assertTrue(
                _has_finite_nonzero_grad(
                    model.raqs[scale_index].parameters()
                )
            )
            self.assertTrue(
                _has_finite_nonzero_grad(
                    model.raqs_rvq_stage2[
                        scale_index
                    ].parameters()
                )
            )

    def test_fixed_validation_layout_round_trip_and_monitor(self):
        torch.manual_seed(8)
        model = _build_model().eval()
        image = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            out = model.forward_test(image)
            reconstructed = model.reconstruct_from_indices(
                out["indices"],
                feature_shapes=out["feature_shapes"],
                codebooks=out["codebooks"],
            )
        self.assertEqual(out["branch"], "independent_raq_rvq")
        self.assertEqual(out["rvq_k_lists"], [[4, 4], [2, 2]])
        self.assertTrue(
            all(
                diagnostic[
                    "independent_codebook_identity_verified"
                ]
                for diagnostic in out["rvq_diagnostics"]
            )
        )
        self.assertEqual(tuple(reconstructed.shape), tuple(image.shape))

        stats = compute_codebook_utilization(
            model, [image], device="cpu"
        )
        self.assertNotIn("raq", stats)
        self.assertEqual(len(stats["raq_stages"]), 2)
        expected_tokens = [64, 16]
        for scale_index, scale_stats in enumerate(
            stats["raq_stages"]
        ):
            self.assertEqual(len(scale_stats), 2)
            for stage_stats in scale_stats:
                self.assertEqual(
                    int(stage_stats["usage_counts"].sum().item()),
                    expected_tokens[scale_index],
                )


if __name__ == "__main__":
    unittest.main()
