import unittest

import torch

from models.deepsc import DeepSC
from utils.raq_rvq import (
    resolve_rvq_stage_k_lists,
    sample_total_codebook_bit_split,
    split_total_codebook_bits,
)


class SplitTotalCodebookBitsTest(unittest.TestCase):
    def test_expected_two_stage_mapping(self):
        expected = {
            2: [2],
            4: [2, 2],
            8: [4, 2],
            16: [4, 4],
            32: [8, 4],
            64: [8, 8],
            128: [16, 8],
            256: [16, 16],
            512: [32, 16],
            1024: [32, 32],
            2048: [64, 32],
        }
        for k_total, stage_sizes in expected.items():
            with self.subTest(k_total=k_total):
                self.assertEqual(split_total_codebook_bits(k_total), stage_sizes)

    def test_rejects_invalid_total_or_depth(self):
        for invalid_total in (0, 1, 3, 12):
            with self.subTest(k_total=invalid_total):
                with self.assertRaises(ValueError):
                    split_total_codebook_bits(invalid_total)
        with self.assertRaises(ValueError):
            split_total_codebook_bits(8, rvq_depth=3)

    def test_custom_ordered_splits_keep_the_same_bit_budget(self):
        self.assertEqual(
            resolve_rvq_stage_k_lists(
                [2048, 16], stage_k_lists=[[32, 64], [8, 2]]
            ),
            [[32, 64], [8, 2]],
        )

    def test_automatic_split_remains_backward_compatible(self):
        self.assertEqual(
            resolve_rvq_stage_k_lists([2048, 16]),
            [[64, 32], [4, 4]],
        )

    def test_custom_split_rejects_changed_bit_budget(self):
        with self.assertRaisesRegex(ValueError, "bit budget mismatch"):
            resolve_rvq_stage_k_lists(
                [2048, 16], stage_k_lists=[[32, 32], [8, 2]]
            )

    def test_custom_split_rejects_non_power_of_two(self):
        with self.assertRaisesRegex(ValueError, "power of two"):
            resolve_rvq_stage_k_lists([16], stage_k_lists=[[3, 4]])

    def test_dynamic_split_preserves_total_bits_and_low_rate_depth(self):
        self.assertEqual(sample_total_codebook_bit_split(2), [2])
        for k_total in (4, 8, 16, 256, 2048):
            for _ in range(20):
                stage_sizes = sample_total_codebook_bit_split(k_total)
                self.assertEqual(len(stage_sizes), 2)
                self.assertEqual(stage_sizes[0] * stage_sizes[1], k_total)


class DeepSCTestTimeRaqRvqTest(unittest.TestCase):
    @staticmethod
    def _build_model(enabled, stage_k_lists=None):
        return DeepSC(
            in_channels=3,
            out_channels=3,
            num_downsample_blocks=1,
            base_channels=4,
            num_embeddings_list=[8],
            embedding_dim_list=[8],
            commitment_cost=0.25,
            device="cpu",
            strides=[2],
            norm_groups=1,
            use_raq=True,
            raq_target_list=[8],
            raq_min_trg=2,
            raq_max_trg=8,
            test_use_raq_rvq=enabled,
            test_raq_rvq_depth=2,
            test_raq_rvq_k_lists=stage_k_lists,
        ).eval()

    def test_control_adds_no_state_dict_entries(self):
        baseline = self._build_model(False)
        rvq = self._build_model(True)
        self.assertEqual(list(baseline.state_dict()), list(rvq.state_dict()))

    def test_enabled_control_requires_raq(self):
        kwargs = {
            "in_channels": 3,
            "out_channels": 3,
            "num_downsample_blocks": 1,
            "base_channels": 4,
            "num_embeddings_list": [8],
            "embedding_dim_list": [8],
            "commitment_cost": 0.25,
            "device": "cpu",
            "strides": [2],
            "norm_groups": 1,
            "use_raq": False,
            "test_use_raq_rvq": True,
        }
        with self.assertRaisesRegex(ValueError, "requires use_raq=True"):
            DeepSC(**kwargs)

    def test_forward_and_nested_reconstruction(self):
        model = self._build_model(True)
        image = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            out = model.forward_test(image)

            self.assertEqual(out["branch"], "raq_rvq")
            self.assertTrue(out["test_raq_rvq_enabled"])
            self.assertEqual(out["rvq_k_lists"], [[4, 2]])
            self.assertEqual(len(out["indices"][0]), 2)
            self.assertEqual([weight.shape[0] for weight in out["codebooks"][0]], [4, 2])
            self.assertTrue(out["rvq_diagnostics"][0]["bit_budget_matches"])

            reconstructed = model.reconstruct_from_indices(
                out["indices"],
                feature_shapes=out["feature_shapes"],
                codebooks=out["codebooks"],
            )
            quantized_stages = [
                model.vector_quantizers[0].get_quantized_features(
                    stage_indices,
                    output_spatial_size=out["feature_shapes"][0],
                    codebook_weight=stage_codebook,
                )
                for stage_indices, stage_codebook in zip(
                    out["indices"][0], out["codebooks"][0]
                )
            ]
            expected = model._decode_features([sum(quantized_stages)])
            torch.testing.assert_close(reconstructed, expected)

    def test_forward_uses_custom_ordered_stage_sizes(self):
        model = self._build_model(True, stage_k_lists=[[2, 4]])
        with torch.no_grad():
            out = model.forward_test(torch.randn(1, 3, 8, 8))
        self.assertEqual(out["rvq_k_lists"], [[2, 4]])
        self.assertEqual([weight.shape[0] for weight in out["codebooks"][0]], [2, 4])
        self.assertTrue(out["rvq_diagnostics"][0]["bit_budget_matches"])

    def test_disabled_output_keeps_single_stage_contract(self):
        model = self._build_model(False)
        with torch.no_grad():
            out = model.forward_test(torch.randn(1, 3, 8, 8))
        self.assertEqual(
            set(out),
            {"indices", "feature_shapes", "num_embeddings_list", "branch", "codebooks"},
        )
        self.assertIsInstance(out["indices"][0], torch.Tensor)
        self.assertIsInstance(out["codebooks"][0], torch.Tensor)
        self.assertEqual(out["branch"], "raq")


class DeepSCDynamicRaqRvqTrainingTest(unittest.TestCase):
    @staticmethod
    def _build_model(target_list=(8,)):
        return DeepSC(
            in_channels=3,
            out_channels=3,
            num_downsample_blocks=1,
            base_channels=4,
            num_embeddings_list=[8],
            embedding_dim_list=[8],
            commitment_cost=0.25,
            device="cpu",
            strides=[2],
            norm_groups=1,
            use_raq=True,
            raq_target_list=list(target_list),
            raq_min_trg=2,
            raq_max_trg=8,
            raq_recon_grad_mode="dual",
            use_dynamic_raq_rvq=True,
            dynamic_raq_rvq_zero_codeword=True,
        )

    def test_dynamic_architecture_adds_independent_stage2_state(self):
        model = self._build_model()
        self.assertEqual(len(model.raqs), 1)
        self.assertEqual(len(model.raqs_rvq_stage2), 1)
        self.assertTrue(
            any(key.startswith("raqs_rvq_stage2.0.") for key in model.state_dict())
        )

    def test_training_forward_uses_residual_stage_and_single_sum(self):
        model = self._build_model().train()
        image = torch.randn(1, 3, 8, 8)
        out = model.forward_train(
            image,
            raq_trg_list=[8],
            raq_rvq_k_lists=[[2, 4]],
        )
        self.assertEqual(out["raq_target_list"], [8])
        self.assertEqual(out["rvq_k_lists"], [[2, 4]])
        self.assertEqual(len(out["vq_losses_raq"]), 1)
        self.assertEqual(len(out["rvq_codebooks_list"][0]), 2)
        torch.testing.assert_close(
            out["rvq_codebooks_list"][0][1][0],
            torch.zeros_like(out["rvq_codebooks_list"][0][1][0]),
        )
        loss = out["reconstructed_images_raq"].pow(2).mean()
        loss = loss + sum(out["vq_losses_raq"])
        loss.backward()
        stage2_grads = [
            parameter.grad
            for parameter in model.raqs_rvq_stage2.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(any(grad is not None for grad in stage2_grads))

    def test_one_bit_total_uses_single_active_stage(self):
        model = self._build_model(target_list=(2,)).train()
        out = model.forward_train(
            torch.randn(1, 3, 8, 8),
            raq_trg_list=[2],
            raq_rvq_k_lists=[[2]],
        )
        self.assertEqual(out["rvq_k_lists"], [[2]])
        self.assertEqual(len(out["rvq_codebooks_list"][0]), 1)


if __name__ == "__main__":
    unittest.main()
