"""Checks for the strict-original single-scale RAQVAE_ONE x2 baseline."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from eval_seq2seq_channel import transmission_plan
from nets.cbk2cbk import CdBk2CdBk
from nets.model import RAQVAE_ONE
from nets.model_one_ds2_src256_k32_2048 import (
    RAQVAE_ONE_DS2_SRC256_K32_2048,
)
from nets.quantizer import Quantizer


def model_args():
    return SimpleNamespace(
        n_hid=8,
        embedding_dim=64,
        num_embeddings=256,
        num_embeddings_min=32,
        num_embeddings_max=2048,
        num_embeddings_test=256,
        lr=5e-4,
        device=torch.device("cpu"),
        raq_type="dd",
        model_type="vqvae_one_ds2_src256_k32_2048",
    )


class OneScaleDS2ModelTest(unittest.TestCase):
    def test_shapes_and_target_range(self):
        model = RAQVAE_ONE_DS2_SRC256_K32_2048(model_args()).eval()
        image = torch.randn(1, 3, 32, 32)
        target = torch.arange(32).unsqueeze(1)
        with torch.no_grad():
            result = model(image, target)
            generated = model.cbk2cbk(model.src, target).squeeze(1)
        self.assertEqual(tuple(result[3].shape), (1, 3, 32, 32))
        self.assertEqual(tuple(result[5].shape), (1, 16, 16))
        self.assertEqual(tuple(generated.shape), (32, 64))

    def test_original_modules_and_sampling_are_inherited(self):
        model = RAQVAE_ONE_DS2_SRC256_K32_2048(model_args())
        self.assertIs(type(model.cbk2cbk), CdBk2CdBk)
        self.assertIs(type(model.quantize_cbk), Quantizer)
        self.assertNotIn("sample_train_trg", type(model).__dict__)
        self.assertIs(type(model).sample_train_trg, RAQVAE_ONE.sample_train_trg)
        with mock.patch("nets.model.random.randint", return_value=777) as randint:
            target = model.sample_train_trg(32, 2048)
        randint.assert_called_once_with(32, 2048)
        self.assertEqual(tuple(target.shape), (777, 1))
        self.assertEqual(int(target[-1]), 776)

    def test_rate_grid_matches_ds2_ours_raq(self):
        plan = transmission_plan(
            source_indices=128 * 128,
            source_values=3 * 256 * 256,
            ldpc_k=128,
            ldpc_n=256,
            modulation_bits=1,
            denominator=48,
            policy="fixed",
            min_k=32,
            max_k=2048,
            target_k=256,
        )
        selected = plan["selected"]
        self.assertEqual(selected["source_payload_bits"], 128 * 128 * 8)
        self.assertEqual(selected["transmission_ratio"], 4 / 3)

    def test_optimizer_and_objective_methods_are_original(self):
        cls = RAQVAE_ONE_DS2_SRC256_K32_2048
        self.assertIs(cls.training_step, RAQVAE_ONE.training_step)
        self.assertIs(cls.configure_optimizers, RAQVAE_ONE.configure_optimizers)


if __name__ == "__main__":
    unittest.main()
