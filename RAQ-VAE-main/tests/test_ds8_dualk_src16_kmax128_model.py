"""CPU checks for the source-K=16, target-K<=128 dual-K model."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_seq2seq_cars import torch
from eval_seq2seq_channel import dual_k_transmission_plan
from nets.model_ds8_dualk import RAQVAE_TWO_DS8_DUALK
from nets.model_ds8_dualk_src16_kmax128 import (
    RAQVAE_TWO_DS8_DUALK_SRC16_KMAX128,
)


def args():
    return SimpleNamespace(
        n_hid=64, embedding_dim=64, num_embeddings=16,
        num_embeddings_min=2, num_embeddings_max=128, num_embeddings_test=2,
        lr=5e-4, device=torch.device("cpu"), raq_type="dd",
        model_type="vqvae2_ds8_dualk_src16_kmax128",
    )


class Src16KMax128ModelTests(unittest.TestCase):
    def test_configuration_and_original_variant_are_isolated(self):
        model = RAQVAE_TWO_DS8_DUALK_SRC16_KMAX128(args()).eval()
        self.assertEqual(model.quantize.embed.weight.shape, (16, 64))
        self.assertEqual(model.embed_layer_dec.weight.shape, (128, 64))
        self.assertEqual(model.K_CHOICES, (2, 4, 8, 16, 32, 64, 128))
        self.assertEqual(RAQVAE_TWO_DS8_DUALK.K_CHOICES, (2, 4, 8, 16, 32, 64))

    def test_generates_128_codewords_and_runs_dual_scale_forward(self):
        torch.manual_seed(42)
        model = RAQVAE_TWO_DS8_DUALK_SRC16_KMAX128(args()).eval()
        with torch.no_grad():
            generated = model.cbk2cbk(
                model.src, model.target_tensor(128)
            ).squeeze(1)
            image = torch.randn(1, 3, 64, 64)
            result = model(image, model.target_tensor(128), model.target_tensor(2))
        self.assertEqual(generated.shape, (128, 64))
        self.assertEqual(result[7].shape, (1, 8, 8))
        self.assertEqual(result[6].shape, (1, 4, 4))
        self.assertEqual(result[4].shape, image.shape)

    def test_k128_rate_plan_and_default_test_rate(self):
        expanded = dual_k_transmission_plan(
            1024, 256, 128, 128, 196608, 192, 256, 2, 48,
            min_k=2, max_k=128,
        )
        self.assertEqual(expanded["selected"]["bits_per_bottom_index"], 7)
        self.assertEqual(expanded["selected"]["bits_per_top_index"], 7)
        default = dual_k_transmission_plan(
            1024, 256, 32, 16, 196608, 192, 256, 2, 48,
            min_k=2, max_k=128,
        )
        self.assertEqual(default["selected"]["source_payload_bits"], 6144)
        self.assertEqual(default["selected"]["ldpc_padding_bits"], 0)
        self.assertEqual(default["selected"]["transmission_ratio"], 1 / 48)


if __name__ == "__main__":
    unittest.main()
