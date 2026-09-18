"""CPU checks for independent dual-K sampling and branch-specific generation."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_seq2seq_cars import torch
from nets.model_ds8_dualk import RAQVAE_TWO_DS8_DUALK


def args():
    return SimpleNamespace(
        n_hid=64, embedding_dim=64, num_embeddings=64,
        num_embeddings_min=2, num_embeddings_max=64, num_embeddings_test=2,
        lr=5e-4, device=torch.device("cpu"), raq_type="dd",
        model_type="vqvae2_ds8_dualk",
    )


class DualKModelTests(unittest.TestCase):
    def test_independent_power_of_two_sampling(self):
        model = RAQVAE_TWO_DS8_DUALK(args())
        with mock.patch("nets.model_ds8_dualk.random.choice", side_effect=[4, 32]) as choice:
            self.assertEqual((model.sample_target_k(), model.sample_target_k()), (4, 32))
            self.assertEqual(choice.call_count, 2)
        self.assertEqual(model.K_CHOICES, (2, 4, 8, 16, 32, 64))

    def test_distinct_shapes_and_two_calls_even_when_equal(self):
        torch.manual_seed(42)
        model = RAQVAE_TWO_DS8_DUALK(args()).eval()
        image = torch.randn(1, 3, 256, 256)
        calls = []
        handle = model.cbk2cbk.register_forward_hook(
            lambda module, inputs, output: calls.append((len(inputs[1]), output.detach().clone()))
        )
        with torch.no_grad():
            result = model(image, model.target_tensor(4), model.target_tensor(16))
        self.assertEqual([call[0] for call in calls], [4, 16])
        self.assertEqual(result[7].shape, (1, 32, 32))
        self.assertEqual(result[6].shape, (1, 16, 16))
        self.assertEqual(result[4].shape, image.shape)
        calls.clear()
        with torch.no_grad():
            model(image, model.target_tensor(8), model.target_tensor(8))
        handle.remove()
        self.assertEqual([call[0] for call in calls], [8, 8])
        torch.testing.assert_close(calls[0][1], calls[1][1], rtol=0, atol=0)

    def test_state_dict_strict_reload(self):
        model = RAQVAE_TWO_DS8_DUALK(args()).eval()
        clone = RAQVAE_TWO_DS8_DUALK(args()).eval()
        clone.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(set(model.state_dict()), set(clone.state_dict()))


if __name__ == "__main__":
    unittest.main()

