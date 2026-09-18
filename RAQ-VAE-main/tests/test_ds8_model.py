"""Shape, inheritance, loss and serialization checks for RAQVAE_TWO_DS8."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_seq2seq_cars import torch
from nets.model import RAQVAE_TWO
from nets.model_ds8 import RAQVAE_TWO_DS8


def args():
    return SimpleNamespace(
        n_hid=64, embedding_dim=64, num_embeddings=64,
        num_embeddings_min=2, num_embeddings_max=64, num_embeddings_test=2,
        lr=5e-4, device=torch.device("cpu"), raq_type="dd", model_type="vqvae2_ds8",
    )


class DS8ModelTests(unittest.TestCase):
    def test_inherits_original_training_behavior(self):
        for method in ("encode", "decode", "forward", "training_step", "validation_step",
                       "sample_train_trg", "sample_val_trg", "configure_optimizers"):
            self.assertIs(getattr(RAQVAE_TWO_DS8, method), getattr(RAQVAE_TWO, method))

    def test_shapes_and_state_reload(self):
        torch.manual_seed(42)
        model = RAQVAE_TWO_DS8(args()).eval()
        image = torch.randn(1, 3, 256, 256)
        target = torch.arange(4).unsqueeze(1)
        with torch.no_grad():
            bottom = model.encoder_b(image)
            top = model.encoder_t(bottom)
            result = model(image, target)
        self.assertEqual(bottom.shape[-2:], (32, 32))
        self.assertEqual(top.shape[-2:], (16, 16))
        self.assertEqual(result[4].shape, image.shape)
        self.assertEqual(result[7].shape[-2:], (32, 32))
        self.assertEqual(result[6].shape[-2:], (16, 16))
        self.assertEqual(result[7].numel() + result[6].numel(), 1280)
        clone = RAQVAE_TWO_DS8(args()).eval()
        clone.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            restored = clone(image, target)[4]
        torch.testing.assert_close(restored, result[4], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
