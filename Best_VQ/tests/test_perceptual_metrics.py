import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from evaluation.perceptual import PerceptualMetrics, reconstruction_fid
from evaluation.quality import evaluate_no_channel
from test_real import test_real as run_eval


class _FeatureNetwork(torch.nn.Module):
    def forward(self, image):
        assert image.min() >= 0 and image.max() <= 1
        return [image.mean(dim=(-2, -1), keepdim=True)]


class _DistanceNetwork(torch.nn.Module):
    def forward(self, real, reconstructed):
        assert real.min() >= -1 and real.max() <= 1
        assert reconstructed.min() >= -1 and reconstructed.max() <= 1
        return (real - reconstructed).abs().mean(dim=(1, 2, 3), keepdim=True)


def _metric_accumulator():
    metrics = PerceptualMetrics.__new__(PerceptualMetrics)
    metrics.device = torch.device("cpu")
    metrics.inception = _FeatureNetwork()
    metrics.lpips = _DistanceNetwork()
    metrics.reset()
    return metrics


class _IdentityModel:
    quantizer_type = "none"

    def eval(self):
        return self

    def forward_test(self, image):
        return {"indices": [image]}

    def reconstruct_from_indices(self, indices, feature_shapes=None):
        return indices[0]


class ReconstructionFIDTests(unittest.TestCase):
    def test_identical_features_and_known_mean_shift(self):
        features = np.random.default_rng(42).normal(size=(4, 8))
        self.assertAlmostEqual(reconstruction_fid(features, features), 0.0)
        shift = np.arange(8) * 0.1
        self.assertAlmostEqual(reconstruction_fid(features, features + shift), shift.dot(shift))

    def test_matches_standard_fid_for_singular_and_full_rank_covariances(self):
        from pytorch_fid.fid_score import calculate_frechet_distance

        rng = np.random.default_rng(42)
        for n, dims in ((4, 8), (12, 3)):
            with self.subTest(num_images=n, feature_dims=dims):
                real = rng.normal(size=(n, dims))
                reconstructed = rng.normal(size=(n, dims))
                expected = calculate_frechet_distance(
                    real.mean(0), np.cov(real, rowvar=False),
                    reconstructed.mean(0), np.cov(reconstructed, rowvar=False),
                )
                self.assertAlmostEqual(reconstruction_fid(real, reconstructed), expected, places=5)

    def test_insufficient_samples_and_invalid_shapes(self):
        self.assertIsNone(reconstruction_fid(np.zeros((1, 8)), np.zeros((1, 8))))
        with self.assertRaises(ValueError):
            reconstruction_fid(np.zeros((2, 8)), np.zeros((3, 8)))


class PerceptualAccumulatorTests(unittest.TestCase):
    def test_input_ranges_and_lpips_mean_weight_each_image(self):
        metrics = _metric_accumulator()
        metrics.update(torch.full((2, 3, 32, 32), -2.0), torch.full((2, 3, 32, 32), 2.0))
        metrics.update(torch.zeros(1, 3, 32, 32), torch.zeros(1, 3, 32, 32))
        result = metrics.compute()
        self.assertAlmostEqual(result["lpips"], 4 / 3)
        self.assertIsInstance(result["rfid"], float)
        self.assertEqual(metrics.num_images, 3)

    def test_resets_between_evaluation_conditions(self):
        metrics = _metric_accumulator()
        loader = [torch.zeros(1, 3, 32, 32), torch.ones(1, 3, 32, 32)]
        for _ in range(2):
            evaluate_no_channel(_IdentityModel(), loader, "cpu", perceptual_metrics=metrics)
            self.assertEqual(metrics.num_images, 2)
            self.assertAlmostEqual(metrics.compute()["rfid"], 0.0)
            self.assertAlmostEqual(metrics.compute()["lpips"], 0.0)

    def test_empty_dataset_is_rejected(self):
        with self.assertRaises(ValueError):
            _metric_accumulator().compute()

    def test_evaluation_outputs_new_metrics_and_json(self):
        cfg = SimpleNamespace(
            DEVICE="cpu", CHECKPOINT_DIR="unused", TEST_DATASET_PATH="unused",
            NUM_WORKERS=0, PIN_MEMORY=False,
        )
        loader = [torch.zeros(1, 3, 32, 32), torch.ones(1, 3, 32, 32)]
        inferred = {"quantizer_type": "none", "num_embeddings_list": [4, 16]}
        with tempfile.TemporaryDirectory() as tmp:
            output = str(Path(tmp) / "result.json")
            stdout = io.StringIO()
            with (
                patch("test_real.Config", return_value=cfg),
                patch("test_real.setup_seed"),
                patch("test_real.build_model_from_checkpoint", return_value=(_IdentityModel(), inferred)),
                patch("test_real.get_dataloader", return_value=loader),
                patch("test_real.PerceptualMetrics", return_value=_metric_accumulator()),
                redirect_stdout(stdout),
            ):
                results = run_eval(no_channel=True, json_output=output)
            payload = json.loads(Path(output).read_text())
            self.assertAlmostEqual(results["no_channel"]["rfid"], 0.0)
            self.assertAlmostEqual(results["no_channel"]["lpips"], 0.0)
            self.assertEqual(payload["results"]["no_channel"], results["no_channel"])
            self.assertEqual(payload["perceptual_metric_config"]["lpips"]["net"], "alex")
            self.assertIn("rFID", stdout.getvalue())
            self.assertIn("LPIPS", stdout.getvalue())
            self.assertNotIn("Source ratio", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
