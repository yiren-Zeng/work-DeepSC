import inspect
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import test_jpeg
from Best_VQ.evaluation.perceptual import PerceptualMetrics


class _RecordingMetrics:
    def __init__(self):
        self.pairs = []

    def update(self, original, reconstructed):
        self.pairs.append((original.clone(), reconstructed.clone()))

    def compute(self):
        return {"rfid": 12.5, "lpips": 0.25}


class JPEGPerceptualTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.original = Path(self.folder.name) / "original"
        self.decoded = Path(self.folder.name) / "decoded"
        self.original.mkdir()
        self.decoded.mkdir()
        for k, original_value, decoded_value in ((1, 0, 0), (2, 255, 128)):
            Image.fromarray(np.full((32, 32, 3), original_value, dtype=np.uint8)).save(
                self.original / f"kodim{k:02d}.png"
            )
            Image.fromarray(np.full((32, 32, 3), decoded_value, dtype=np.uint8)).save(
                self.decoded / f"val_0000{k:02d}.jp2", format="JPEG2000", irreversible=False
            )

    def run_evaluation(self):
        metrics = _RecordingMetrics()
        stdout = io.StringIO()
        with (
            patch.object(test_jpeg, "device", torch.device("cpu")),
            patch.object(test_jpeg, "PerceptualMetrics", return_value=metrics),
            patch.object(test_jpeg, "range", return_value=range(1, 3), create=True),
            patch.object(test_jpeg, "tqdm", side_effect=lambda items, **kwargs: items),
            patch.object(test_jpeg, "calculate_ms_ssim", return_value=0.9),
            redirect_stdout(stdout),
        ):
            results = test_jpeg.main(self.original, self.decoded)
        return results, metrics.pairs, stdout.getvalue()

    def test_reuses_the_exact_best_vq_metric_implementation(self):
        self.assertIs(test_jpeg.PerceptualMetrics, PerceptualMetrics)
        expected = Path(test_jpeg.__file__).resolve().parents[1] / "Best_VQ/evaluation/perceptual.py"
        self.assertEqual(Path(inspect.getsourcefile(test_jpeg.PerceptualMetrics)).resolve(), expected)

    def test_jpeg2000_pairs_use_the_shared_normalization_and_report_metrics(self):
        results, pairs, stdout = self.run_evaluation()
        self.assertEqual(len(pairs), 2)
        torch.testing.assert_close(pairs[0][0], -torch.ones(1, 3, 32, 32))
        torch.testing.assert_close(pairs[0][1], -torch.ones(1, 3, 32, 32))
        torch.testing.assert_close(pairs[1][0], torch.ones(1, 3, 32, 32))
        torch.testing.assert_close(pairs[1][1], torch.full((1, 3, 32, 32), 128 / 255 * 2 - 1))
        self.assertEqual(results["rfid"], 12.5)
        self.assertEqual(results["lpips"], 0.25)
        self.assertEqual(results["num_images"], 2)
        self.assertEqual(results["num_decode_failures"], 0)
        self.assertIn("rFID         = 12.5000", stdout)
        self.assertIn("平均 LPIPS   = 0.2500", stdout)
        self.assertAlmostEqual(results["ms_ssim"], 0.9)
        self.assertAlmostEqual(results["psnr"], 53.027365, places=5)

    def test_missing_jpeg2000_files_are_counted_as_black_reconstructions(self):
        (self.decoded / "val_000002.jp2").unlink()
        results, pairs, stdout = self.run_evaluation()
        self.assertEqual(len(pairs), 2)
        torch.testing.assert_close(pairs[1][0], torch.ones(1, 3, 32, 32))
        torch.testing.assert_close(pairs[1][1], -torch.ones(1, 3, 32, 32))
        self.assertEqual(results["num_decode_failures"], 1)
        self.assertEqual(results["psnr"], 50.0)
        self.assertAlmostEqual(results["ms_ssim"], 0.45)
        self.assertIn("解码失败 1 张", stdout)


if __name__ == "__main__":
    unittest.main()
