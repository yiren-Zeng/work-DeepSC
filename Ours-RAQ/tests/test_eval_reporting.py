"""Regression checks for exact CBR, mean MS-SSIM dB, and reconstruction export."""

import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import test_real as evaluator
from utils.image_export import ReconstructionImageWriter


class ToyModel(torch.nn.Module):
    def forward_test(self, image):
        self.image = image
        return {
            "indices": [torch.zeros((1, 2, 2), dtype=torch.long)],
            "num_embeddings_list": [2],
        }

    def reconstruct_from_indices(self, indices, **kwargs):
        return self.image * 0.5


class NamedLoader(list):
    def __init__(self):
        super().__init__([
            torch.full((1, 3, 16, 16), -1.0),
            torch.full((1, 3, 16, 16), 1.0),
        ])
        self.dataset = SimpleNamespace(image_files=["z.png", "a.png"])


class EvaluationReportingTests(unittest.TestCase):
    def test_cbr_uses_actual_symbols_including_padding(self):
        diagnostics = {
            "mode": "ldpc",
            "total": {
                "source_values": 256 * 256 * 3,
                "payload_bits": 131071,
                "payload_bpp": 131071 / (256 * 256),
                "transmitted_bits": 262144,
                "channel_symbols": 262144,
                "transmission_ratio": 4 / 3,
            },
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            stats = evaluator._print_compression_stats(diagnostics)
        self.assertEqual(stats["transmission_ratio_fraction"], "4/3")
        self.assertIn("4/3 (1.33333333)", output.getvalue())

        diagnostics["total"].update({
            "channel_symbols": 262145,
            "transmission_ratio": 262145 / 196608,
        })
        with contextlib.redirect_stdout(io.StringIO()):
            stats = evaluator._print_compression_stats(diagnostics)
        self.assertEqual(stats["transmission_ratio_fraction"], "262145/196608")

    def test_ms_ssim_db_endpoints(self):
        self.assertEqual(evaluator._ms_ssim_db(0), 0)
        self.assertAlmostEqual(evaluator._ms_ssim_db(0.99), 20)
        self.assertTrue(math.isinf(evaluator._ms_ssim_db(1)))

    def test_ms_ssim_db_matches_four_decimal_display_in_reported_example(self):
        mean_ms_ssim = 0.993566460081
        self.assertEqual(f"{mean_ms_ssim:.4f}", "0.9936")
        self.assertEqual(f"{evaluator._ms_ssim_db(mean_ms_ssim):.4f}", "21.9382")
        self.assertAlmostEqual(
            evaluator._ms_ssim_db(mean_ms_ssim), -10 * math.log10(1 - 0.9936)
        )

    def test_ms_ssim_db_handles_rounding_to_one(self):
        self.assertTrue(math.isinf(evaluator._ms_ssim_db(0.99996)))
        self.assertAlmostEqual(evaluator._ms_ssim_db(0.99994), 40)

    def test_mean_conversion_and_separate_runs_with_all_images(self):
        cfg = SimpleNamespace(
            validate=lambda: None,
            DEVICE="cpu",
            CHECKPOINT_DIR="unused",
            TEST_DATASET_PATH="unused",
            NUM_WORKERS=0,
            PIN_MEMORY=False,
            USE_RAQ=False,
        )
        loader = NamedLoader()
        with tempfile.TemporaryDirectory() as root:
            for _ in range(2):
                output = io.StringIO()
                with (
                    patch.object(evaluator, "Config", return_value=cfg),
                    patch.object(evaluator, "get_dataloader", return_value=loader),
                    patch.object(evaluator, "build_model_from_checkpoint", return_value=(
                        ToyModel(),
                        {"num_embeddings_list": [2], "quantizer_type": "simvq"},
                    )),
                    patch("evaluation.quality._image_quality", side_effect=[(0.90001, 20), (0.99001, 30)]),
                    contextlib.redirect_stdout(output),
                ):
                    results = evaluator.test_real(
                        no_channel=True, report_rate=True, image_output_dir=root
                    )
                metrics = results["no_channel"]
                self.assertAlmostEqual(metrics["ms_ssim"], 0.94501)
                self.assertAlmostEqual(metrics["ms_ssim_db"], -10 * math.log10(0.055))
                self.assertNotAlmostEqual(metrics["ms_ssim_db"], 15)
                self.assertIn("Avg MS-SSIM: 0.9450", output.getvalue())
                self.assertIn("MS-SSIM (dB, from 4dp mean): 12.5964", output.getvalue())
                self.assertIn("MS-SSIM (dB)", output.getvalue())
                self.assertIn("CBR: N/A", output.getvalue())
                image_dir = Path(metrics["image_output_dir"])
                self.assertEqual({p.name for p in image_dir.iterdir()}, {"z.png", "a.png"})
                with Image.open(image_dir / "z.png") as image:
                    self.assertTrue(np.all(np.asarray(image) == 64))
                with Image.open(image_dir / "a.png") as image:
                    self.assertTrue(np.all(np.asarray(image) == 191))
            self.assertEqual(len(list(Path(root).iterdir())), 2)

    def test_png_export_clamps_without_mutating_metric_tensors(self):
        reconstruction = torch.tensor([-2.0, 0.0, 2.0]).reshape(1, 3, 1, 1)
        original = reconstruction.clone()
        with tempfile.TemporaryDirectory() as root:
            writer = ReconstructionImageWriter(Path(root) / "snr_0", ["kodim01.png"])
            writer(reconstruction)
            with Image.open(writer.output_dir / "kodim01.png") as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.getpixel((0, 0)), (0, 128, 255))
        torch.testing.assert_close(reconstruction, original)

    def test_duplicate_stems_do_not_overwrite_and_batches_keep_names(self):
        with tempfile.TemporaryDirectory() as root:
            output_dir = Path(root) / "snr_3"
            writer = ReconstructionImageWriter(output_dir, ["z.png", "a.jpg", "a.png"])
            writer(torch.zeros((2, 3, 2, 4)))
            writer(torch.ones((1, 3, 2, 4)))
            self.assertEqual(writer.saved_count, 3)
            self.assertEqual({p.name for p in output_dir.iterdir()}, {"z.png", "a.png", "a_1.png"})
            with Image.open(output_dir / "a_1.png") as image:
                self.assertEqual(image.size, (4, 2))
                self.assertTrue(np.all(np.asarray(image) == 255))
            with self.assertRaises(FileExistsError):
                ReconstructionImageWriter(output_dir)


if __name__ == "__main__":
    unittest.main()
