"""Check native ds2 geometry, PNG-only writes, and three-line summaries."""

import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import torch

import eval_seq2seq_one_ds2_src256_k32_2048_channel as evaluator


class ToyModel(torch.nn.Module):
    def __init__(self, _args):
        super().__init__()
        self.src = torch.zeros(256, 64)

    def load_state_dict(self, state, strict=True):
        pass

    def cbk2cbk(self, source, target):
        return torch.zeros(len(target), 1, 64)

    def forward(self, image, target):
        self.image = image
        h, w = image.shape[-2:]
        return (None, None, None, image * 0.5, None, torch.zeros(1, h // 2, w // 2, dtype=torch.long))

    def decode_latent(self, indices, target):
        return self.image * 0.5

    def decode(self, quantized):
        return self.image * 0.25


class NamedLoader(list):
    def __init__(self):
        super().__init__([
            torch.ones(1, 3, *( (4, 6) if i % 2 == 0 else (6, 4) ))
            for i in range(24)
        ])
        self.dataset = self
        self.image_files = [f"kodim{i + 1:02}.png" for i in range(24)]


class NativeImageEvaluationTests(unittest.TestCase):
    def test_geometry_uses_original_kodak_area_and_both_orientations(self):
        geometry = evaluator.evaluation_geometry([(512, 768)] * 18 + [(768, 512)] * 6)
        self.assertEqual(geometry["shapes"], ((512, 768), (768, 512)))
        self.assertEqual(geometry["pixels"], 393216)
        self.assertEqual(geometry["source_indices"], 98304)
        self.assertEqual(geometry["source_values"], 1179648)
        plan = evaluator.shared.transmission_plan(
            geometry["source_indices"], geometry["source_values"],
            112, 256, 4, 48, "fixed", 32, 2048, 32,
        )["selected"]
        self.assertEqual(plan["source_payload_bits"], 491520)
        self.assertEqual(plan["ldpc_padding_bits"], math.ceil(491520 / 112) * 112 - 491520)

    def test_geometry_rejects_incompatible_dimensions(self):
        for shapes in ([], [(513, 768)], [(512, 768), (256, 256)]):
            with self.subTest(shapes=shapes), self.assertRaises(ValueError):
                evaluator.evaluation_geometry(shapes)

    def test_summary_has_three_lines_exact_cbr_and_matching_db(self):
        report = {
            "metadata": {
                "image_shapes": [(512, 768), (768, 512)], "target_K": 32,
                "channel_enabled": True, "channel_label": "AWGN", "snr_db": 10,
                "ldpc": {"rate": 0.5}, "modulation": "bpsk", "seed": 42,
            },
            "metrics": {"psnr": 35.7371, "ms_ssim": 0.993566460081},
            "totals": {"channel_symbols": 983040, "source_values": 1179648},
            "num_images": 24, "image_output_dir": "/unused/generated_images/k32/awgn_snr10",
        }
        lines = evaluator.format_image_result(report).splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("768x512/512x768", lines[0])
        self.assertIn("seed=42", lines[0])
        self.assertIn("MS-SSIM=0.9936 (21.9382 dB)", lines[1])
        self.assertIn("CBR=5/6 (0.83333333)", lines[1])
        self.assertTrue(lines[2].startswith("图片："))

    def _evaluate_png_only(self, no_channel):
        loader = NamedLoader()
        bit_utils = SimpleNamespace(
            index_tensor_to_bits=lambda indices, k: (
                np.zeros(indices.numel() * int(math.log2(k)), dtype=np.uint8), tuple(indices.shape), k,
            ),
            bits_to_index_tensor=lambda bits, shape, k: torch.zeros(shape, dtype=torch.long),
        )
        modulation = SimpleNamespace(**{name: mock.Mock() for name in (
            "bpsk_llr", "bpsk_modulate", "qam16_llr", "qam16_modulate", "qpsk_llr", "qpsk_modulate",
        )})
        ldpc = SimpleNamespace(
            get_ldpc_code=lambda k, rate: {"k": k, "n": 256},
            ldpc_encode=mock.Mock(), ldpc_decode=mock.Mock(),
        )
        modules = {
            "utils.bit_utils": bit_utils,
            "utils.reproducibility": SimpleNamespace(_reset_eval_seed=mock.Mock()),
            "communications.modulation": modulation,
            "communications.ldpc_coding": ldpc,
        }
        quality = SimpleNamespace(_image_quality=lambda image, reconstruction: (0.993566460081, 35.7371))
        with tempfile.TemporaryDirectory() as folder, contextlib.ExitStack() as stack:
            root = Path(folder)
            checkpoint = root / "weights.ckpt"
            checkpoint.write_bytes(b"read-only fixture")
            (root / "config.json").write_text(json.dumps({
                "variant": "RAQVAE_ONE_DS2_SRC256_K32_2048_ORIGINAL",
                "model_args": {"num_embeddings": 256, "num_embeddings_min": 32,
                               "num_embeddings_max": 2048, "embedding_dim": 64},
            }))
            (root / "status.json").write_text(json.dumps({"best_model_path": str(checkpoint)}))
            stack.enter_context(mock.patch.dict(sys.modules, modules))
            stack.enter_context(mock.patch.object(evaluator, "local_path", side_effect=Path))
            stack.enter_context(mock.patch.object(evaluator.shared, "dataset_image_shapes", return_value=[(4, 6), (6, 4)]))
            stack.enter_context(mock.patch.object(evaluator, "load_shiyan_loader", return_value=(lambda *a, **k: loader, mock.Mock())))
            stack.enter_context(mock.patch.object(evaluator, "load_file_module", return_value=quality))
            stack.enter_context(mock.patch.object(evaluator, "RAQVAE_ONE_DS2_SRC256_K32_2048", ToyModel))
            real_device = torch.device
            stack.enter_context(mock.patch.object(evaluator.torch, "device", return_value=real_device("cpu")))
            stack.enter_context(mock.patch.object(evaluator.torch.cuda, "get_device_name", return_value="CPU fixture"))
            load = stack.enter_context(mock.patch.object(evaluator.torch, "load", return_value={"state_dict": {}, "epoch": 0, "global_step": 1}))
            writers = [stack.enter_context(mock.patch.object(owner, name)) for owner, name in (
                (evaluator, "write_json"), (evaluator.torch, "save"), (evaluator.shutil, "copy2"),
            )]
            stack.enter_context(mock.patch.object(evaluator.shared, "transmit_ldpc_stream_exact", side_effect=lambda bits, *a, **k: (bits, {
                "ldpc_input_bits": 128, "ldpc_padding_bits": 98, "coded_bits": 256,
                "transmitted_bits": 256, "channel_symbols": 256, "bit_errors": 0,
            })))
            output = io.StringIO()
            args = ["--run-dir", str(root), "--output", str(root / "images"),
                    "--target-k", "32", "--images-only", "--concise"]
            if no_channel:
                args.append("--no-channel")
            with contextlib.redirect_stdout(output):
                report = evaluator.main(args)
            self.assertEqual(len(output.getvalue().splitlines()), 3)
            self.assertEqual(len(list((root / "images").iterdir())), 24)
            self.assertEqual({p.suffix for p in (root / "images").iterdir()}, {".png"})
            self.assertEqual(report["totals"]["source_pixels"], 24 * 24)
            self.assertEqual(report["totals"]["source_values"], 24 * 72)
            self.assertAlmostEqual(report["metrics"]["ms_ssim_db"], -10 * math.log10(1 - 0.9936))
            with Image.open(root / "images/kodim01.png") as image:
                self.assertEqual(image.size, (6, 4))
                self.assertEqual(image.getpixel((0, 0)), (191, 191, 191) if no_channel else (159, 159, 159))
            for writer in writers:
                writer.assert_not_called()
            load.assert_called_once_with(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint.read_bytes(), b"read-only fixture")
            self.assertIn("CBR=N/A" if no_channel else "CBR=32/9", output.getvalue())

    def test_no_channel_writes_only_final_pngs(self):
        self._evaluate_png_only(no_channel=True)

    def test_channel_writes_only_received_reconstructions(self):
        self._evaluate_png_only(no_channel=False)


if __name__ == "__main__":
    unittest.main()
