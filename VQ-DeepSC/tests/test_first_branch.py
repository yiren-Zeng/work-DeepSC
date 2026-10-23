import json
import io
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from config import Config
from config_first_branch import FirstBranchConfig
from data.datasets import get_dataloader
from evaluation.quality import _transmit_ldpc_stream
from evaluation.first_branch_images import (
    ReconstructionWriter, evaluate_with_images, reconstruction_directory,
)
from losses.deepsc_loss import DeepSCLoss
from test_real import load_model
from test_real_first_branch import main as evaluate_first_branch
from train import build_model
from utils.bit_utils import bits_to_index_tensor, index_tensor_to_bits


class FirstBranchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_single_branch_backward_and_checkpoint_roundtrip(self):
        cfg = FirstBranchConfig()
        cfg.validate()
        torch.manual_seed(42)
        model = build_model(cfg)
        self.assertEqual(len(model.semantic_encoder.blocks), 1)
        self.assertEqual(len(model.semantic_decoder.up_blocks), 1)
        self.assertEqual(len(model.vector_quantizers), 1)
        self.assertEqual(tuple(model.vector_quantizers[0].embedding.weight.shape), (32, 128))

        images = torch.rand(1, 3, 256, 256) * 2 - 1
        output = model.forward_train(images)
        self.assertEqual(output["reconstructed_images"].shape, images.shape)
        self.assertEqual(len(output["vq_losses"]), 1)
        self.assertEqual(tuple(output["indices"][0].shape), (1, 128, 128))
        self.assertGreaterEqual(int(output["indices"][0].min()), 0)
        self.assertLess(int(output["indices"][0].max()), 32)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, betas=cfg.BETAS)
        before = model.vector_quantizers[0].embedding.weight.detach().clone()
        reconstruction, vq = DeepSCLoss()(images, output["reconstructed_images"], output["vq_losses"])
        loss = reconstruction + vq
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for parameter in (
            model.semantic_encoder.init[0].weight,
            model.vector_quantizers[0].embedding.weight,
            model.semantic_decoder.final.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.vector_quantizers[0].embedding.weight))

        model.eval()
        with torch.no_grad():
            expected = model(images)["reconstructed_images"]
            indices = model.forward_test(images)["indices"]
            torch.testing.assert_close(model.reconstruct_from_indices(indices), expected)
        with tempfile.TemporaryDirectory(prefix="vq_first_branch_checkpoint_") as temporary:
            checkpoint = Path(temporary) / "best_vq_deepsc.pth"
            torch.save(model.state_dict(), checkpoint)
            restored = load_model(str(checkpoint), cfg, torch.device("cpu"))
            with torch.no_grad():
                torch.testing.assert_close(restored(images)["reconstructed_images"], expected)

    def test_original_configuration_and_training_settings_are_preserved(self):
        original = Config()
        cfg = FirstBranchConfig()
        self.assertEqual(original.NUM_DOWNSAMPLE_BLOCKS, 4)
        self.assertEqual(original.EMBEDDING_DIM_LIST, [128, 256, 512, 1024])
        for name in (
            "LEARNING_RATE", "BETAS", "TOTAL_BATCH_SIZE", "MICRO_BATCH_SIZE",
            "COMMITMENT_COST", "TRAIN_DATASET_PATH", "VAL_DATASET_PATH",
            "TEST_BATCH_SIZE", "NUM_WORKERS", "PIN_MEMORY",
        ):
            self.assertEqual(getattr(cfg, name), getattr(original, name), name)
        self.assertNotEqual(cfg.CHECKPOINT_DIR, original.CHECKPOINT_DIR)
        self.assertNotEqual(cfg.LOG_DIR, original.LOG_DIR)
        self.assertEqual(cfg.NUM_EPOCHS, 200)
        self.assertEqual(cfg.TEST_DATASET_PATH, "/workspace/yi/work/Kodak")
        self.assertEqual(original.TEST_DATASET_PATH, "/workspace/yi/work/Kodak-256-transform-resize")
        with torch.device("meta"):
            original_model = build_model(original)
        self.assertEqual(len(original_model.semantic_encoder.blocks), 4)
        self.assertEqual(len(original_model.vector_quantizers), 4)
        self.assertEqual(len(original_model.semantic_decoder.up_blocks), 4)

    def test_new_evaluation_entry_defaults_to_bpsk_half_rate(self):
        with patch("test_real_first_branch.evaluate_with_images", return_value={}) as evaluate:
            evaluate_first_branch([])
        arguments = evaluate.call_args.kwargs
        self.assertIsInstance(arguments["cfg"], FirstBranchConfig)
        self.assertEqual(arguments["cfg"].TEST_DATASET_PATH, "/workspace/yi/work/Kodak")
        self.assertEqual(arguments["modulation"], "bpsk")
        self.assertEqual(arguments["ldpc_rate"], 0.5)
        self.assertEqual(arguments["ldpc_n"], 256)
        self.assertEqual(arguments["test_snrs"], [6])
        self.assertEqual(arguments["stream_packing"], "combined")
        self.assertNotIn("json_output", arguments)
        self.assertTrue(arguments["image_output_dir"].endswith("generated_images/k32_first_branch"))

    def test_loader_preserves_native_landscape_and_portrait_sizes(self):
        sizes = [(768, 512), (512, 768)]
        with tempfile.TemporaryDirectory(prefix="vq_first_branch_native_") as temporary:
            for index, size in enumerate(sizes):
                Image.new("RGB", size, (31, 63, 127)).save(Path(temporary) / f"{index}.png")
            loader = get_dataloader(
                temporary, batch_size=1, shuffle=False, mode="test",
                num_workers=0, pin_memory=False,
            )
            observed = []
            for image in loader:
                observed.append((int(image.shape[-1]), int(image.shape[-2])))
                self.assertEqual(tuple(image.shape[:2]), (1, 3))
                torch.testing.assert_close(
                    image[0, :, 0, 0], torch.tensor([31, 63, 127]) / 255.0 * 2 - 1
                )
            self.assertCountEqual(observed, sizes)

    def test_console_summary_suppresses_diagnostics_and_does_not_save_json(self):
        result = {
            "6": {
                "psnr": 25.12341,
                "ms_ssim": 0.91234,
                "diagnostics": {
                    "ldpc": {"k": 128, "n": 256},
                    "combined_stream": {"channel_symbols": 163840, "source_values": 196608},
                },
            },
        }

        def noisy_evaluation(**arguments):
            self.assertNotIn("json_output", arguments)
            print("verbose scale diagnostics")
            os.write(2, b"native framework initialization\n")
            return result

        output = io.StringIO()
        with patch("test_real_first_branch.evaluate_with_images", side_effect=noisy_evaluation), redirect_stdout(output):
            actual = evaluate_first_branch([])
        self.assertIs(actual, result)
        self.assertEqual(output.getvalue().splitlines(), [
            "K=32 | BPSK | LDPC=1/2 | CBR=5/6 (0.833333)",
            "SNR=6 dB | PSNR=25.1234 dB | MS-SSIM=0.9123",
        ])

    def test_evaluation_errors_restore_output_and_propagate(self):
        output = io.StringIO()
        errors = io.StringIO()
        with patch("test_real_first_branch.evaluate_with_images", side_effect=FileNotFoundError("missing checkpoint")), redirect_stdout(output), redirect_stderr(errors):
            with self.assertRaisesRegex(FileNotFoundError, "missing checkpoint"):
                evaluate_first_branch([])
            print("stdout restored")
            print("stderr restored", file=sys.stderr)
        self.assertEqual(output.getvalue(), "stdout restored\n")
        self.assertEqual(errors.getvalue(), "stderr restored\n")

    def test_reconstruction_pngs_preserve_names_sizes_and_evaluation_tensor(self):
        class FixedReconstruction(torch.nn.Module):
            def __init__(self, image):
                super().__init__()
                self.image = image

            def reconstruct_from_indices(self, indices):
                return self.image

        image = torch.zeros(1, 3, 7, 11)
        image[:, 0] = -2
        image[:, 2] = 2
        before = image.clone()
        with tempfile.TemporaryDirectory(prefix="vq_first_branch_export_") as temporary:
            directory = Path(temporary) / "bpsk_ldpc12_snr10"
            writer = ReconstructionWriter(FixedReconstruction(image), ["kodim07.png"], directory)
            self.assertIs(writer.reconstruct_from_indices([]), image)
            writer.assert_complete()
            torch.testing.assert_close(image, before)
            with Image.open(directory / "kodim07.png") as saved:
                self.assertEqual(saved.size, (11, 7))
                self.assertEqual(saved.mode, "RGB")
                self.assertEqual(saved.getpixel((0, 0)), (0, 128, 255))
            replacement = ReconstructionWriter(
                FixedReconstruction(torch.ones_like(image)), ["kodim07.png"], directory
            )
            replacement.reconstruct_from_indices([])
            replacement.assert_complete()
            with Image.open(directory / "kodim07.png") as saved:
                self.assertEqual(saved.getpixel((0, 0)), (255, 255, 255))
            self.assertEqual([path.name for path in directory.iterdir()], ["kodim07.png"])

    def test_reconstruction_directories_separate_link_settings(self):
        code = {"k": 128, "n": 256}
        root = Path("generated_images/k32_first_branch")
        self.assertEqual(
            reconstruction_directory(root, "bpsk", 10, code), root / "bpsk_ldpc12_snr10"
        )
        self.assertNotEqual(
            reconstruction_directory(root, "bpsk", 10, code),
            reconstruction_directory(root, "bpsk", 6, code),
        )
        self.assertNotEqual(
            reconstruction_directory(root, "bpsk", 10, code),
            reconstruction_directory(root, "qpsk", 10, code),
        )
        self.assertNotEqual(
            reconstruction_directory(root, "bpsk", 10, code),
            reconstruction_directory(root, "bpsk", 10, {"k": 192, "n": 256}),
        )
        self.assertEqual(reconstruction_directory(root), root / "no_channel")

    def test_no_channel_export_uses_source_filenames_in_loader_order(self):
        class HalfIntensityReconstruction(torch.nn.Module):
            def forward_test(self, images):
                self.images = images
                batch, _, height, width = images.shape
                return {"indices": [torch.zeros(batch, height // 2, width // 2, dtype=torch.long)]}

            def reconstruct_from_indices(self, indices):
                return self.images * 0.5

        with tempfile.TemporaryDirectory(prefix="vq_first_branch_export_eval_") as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            Image.new("RGB", (24, 16), (0, 0, 0)).save(source / "kodim20.png")
            Image.new("RGB", (16, 24), (255, 255, 255)).save(source / "kodim03.png")
            cfg = FirstBranchConfig()
            cfg.DEVICE = "cpu"
            cfg.TEST_DATASET_PATH = str(source)
            cfg.NUM_WORKERS = 0
            cfg.PIN_MEMORY = False
            root = Path(temporary) / "reconstructed"
            with patch("evaluation.first_branch_images.load_model", return_value=HalfIntensityReconstruction()):
                results = evaluate_with_images(cfg=cfg, image_output_dir=root, no_channel=True)
            result = results["no_channel"]
            self.assertEqual(result["num_saved_images"], 2)
            self.assertEqual(result["diagnostics"]["num_images"], 2)
            self.assertEqual(result["image_output_dir"], str(root / "no_channel"))
            for name, size, color in [
                ("kodim20.png", (24, 16), (64, 64, 64)),
                ("kodim03.png", (16, 24), (191, 191, 191)),
            ]:
                with Image.open(root / "no_channel" / name) as saved:
                    self.assertEqual(saved.size, size)
                    self.assertEqual(saved.getpixel((0, 0)), color)
            self.assertEqual(len(list(root.rglob("*.*"))), 2)

    def test_real_ldpc_bpsk_symbol_count_is_five_sixths(self):
        from communications.ldpc_coding import get_ldpc_code

        indices = torch.arange(128 * 128).remainder(32).reshape(1, 128, 128)
        bits, shape, _ = index_tensor_to_bits(indices, 32)
        self.assertEqual(len(bits), 81920)
        code = get_ldpc_code(128, rate=0.5)
        decoded, stats = _transmit_ldpc_stream(
            bits, target_snr=20, ldpc_code=code,
            device=torch.device("cpu"), modulation="bpsk",
        )
        np.testing.assert_array_equal(decoded, bits)
        torch.testing.assert_close(bits_to_index_tensor(decoded, shape, 32), indices)
        self.assertEqual(stats["ldpc_input_bits"], 81920)
        self.assertEqual(stats["ldpc_padding_bits"], 0)
        self.assertEqual(stats["coded_bits"], 163840)
        self.assertEqual(stats["transmitted_bits"], 163840)
        self.assertEqual(stats["modulation_padding_bits"], 0)
        self.assertEqual(stats["channel_symbols"], 163840)
        self.assertEqual(stats["channel_symbols"] * 6, (3 * 256 * 256) * 5)
        print(json.dumps({"first_branch_payload_bits": len(bits), **stats,
                          "source_values": 3 * 256 * 256,
                          "cbr": stats["channel_symbols"] / (3 * 256 * 256),
                          "cbr_fraction": "5/6"}, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
