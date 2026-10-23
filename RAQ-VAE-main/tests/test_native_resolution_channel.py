"""Check native Kodak geometry and preserve other evaluation entry defaults."""

import contextlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval_seq2seq_channel as evaluator
import eval_seq2seq_src256_k4_4096_channel as native_entry


class NativeResolutionTests(unittest.TestCase):
    def test_legacy_geometry_stays_256_square(self):
        geometry = evaluator.evaluation_geometry()
        self.assertEqual(geometry["shapes"], ((256, 256),))
        self.assertEqual(geometry["pixels"], 65536)
        self.assertEqual(geometry["source_indices"], 5120)
        self.assertEqual(geometry["bottom_indices"], 4096)
        self.assertEqual(geometry["top_indices"], 1024)

    def test_kodak_orientations_share_one_exact_rate_plan(self):
        geometry = evaluator.evaluation_geometry([(512, 768)] * 18 + [(768, 512)] * 6)
        self.assertEqual(geometry["shapes"], ((512, 768), (768, 512)))
        self.assertEqual(geometry["pixels"], 393216)
        self.assertEqual(geometry["source_values"], 1179648)
        self.assertEqual(geometry["bottom_indices"], 24576)
        self.assertEqual(geometry["top_indices"], 6144)
        self.assertEqual(geometry["source_indices"], 30720)
        self.assertEqual(evaluator.evaluation_geometry()["source_indices"], 5120)

    def test_fixed_k4096_plan_uses_full_image_payload(self):
        geometry = evaluator.evaluation_geometry([(512, 768)])
        plan = evaluator.transmission_plan(
            geometry["source_indices"], geometry["source_values"],
            128, 256, 4, 48, "fixed", 4, 4096, 4096,
        )
        self.assertEqual(plan["selected"]["source_payload_bits"], 368640)
        self.assertEqual(plan["selected"]["channel_symbols"], 184320)
        self.assertEqual(plan["selected"]["transmission_ratio"], 5 / 32)
        self.assertEqual(plan["budget_padding_bits"], 0)

    def test_native_ldpc_padding_is_recomputed_instead_of_scaled(self):
        native = evaluator.evaluation_geometry([(512, 768)])
        legacy = evaluator.evaluation_geometry()
        plans = [evaluator.transmission_plan(
            g["source_indices"], g["source_values"],
            112, 256, 4, 48, "fixed", 4, 4096, 4,
        )["selected"] for g in (native, legacy)]
        self.assertEqual(plans[0]["ldpc_padding_bits"], 48)
        self.assertEqual(plans[1]["ldpc_padding_bits"], 64)
        self.assertEqual(plans[0]["channel_symbols"], 35136)
        self.assertNotEqual(plans[0]["transmission_ratio"], plans[1]["transmission_ratio"])

    def test_different_image_areas_are_rejected_before_transmission(self):
        with self.assertRaisesRegex(ValueError, "equal pixel counts"):
            evaluator.evaluation_geometry([(512, 768), (256, 256)])

    def test_incompatible_stride_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "multiples of 8"):
            evaluator.evaluation_geometry([(513, 768)])

    def test_header_scan_preserves_portrait_and_landscape_without_transforming(self):
        files = [Path(f"kodim{i:02}.png") for i in range(24)]
        sizes = [(768, 512)] * 18 + [(512, 768)] * 6
        with mock.patch.object(Path, "is_dir", return_value=True), \
                mock.patch.object(Path, "is_file", return_value=True), \
                mock.patch.object(Path, "iterdir", return_value=files), \
                mock.patch("PIL.Image.open", side_effect=[
                    contextlib.nullcontext(SimpleNamespace(size=size)) for size in sizes
                ]):
            shapes = evaluator.dataset_image_shapes("/unused")
        self.assertEqual(shapes, [(height, width) for width, height in sizes])

    def test_only_specialized_entry_requests_native_geometry(self):
        argv = ["--run-dir", "/unused", "--target-k", "4096"]
        with mock.patch.object(evaluator, "OUTPUT_TAG", evaluator.OUTPUT_TAG), \
                mock.patch.object(evaluator, "main", return_value={"ok": True}) as call:
            self.assertEqual(native_entry.main(argv), {"ok": True})
        call.assert_called_once_with(argv, native_resolution=True)


if __name__ == "__main__":
    unittest.main()
