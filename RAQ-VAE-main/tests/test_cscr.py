"""Check CSCR label assignment, inverse lookup, and trial compatibility."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aggregate_seq2seq_channel_trials import aggregate_reports
from cscr import build_cscr_mapping


class CSCRTests(unittest.TestCase):
    def test_known_cosine_path_and_gray_labels_preserve_codewords_and_rng(self):
        # Distinct angular neighbors; the largest-norm vector starts the path.
        order = torch.tensor([5, 2, 7, 0, 6, 1, 4, 3])
        angles = torch.arange(8, dtype=torch.float64) / 10
        codebook = torch.empty(8, 2, dtype=torch.float64)
        codebook[order] = torch.stack((angles.cos(), angles.sin()), dim=1)
        codebook[5] *= 3
        original = codebook.clone()
        rng = torch.random.get_rng_state().clone()
        mapping = build_cscr_mapping(codebook)

        torch.testing.assert_close(mapping["greedy_order"], order)
        gray = torch.tensor([0, 1, 3, 2, 6, 7, 5, 4])
        torch.testing.assert_close(mapping["index_to_label"][order], gray)
        torch.testing.assert_close(mapping["label_to_index"][gray], order)
        torch.testing.assert_close(codebook, original, rtol=0, atol=0)
        torch.testing.assert_close(torch.random.get_rng_state(), rng)

        # Inverse lookup is exactly the physically reordered codebook, even for
        # corrupted labels, not just a successful round trip of sent indices.
        reordered = torch.empty_like(codebook)
        reordered[gray] = codebook[order]
        labels = torch.arange(8)
        torch.testing.assert_close(
            codebook[mapping["label_to_index"][labels]], reordered[labels],
            rtol=0, atol=0,
        )

    def test_both_branch_sizes_keep_fixed_width_and_exact_noiseless_embeddings(self):
        for size in (2, 4, 8, 16, 32, 64):
            with self.subTest(size=size):
                values = torch.arange(size * 3, dtype=torch.float64).reshape(size, 3)
                codebook = (values + 1).sin()
                mapping = build_cscr_mapping(codebook)
                indices = torch.arange(size).reshape(1, 2, size // 2)
                sent = mapping["index_to_label"][indices]
                self.assertEqual(sent.shape, indices.shape)
                self.assertEqual(sent.min().item(), 0)
                self.assertEqual(sent.max().item(), size - 1)
                restored = mapping["label_to_index"][sent]
                torch.testing.assert_close(restored, indices)
                torch.testing.assert_close(
                    torch.nn.functional.embedding(restored, codebook),
                    torch.nn.functional.embedding(indices, codebook),
                    rtol=0, atol=0,
                )

    def test_zero_vectors_and_ties_have_a_deterministic_valid_permutation(self):
        mapping = build_cscr_mapping(torch.zeros(4, 3))
        torch.testing.assert_close(mapping["greedy_order"], torch.arange(4))
        torch.testing.assert_close(
            mapping["index_to_label"], torch.tensor([0, 1, 3, 2])
        )

    def test_rejects_invalid_codebooks(self):
        invalid = (
            torch.ones(3, 2), torch.ones(1, 2), torch.ones(4, 0),
            torch.ones(4), torch.ones(4, 2, dtype=torch.long),
            torch.full((4, 2), float("nan")),
            torch.full((4, 2), float("inf")),
        )
        for codebook in invalid:
            with self.subTest(shape=tuple(codebook.shape), dtype=codebook.dtype):
                with self.assertRaises(ValueError):
                    build_cscr_mapping(codebook)

    def test_aggregation_accepts_legacy_off_but_rejects_mixed_cscr_configs(self):
        report = {
            "metadata": {
                "checkpoint_sha256": "checkpoint", "target_K_bottom": 32,
                "target_K_top": 16, "dataset_sha256": {"image": "hash"},
                "snr_db": 4.5, "channel": "awgn", "rician_k_factor": 10,
                "modulation": "qpsk", "ldpc": {"rate": 0.625},
                "rate_plan": {}, "seed": 42,
            },
            "metrics": {"psnr": 25, "ms_ssim": 0.9, "ms_ssim_db": 10},
            "totals": {
                "source_bit_errors": 0, "source_payload_bits": 100,
                "combined_bit_errors": 0, "ldpc_input_bits": 160,
                "channel_symbols": 128,
            },
            "num_images": 24,
        }
        other = copy.deepcopy(report)
        other["metadata"].update(seed=43, cscr={"enabled": False})
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / f"trial_{seed}.json" for seed in (42, 43)]
            paths[0].write_text(json.dumps(report))
            paths[1].write_text(json.dumps(other))
            result = aggregate_reports(paths, [42, 43])
            self.assertFalse(result["configuration"]["cscr"]["enabled"])
            other["metadata"]["cscr"] = {"enabled": True}
            paths[1].write_text(json.dumps(other))
            with self.assertRaisesRegex(ValueError, "Incompatible trial"):
                aggregate_reports(paths, [42, 43])


if __name__ == "__main__":
    unittest.main()
