"""Cosine/Gray mapping contracts used by joint CSCR."""

import unittest

import torch

from utils.cscr import build_cscr_mapping


class CSCRMappingTests(unittest.TestCase):
    def test_known_cosine_order_gray_labels_and_unchanged_inputs_and_rng(self):
        order = torch.tensor([5, 2, 7, 0, 6, 1, 4, 3])
        angles = torch.arange(8, dtype=torch.float64) / 10
        codebook = torch.empty(8, 2, dtype=torch.float64)
        codebook[order] = torch.stack((angles.cos(), angles.sin()), dim=1)
        codebook[5] *= 3
        original = codebook.clone()
        rng = torch.random.get_rng_state().clone()
        mapping = build_cscr_mapping(codebook)
        gray = torch.tensor([0, 1, 3, 2, 6, 7, 5, 4])
        torch.testing.assert_close(mapping["greedy_order"], order)
        torch.testing.assert_close(mapping["index_to_label"][order], gray)
        torch.testing.assert_close(mapping["label_to_index"][gray], order)
        torch.testing.assert_close(codebook, original, rtol=0, atol=0)
        torch.testing.assert_close(torch.random.get_rng_state(), rng)
        reordered = torch.empty_like(codebook)
        reordered[gray] = codebook[order]
        torch.testing.assert_close(
            codebook[mapping["label_to_index"]], reordered, rtol=0, atol=0,
        )

    def test_zero_vectors_are_deterministic_and_invalid_codebooks_are_rejected(self):
        mapping = build_cscr_mapping(torch.zeros(4, 3))
        torch.testing.assert_close(mapping["greedy_order"], torch.arange(4))
        for invalid in (torch.ones(3, 2), torch.ones(4, 0), torch.ones(4),
                        torch.full((4, 2), float("nan")), torch.full((4, 2), float("inf"))):
            with self.subTest(shape=tuple(invalid.shape)):
                with self.assertRaises(ValueError):
                    build_cscr_mapping(invalid)


if __name__ == "__main__":
    unittest.main()
