import unittest

import torch

from evaluation.quality import evaluate_ldpc_channel, evaluate_no_channel
from test_real import test_real as run_eval


class _IndicesModel:
    quantizer_type = "simvq"

    def eval(self):
        return self

    def forward_test(self, image):
        self.image = image
        self.indices = [
            (torch.arange(72) % 4).reshape(1, 8, 9),
            torch.arange(4).reshape(1, 2, 2),
        ]
        return {"indices": self.indices}

    def reconstruct_from_indices(self, indices, feature_shapes=None):
        for sent, received in zip(self.indices, indices):
            torch.testing.assert_close(sent.flatten(), received.flatten())
        return self.image


class EvalRateTests(unittest.TestCase):
    def setUp(self):
        self.model = _IndicesModel()
        self.loader = [torch.zeros(1, 3, 16, 16)]

    def test_source_rate_uses_actual_indices_and_image_sizes(self):
        loader = self.loader + [torch.zeros(1, 3, 17, 19)]
        ms_ssim, psnr, stats = evaluate_no_channel(
            self.model, loader, "cpu", [4, 16], return_rate_stats=True,
        )
        self.assertAlmostEqual(ms_ssim, 1.0, places=5)
        self.assertEqual(psnr, 100.0)
        self.assertEqual(stats["num_images"], 2)
        self.assertEqual(stats["payload_bits"], 320)
        self.assertEqual(stats["source_pixels"], 579)
        self.assertEqual(stats["raw_image_bits"], 579 * 24)
        self.assertAlmostEqual(stats["payload_bpp"], 320 / 579)
        self.assertAlmostEqual(stats["source_bit_ratio"], 320 / (579 * 24))
        self.assertAlmostEqual(stats["source_compression_factor"], (579 * 24) / 320)
        self.assertNotIn("transmission_ratio", stats)

    def test_ldpc_rate_changes_transmission_size(self):
        from communications.ldpc_coding import get_ldpc_code

        stats_by_rate = {}
        for rate in (0.5, 0.75):
            code = get_ldpc_code(int(256 * rate), rate, coded_block_length=256)
            ms_ssim, psnr, stats = evaluate_ldpc_channel(
                self.model, self.loader, [4, 16], 100, code, "cpu",
                modulation="qpsk", return_rate_stats=True,
            )
            self.assertAlmostEqual(ms_ssim, 1.0, places=5)
            self.assertEqual(psnr, 100.0)
            self.assertEqual(stats["payload_bits"], 160)
            stats_by_rate[rate] = stats
        half, three_quarters = stats_by_rate[0.5], stats_by_rate[0.75]
        self.assertEqual(half["ldpc_padding_bits"], 96)
        self.assertEqual(three_quarters["ldpc_padding_bits"], 32)
        self.assertEqual(half["coded_bits"], 512)
        self.assertEqual(three_quarters["coded_bits"], 256)
        self.assertAlmostEqual(half["transmission_ratio"], 256 / 768)
        self.assertAlmostEqual(three_quarters["transmission_ratio"], 128 / 768)
        self.assertAlmostEqual(half["source_bit_ratio"], three_quarters["source_bit_ratio"])

    def test_modulation_padding_is_counted_and_removed_before_decoding(self):
        from communications.ldpc_coding import get_ldpc_code

        for modulation, n, padding, symbols in (
            ("qpsk", 257, 1, 129),
            ("16qam", 258, 2, 65),
        ):
            with self.subTest(modulation=modulation):
                code = get_ldpc_code(int(n * 0.75), 0.75, coded_block_length=n)
                _, _, stats = evaluate_ldpc_channel(
                    self.model, self.loader, [4, 16], 100, code, "cpu",
                    modulation=modulation, return_rate_stats=True,
                )
                self.assertEqual(stats["coded_bits"], n)
                self.assertEqual(stats["modulation_padding_bits"], padding)
                self.assertEqual(stats["transmitted_bits"], n + padding)
                self.assertEqual(stats["channel_symbols"], symbols)
                self.assertAlmostEqual(stats["transmitted_bit_ratio"], (n + padding) / 6144)

    def test_existing_quality_return_and_no_quantization(self):
        self.assertEqual(len(evaluate_no_channel(self.model, self.loader, "cpu")), 2)
        self.model.quantizer_type = "none"
        _, _, stats = evaluate_no_channel(
            self.model, self.loader, "cpu", return_rate_stats=True,
        )
        self.assertIsNone(stats)

    def test_invalid_ldpc_parameters_fail_before_loading_checkpoint(self):
        for n, rate in ((0, 0.5), (256, 0), (256, 1), (256, -0.5), (1, 0.5)):
            with self.subTest(n=n, rate=rate), self.assertRaises(ValueError):
                run_eval(ldpc_n=n, ldpc_rate=rate)


if __name__ == "__main__":
    unittest.main()
