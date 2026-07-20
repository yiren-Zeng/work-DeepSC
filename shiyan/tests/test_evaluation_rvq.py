import json
import sys
import types
import unittest
from unittest import mock

import numpy as np
import torch

from evaluation import quality


def _fake_ldpc_module(call_lengths):
    module = types.ModuleType("communications.ldpc_coding")

    def encode(bits, code=None):
        bits = np.asarray(bits, dtype=np.uint8)
        call_lengths.append(len(bits))
        k = code["k"]
        padded = np.pad(bits, (0, (-len(bits)) % k), "constant")
        return np.repeat(padded, 2)

    def decode(llrs, code=None):
        hard = (np.asarray(llrs) > 0).astype(np.uint8)
        return (hard.reshape(-1, 2).mean(axis=1) >= 0.5).astype(np.uint8)

    module.ldpc_encode = encode
    module.ldpc_decode = decode
    return module


class _RvqModel:
    def __init__(self):
        self.received = None
        self._real_image = None

    def eval(self):
        return self

    def forward_test(self, real_image):
        self._real_image = real_image
        indices = [
            [torch.tensor([[[5]]]), torch.tensor([[[3]]])],
            [torch.tensor([[[7]]]), torch.tensor([[[2]]])],
        ]
        scale_diagnostics = []
        for scale, stage_indices in enumerate(indices):
            input_energy = float(4 * (scale + 1))
            residuals = [input_energy / 2, input_energy / 4]
            stages = []
            for stage, (tensor, k, residual) in enumerate(
                zip(stage_indices, [64, 32], residuals)
            ):
                stages.append({
                    "stage_index": stage,
                    "k": k,
                    "bits_per_index": int(np.log2(k)),
                    "index_min": int(tensor.min()),
                    "index_max": int(tensor.max()),
                    "codebook_size": k,
                    "payload_bits": int(tensor.numel() * np.log2(k)),
                    "residual_mse_energy": residual,
                })
            scale_diagnostics.append({
                "scale_index": scale,
                "source_route": "src",
                "k_total": 2048,
                "stage_k_list": [64, 32],
                "input_mse_energy": input_energy,
                "residual_mse_energies": residuals,
                "final_residual_mse_energy": residuals[-1],
                "quantized_sum_mse_energy": input_energy - residuals[-1],
                "stage_diagnostics": stages,
                "payload_bits": 11,
                "baseline_payload_bits": 11,
                "bit_budget_matches": True,
            })
        return {
            "indices": indices,
            "feature_shapes": [(1, 1), (1, 1)],
            "codebooks": [
                [torch.zeros(64, 1), torch.zeros(32, 1)],
                [torch.zeros(64, 1), torch.zeros(32, 1)],
            ],
            "num_embeddings_list": [[64, 32], [64, 32]],
            "rvq_k_lists": [[64, 32], [64, 32]],
            "test_raq_rvq_enabled": True,
            "rvq_diagnostics": scale_diagnostics,
        }

    def reconstruct_from_indices(self, indices, feature_shapes=None, codebooks=None):
        self.received = indices
        return self._real_image.clone()


class _FlatModel:
    def __init__(self):
        self._real_image = None
        self.received = None

    def eval(self):
        return self

    def forward_test(self, real_image):
        self._real_image = real_image
        return {
            "indices": [torch.tensor([[[5]]]), torch.tensor([[[3]]])],
            "feature_shapes": [(1, 1), (1, 1)],
            "num_embeddings_list": [64, 32],
        }

    def reconstruct_from_indices(self, indices, feature_shapes=None, codebooks=None):
        self.received = indices
        return self._real_image.clone()


class RvqEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.loader = [torch.zeros(1, 3, 2, 2)]
        self.code = {"k": 5, "n": 10, "rate": 0.5}

    def test_no_channel_preserves_nested_layout_and_optional_return(self):
        model = _RvqModel()
        with mock.patch.object(quality, "_image_quality", return_value=(1.0, 100.0)):
            legacy_result = quality.evaluate_no_channel(model, self.loader, "cpu")
            result = quality.evaluate_no_channel(
                model, self.loader, "cpu", return_diagnostics=True
            )

        self.assertEqual(len(legacy_result), 2)
        self.assertEqual(len(result), 3)
        diagnostics = result[2]
        self.assertTrue(diagnostics["rvq_enabled"])
        self.assertEqual(diagnostics["total"]["payload_bits"], 22)
        self.assertEqual(len(diagnostics["per_stage"]), 4)
        self.assertTrue(
            diagnostics["rvq_quantization"]["all_bit_budgets_match"]
        )
        self.assertEqual(
            diagnostics["rvq_quantization"]["per_scale"][0][
                "mean_residual_mse_energies"
            ],
            [2.0, 1.0],
        )
        self.assertIsInstance(model.received[0], list)
        json.dumps(diagnostics)

    def test_each_stage_has_independent_ldpc_and_modulation_padding(self):
        model = _RvqModel()
        call_lengths = []
        fake_ldpc = _fake_ldpc_module(call_lengths)
        with mock.patch.dict(
            sys.modules, {"communications.ldpc_coding": fake_ldpc}
        ), mock.patch.object(
            quality, "awgn_channel", side_effect=lambda symbols, snr: symbols
        ), mock.patch.object(
            quality, "_image_quality", return_value=(1.0, 100.0)
        ):
            ms_ssim, psnr, diagnostics = quality.evaluate_ldpc_channel(
                model,
                self.loader,
                [2048, 2048],
                100,
                self.code,
                "cpu",
                modulation="16qam",
                return_diagnostics=True,
            )

        self.assertEqual((ms_ssim, psnr), (1.0, 100.0))
        self.assertEqual(call_lengths, [6, 5, 6, 5])
        self.assertIsInstance(model.received[0], list)
        for scale in range(2):
            self.assertEqual(int(model.received[scale][0]), [5, 7][scale])
            self.assertEqual(int(model.received[scale][1]), [3, 2][scale])

        total = diagnostics["total"]
        self.assertEqual(total["payload_bits"], 22)
        self.assertEqual(total["ldpc_input_bits"], 30)
        self.assertEqual(total["ldpc_padding_bits"], 8)
        self.assertEqual(total["coded_bits"], 60)
        self.assertEqual(total["modulation_padding_bits"], 4)
        self.assertEqual(total["transmitted_bits"], 64)
        self.assertEqual(total["channel_symbols"], 16)
        self.assertEqual(total["bit_errors"], 0)
        self.assertEqual(total["index_errors"], 0)
        self.assertEqual(total["single_stream_coded_bits"], 50)
        self.assertEqual(total["single_stream_transmitted_bits"], 52)
        self.assertFalse(total["coded_bits_match_single_stream"])
        self.assertFalse(total["transmitted_bits_match_single_stream"])
        self.assertTrue(total["payload_bits_match_single_stage_budget"])
        json.dumps(diagnostics)

    def test_flat_path_keeps_one_ldpc_stream_and_two_value_default(self):
        model = _FlatModel()
        call_lengths = []
        fake_ldpc = _fake_ldpc_module(call_lengths)
        with mock.patch.dict(
            sys.modules, {"communications.ldpc_coding": fake_ldpc}
        ), mock.patch.object(
            quality, "awgn_channel", side_effect=lambda symbols, snr: symbols
        ), mock.patch.object(
            quality, "_image_quality", return_value=(1.0, 100.0)
        ):
            result = quality.evaluate_ldpc_channel(
                model, self.loader, [64, 32], 100, self.code, "cpu"
            )

        self.assertEqual(len(result), 2)
        self.assertEqual(call_lengths, [11])
        self.assertIsInstance(model.received[0], torch.Tensor)


if __name__ == "__main__":
    unittest.main()
