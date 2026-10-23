"""Two-mode controls, mixed-radix wire labels, and original RVQ reconstruction."""

import argparse
import ast
import itertools
import io
import os
from contextlib import redirect_stderr
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from evaluation import quality
from models.deepsc import DeepSC
from utils.bit_utils import bits_to_index_tensor, index_tensor_to_bits
from utils.cscr import build_cscr_mapping
from utils.joint_cscr import (
    CSCR_MODES, build_joint_codebook, build_joint_cscr_mappings,
    pack_joint_indices, resolve_cscr_mode, unpack_joint_indices,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/eval/test_independent_raq_rvq_src64_64_d2_res6_6_mixture_per_k_combined.sh"
SCRIPTS = (
    SCRIPT,
    ROOT / "scripts/eval/test_independent_raq_rvq_src64_64_d2_res6_6_uniform_channel1_combined.sh",
)


class JointIndexTests(unittest.TestCase):
    def test_exhaustive_mixed_radix_roundtrip_and_original_shapes(self):
        for sizes in ([2, 4], [8, 2], [4], [2, 4, 2]):
            pairs = list(itertools.product(*(range(k) for k in sizes)))
            stages = [torch.tensor([pair[s] for pair in pairs]).reshape(2, -1)
                      for s in range(len(sizes))]
            before = [stage.clone() for stage in stages]
            joint = pack_joint_indices(stages, sizes)
            torch.testing.assert_close(joint.flatten(), torch.arange(len(pairs)))
            recovered = unpack_joint_indices(joint, sizes)
            for original, restored, snapshot in zip(stages, recovered, before):
                torch.testing.assert_close(original, restored, rtol=0, atol=0)
                torch.testing.assert_close(original, snapshot, rtol=0, atol=0)

    def test_joint_codebook_uses_sums_and_matches_index_enumeration(self):
        books = [torch.tensor([[2., 0.], [0., 3.]]),
                 torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])]
        snapshots = [book.clone() for book in books]
        rng = torch.random.get_rng_state().clone()
        expected = torch.stack([a + b for a in books[0] for b in books[1]])
        joint = build_joint_codebook(books, [2, 4])
        torch.testing.assert_close(joint, expected, rtol=0, atol=0)
        actual_map = build_joint_cscr_mappings([books], [[2, 4]])[0]
        expected_map = build_cscr_mapping(expected)
        for key in expected_map:
            torch.testing.assert_close(actual_map[key], expected_map[key])
        for book, snapshot in zip(books, snapshots):
            torch.testing.assert_close(book, snapshot, rtol=0, atol=0)
        torch.testing.assert_close(torch.random.get_rng_state(), rng)

    def test_invalid_layout_indices_and_joint_codebooks_are_rejected(self):
        for sizes in ([], [3], [0], [True], [2 ** 32, 2 ** 32]):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                pack_joint_indices([torch.tensor([0])], sizes)
        for stages in ([torch.tensor([2]), torch.tensor([0])],
                       [torch.tensor([0]), torch.tensor([[0]])],
                       [torch.tensor([0.]), torch.tensor([0])],
                       [torch.tensor([0])]):
            with self.assertRaises(ValueError):
                pack_joint_indices(stages, [2, 4])
        with self.assertRaises(ValueError):
            unpack_joint_indices(torch.tensor([8]), [2, 4])
        for books in ([torch.ones(2, 2), torch.ones(4, 3)],
                      [torch.ones(2, 2), torch.ones(2, 2)],
                      [torch.ones(2, 2), torch.full((4, 2), float("nan"))]):
            with self.assertRaises(ValueError):
                build_joint_codebook(books, [2, 4])
        with self.assertRaisesRegex(ValueError, "at most"):
            build_joint_codebook([torch.ones(128, 2)] * 2, [128, 128])
        with self.assertRaises(ValueError):
            build_joint_cscr_mappings(None, [[2, 4]])


class JointTransmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.manual_seed(321)
        cls.device = torch.device("cpu")
        cls.model = DeepSC(
            in_channels=3, out_channels=3, num_downsample_blocks=2,
            base_channels=4, num_embeddings_list=[8, 8], embedding_dim_list=[8, 16],
            commitment_cost=0.25, device=cls.device, strides=[2, 2],
            norm_type="group", norm_groups=1, activation="silu",
            encoder_res_blocks=1, decoder_res_blocks=1,
            raq_min_trg=2, raq_max_trg=8, independent_raq_rvq_depth=2,
            independent_raq_rvq_k_lists=[[2, 4], [8, 2]],
        ).eval().requires_grad_(False)
        cls.image = torch.linspace(-1, 1, 3 * 192 * 192).reshape(1, 3, 192, 192)
        cls.code = {"k": 128, "n": 256, "rate": 0.5}
        cls.fake_ldpc = SimpleNamespace(ldpc_encode=None, ldpc_decode=None)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def evaluate(self, mode=None, *, flip=False, packing="combined"):
        streams, reconstructions = [], []

        def transmit(bits, _snr, code, _device, _modulate, _llr, modulation_bits,
                     _encode, _decode, **kwargs):
            sent = np.asarray(bits, dtype=np.uint8).copy()
            received = sent.copy()
            if flip:
                received[::17] ^= 1
            streams.append((sent, received))
            lengths = quality._stream_length_reference(len(sent), code, modulation_bits)
            return received, {k: v for k, v in lengths.items() if k != "payload_bits"} | {
                "bit_errors": int(np.count_nonzero(sent != received)),
            }

        hook = self.model.semantic_decoder.register_forward_hook(
            lambda _module, _inputs, output: reconstructions.append(output.detach().clone())
        )
        try:
            with patch.dict(sys.modules, {"communications.ldpc_coding": self.fake_ldpc}):
                with patch("evaluation.quality._transmit_ldpc_stream", side_effect=transmit):
                    result = quality.evaluate_ldpc_channel(
                        self.model, [self.image], -10., self.code, self.device,
                        modulation="qpsk", stream_packing=packing,
                        return_diagnostics=True,
                        **({"cscr_mode": mode} if mode is not None else {}),
                    )
        finally:
            hook.remove()
        self.assertEqual(len(streams), 1 if packing == "combined" else 4)
        combined = tuple(np.concatenate([stream[i] for stream in streams]) for i in (0, 1))
        return result, combined, reconstructions[-1]

    def test_both_noiseless_modes_preserve_reconstruction_weights_and_physical_lengths(self):
        snapshots = {name: value.clone() for name, value in self.model.state_dict().items()}
        reference, _, reference_image = self.evaluate("off")
        lengths = ("payload_bits", "ldpc_input_bits", "ldpc_padding_bits", "coded_bits",
                   "modulation_padding_bits", "transmitted_bits", "channel_symbols",
                   "transmission_ratio")
        for mode in CSCR_MODES:
            with self.subTest(mode=mode):
                result, (sent, _), image = self.evaluate(mode)
                self.assertEqual(result[:2], reference[:2])
                torch.testing.assert_close(image, reference_image, rtol=0, atol=0)
                self.assertEqual(result[2]["rvq_quantization"], reference[2]["rvq_quantization"])
                for field in lengths:
                    self.assertEqual(result[2]["total"][field], reference[2]["total"][field])
                self.assertEqual(len(sent), reference[2]["total"]["payload_bits"])
                self.assertEqual(result[2]["total"]["index_errors"], 0)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, snapshots[name], rtol=0, atol=0)

    def test_default_off_matches_explicit_off_with_corruption_in_both_packings(self):
        for packing in ("combined", "per_stage"):
            with self.subTest(packing=packing):
                explicit, streams, image = self.evaluate("off", flip=True, packing=packing)
                default, default_streams, default_image = self.evaluate(flip=True, packing=packing)
                self.assertEqual(explicit, default)
                np.testing.assert_array_equal(streams[0], default_streams[0])
                np.testing.assert_array_equal(streams[1], default_streams[1])
                torch.testing.assert_close(image, default_image, rtol=0, atol=0)

    def test_joint_wire_order_and_corrupted_reconstruction_match_physical_joint_tables(self):
        with torch.no_grad():
            out = self.model.forward_test(self.image)
        expected_bits, physical_books = [], []
        for indices, books, sizes in zip(out["indices"], out["codebooks"], out["rvq_k_lists"]):
            # Independently enumerate the physical sum table and joint ids.
            joint_book = torch.stack([a + b for a in books[0] for b in books[1]])
            joint_ids = indices[0] * sizes[1] + indices[1]
            mapping = build_cscr_mapping(joint_book)
            physical = torch.empty_like(joint_book)
            gray = torch.tensor([i ^ (i >> 1) for i in range(len(joint_book))])
            physical[gray] = joint_book[mapping["greedy_order"]]
            wire_ids = mapping["index_to_label"][joint_ids]
            physical_books.append(physical)
            expected_bits.append(index_tensor_to_bits(wire_ids, len(joint_book))[0])
        result, (sent, received), actual = self.evaluate("joint-cscr", flip=True)
        np.testing.assert_array_equal(sent, np.concatenate(expected_bits))
        features, offset, wrong_tuples = [], 0, 0
        for scale, physical in enumerate(physical_books):
            shape = out["indices"][scale][0].shape
            count = int(np.prod(shape)) * (len(physical).bit_length() - 1)
            labels = bits_to_index_tensor(received[offset:offset + count], shape, len(physical))
            sent_labels = bits_to_index_tensor(sent[offset:offset + count], shape, len(physical))
            wrong_tuples += int(torch.count_nonzero(labels != sent_labels))
            offset += count
            features.append(self.model.vector_quantizers[scale].get_quantized_features(
                labels, output_spatial_size=out["feature_shapes"][scale], codebook_weight=physical,
            ))
        with torch.no_grad():
            expected_image = self.model.semantic_decoder(features)
        torch.testing.assert_close(actual, expected_image, rtol=0, atol=0)
        self.assertEqual(offset, len(received))
        self.assertEqual(result[2]["total"]["bit_errors"], int(np.count_nonzero(sent != received)))
        self.assertEqual(result[2]["total"]["index_errors"], wrong_tuples)
        self.assertEqual([s["stage"] for s in result[2]["per_stage"]], [None, None])
        self.assertEqual(result[2]["cscr"]["mode"], "joint-cscr")

    def test_off_skips_mapping_and_joint_no_channel_matches_original(self):
        with patch("evaluation.joint_cscr.build_joint_cscr_mappings",
                   side_effect=AssertionError("off must not build CSCR mappings")):
            reference = quality.evaluate_no_channel(self.model, [self.image], self.device)
            self.evaluate("off", flip=True)
        result = quality.evaluate_no_channel(
            self.model, [self.image], self.device, return_diagnostics=True, cscr_mode="joint-cscr",
        )
        self.assertEqual(result[:2], reference)
        self.assertEqual(result[2]["joint_packing"]["joint_k_per_scale"], [8, 16])
        self.assertTrue(result[2]["cscr"]["enabled"])

    def test_joint_modes_reject_per_stage_streams(self):
        with patch.dict(sys.modules, {"communications.ldpc_coding": self.fake_ldpc}):
            with self.assertRaisesRegex(ValueError, "combined"):
                quality.evaluate_ldpc_channel(
                    self.model, [self.image], -10., self.code, self.device,
                    cscr_mode="joint-cscr", stream_packing="per_stage",
                )


class ModeSelectionTests(unittest.TestCase):
    @staticmethod
    def cli_main():
        return next(node for node in ast.parse((ROOT / "test_real.py").read_text()).body
                    if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'")

    def test_mode_resolution_and_cli_forwarding(self):
        self.assertEqual(CSCR_MODES, ("off", "joint-cscr"))
        self.assertEqual(resolve_cscr_mode(), "off")
        for mode in CSCR_MODES:
            self.assertEqual(resolve_cscr_mode(mode), mode)
        for invalid in ("stage", "joint-pack", "all", "", None, True, False):
            with self.subTest(mode=invalid), self.assertRaises(ValueError):
                resolve_cscr_mode(invalid)
        main = self.cli_main()
        for mode in (None, *CSCR_MODES):
            evaluate = Mock()
            args = ["test_real.py", "--stream-packing", "combined"]
            if mode is not None:
                args += ["--cscr-mode", mode]
            with patch.object(sys, "argv", args):
                exec(compile(ast.Module(body=main.body, type_ignores=[]), "test_real.py", "exec"), {
                    "argparse": argparse, "CSCR_MODES": CSCR_MODES, "test_real": evaluate,
                })
            evaluate.assert_called_once()
            self.assertEqual(evaluate.call_args.kwargs, {"cscr_mode": mode or "off"})

    def test_cli_rejects_retired_modes_and_boolean_flags(self):
        main = self.cli_main()
        for args in (["--cscr-mode", "stage"], ["--cscr-mode", "joint-pack"],
                     ["--cscr"], ["--no-cscr"], ["--cscr", "joint-cscr"]):
            evaluate = Mock()
            with self.subTest(args=args), redirect_stderr(io.StringIO()):
                with patch.object(sys, "argv", ["test_real.py", *args]):
                    with self.assertRaises(SystemExit) as raised:
                        exec(compile(ast.Module(body=main.body, type_ignores=[]), "test_real.py", "exec"), {
                            "argparse": argparse, "CSCR_MODES": CSCR_MODES, "test_real": evaluate,
                        })
                self.assertEqual(raised.exception.code, 2)
            evaluate.assert_not_called()

    def test_both_evaluators_reject_retired_modes(self):
        fake_ldpc = SimpleNamespace(ldpc_encode=None, ldpc_decode=None)
        with patch.dict(sys.modules, {"communications.ldpc_coding": fake_ldpc}):
            for mode in ("stage", "joint-pack"):
                with self.subTest(mode=mode):
                    with self.assertRaises(ValueError):
                        quality.evaluate_no_channel(None, [], "cpu", cscr_mode=mode)
                    with self.assertRaises(ValueError):
                        quality.evaluate_ldpc_channel(None, [], 12, None, "cpu", cscr_mode=mode)

    def test_shell_selects_one_mode_and_defaults_to_joint_cscr(self):
        wrapper = 'python() { printf "ARG=%s\\n" "$@"; }; export -f python; bash "$1"'
        for script in SCRIPTS:
            for mode in (None, "", *CSCR_MODES):
                with self.subTest(script=script.name, mode=mode):
                    env = dict(os.environ)
                    env.pop("CSCR_MODE", None)
                    if mode is not None:
                        env["CSCR_MODE"] = mode
                    completed = subprocess.run(
                        ["bash", "--noprofile", "--norc", "-c", wrapper, "--", str(script)],
                        env=env, capture_output=True, text=True, timeout=20, check=True,
                    )
                    args = [line[4:] for line in completed.stdout.splitlines() if line.startswith("ARG=")]
                    self.assertEqual(args.count("test_real.py"), 1)
                    selected = args[args.index("--concise") + 1:]
                    self.assertEqual(selected, ["--cscr-mode", mode or "joint-cscr"])
            for mode in ("stage", "joint-pack", "all"):
                with self.subTest(script=script.name, invalid=mode):
                    completed = subprocess.run(
                        ["bash", "--noprofile", "--norc", "-c", wrapper, "--", str(script)],
                        env=dict(os.environ, CSCR_MODE=mode),
                        capture_output=True, text=True, timeout=20,
                    )
                    self.assertEqual(completed.returncode, 2)
                    self.assertNotIn("ARG=test_real.py", completed.stdout)


if __name__ == "__main__":
    unittest.main()
