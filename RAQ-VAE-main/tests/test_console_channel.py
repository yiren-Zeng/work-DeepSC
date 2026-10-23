"""Check terminal-only evaluation, seed output, averaging, and error visibility."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval_seq2seq_channel as evaluator
from aggregate_seq2seq_channel_trials import _compatibility_signature
from eval_seq2seq_ds8_dualk_console import main as console_main, quiet_evaluation, run_trials


def report(psnr=20.0, ms_ssim=0.8):
    return {
        "metadata": {
            "dual_k": True, "target_K_bottom": 32, "target_K_top": 16,
            "ldpc": {"rate": 0.625}, "modulation": "qpsk", "channel": "awgn",
            "rician_k_factor": 10, "snr_db": 4.5, "cscr": {"enabled": True},
            "checkpoint_sha256": "weights", "dataset_sha256": {"image": "hash"},
            "rate_plan": {},
        },
        "metrics": {"psnr": psnr, "ms_ssim": ms_ssim},
        "transmission_ratio": 0.025390625, "num_images": 24,
    }


def fake_evaluator(reports):
    return SimpleNamespace(
        main=mock.Mock(side_effect=reports),
        channel_label=evaluator.channel_label,
        format_ms_ssim=evaluator.format_ms_ssim,
    )


class ConsoleChannelTests(unittest.TestCase):
    def test_channel_argument_is_forwarded_without_becoming_channel_seeds(self):
        backend = fake_evaluator([report(), report(22.0, 0.9)])
        entry = SimpleNamespace(evaluator=backend)
        with mock.patch.dict(sys.modules, {"eval_seq2seq_ds8_dualk_channel": entry}):
            with contextlib.redirect_stdout(io.StringIO()):
                console_main([
                    "--channel", "awgn", "--snr", "4.5", "--channel-seeds", "42", "43",
                ])
        for call, seed in zip(backend.main.call_args_list, (42, 43)):
            self.assertEqual(call.args[0], [
                "--channel", "awgn", "--snr", "4.5", "--console-only", "--seed", str(seed),
            ])

    def test_seeds_print_metrics_then_mean_raw_ms_ssim_with_one_configuration(self):
        responses = iter((report(), report(22.0, 0.9)))

        def evaluate(_argv):
            print("dependency stdout")
            os.write(2, b"native dependency stderr\n")
            return next(responses)

        backend = fake_evaluator([])
        backend.main.side_effect = evaluate
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            run_trials(backend, ["--cscr"], [42, 43], _compatibility_signature)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 5)
        self.assertEqual(sum(line.startswith("配置：") for line in lines), 1)
        self.assertEqual(sum(line.startswith("实际传输比：") for line in lines), 1)
        self.assertTrue(lines[2].startswith("seed=42 | PSNR：20.0000 dB | MS-SSIM："))
        self.assertTrue(lines[3].startswith("seed=43 | PSNR：22.0000 dB | MS-SSIM："))
        self.assertEqual(lines[4], "平均 | PSNR：21.0000 dB | MS-SSIM：0.850000 (8.2391 dB)")
        self.assertNotIn("dependency", output.getvalue())
        for call, seed in zip(backend.main.call_args_list, (42, 43)):
            self.assertEqual(call.args[0], ["--cscr", "--console-only", "--seed", str(seed)])

    def test_failure_restores_stderr_and_replays_python_and_native_diagnostics(self):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                with quiet_evaluation():
                    print("python diagnostic")
                    os.write(2, b"native diagnostic\n")
                    raise RuntimeError("test failure")
        self.assertIn("python diagnostic", errors.getvalue())
        self.assertIn("native diagnostic", errors.getvalue())

    def test_incompatible_trials_do_not_produce_a_misleading_average(self):
        changed = report(22.0, 0.9)
        changed["metadata"]["snr_db"] = 6.0
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaisesRegex(ValueError, "Incompatible trial"):
                run_trials(fake_evaluator([report(), changed]), [], [42, 43],
                           _compatibility_signature)
        self.assertNotIn("平均", output.getvalue())

    def test_console_only_loads_original_checkpoint_without_creating_output(self):
        class CheckpointRead(Exception):
            pass

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "original.ckpt"
            checkpoint.write_bytes(b"checkpoint fixture")
            (root / "config.json").write_text(json.dumps({
                "model_args": {"num_embeddings_min": 2, "num_embeddings_max": 64},
            }))
            (root / "status.json").write_text(json.dumps({
                "best_model_path": str(checkpoint),
            }))
            output = root / "must_not_be_created"
            before = sorted(root.iterdir())
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(evaluator, "DUAL_K", True))
                stack.enter_context(mock.patch.object(evaluator, "local_path", side_effect=Path))
                writers = [stack.enter_context(mock.patch.object(owner, name))
                           for owner, name in (
                               (evaluator.shutil, "copy2"), (evaluator.torch, "save"),
                               (evaluator, "write_json"), (evaluator, "save_image"),
                           )]
                stack.enter_context(mock.patch.object(Path, "mkdir", side_effect=AssertionError("mkdir")))
                load = stack.enter_context(mock.patch.object(
                    evaluator.torch, "load", side_effect=CheckpointRead,
                ))
                with self.assertRaises(CheckpointRead):
                    evaluator.main([
                        "--run-dir", str(root), "--bottom-k", "32", "--top-k", "16",
                        "--output", str(output), "--console-only",
                    ])
                load.assert_called_once_with(checkpoint, map_location="cpu", weights_only=False)
                for writer in writers:
                    writer.assert_not_called()
            self.assertEqual(sorted(root.iterdir()), before)
            self.assertFalse(output.exists())

    def test_legacy_summary_keeps_its_path_while_console_summary_has_only_four_fields(self):
        example = report()
        legacy = evaluator.format_console_result(example, Path("results.json"))
        self.assertIn("测试完成：24 张图片", legacy)
        self.assertIn("结果文件：results.json", legacy)
        concise = evaluator.format_console_result(example, None)
        self.assertEqual(len(concise.strip().splitlines()), 4)
        self.assertNotIn("结果文件", concise)
        self.assertNotIn("测试完成", concise)


if __name__ == "__main__":
    unittest.main()
