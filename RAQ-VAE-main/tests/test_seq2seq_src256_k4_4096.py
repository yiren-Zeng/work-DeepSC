"""Checks for sampler isolation, checkpoint durability and automatic recovery."""
import contextlib
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_seq2seq_cars_src256_k4_4096 import (
    AtomicCheckpointIO, CarsRAQVAESource256, TARGET_KS, select_resume_checkpoint, torch,
)
from nets.model import RAQVAE_TWO


class Source256Tests(unittest.TestCase):
    def test_sampler_and_original_model_compatibility(self):
        args = SimpleNamespace(n_hid=64, embedding_dim=64, num_embeddings=256,
                               num_embeddings_min=4, num_embeddings_max=4096,
                               num_embeddings_test=4, lr=5e-4, device=torch.device("cpu"))
        model = CarsRAQVAESource256(args)
        original = RAQVAE_TWO(args)
        original.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(tuple(model.quantize.embed.weight.shape), (256, 64))
        self.assertEqual(tuple(model.embed_layer_dec.weight.shape), (4096, 64))
        self.assertEqual(model.encoder_b.blocks[0].stride, (4, 4))
        self.assertEqual(model.encoder_t.blocks[0].stride, (2, 2))
        self.assertIs(CarsRAQVAESource256.forward, RAQVAE_TWO.forward)
        self.assertIs(CarsRAQVAESource256.configure_optimizers, RAQVAE_TWO.configure_optimizers)
        random.seed(42)
        with contextlib.redirect_stdout(io.StringIO()):
            for sampler in (model.sample_train_trg, model.sample_val_trg):
                observed = {sampler(4, 4096).shape[0] for _ in range(300)}
                self.assertEqual(observed, set(TARGET_KS))

    def test_atomic_checkpoint_failure_preserves_previous(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "recovery_0.ckpt"
            checkpoint_io = AtomicCheckpointIO()
            with patch("train_seq2seq_cars_src256_k4_4096.local_path", side_effect=Path):
                checkpoint_io.save_checkpoint({"value": 123}, path)
                before = path.read_bytes()

                def interrupted_save(checkpoint, handle):
                    handle.write(b"incomplete")
                    raise OSError("simulated disk-write interruption")

                with patch.object(torch, "save", side_effect=interrupted_save):
                    with self.assertRaises(OSError):
                        checkpoint_io.save_checkpoint({"value": 456}, path)
                self.assertEqual(path.read_bytes(), before)
                self.assertFalse(list(directory.glob("*.tmp")))

    def test_resume_chooses_most_progress_and_skips_corrupt_slot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "checkpoints"
            directory.mkdir()
            checkpoint = dict(state_dict={}, optimizer_states=[], loops={}, raq_rng_state={})
            torch.save(dict(checkpoint, global_step=50), directory / "recovery_0.ckpt")
            torch.save(dict(checkpoint, global_step=100), directory / "recovery_1.ckpt")
            torch.save(dict(checkpoint, global_step=40), directory / "last.ckpt")
            self.assertTrue(select_resume_checkpoint(root).endswith("recovery_1.ckpt"))
            (directory / "recovery_1.ckpt").write_bytes(b"interrupted file")
            self.assertTrue(select_resume_checkpoint(root).endswith("recovery_0.ckpt"))

    def test_supervisor_restarts_failed_worker_and_respects_stop(self):
        supervisor_path = ROOT / "scripts/supervise_seq2seq_src256_k4_4096.py"
        for stop in (False, True):
            with self.subTest(stop=stop), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                directory = root / "run"
                directory.mkdir()
                worker = "\n".join([
                    "import json,sys",
                    "from pathlib import Path",
                    "p=Path(sys.argv[1])",
                    "counter=p/'attempts'",
                    "n=int(counter.read_text())+1 if counter.exists() else 1",
                    "counter.write_text(str(n))",
                    "(p/'status.json').write_text(json.dumps({'state':'completed' if n==2 else 'failed'}))",
                    "(p/'STOP').touch()" if stop else "pass",
                    "sys.exit(17 if n==1 else 0)",
                ])
                driver = "\n".join([
                    "import importlib.util,sys",
                    "from pathlib import Path",
                    f"spec=importlib.util.spec_from_file_location('supervisor',{str(supervisor_path)!r})",
                    "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)",
                    "sys.exit(m.supervise(Path(sys.argv[1]),[sys.executable,'-c',sys.argv[2],sys.argv[1]],"
                    "'test',poll_seconds=0.01,retry_seconds=0.01))",
                ])
                result = subprocess.run([sys.executable, "-B", "-c", driver, str(directory), worker],
                                        capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                state = json.loads((directory / "supervisor.json").read_text())
                self.assertEqual(state["state"], "stopped" if stop else "completed")
                self.assertEqual(int((directory / "attempts").read_text()), 1 if stop else 2)
                if not stop:
                    self.assertEqual(state["restarts"], 1)
                    self.assertIn("AUTO_RESUME", result.stdout)


if __name__ == "__main__":
    unittest.main()
