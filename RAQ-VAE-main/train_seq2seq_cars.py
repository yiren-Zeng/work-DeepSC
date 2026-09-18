"""Train the unmodified RAQVAE_TWO on the read-only shiyan image loader.

All run artifacts and extra dependencies live under this project. Model forward,
loss, codebook updates, K sampling and optimizer come directly from nets/model.py.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
for key, value in {
    "TMPDIR": ROOT / ".runtime/tmp",
    "XDG_CACHE_HOME": ROOT / ".runtime/cache",
    "MPLCONFIGDIR": ROOT / ".runtime/cache/matplotlib",
    "TORCH_HOME": ROOT / ".runtime/cache/torch",
}.items():
    os.environ[key] = str(value)
    Path(value).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(ROOT / ".runtime/deps"))
sys.path.insert(0, str(ROOT))

import argparse
import hashlib
import importlib.util
import json
import random
import time
from types import SimpleNamespace

import numpy as np
import torch

# Lightning 1.9.5 (the repository version) uses the NumPy 1.x spelling.
# This process-local compatibility alias has no effect on model arithmetic.
if not hasattr(np, "Inf"):
    np.Inf = np.inf
import pytorch_lightning as pl

from nets.model import RAQVAE_TWO


def load_file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_shiyan_loader(source):
    # shiyan's loader imports utils.reproducibility; keep RAQ's utils namespace
    # while loading exactly that one read-only module from the other project.
    reproducibility = load_file_module(
        "utils.reproducibility", source / "utils/reproducibility.py"
    )
    dataset_module = load_file_module(
        "raq_external_image_datasets", source / "data/datasets.py"
    )
    return dataset_module.get_dataloader, reproducibility.setup_seed


def local_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Output must be inside {ROOT}: {path}")
    return path


def write_json(path, data):
    path = local_path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


class CarsRAQVAE(RAQVAE_TWO):
    """Adapt tensor batches to the original (image, label) step interface."""

    def training_step(self, batch, batch_idx):
        return super().training_step((batch,), batch_idx)

    def validation_step(self, batch, batch_idx):
        return super().validation_step((batch,), batch_idx)

    def log(self, name, value, *args, **kwargs):
        if name.startswith("train_"):
            # Add epoch aggregates for monitoring; do not change the objective.
            kwargs.setdefault("on_step", True)
            kwargs.setdefault("on_epoch", True)
        return super().log(name, value, *args, **kwargs)


class RunStatus(pl.Callback):
    def __init__(self, directory):
        self.directory = directory
        self.started = time.time()
        self.epoch_started = self.started
        self.gradient_checked = False
        self.resume_rng = None

    def status(self, trainer, state, **extra):
        metrics = {
            name: float(value.detach().cpu()) if torch.is_tensor(value) else value
            for name, value in trainer.callback_metrics.items()
        }
        checkpoint = trainer.checkpoint_callback
        write_json(self.directory / "status.json", {
            "state": state,
            "pid": os.getpid(),
            "epoch_index": trainer.current_epoch,
            "global_step": trainer.global_step,
            "elapsed_seconds": time.time() - self.started,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "best_model_path": checkpoint.best_model_path if checkpoint else "",
            "metrics": metrics,
            **extra,
        })

    def on_fit_start(self, trainer, pl_module):
        self.status(trainer, "running")

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_started = time.time()
        print(f"EPOCH_START {trainer.current_epoch + 1}/{trainer.max_epochs}", flush=True)

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        self.resume_rng = checkpoint.get("raq_rng_state")

    def on_train_start(self, trainer, pl_module):
        if self.resume_rng is not None:
            state = self.resume_rng
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch"].cpu())
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
            if state.get("train_loader") is not None:
                trainer.train_dataloader.generator.set_state(state["train_loader"].cpu())
            # Persistent worker augmentation RNG is not serialized by DataLoader.
            print("RESUMED model/optimizer/epoch and main-process RNG; "
                  "worker augmentation sequence may differ from uninterrupted training", flush=True)

    def on_after_backward(self, trainer, pl_module):
        if self.gradient_checked:
            return
        norms = {}
        for name in ("encoder_b", "encoder_t", "cbk_enc", "cbk_dec", "decoder"):
            grads = [p.grad for p in getattr(pl_module, name).parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all().item() for g in grads):
                raise RuntimeError(f"Missing or nonfinite gradients: {name}")
            norms[name] = sum(float(g.detach().float().norm()) for g in grads)
            if norms[name] == 0:
                raise RuntimeError(f"Zero gradient in first training step: {name}")
        write_json(self.directory / "first_step_gradients.json", norms)
        print("FIRST_STEP_GRADIENTS " + json.dumps(norms), flush=True)
        self.gradient_checked = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if not torch.isfinite(loss).all():
            raise RuntimeError("Nonfinite training loss")
        if batch_idx % 50 == 0:
            print(f"TRAIN epoch={trainer.current_epoch + 1} batch={batch_idx} "
                  f"step={trainer.global_step} loss={float(loss):.6f}", flush=True)
            self.status(trainer, "running")

    def on_train_epoch_end(self, trainer, pl_module):
        duration = time.time() - self.epoch_started
        print(f"EPOCH_END {trainer.current_epoch + 1} seconds={duration:.1f}", flush=True)
        self.status(trainer, "running", last_epoch_seconds=duration)

    def on_validation_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            print("VALIDATION " + json.dumps({
                key: float(value) for key, value in trainer.callback_metrics.items()
                if key.startswith("val_")
            }), flush=True)
            self.status(trainer, "running")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Preserve random K sampling and augmentation generators for continuation.
        checkpoint["raq_rng_state"] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "train_loader": trainer.train_dataloader.generator.get_state()
            if getattr(trainer.train_dataloader, "generator", None) is not None else None,
        }

    def on_exception(self, trainer, pl_module, exception):
        self.status(trainer, "failed", error=repr(exception))


class LocalCheckpointIO(pl.plugins.io.TorchCheckpointIO):
    def load_checkpoint(self, path, map_location=None):
        # Only load self-produced checkpoints under this project. Full checkpoints
        # include optimizer and RNG state, beyond torch 2.6+'s weights-only loader.
        return torch.load(local_path(path), map_location=map_location or "cpu", weights_only=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--shiyan-root", default="/workspace/yi/work/shiyan")
    parser.add_argument("--train-data", default="/workspace/yi/work/Cars196/train_data")
    parser.add_argument("--val-data", default="/workspace/yi/work/Cars196/val_data")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", help="Resume a trusted checkpoint inside this output directory")
    args = parser.parse_args()
    output = local_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        resume_path = local_path(args.resume)
        if not resume_path.is_relative_to(output) or not resume_path.is_file():
            raise ValueError("Resume checkpoint must exist inside this run's output directory")
        previous = json.loads((output / "config.json").read_text())
        for key in ("batch_size", "val_batch_size", "num_workers", "seed", "train_data", "val_data"):
            if vars(args)[key] != previous[key]:
                raise ValueError(f"Resume configuration differs: {key}")
    elif (output / "config.json").exists():
        raise FileExistsError(f"Run already exists: {output}; use a fresh output directory")

    # Eliminate inherited resize overrides: match the reference script defaults.
    os.environ.pop("SIMVQ_TRAIN_RESIZE", None)
    os.environ.pop("SIMVQ_VAL_RESIZE", None)
    get_dataloader, setup_seed = load_shiyan_loader(Path(args.shiyan_root))
    setup_seed(args.seed)
    torch.set_num_threads(8)
    train_loader = get_dataloader(args.train_data, args.batch_size, shuffle=True,
                                  mode="train", num_workers=args.num_workers, pin_memory=True)
    val_loader = get_dataloader(args.val_data, args.val_batch_size, shuffle=False,
                                mode="val", num_workers=args.num_workers, pin_memory=True)
    model_args = SimpleNamespace(
        n_hid=64, embedding_dim=64, num_embeddings=64,
        num_embeddings_min=2, num_embeddings_max=64, num_embeddings_test=2,
        lr=5e-4, device=torch.device("cuda:0"), raq_type="dd", model_type="vqvae2",
    )
    model = CarsRAQVAE(model_args)
    epochs = 1 if args.smoke else args.epochs
    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [ROOT / "nets/model.py", ROOT / "nets/quantizer.py",
                     ROOT / "nets/cbk2cbk.py", ROOT / "nets/enc_dec.py",
                     Path(args.shiyan_root) / "data/datasets.py",
                     Path(args.shiyan_root) / "utils/reproducibility.py"]
    }
    configuration = {
        **vars(args), "output": str(output), "epochs": epochs,
        "model_args": {key: str(value) if isinstance(value, torch.device) else value
                       for key, value in vars(model_args).items()},
        "torch": torch.__version__, "pytorch_lightning": pl.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "train_images": len(train_loader.dataset), "val_images": len(val_loader.dataset),
        "parameters": sum(p.numel() for p in model.parameters()),
        "optimizer": "Original RAQVAE_TWO.configure_optimizers: AdamW, lr=0.0005, defaults",
        "loss": "Original RAQVAE_TWO.training_step; no extra losses or changed gradients",
        "train_K": "Uniform integer 2..64 per batch, shared across top/bottom",
        "val_K": "Uniform exponent: K in [2,4,8,16,32,64], per validation batch",
        "checkpoint_monitor": "val_recon_loss = MSE_src + MSE_trg",
        "check_val_every_n_epoch": 1 if args.smoke else 2,
        "precision": 32, "channel_training": False, "source_sha256": hashes,
    }
    if args.resume:
        write_json(output / f"resume_{time.time_ns()}.json", configuration)
    else:
        write_json(output / "config.json", configuration)
    print("CONFIG " + json.dumps(configuration, ensure_ascii=False), flush=True)
    # A resumed run gets new log segments, preserving previous CSV history.
    log_version = f"resume_{time.time_ns()}" if args.resume else ""
    loggers = [
        pl.loggers.TensorBoardLogger(str(output), name="tensorboard", version=log_version),
        pl.loggers.CSVLogger(str(output), name="csv", version=log_version),
    ]
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=output / "checkpoints", filename="epoch={epoch:03d}-step={step}",
        auto_insert_metric_name=False, monitor="val_recon_loss", mode="min",
        save_top_k=5, save_last=True, save_weights_only=False,
    )
    status = RunStatus(output)
    trainer = pl.Trainer(
        default_root_dir=str(output), accelerator="gpu", devices=[0],
        max_epochs=epochs, logger=loggers, callbacks=[checkpoint, status],
        check_val_every_n_epoch=1 if args.smoke else 2,
        num_sanity_val_steps=0, log_every_n_steps=1 if args.smoke else 50,
        enable_progress_bar=False, deterministic=True, precision=32,
        limit_train_batches=4 if args.smoke else 1.0,
        limit_val_batches=2 if args.smoke else 1.0,
        plugins=[LocalCheckpointIO()],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=args.resume)
    status.status(trainer, "completed")
    print(f"TRAINING_COMPLETE best={checkpoint.best_model_path}", flush=True)


if __name__ == "__main__":
    main()
