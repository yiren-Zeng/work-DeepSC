"""Original two-scale RAQVAE_TWO, source K256, power-of-two target K4..4096.

The only model-method override is target-K sampling. Core nets files are untouched.
Recovery checkpoints include optimizer, Lightning loops and main-process RNG state.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True

from train_seq2seq_cars import (
    ROOT, CarsRAQVAE, LocalCheckpointIO, RunStatus, load_shiyan_loader,
    local_path, pl, torch, write_json,
)

VARIANT = "RAQVAE_TWO_SRC256_K4_4096_POW2"
TARGET_KS = tuple(2 ** exponent for exponent in range(2, 13))
CORE_FILES = ("nets/model.py", "nets/enc_dec.py", "nets/quantizer.py", "nets/cbk2cbk.py")


class CarsRAQVAESource256(CarsRAQVAE):
    def __init__(self, args, smoke=False):
        super().__init__(args)
        self.smoke = smoke
        self.last_train_k = None
        self.smoke_val_batches = 0

    def sample_train_trg(self, num_min, num_max):
        choices = tuple(k for k in TARGET_KS if num_min <= k <= num_max)
        k = (4 if self.global_step % 2 == 0 else 4096) if self.smoke else random.choice(choices)
        self.last_train_k = k
        return torch.arange(k).unsqueeze(1)

    def sample_val_trg(self, num_min, num_max):
        if self.smoke:
            k = 4 if self.smoke_val_batches % 2 == 0 else 4096
            self.smoke_val_batches += 1
        else:
            k = random.choice(tuple(k for k in TARGET_KS if num_min <= k <= num_max))
        print(f"VAL_K {k}", flush=True)
        return torch.arange(k).unsqueeze(1)


class AtomicCheckpointIO(LocalCheckpointIO):
    """Keep the previous complete file until the replacement is durable."""

    def save_checkpoint(self, checkpoint, path, storage_options=None):
        if storage_options is not None:
            raise TypeError("Local atomic checkpoints do not use storage_options")
        destination = local_path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as handle:
                torch.save(checkpoint, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)


class RecoveryCheckpoint(pl.Callback):
    def __init__(self, directory, interval=50):
        self.directory = directory
        self.interval = interval
        self.last_saved_step = -1
        self.slot = 0

    def _save(self, trainer):
        step = int(trainer.global_step)
        if step == self.last_saved_step:
            return
        # Two independent atomic slots allow fallback if one file is damaged.
        path = self.directory / "checkpoints" / f"recovery_{self.slot}.ckpt"
        trainer.save_checkpoint(str(path), weights_only=False)
        write_json(self.directory / "recovery.json", {
            "checkpoint": str(path), "global_step": step,
            "epoch_index": trainer.current_epoch, "saved_at": time.time(),
        })
        self.last_saved_step = step
        self.slot = 1 - self.slot
        print(f"RECOVERY_SAVED step={step} path={path}", flush=True)

    def on_train_start(self, trainer, pl_module):
        existing = list((self.directory / "checkpoints").glob("recovery_*.ckpt"))
        if existing:
            # Overwrite the older slot first, preserving the newest complete file.
            self.slot = int(min(existing, key=lambda p: p.stat().st_mtime_ns).stem[-1])
            if len(existing) == 1:
                self.slot = 1 - int(existing[0].stem[-1])
        else:
            self._save(trainer)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.interval == 0:
            self._save(trainer)

    def on_train_epoch_end(self, trainer, pl_module):
        self._save(trainer)


class Source256RunStatus(RunStatus):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if trainer.global_step <= 5 or trainer.global_step % 10 == 0:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            print(
                f"PROGRESS epoch={trainer.current_epoch + 1} step={trainer.global_step} "
                f"K={pl_module.last_train_k} loss={float(loss):.6f} "
                f"peak_gpu_mib={torch.cuda.max_memory_allocated() / 2**20:.1f}",
                flush=True,
            )
            self.status(trainer, "running", target_k=pl_module.last_train_k)


def select_resume_checkpoint(directory):
    """Validate full local checkpoints, preferring greatest optimizer progress."""
    candidates = list((directory / "checkpoints").glob("*.ckpt"))
    valid = []
    for path in candidates:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            for key in ("state_dict", "optimizer_states", "loops", "raq_rng_state"):
                if key not in checkpoint:
                    raise ValueError(f"Missing {key}")
            valid.append((int(checkpoint["global_step"]), path.stat().st_mtime_ns, path))
            del checkpoint
        except Exception as error:
            print(f"RESUME_REJECTED {path}: {error!r}", flush=True)
    if candidates and not valid:
        raise RuntimeError("No valid full checkpoint remains; refusing to discard training progress")
    return str(max(valid)[2]) if valid else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shiyan-root", default="/workspace/yi/work/shiyan")
    parser.add_argument("--train-data", default="/workspace/yi/work/Cars196/train_data")
    parser.add_argument("--val-data", default="/workspace/yi/work/Cars196/val_data")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recovery-every-steps", type=int, default=50)
    parser.add_argument("--resume")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="K4/K4096 train and validation, full batch sizes")
    parser.add_argument("--max-steps", type=int, default=-1, help="Smoke-only interruption/recovery check")
    args = parser.parse_args()
    if args.max_steps != -1 and not args.smoke:
        parser.error("--max-steps is only supported for smoke checks")
    if min(args.epochs, args.batch_size, args.val_batch_size, args.recovery_every_steps) <= 0:
        parser.error("Epochs, batch sizes and recovery interval must be positive")
    if args.resume and args.auto_resume:
        parser.error("Choose --resume or --auto-resume")
    output = local_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    configuration_path = output / "config.json"
    previous = json.loads(configuration_path.read_text()) if configuration_path.is_file() else None
    if previous:
        if not (args.resume or args.auto_resume):
            raise FileExistsError("Existing run; use --auto-resume or a fresh --output")
        if previous.get("variant") != VARIANT:
            raise ValueError("Run belongs to another variant")
        for key in ("epochs", "batch_size", "val_batch_size", "num_workers", "seed",
                    "train_data", "val_data", "shiyan_root", "smoke", "recovery_every_steps"):
            if previous[key] != getattr(args, key):
                raise ValueError(f"Resume configuration differs: {key}")
    resume = args.resume or (select_resume_checkpoint(output) if args.auto_resume else None)
    if resume:
        resume = local_path(resume)
        if not resume.is_relative_to(output) or not resume.is_file():
            raise ValueError("Resume checkpoint must exist inside this run")
        print(f"RESUMING {resume}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; launch outside the restricted sandbox")
    os.environ.pop("SIMVQ_TRAIN_RESIZE", None)
    os.environ.pop("SIMVQ_VAL_RESIZE", None)
    get_dataloader, setup_seed = load_shiyan_loader(Path(args.shiyan_root))
    setup_seed(args.seed)
    torch.set_num_threads(8)
    train_loader = get_dataloader(
        args.train_data, args.batch_size, shuffle=True, mode="train",
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = get_dataloader(
        args.val_data, args.val_batch_size, shuffle=False, mode="val",
        num_workers=args.num_workers, pin_memory=True,
    )
    model_args = SimpleNamespace(
        n_hid=64, embedding_dim=64, num_embeddings=256,
        num_embeddings_min=4, num_embeddings_max=4096, num_embeddings_test=4,
        lr=5e-4, device=torch.device("cuda:0"), raq_type="dd", model_type="vqvae2",
    )
    model = CarsRAQVAESource256(model_args, smoke=args.smoke)
    hashed_paths = [ROOT / name for name in CORE_FILES] + [
        Path(__file__), Path(args.shiyan_root) / "data/datasets.py",
        Path(args.shiyan_root) / "utils/reproducibility.py",
    ]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in hashed_paths}
    if previous and hashes != previous["source_sha256"]:
        raise ValueError("Source files differ from the initial run; refusing a silent method change")
    configuration = {
        **vars(args), "variant": VARIANT, "output": str(output),
        "model_args": {k: str(v) if isinstance(v, torch.device) else v for k, v in vars(model_args).items()},
        "torch": torch.__version__, "pytorch_lightning": pl.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "train_images": len(train_loader.dataset), "val_images": len(val_loader.dataset),
        "parameters": sum(p.numel() for p in model.parameters()),
        "target_ks": list(TARGET_KS), "train_K": "uniform power of two per batch, shared top/bottom",
        "val_K": "uniform power of two per batch, shared top/bottom",
        "optimizer": "Original RAQVAE_TWO AdamW, lr=0.0005, defaults",
        "loss": "Original RAQVAE_TWO.training_step, unchanged",
        "spatial_scales": {"bottom": "1/4", "top": "1/8"},
        "precision": 32, "channel_training": False,
        "check_val_every_n_epoch": 1 if args.smoke else 2,
        "checkpoint_monitor": "val_recon_loss = MSE_src + MSE_trg",
        "source_sha256": hashes,
    }
    write_json(output / (f"resume_{time.time_ns()}.json" if previous else "config.json"), configuration)
    print("CONFIG " + json.dumps(configuration, ensure_ascii=False), flush=True)
    log_version = f"resume_{time.time_ns()}" if previous else ""
    loggers = [
        pl.loggers.TensorBoardLogger(str(output), name="tensorboard", version=log_version),
        pl.loggers.CSVLogger(str(output), name="csv", version=log_version),
    ]
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=output / "checkpoints", filename="epoch={epoch:03d}-step={step}",
        auto_insert_metric_name=False, monitor="val_recon_loss", mode="min",
        save_top_k=5, save_last=True, save_weights_only=False,
    )
    status = Source256RunStatus(output)
    recovery = RecoveryCheckpoint(output, interval=1 if args.smoke else args.recovery_every_steps)
    trainer = pl.Trainer(
        default_root_dir=str(output), accelerator="gpu", devices=[0],
        max_epochs=1 if args.smoke else args.epochs, max_steps=args.max_steps,
        logger=loggers, callbacks=[status, recovery, checkpoint],
        check_val_every_n_epoch=1 if args.smoke else 2, num_sanity_val_steps=0,
        log_every_n_steps=1 if args.smoke else 50, enable_progress_bar=False,
        deterministic=True, precision=32,
        limit_train_batches=2 if args.smoke else 1.0,
        limit_val_batches=2 if args.smoke else 1.0,
        plugins=[AtomicCheckpointIO()],
    )
    termination = {"signal": None}

    def request_stop(signum, frame):
        termination["signal"] = signum
        trainer.should_stop = True
        print(f"STOP_REQUESTED signal={signum}; finish current batch and save recovery", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=str(resume) if resume else None)
    complete = termination["signal"] is None and args.max_steps == -1
    status.status(trainer, "completed" if complete else "interrupted",
                  peak_gpu_mib=torch.cuda.max_memory_allocated() / 2**20)
    print(f"TRAINING_{'COMPLETE' if complete else 'INTERRUPTED'} best={checkpoint.best_model_path}", flush=True)


if __name__ == "__main__":
    main()
