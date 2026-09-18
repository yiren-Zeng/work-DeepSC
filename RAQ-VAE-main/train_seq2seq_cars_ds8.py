"""Train the isolated 1/8 + 1/16 RAQVAE_TWO variant on Cars196."""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
from train_seq2seq_cars import (
    ROOT, LocalCheckpointIO, RunStatus, load_shiyan_loader, local_path, pl, torch, write_json,
)
from nets.model_ds8 import RAQVAE_TWO_DS8


class CarsRAQVAEDS8(RAQVAE_TWO_DS8):
    """Adapt tensor-only Cars196 batches while retaining original model steps."""

    def training_step(self, batch, batch_idx):
        return super().training_step((batch,), batch_idx)

    def validation_step(self, batch, batch_idx):
        return super().validation_step((batch,), batch_idx)

    def log(self, name, value, *args, **kwargs):
        if name.startswith("train_"):
            kwargs.setdefault("on_step", True)
            kwargs.setdefault("on_epoch", True)
        return super().log(name, value, *args, **kwargs)


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
        lr=5e-4, device=torch.device("cuda:0"), raq_type="dd", model_type="vqvae2_ds8",
    )
    model = CarsRAQVAEDS8(model_args)
    epochs = 1 if args.smoke else args.epochs
    source_paths = [
        ROOT / "nets/model.py", ROOT / "nets/model_ds8.py",
        ROOT / "nets/quantizer.py", ROOT / "nets/cbk2cbk.py",
        ROOT / "nets/enc_dec.py", ROOT / "nets/blocks.py",
        Path(args.shiyan_root) / "data/datasets.py",
        Path(args.shiyan_root) / "utils/reproducibility.py",
    ]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    configuration = {
        **vars(args), "output": str(output), "epochs": epochs,
        "variant": "RAQVAE_TWO_DS8", "training_from_scratch": True,
        "spatial_scales": {"bottom": "1/8", "top": "1/16"},
        "latent_shapes_at_256": {"bottom": [32, 32], "top": [16, 16]},
        "indices_per_256_image": 1280,
        "sampling_layers": {
            "encoder_bottom": "Conv2d(kernel=16,stride=8,padding=4)",
            "encoder_top": "original Conv2d(kernel=4,stride=2,padding=1)",
            "top_upsample": "original ConvTranspose2d(kernel=4,stride=2,padding=1)",
            "decoder": "ConvTranspose2d(kernel=16,stride=8,padding=4)",
        },
        "model_args": {key: str(value) if isinstance(value, torch.device) else value
                       for key, value in vars(model_args).items()},
        "torch": torch.__version__, "pytorch_lightning": pl.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "train_images": len(train_loader.dataset), "val_images": len(val_loader.dataset),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer": "Inherited RAQVAE_TWO.configure_optimizers: AdamW, lr=0.0005, defaults",
        "loss": "Inherited RAQVAE_TWO.training_step: unchanged reconstruction + quantization losses",
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
