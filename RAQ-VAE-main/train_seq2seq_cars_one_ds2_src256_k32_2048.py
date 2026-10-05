"""Train the x2 single-scale RAQVAE_ONE comparison baseline on Cars196."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True

from train_seq2seq_cars import (
    ROOT,
    LocalCheckpointIO,
    RunStatus,
    load_shiyan_loader,
    local_path,
    pl,
    torch,
    write_json,
)
from nets.model_one_ds2_src256_k32_2048 import (
    RAQVAE_ONE_DS2_SRC256_K32_2048,
)


class CarsRAQVAEOneDS2(RAQVAE_ONE_DS2_SRC256_K32_2048):
    """Adapt tensor-only Cars196 batches without changing the model objective."""

    def training_step(self, batch, batch_idx):
        return super().training_step((batch,), batch_idx)

    def validation_step(self, batch, batch_idx):
        # Preserve the original validation arithmetic while omitting image-grid
        # logging. This affects artifacts only, not training or model selection.
        target = self.sample_val_trg(self.n_embed_min, self.n_embed_max)
        result = self.forward(batch, target)
        recon_loss_src = self.criterion(result[0], batch)
        recon_loss_trg = self.criterion(result[3], batch)
        batch_size = int(batch.shape[0])
        self.log(
            "val_recon_loss",
            recon_loss_src + recon_loss_trg,
            batch_size=batch_size,
        )
        self.log(
            "val_recon_loss(src)", recon_loss_src, batch_size=batch_size
        )
        self.log(
            "val_recon_loss(trg)", recon_loss_trg, batch_size=batch_size
        )

    def log(self, name, value, *args, **kwargs):
        if name.startswith("train_"):
            kwargs.setdefault("on_step", True)
            kwargs.setdefault("on_epoch", True)
        return super().log(name, value, *args, **kwargs)


class OneScaleRunStatus(RunStatus):
    """RunStatus gradient audit for RAQVAE_ONE module names."""

    def on_after_backward(self, trainer, pl_module):
        if self.gradient_checked:
            return
        norms = {}
        for name in ("encoder", "cbk_enc", "cbk_dec", "decoder"):
            gradients = [
                parameter.grad
                for parameter in getattr(pl_module, name).parameters()
                if parameter.grad is not None
            ]
            if not gradients or not all(
                torch.isfinite(gradient).all().item() for gradient in gradients
            ):
                raise RuntimeError(f"Missing or nonfinite gradients: {name}")
            norms[name] = sum(
                float(gradient.detach().float().norm()) for gradient in gradients
            )
            if norms[name] == 0:
                raise RuntimeError(f"Zero gradient in first training step: {name}")
        write_json(self.directory / "first_step_gradients.json", norms)
        print("FIRST_STEP_GRADIENTS " + json.dumps(norms), flush=True)
        self.gradient_checked = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--shiyan-root", default="/workspace/yi/work/shiyan")
    parser.add_argument("--train-data", default="/workspace/yi/work/Cars196/train_data")
    parser.add_argument("--val-data", default="/workspace/yi/work/Cars196/val_data")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-hid", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--resume", help="Resume a trusted checkpoint inside this output directory"
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.accumulate_grad_batches <= 0:
        parser.error("batch sizes and accumulation must be positive")
    if args.embedding_dim <= 0 or args.n_hid <= 0:
        parser.error("model dimensions must be positive")

    output = local_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        resume_path = local_path(args.resume)
        if not resume_path.is_relative_to(output) or not resume_path.is_file():
            raise ValueError(
                "Resume checkpoint must exist inside this run's output directory"
            )
        previous = json.loads((output / "config.json").read_text())
        for key in (
            "batch_size",
            "accumulate_grad_batches",
            "val_batch_size",
            "num_workers",
            "seed",
            "train_data",
            "val_data",
            "n_hid",
            "embedding_dim",
            "learning_rate",
        ):
            if vars(args)[key] != previous[key]:
                raise ValueError(f"Resume configuration differs: {key}")
    elif (output / "config.json").exists():
        raise FileExistsError(
            f"Run already exists: {output}; use a fresh output directory"
        )

    os.environ.pop("SIMVQ_TRAIN_RESIZE", None)
    os.environ.pop("SIMVQ_VAL_RESIZE", None)
    get_dataloader, setup_seed = load_shiyan_loader(Path(args.shiyan_root))
    setup_seed(args.seed)
    torch.set_num_threads(8)
    train_loader = get_dataloader(
        args.train_data,
        args.batch_size,
        shuffle=True,
        mode="train",
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = get_dataloader(
        args.val_data,
        args.val_batch_size,
        shuffle=False,
        mode="val",
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model_args = SimpleNamespace(
        n_hid=args.n_hid,
        embedding_dim=args.embedding_dim,
        num_embeddings=256,
        num_embeddings_min=32,
        num_embeddings_max=2048,
        num_embeddings_test=256,
        lr=args.learning_rate,
        device=torch.device("cuda:0"),
        raq_type="dd",
        model_type="vqvae_one_ds2_src256_k32_2048",
    )
    model = CarsRAQVAEOneDS2(model_args)
    epochs = 1 if args.smoke else args.epochs
    source_paths = [
        Path(__file__).resolve(),
        ROOT / "nets/model.py",
        ROOT / "nets/model_one_ds2_src256_k32_2048.py",
        ROOT / "nets/quantizer.py",
        ROOT / "nets/cbk2cbk.py",
        ROOT / "nets/enc_dec.py",
        ROOT / "nets/blocks.py",
        Path(args.shiyan_root) / "data/datasets.py",
        Path(args.shiyan_root) / "utils/reproducibility.py",
    ]
    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    configuration = {
        **vars(args),
        "output": str(output),
        "epochs": epochs,
        "variant": "RAQVAE_ONE_DS2_SRC256_K32_2048_ORIGINAL",
        "implementation": (
            "RAQVAE_ONE unchanged except encoder/decoder spatial stride x4->x2"
        ),
        "baseline_for": (
            "/workspace/yi/work/Ours-RAQ/scripts/eval/"
            "test_src256_raq32_2048_curriculum_ch256_res6_unet1_ds2.sh"
        ),
        "training_from_scratch": True,
        "spatial_scales": {"single": "1/2"},
        "latent_shapes_at_256": {"single": [128, 128]},
        "indices_per_256_image": 16384,
        "sampling_layers": {
            "encoder": "Conv2d(kernel=4,stride=2,padding=1)",
            "decoder": "ConvTranspose2d(kernel=4,stride=2,padding=1)",
        },
        "model_args": {
            key: str(value) if isinstance(value, torch.device) else value
            for key, value in vars(model_args).items()
        },
        "torch": torch.__version__,
        "pytorch_lightning": pl.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "train_images": len(train_loader.dataset),
        "val_images": len(val_loader.dataset),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "optimizer": (
            "Inherited RAQVAE_ONE.configure_optimizers: AdamW, "
            f"lr={args.learning_rate}, defaults"
        ),
        "loss": (
            "Inherited RAQVAE_ONE objective: "
            "MSE_src + MSE_trg + latent_src + latent_trg"
        ),
        "effective_batch_size": (
            args.batch_size * args.accumulate_grad_batches
        ),
        "train_K": (
            "Inherited RAQVAE_ONE.sample_train_trg: uniform random integer "
            "K in [32,2048] per training batch"
        ),
        "val_K": "Uniform exponent: K in [32,64,128,256,512,1024,2048] per validation batch",
        "checkpoint_monitor": "val_recon_loss = MSE_src + MSE_trg",
        "check_val_every_n_epoch": 1 if args.smoke else 2,
        "precision": 32,
        "channel_training": False,
        "source_sha256": hashes,
    }
    if args.resume:
        write_json(output / f"resume_{time.time_ns()}.json", configuration)
    else:
        write_json(output / "config.json", configuration)
    print("CONFIG " + json.dumps(configuration, ensure_ascii=False), flush=True)

    log_version = f"resume_{time.time_ns()}" if args.resume else ""
    loggers = [
        pl.loggers.TensorBoardLogger(
            str(output), name="tensorboard", version=log_version
        ),
        pl.loggers.CSVLogger(str(output), name="csv", version=log_version),
    ]
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=output / "checkpoints",
        filename="epoch={epoch:03d}-step={step}",
        auto_insert_metric_name=False,
        monitor="val_recon_loss",
        mode="min",
        save_top_k=5,
        save_last=True,
        save_weights_only=False,
    )
    status = OneScaleRunStatus(output)
    trainer = pl.Trainer(
        default_root_dir=str(output),
        accelerator="gpu",
        devices=[0],
        max_epochs=epochs,
        logger=loggers,
        callbacks=[checkpoint, status],
        check_val_every_n_epoch=1 if args.smoke else 2,
        num_sanity_val_steps=0,
        log_every_n_steps=1 if args.smoke else 50,
        enable_progress_bar=False,
        deterministic=True,
        precision=32,
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_train_batches=1 if args.smoke else 1.0,
        limit_val_batches=1 if args.smoke else 1.0,
        plugins=[LocalCheckpointIO()],
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume,
    )
    status.status(trainer, "completed")
    print(f"TRAINING_COMPLETE best={checkpoint.best_model_path}", flush=True)


if __name__ == "__main__":
    main()
