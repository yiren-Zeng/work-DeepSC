import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from config import Config
from data.datasets import get_dataloader
from losses.deepsc_loss import DeepSCLoss
from models.deepsc_last2_cascade import DeepSCLast2Cascade


def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[Info] Random seed set to {seed}")


def build_model(cfg):
    return DeepSCLast2Cascade(
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        num_downsample_blocks=cfg.NUM_DOWNSAMPLE_BLOCKS,
        base_channels=cfg.BASE_CHANNELS,
        full_num_embeddings_list=cfg.NUM_EMBEDDINGS_LIST,
        full_embedding_dim_list=cfg.EMBEDDING_DIM_LIST,
        commitment_cost=cfg.COMMITMENT_COST,
    )


def _checkpoint_metadata(model):
    return {
        "model_type": model.MODEL_TYPE,
        "active_scales": list(model.ACTIVE_SCALES),
        "full_num_embeddings_list": list(model.full_num_embeddings_list),
        "active_num_embeddings_list": list(model.num_embeddings_list),
        "active_embedding_dim_list": list(model.embedding_dim_list),
    }


def _validate_resume_checkpoint(checkpoint, model):
    expected = _checkpoint_metadata(model)
    for key, expected_value in expected.items():
        actual_value = checkpoint.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Resume checkpoint {key}={actual_value!r} does not match "
                f"the requested last-two model value {expected_value!r}"
            )


def save_best_checkpoint(path, epoch, model, best_val_loss):
    torch.save(
        {
            **_checkpoint_metadata(model),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def save_training_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    best_val_loss,
):
    torch.save(
        {
            **_checkpoint_metadata(model),
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
        },
        path,
    )


def main():
    cfg = Config()
    cfg.validate()
    setup_seed(42)
    device = torch.device(cfg.DEVICE)

    if cfg.TOTAL_BATCH_SIZE % cfg.MICRO_BATCH_SIZE != 0:
        raise ValueError("TOTAL_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE")
    accumulation_steps = cfg.TOTAL_BATCH_SIZE // cfg.MICRO_BATCH_SIZE

    print(f"Start last-two copy-cascade VQ training on {device}")
    print(f"Total batch size: {cfg.TOTAL_BATCH_SIZE}")
    print(f"Micro batch size: {cfg.MICRO_BATCH_SIZE}")
    print(f"Gradient accumulation steps: {accumulation_steps}")
    print(f"Experiment: {cfg.EXPERIMENT_NAME}")
    print(f"Full historical codebook label: {cfg.NUM_EMBEDDINGS_LIST}")
    print(f"Active scales: {list(DeepSCLast2Cascade.ACTIVE_SCALES)}")
    print(f"Active codebooks: {cfg.NUM_EMBEDDINGS_LIST[-2:]}")
    print(f"Checkpoint directory: {cfg.CHECKPOINT_DIR}")

    log_dir = os.path.join(cfg.LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    model = build_model(cfg).to(device)
    if accumulation_steps > 1:
        base_momentum = 0.1
        adjusted_momentum = 1 - (1 - base_momentum) ** (1 / accumulation_steps)
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.momentum = adjusted_momentum

    loss_fn = DeepSCLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, betas=cfg.BETAS)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    start_epoch = 0
    best_val_loss = float("inf")
    if cfg.RESUME and os.path.isfile(cfg.RESUME_PATH):
        print(f"Loading checkpoint: {cfg.RESUME_PATH}")
        checkpoint = torch.load(
            cfg.RESUME_PATH,
            map_location=device,
            weights_only=False,
        )
        _validate_resume_checkpoint(checkpoint, model)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
            cuda_states = [state.cpu() for state in checkpoint["cuda_rng_state"]]
            torch.cuda.set_rng_state_all(cuda_states[: torch.cuda.device_count()])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        print(f"Resuming from epoch {start_epoch + 1}")

    train_loader = get_dataloader(
        root_dir=cfg.TRAIN_DATASET_PATH,
        batch_size=cfg.MICRO_BATCH_SIZE,
        shuffle=True,
        mode="train",
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    val_loader = get_dataloader(
        root_dir=cfg.VAL_DATASET_PATH,
        batch_size=cfg.MICRO_BATCH_SIZE,
        shuffle=False,
        mode="val",
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    global_step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        reconstruction_sum = 0.0
        vq_sum = 0.0

        for step, images in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            should_step = (
                (step + 1) % accumulation_steps == 0
                or step + 1 == len(train_loader)
            )

            output = model(images)
            reconstruction_loss, vq_loss = loss_fn(
                images,
                output["reconstructed_images"],
                output["vq_losses"],
            )
            total_loss = reconstruction_loss + vq_loss
            (total_loss / accumulation_steps).backward()
            reconstruction_sum += reconstruction_loss.item()
            vq_sum += vq_loss.item()

            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % (accumulation_steps * 10) == 0:
                print(
                    f"Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}] "
                    f"Step [{step + 1}/{len(train_loader)}] "
                    f"Recon: {reconstruction_loss.item():.6f} "
                    f"VQ: {vq_loss.item():.6f}"
                )
                writer.add_scalar("Train/Loss_Step", total_loss.item(), global_step)
            global_step += 1

        scheduler.step()
        average_reconstruction = reconstruction_sum / len(train_loader)
        average_vq = vq_sum / len(train_loader)
        writer.add_scalar("Loss/Train/Recon", average_reconstruction, epoch)
        writer.add_scalar("Loss/Train/VQ", average_vq, epoch)
        writer.add_scalar(
            "Loss/Train/Total", average_reconstruction + average_vq, epoch
        )

        model.eval()
        validation_sum = 0.0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device, non_blocking=True)
                output = model(images)
                validation_loss, _ = loss_fn(
                    images,
                    output["reconstructed_images"],
                    output["vq_losses"],
                )
                validation_sum += validation_loss.item()

        average_validation = validation_sum / len(val_loader)
        writer.add_scalar("Loss/Val/Recon", average_validation, epoch)
        print(
            f"[VAL] Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}] "
            f"Recon: {average_validation:.6f}"
        )

        if (epoch + 1) % 10 == 0:
            stats = model.compute_codebook_utilization(
                val_loader, max_batches=20, device=device
            )
            model.print_codebook_utilization(stats)
            for scale_stats in stats["src"]:
                physical_scale = scale_stats["scale"]
                writer.add_scalar(
                    f"Codebook/L{physical_scale}/ActiveRatio",
                    scale_stats["active_ratio"],
                    epoch,
                )
                writer.add_scalar(
                    f"Codebook/L{physical_scale}/Perplexity",
                    scale_stats["perplexity"],
                    epoch,
                )
                writer.add_scalar(
                    f"Codebook/L{physical_scale}/MinL2Dist",
                    scale_stats["min_l2_dist"],
                    epoch,
                )
                writer.add_scalar(
                    f"Codebook/L{physical_scale}/CollapseRatio",
                    scale_stats["collapse_ratio"],
                    epoch,
                )

        if average_validation < best_val_loss:
            best_val_loss = average_validation
            save_best_checkpoint(
                os.path.join(cfg.CHECKPOINT_DIR, "best_vq_deepsc.pth"),
                epoch,
                model,
                best_val_loss,
            )
            print(f"Saved best model with validation loss {best_val_loss:.6f}")

        save_training_checkpoint(
            cfg.RESUME_PATH,
            epoch,
            model,
            optimizer,
            scheduler,
            best_val_loss,
        )

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
