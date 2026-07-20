#!/usr/bin/env python3
"""Five-stage single-teacher variable-rate RAQ training entry point.

The stage is selected exclusively through ``SIMVQ_RAQ_STAGE``.  This keeps the
shell pipeline auditable and, importantly, never imports or mutates the legacy
``train.py`` behavior.  Stages 2--5 always instantiate one separate frozen
``[2048, 2048]`` teacher and never use a detached student SRC branch as a
teacher substitute.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch
from torch.utils.tensorboard import SummaryWriter

from config_variable_rate import VariableRateConfig
from data.datasets import get_dataloader
from evaluation.profile_validation import _build_lpips, profile_key, validate_profiles
from losses.deepsc_loss import DeepSCLoss
from losses.variable_rate_raq_loss import VariableRateRAQLoss
from models.variable_rate_deepsc import VariableRateDeepSC
from training.frozen_teacher import (
    assert_teacher_has_no_grad,
    build_source_teacher,
    copy_teacher_into_student,
    load_frozen_teacher,
    teacher_forward,
)
from training.profile_sampler import ProfileSampler
from utils.checkpoint_utils import load_state_dict_compatible
from utils.reproducibility import setup_seed
from utils.variable_rate_checkpoint import (
    load_checkpoint,
    restore_rng,
    save_checkpoint,
)


def _extract_images(batch: Any) -> torch.Tensor:
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, (tuple, list)) and batch and torch.is_tensor(batch[0]):
        return batch[0]
    if isinstance(batch, Mapping):
        for key in ("image", "images", "input", "pixel_values"):
            value = batch.get(key)
            if torch.is_tensor(value):
                return value
    raise TypeError("training dataloader must yield an image tensor")


def _amp_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # compatibility with older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _cosine_scheduler(optimizer, epochs: int):
    def factor(epoch: int) -> float:
        progress = min(1.0, max(0.0, float(epoch) / max(1, epochs)))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def _append_csv(path: str | Path, row: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _model_common_config(cfg) -> Dict[str, Any]:
    """Constructor metadata sufficient for both SRC and variable-rate models."""

    return {
        "in_channels": cfg.IN_CHANNELS,
        "out_channels": cfg.OUT_CHANNELS,
        "num_downsample_blocks": cfg.NUM_DOWNSAMPLE_BLOCKS,
        "base_channels": cfg.BASE_CHANNELS,
        "num_embeddings_list": list(cfg.NUM_EMBEDDINGS_LIST),
        "source_num_embeddings": list(cfg.NUM_EMBEDDINGS_LIST),
        "embedding_dim_list": list(cfg.EMBEDDING_DIM_LIST),
        "embedding_dims": list(cfg.EMBEDDING_DIM_LIST),
        "commitment_cost": cfg.COMMITMENT_COST,
        "strides": list(cfg.DOWNSAMPLE_STRIDES),
        "skip_dropout_p": [0.0] * max(0, cfg.NUM_DOWNSAMPLE_BLOCKS - 1),
        "norm_type": cfg.NORM_TYPE,
        "norm_groups": cfg.GROUP_NORM_GROUPS,
        "activation": cfg.ACTIVATION,
        "encoder_res_blocks": cfg.ENCODER_RES_BLOCKS,
        "decoder_res_blocks": cfg.DECODER_RES_BLOCKS,
        "upsample_mode": cfg.UPSAMPLE_MODE,
        "use_cascade_downsample": cfg.USE_CASCADE_DOWNSAMPLE,
        "use_bottleneck_attention": cfg.USE_BOTTLENECK_ATTENTION,
        "bottleneck_attention_blocks": cfg.BOTTLENECK_ATTENTION_BLOCKS,
        "use_swinir_enhance": False,
        "channel_coding_rate_train": cfg.CHANNEL_CODING_RATE_TRAIN,
        "channel_coding_rate_val": cfg.CHANNEL_CODING_RATE_VAL,
        "block_length": cfg.BLOCK_LENGTH,
        "snr_range_db": list(cfg.SNR_RANGE_DB),
    }


def _build_loaders(cfg):
    train_loader = get_dataloader(
        cfg.TRAIN_DATASET_PATH,
        batch_size=cfg.MICRO_BATCH_SIZE,
        shuffle=True,
        mode="train",
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    val_loader = get_dataloader(
        cfg.VAL_DATASET_PATH,
        batch_size=cfg.MICRO_BATCH_SIZE,
        shuffle=False,
        mode="val",
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise RuntimeError("training and validation dataloaders must be non-empty")
    return train_loader, val_loader


def _validate_source(model, dataloader, device, max_batches: int) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    image_count = 0
    mse_sum = 0.0
    psnr_sum = 0.0
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(dataloader):
                if batch_index >= max_batches:
                    break
                images = _extract_images(batch).to(device, non_blocking=True)
                output = teacher_forward(model, images)
                reconstruction = output["reconstructed_images"]
                per_image_mse = (
                    ((reconstruction.clamp(-1, 1) - images.clamp(-1, 1)) / 2.0)
                    .square()
                    .flatten(1)
                    .mean(1)
                )
                per_image_psnr = torch.where(
                    per_image_mse <= torch.finfo(per_image_mse.dtype).eps,
                    torch.full_like(per_image_mse, 100.0),
                    10.0 * torch.log10(1.0 / per_image_mse),
                )
                image_count += int(images.shape[0])
                mse_sum += float(per_image_mse.sum().item())
                psnr_sum += float(per_image_psnr.sum().item())
    finally:
        model.train(was_training)
    if image_count == 0:
        raise RuntimeError("source validation consumed no images")
    return {
        "num_images": image_count,
        "mse_0_1": mse_sum / image_count,
        "psnr": psnr_sum / image_count,
    }


def _restore_training_state(
    checkpoint: Mapping[str, Any], optimizer, scheduler, scaler, sampler=None
) -> tuple[int, int, float]:
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler_state:
        scheduler.load_state_dict(scheduler_state)
    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler_state:
        scaler.load_state_dict(scaler_state)
    sampler_state = checkpoint.get("profile_sampler_state")
    if sampler is not None and sampler_state:
        sampler.load_state_dict(sampler_state, strict=True)
    restore_rng(dict(checkpoint))
    return (
        int(checkpoint.get("epoch", -1)) + 1,
        int(checkpoint.get("global_step", 0)),
        float(checkpoint.get("best_score", float("-inf"))),
    )


def _optimizer_step(optimizer, scaler, parameters, grad_clip_norm: float) -> None:
    if scaler.is_enabled():
        scaler.unscale_(optimizer)
    if grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(parameters, grad_clip_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def _train_source_teacher(cfg, device: torch.device) -> None:
    train_loader, val_loader = _build_loaders(cfg)
    model = build_source_teacher(cfg, device)
    model.set_channel_prob(0.0)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("SRC teacher has no trainable parameters")
    optimizer = torch.optim.Adam(
        trainable,
        lr=cfg.STAGE1_SRC_LR,
        betas=cfg.BETAS,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = _cosine_scheduler(optimizer, cfg.NUM_EPOCHS)
    amp_enabled = bool(cfg.AMP_ENABLED and device.type == "cuda")
    scaler = _make_scaler(amp_enabled)
    criterion = DeepSCLoss(
        layer_weights=cfg.RAQ_LAYER_VQ_WEIGHTS,
        mse_weight=cfg.MSE_LOSS_WEIGHT,
        ms_ssim_weight=cfg.MS_SSIM_LOSS_WEIGHT,
        lpips_weight=cfg.LPIPS_LOSS_WEIGHT,
    ).to(device)

    start_epoch, global_step, best_score = 0, 0, float("-inf")
    if cfg.RESUME:
        if not cfg.RESUME_PATH:
            raise ValueError("SIMVQ_RESUME=1 requires SIMVQ_RESUME_PATH")
        checkpoint = load_checkpoint(cfg.RESUME_PATH, map_location="cpu")
        if checkpoint.get("stage") not in (None, "src_teacher"):
            raise ValueError("Stage-1 resume checkpoint has the wrong stage")
        load_state_dict_compatible(model, checkpoint["model_state_dict"], strict=True)
        start_epoch, global_step, best_score = _restore_training_state(
            checkpoint, optimizer, scheduler, scaler
        )

    checkpoint_dir = Path(cfg.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(cfg.LOG_DIR)
    accumulation = cfg.GRADIENT_ACCUMULATION_STEPS
    num_batches = (
        min(len(train_loader), cfg.TRAIN_MAX_BATCHES)
        if cfg.TRAIN_MAX_BATCHES
        else len(train_loader)
    )
    optimizer.zero_grad(set_to_none=True)
    print(
        f"[Stage 1] SRC [2048,2048], epochs={cfg.NUM_EPOCHS}, "
        f"micro_batch={cfg.MICRO_BATCH_SIZE}, accumulation={accumulation}"
    )

    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        model.train()
        model.set_channel_prob(0.0)
        epoch_loss = 0.0
        epoch_images = 0
        epoch_start = time.time()
        window_size = accumulation
        for batch_index, batch in enumerate(train_loader):
            if batch_index >= num_batches:
                break
            if batch_index % accumulation == 0:
                window_size = min(accumulation, num_batches - batch_index)
            images = _extract_images(batch).to(device, non_blocking=True)
            with _amp_context(device, amp_enabled):
                output = model.forward_train(images)
                recon_loss, vq_loss = criterion(
                    images,
                    output["reconstructed_images"],
                    output["vq_losses"],
                )
                total_loss = recon_loss + cfg.RAQ_VQ_WEIGHT * vq_loss
                scaled_loss = total_loss / float(window_size)
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"non-finite Stage-1 loss at epoch {epoch + 1}")
            scaler.scale(scaled_loss).backward()
            epoch_loss += float(total_loss.detach().item()) * int(images.shape[0])
            epoch_images += int(images.shape[0])

            window_end = (
                (batch_index + 1) % accumulation == 0 or batch_index + 1 == num_batches
            )
            if window_end:
                _optimizer_step(
                    optimizer, scaler, trainable, cfg.GRAD_CLIP_NORM
                )
                global_step += 1
                writer.add_scalar("train/src_total_loss", total_loss.detach(), global_step)
                writer.add_scalar("train/src_reconstruction_loss", recon_loss.detach(), global_step)
                writer.add_scalar("train/src_vq_loss", vq_loss.detach(), global_step)
                if global_step % cfg.LOG_INTERVAL == 0:
                    print(
                        f"[Stage 1][{epoch + 1}/{cfg.NUM_EPOCHS}] "
                        f"step={global_step} loss={float(total_loss.detach()):.6f}"
                    )

        scheduler.step()
        if epoch_images == 0:
            raise RuntimeError("Stage-1 epoch consumed no images")
        train_mean = epoch_loss / epoch_images
        validation = None
        if (epoch + 1) % cfg.VAL_INTERVAL == 0 or epoch + 1 == cfg.NUM_EPOCHS:
            validation = _validate_source(
                model, val_loader, device, max_batches=cfg.VAL_MAX_BATCHES
            )
            score = validation["psnr"]
            writer.add_scalar("validation/src_psnr", score, epoch + 1)
            writer.add_scalar("validation/src_mse_0_1", validation["mse_0_1"], epoch + 1)
            if score > best_score:
                best_score = score
                save_checkpoint(
                    checkpoint_dir / "best_src_teacher.pth",
                    model=model,
                    stage="src_teacher",
                    epoch=epoch,
                    config=cfg,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    global_step=global_step,
                    best_score=best_score,
                    validation=validation,
                    model_config=_model_common_config(cfg),
                )

        save_checkpoint(
            checkpoint_dir / "last_checkpoint.pth",
            model=model,
            stage="src_teacher",
            epoch=epoch,
            config=cfg,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            global_step=global_step,
            best_score=best_score,
            validation=validation,
            model_config=_model_common_config(cfg),
        )
        if cfg.SAVE_EVERY and (epoch + 1) % cfg.SAVE_EVERY == 0:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch + 1:04d}.pth",
                model=model,
                stage="src_teacher",
                epoch=epoch,
                config=cfg,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                global_step=global_step,
                best_score=best_score,
                validation=validation,
                model_config=_model_common_config(cfg),
            )
        _append_csv(
            cfg.METRICS_PATH,
            {
                "epoch": epoch + 1,
                "stage": "src_teacher",
                "train_loss": train_mean,
                "val_psnr": "" if validation is None else validation["psnr"],
                "val_mse_0_1": "" if validation is None else validation["mse_0_1"],
                "best_score": best_score,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": time.time() - epoch_start,
            },
        )
        print(
            f"[Stage 1] epoch={epoch + 1} train={train_mean:.6f} "
            f"val_psnr={None if validation is None else round(validation['psnr'], 4)} "
            f"best={best_score:.4f}"
        )
        writer.flush()
    writer.close()


def _set_trainable(module: torch.nn.Module, value: bool = True) -> list[torch.nn.Parameter]:
    parameters = list(module.parameters())
    for parameter in parameters:
        parameter.requires_grad_(value)
    return parameters if value else []


def _stage_generator_lr(cfg) -> float:
    return {
        2: cfg.STAGE2_RAQ_LR,
        3: cfg.STAGE3_RAQ_LR,
        4: cfg.STAGE4_RAQ_LR,
        5: cfg.STAGE5_RAQ_LR,
    }[cfg.STAGE_INDEX]


def _configure_student_optimizer(student: VariableRateDeepSC, cfg):
    for parameter in student.parameters():
        parameter.requires_grad_(False)

    base_raq_lr = _stage_generator_lr(cfg)
    groups = []
    core_parameters = _set_trainable(student.raq_generator.layer_generators, True)
    groups.append({"params": core_parameters, "lr": base_raq_lr, "name": "raq_generator"})

    rate_parameters = _set_trainable(student.raq_generator.rate_conditioner, True)
    groups.append(
        {
            "params": rate_parameters,
            "lr": min(base_raq_lr, cfg.RATE_MODULE_LR),
            "name": "rate_embedding",
        }
    )
    film_parameters = _set_trainable(student.encoder_rate_affines, True)
    film_parameters += _set_trainable(student.decoder_rate_affines, True)
    groups.append(
        {
            "params": film_parameters,
            "lr": min(base_raq_lr, cfg.FILM_LR),
            "name": "film",
        }
    )

    if cfg.STAGE_INDEX >= 4:
        decoder_parameters: list[torch.nn.Parameter] = []
        tail_count = min(cfg.DECODER_TAIL_BLOCKS, len(student.semantic_decoder.up_blocks))
        if tail_count:
            for block in student.semantic_decoder.up_blocks[-tail_count:]:
                decoder_parameters += _set_trainable(block, True)
        decoder_parameters += _set_trainable(student.semantic_decoder.final, True)
        decoder_lr = (
            cfg.STAGE4_DECODER_LR if cfg.STAGE_INDEX == 4 else cfg.STAGE5_DECODER_LR
        )
        groups.append(
            {"params": decoder_parameters, "lr": decoder_lr, "name": "decoder_tail"}
        )

        train_encoder = (
            cfg.TRAIN_ENCODER_STAGE4
            if cfg.STAGE_INDEX == 4
            else cfg.TRAIN_ENCODER_STAGE5
        )
        if train_encoder:
            encoder_parameters: list[torch.nn.Parameter] = []
            tail_count = min(cfg.ENCODER_TAIL_BLOCKS, len(student.semantic_encoder.blocks))
            if tail_count:
                for block in student.semantic_encoder.blocks[-tail_count:]:
                    encoder_parameters += _set_trainable(block, True)
            encoder_parameters += _set_trainable(student.bottleneck_attention, True)
            encoder_lr = (
                cfg.STAGE4_ENCODER_LR if cfg.STAGE_INDEX == 4 else cfg.STAGE5_ENCODER_LR
            )
            groups.append(
                {"params": encoder_parameters, "lr": encoder_lr, "name": "encoder_tail"}
            )

    student.freeze_source_codebooks()
    groups = [group for group in groups if group["params"]]
    parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise RuntimeError("student optimizer parameter groups overlap")
    trainable_ids = {id(parameter) for parameter in student.parameters() if parameter.requires_grad}
    if trainable_ids != set(parameter_ids):
        raise RuntimeError("some trainable student parameters are missing from optimizer groups")
    if not trainable_ids:
        raise RuntimeError("student stage has no trainable parameters")

    optimizer = torch.optim.Adam(
        groups,
        betas=cfg.BETAS,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    total = sum(parameter.numel() for parameter in student.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in student.parameters() if parameter.requires_grad
    )
    print(f"[Trainability] {trainable_count:,}/{total:,} parameters trainable")
    for group in optimizer.param_groups:
        print(
            f"  - {group.get('name')}: "
            f"{sum(parameter.numel() for parameter in group['params']):,} @ {group['lr']:.3g}"
        )
    return optimizer


def _frozen_submodules_eval(model: torch.nn.Module) -> None:
    """Prevent running-stat updates in parameter-frozen BatchNorm subtrees."""

    for module in model.modules():
        if module is model:
            continue
        if not any(parameter.requires_grad for parameter in module.parameters(recurse=True)):
            module.eval()


def _assert_source_codebooks_frozen(student: VariableRateDeepSC) -> None:
    offenders = []
    for layer, quantizer in enumerate(student.vector_quantizers):
        for name, parameter in quantizer.named_parameters():
            if parameter.requires_grad or parameter.grad is not None:
                offenders.append(f"layer{layer}.{name}")
    if offenders:
        raise RuntimeError("student source codebooks changed trainability: " + ", ".join(offenders))


def _assert_generator_gradient(student: VariableRateDeepSC, profile: Sequence[int]) -> float:
    active_layers = [index for index, k in enumerate(profile) if int(k) < 2048]
    if not active_layers:
        raise ValueError("generator-gradient assertion needs a non-bypassed profile")
    magnitude = 0.0
    missing = []
    for layer in active_layers:
        layer_magnitude = 0.0
        for parameter in student.raq_generator.layer_generators[layer].parameters():
            if parameter.grad is not None:
                layer_magnitude += float(parameter.grad.detach().abs().sum().item())
        if layer_magnitude <= 0.0:
            missing.append(layer)
        magnitude += layer_magnitude
    if missing:
        raise RuntimeError(
            f"dual reconstruction produced no RAQ generator gradient for layers {missing} "
            f"at profile {tuple(profile)}"
        )
    return magnitude


def _load_student_checkpoint(
    student: VariableRateDeepSC,
    path: str | Path,
    cfg,
    *,
    resume: bool,
) -> Dict[str, Any]:
    checkpoint = load_checkpoint(path, map_location="cpu")
    expected_stage = (
        cfg.TRAIN_STAGE
        if resume
        else {3: "identity_warmup", 4: "variable_rate", 5: "joint_lite"}[cfg.STAGE_INDEX]
    )
    saved_stage = checkpoint.get("stage")
    if saved_stage != expected_stage:
        raise ValueError(
            f"student checkpoint stage is {saved_stage!r}; expected {expected_stage!r}"
        )
    recorded_teacher = checkpoint.get("teacher_checkpoint")
    if recorded_teacher:
        expected_teacher = Path(cfg.SRC_TEACHER_CHECKPOINT).expanduser().resolve()
        if Path(str(recorded_teacher)).expanduser().resolve() != expected_teacher:
            raise ValueError("student checkpoint was trained with a different SRC teacher")
    load_state_dict_compatible(student, checkpoint["model_state_dict"], strict=True)
    return checkpoint


def _channel_probability(cfg, epoch: int) -> float:
    if cfg.STAGE_INDEX != 5:
        return 0.0
    denominator = max(1, cfg.CHANNEL_RAMP_EPOCHS - 1)
    progress = min(1.0, float(epoch) / denominator)
    return cfg.CHANNEL_PROB_START + (
        cfg.CHANNEL_PROB_END - cfg.CHANNEL_PROB_START
    ) * progress


def _scheduled_layer_vq_weights(cfg, epoch: int) -> tuple[float, float]:
    """Linearly preserve the legacy per-layer VQ-weight schedule contract."""

    progress = min(1.0, float(epoch) / max(1, cfg.NUM_EPOCHS - 1))
    values = tuple(
        float(start) + (float(end) - float(start)) * progress
        for start, end in zip(
            cfg.RAQ_LAYER_VQ_WEIGHTS, cfg.LAYER_LOSS_WEIGHTS_FINAL
        )
    )
    return values  # type: ignore[return-value]


def _reference_psnr_from_env() -> Optional[Dict[str, float]]:
    raw = os.environ.get("SIMVQ_RAQ_SRC_REFERENCE_PSNR", "").strip()
    if not raw:
        return None
    path = Path(raw)
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("SIMVQ_RAQ_SRC_REFERENCE_PSNR must be a JSON object or JSON file")
    return {str(key): float(value) for key, value in payload.items()}


def _save_sampler_counts(cfg, sampler: ProfileSampler, epoch: int) -> None:
    counts = {
        profile_key(profile): count for profile, count in sampler.counts.items()
    }
    _write_json(
        Path(cfg.SNAPSHOT_DIR) / "profile_sampling_counts.json",
        {
            "epoch": epoch + 1,
            "coverage": sampler.coverage_summary(),
            "counts": counts,
        },
    )


def _train_variable_rate_student(cfg, device: torch.device) -> None:
    train_loader, val_loader = _build_loaders(cfg)
    teacher_path = Path(cfg.SRC_TEACHER_CHECKPOINT).expanduser().resolve()
    teacher = load_frozen_teacher(cfg, teacher_path, device)
    student = VariableRateDeepSC.from_config(cfg).to(device)

    resume_checkpoint = None
    if cfg.RESUME:
        if not cfg.RESUME_PATH:
            raise ValueError("SIMVQ_RESUME=1 requires SIMVQ_RESUME_PATH")
        resume_checkpoint = _load_student_checkpoint(
            student, cfg.RESUME_PATH, cfg, resume=True
        )
    elif cfg.STAGE_INDEX == 2:
        # State dictionaries are copied, never aliased.  The teacher stays a
        # completely independent immutable model for all later supervision.
        copy_teacher_into_student(student, teacher)
    else:
        _load_student_checkpoint(student, cfg.STUDENT_CHECKPOINT, cfg, resume=False)
    student.freeze_source_codebooks()
    student.set_channel_prob(0.0)

    teacher_parameter_ids = {id(parameter) for parameter in teacher.parameters()}
    student_parameter_ids = {id(parameter) for parameter in student.parameters()}
    if teacher_parameter_ids & student_parameter_ids:
        raise RuntimeError("teacher and student unexpectedly share parameter objects")
    assert_teacher_has_no_grad(teacher)
    _assert_source_codebooks_frozen(student)

    sampler = ProfileSampler(
        cfg.TARGET_PROFILES,
        num_random=cfg.SANDWICH_NUM_RANDOM,
        min_profile=cfg.MIN_PROFILE,
        max_profile=cfg.MAX_PROFILE,
        supported_k=cfg.SUPPORTED_K_VALUES,
        seed=cfg.PROFILE_SAMPLER_SEED,
    )
    optimizer = _configure_student_optimizer(student, cfg)
    scheduler = _cosine_scheduler(optimizer, cfg.NUM_EPOCHS)
    amp_enabled = bool(cfg.AMP_ENABLED and device.type == "cuda")
    scaler = _make_scaler(amp_enabled)
    criterion = VariableRateRAQLoss.from_config(cfg).to(device)
    criterion.eval()

    start_epoch, global_step, best_score = 0, 0, float("-inf")
    if resume_checkpoint is not None:
        start_epoch, global_step, best_score = _restore_training_state(
            resume_checkpoint, optimizer, scheduler, scaler, sampler
        )

    checkpoint_dir = Path(cfg.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(cfg.LOG_DIR)
    lpips_model = _build_lpips(device, "vgg")
    src_reference_psnr = _reference_psnr_from_env()
    worst_weight = cfg.VAL_WORST_WEIGHT / (
        cfg.VAL_AVERAGE_WEIGHT + cfg.VAL_WORST_WEIGHT
    )
    accumulation = cfg.GRADIENT_ACCUMULATION_STEPS
    num_batches = (
        min(len(train_loader), cfg.TRAIN_MAX_BATCHES)
        if cfg.TRAIN_MAX_BATCHES
        else len(train_loader)
    )
    trainable_parameters = [
        parameter for parameter in student.parameters() if parameter.requires_grad
    ]
    optimizer.zero_grad(set_to_none=True)
    gradient_contract_checked = False

    print(
        f"[Stage {cfg.STAGE_INDEX}] {cfg.TRAIN_STAGE}, profiles={len(sampler.profiles)}, "
        f"sandwich=max+min+{cfg.SANDWICH_NUM_RANDOM}, dual=True"
    )
    print(
        "[Gradient contract] [2048,2048] hard-bypasses both layer generators; "
        f"Stage {cfg.STAGE_INDEX} minimum profile is {cfg.MIN_PROFILE}."
    )

    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        student.train()
        _frozen_submodules_eval(student)
        teacher.eval()
        layer_vq_weights = _scheduled_layer_vq_weights(cfg, epoch)
        criterion.layer_vq_weights = layer_vq_weights
        channel_probability = _channel_probability(cfg, epoch)
        student.set_channel_prob(channel_probability)
        use_channel = cfg.STAGE_INDEX == 5
        epoch_loss_sum = 0.0
        epoch_profile_forwards = 0
        detail_sums: Dict[str, float] = defaultdict(float)
        epoch_start = time.time()
        window_profiles: list[tuple[int, int]] = []
        window_size = accumulation

        for batch_index, batch in enumerate(train_loader):
            if batch_index >= num_batches:
                break
            window_offset = batch_index % accumulation
            if window_offset == 0:
                window_size = min(accumulation, num_batches - batch_index)
                window_profiles = sampler.sample_profiles(update_counts=True)
            else:
                # Counts record actual microbatch/profile forwards, not merely
                # optimizer windows.
                sampler.record_profiles(window_profiles)

            images = _extract_images(batch).to(device, non_blocking=True)
            # Exactly one independent teacher forward per microbatch.  Its
            # detached outputs are reused by all sandwich profiles below.
            with _amp_context(device, amp_enabled):
                frozen_targets = teacher_forward(teacher, images)

            profile_denominator = float(window_size * len(window_profiles))
            for profile in window_profiles:
                with _amp_context(device, amp_enabled):
                    student_output = student.forward_profile(
                        images,
                        profile,
                        use_channel=use_channel,
                        generate_hierarchy=cfg.HIERARCHY_WEIGHT > 0,
                    )
                    total_loss, details = criterion.forward_from_outputs(
                        images,
                        student_output,
                        frozen_targets,
                        return_details=True,
                    )
                    normalized_loss = total_loss / profile_denominator
                if not torch.isfinite(total_loss):
                    raise FloatingPointError(
                        f"non-finite loss at epoch {epoch + 1}, profile={profile}"
                    )
                scaler.scale(normalized_loss).backward()

                if not gradient_contract_checked and profile != cfg.MAX_PROFILE:
                    gradient_magnitude = _assert_generator_gradient(student, profile)
                    assert_teacher_has_no_grad(teacher)
                    _assert_source_codebooks_frozen(student)
                    print(
                        f"[Gradient contract] non-bypass profile {profile} produced "
                        f"generator |grad| sum={gradient_magnitude:.6g}; teacher/source frozen."
                    )
                    gradient_contract_checked = True

                epoch_loss_sum += float(total_loss.detach().item())
                epoch_profile_forwards += 1
                for name, value in details.items():
                    if torch.is_tensor(value) and value.numel() == 1:
                        detail_sums[name] += float(value.detach().item())
                del student_output, total_loss, normalized_loss, details

            assert_teacher_has_no_grad(teacher)
            _assert_source_codebooks_frozen(student)
            window_end = (
                (batch_index + 1) % accumulation == 0 or batch_index + 1 == num_batches
            )
            if window_end:
                _optimizer_step(
                    optimizer, scaler, trainable_parameters, cfg.GRAD_CLIP_NORM
                )
                global_step += 1
                if global_step % cfg.LOG_INTERVAL == 0:
                    recent = epoch_loss_sum / max(1, epoch_profile_forwards)
                    print(
                        f"[Stage {cfg.STAGE_INDEX}][{epoch + 1}/{cfg.NUM_EPOCHS}] "
                        f"step={global_step} mean_profile_loss={recent:.6f} "
                        f"profiles={window_profiles} channel_p={channel_probability:.3f}"
                    )

        scheduler.step()
        if not gradient_contract_checked:
            raise RuntimeError(
                "no non-maximum profile was executed; the RAQ layer generators cannot train"
            )
        if epoch_profile_forwards == 0:
            raise RuntimeError("student epoch consumed no profile forwards")
        train_mean = epoch_loss_sum / epoch_profile_forwards
        for name, value in detail_sums.items():
            writer.add_scalar(
                f"train_profile_mean/{name}", value / epoch_profile_forwards, epoch + 1
            )
        writer.add_scalar("train_profile_mean/total", train_mean, epoch + 1)
        writer.add_scalar("train/layer0_vq_weight", layer_vq_weights[0], epoch + 1)
        writer.add_scalar("train/layer1_vq_weight", layer_vq_weights[1], epoch + 1)
        writer.add_scalar("train/channel_probability", channel_probability, epoch + 1)

        validation = None
        if (epoch + 1) % cfg.VAL_INTERVAL == 0 or epoch + 1 == cfg.NUM_EPOCHS:
            # Fixed-profile checkpoint validation is always clean/no-channel;
            # Stage 5's stochastic channel objective is logged separately.
            validation = validate_profiles(
                student,
                teacher,
                val_loader,
                cfg.VAL_PROFILES,
                device,
                lpips_model=lpips_model,
                max_batches=cfg.VAL_MAX_BATCHES,
                src_reference_psnr=src_reference_psnr,
                profile_weights=cfg.VAL_PROFILE_WEIGHTS or None,
                worst_profile_weight=worst_weight,
                max_profile=cfg.MAX_PROFILE,
                max_teacher_psnr_drop_db=cfg.VAL_MAX_PSNR_DROP_DB,
                require_teacher_guard=cfg.VAL_REQUIRE_MAX_PROTECTION,
                writer=writer,
                global_step=epoch + 1,
                csv_path=cfg.PROFILE_METRICS_PATH,
                per_profile_csv_dir=Path(cfg.PROFILE_METRICS_PATH).with_suffix(""),
            )
            score = float(validation["score"])
            if validation["eligible"] and score > best_score:
                best_score = score
                save_checkpoint(
                    checkpoint_dir / "best_variable_rate_raq.pth",
                    model=student,
                    stage=cfg.TRAIN_STAGE,
                    epoch=epoch,
                    config=cfg,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    sampler=sampler,
                    global_step=global_step,
                    best_score=best_score,
                    teacher_checkpoint=str(teacher_path),
                    validation=validation,
                    model_config=student.export_constructor_config(),
                    extra_state={"channel_probability": channel_probability},
                )
            elif not validation["eligible"]:
                print(
                    f"[Checkpoint guard] not eligible: max-profile teacher drop "
                    f"{validation['teacher_psnr_drop_db']:.4f} dB exceeds "
                    f"{cfg.VAL_MAX_PSNR_DROP_DB:.4f} dB"
                )

        save_checkpoint(
            checkpoint_dir / "last_checkpoint.pth",
            model=student,
            stage=cfg.TRAIN_STAGE,
            epoch=epoch,
            config=cfg,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            sampler=sampler,
            global_step=global_step,
            best_score=best_score,
            teacher_checkpoint=str(teacher_path),
            validation=validation,
            model_config=student.export_constructor_config(),
            extra_state={"channel_probability": channel_probability},
        )
        if cfg.SAVE_EVERY and (epoch + 1) % cfg.SAVE_EVERY == 0:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch + 1:04d}.pth",
                model=student,
                stage=cfg.TRAIN_STAGE,
                epoch=epoch,
                config=cfg,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                sampler=sampler,
                global_step=global_step,
                best_score=best_score,
                teacher_checkpoint=str(teacher_path),
                validation=validation,
                model_config=student.export_constructor_config(),
                extra_state={"channel_probability": channel_probability},
            )
        _save_sampler_counts(cfg, sampler, epoch)
        coverage = sampler.coverage_summary()
        _append_csv(
            cfg.METRICS_PATH,
            {
                "epoch": epoch + 1,
                "stage": cfg.TRAIN_STAGE,
                "train_profile_loss": train_mean,
                "validation_score": "" if validation is None else validation["score"],
                "weighted_mean_psnr": "" if validation is None else validation["weighted_mean_psnr"],
                "worst_profile": "" if validation is None else validation["worst_profile"],
                "worst_psnr": "" if validation is None else validation["worst_psnr"],
                "teacher_drop_db": "" if validation is None else validation["teacher_psnr_drop_db"],
                "eligible": "" if validation is None else validation["eligible"],
                "best_score": best_score,
                "covered_profiles": coverage["covered_profiles"],
                "min_sample_count": coverage["min_count"],
                "max_sample_count": coverage["max_count"],
                "layer0_vq_weight": layer_vq_weights[0],
                "layer1_vq_weight": layer_vq_weights[1],
                "channel_probability": channel_probability,
                "seconds": time.time() - epoch_start,
            },
        )
        print(
            f"[Stage {cfg.STAGE_INDEX}] epoch={epoch + 1} train={train_mean:.6f} "
            f"score={None if validation is None else round(validation['score'], 4)} "
            f"eligible={None if validation is None else validation['eligible']} "
            f"coverage={coverage['covered_profiles']}/{coverage['profiles']}"
        )
        writer.flush()
    writer.close()


def main() -> int:
    cfg = VariableRateConfig()
    cfg.validate(require_checkpoint_paths=True, require_dataset_paths=True)
    setup_seed(cfg.SEED, deterministic=True)
    device = torch.device(cfg.DEVICE)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        print(
            f"[Device] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}; "
            f"logical={device}; name={torch.cuda.get_device_name(device)}"
        )
    else:
        print(f"[Device] {device}")
    print(f"[Isolation] working directory: {Path.cwd().resolve()}")
    if Path.cwd().resolve().name != "shiyan-2":
        raise RuntimeError("train_variable_rate.py must be launched from the shiyan-2 root")

    if cfg.STAGE_INDEX == 1:
        _train_source_teacher(cfg, device)
    else:
        _train_variable_rate_student(cfg, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
