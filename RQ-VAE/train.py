import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import math
from datetime import datetime
from config import Config
from models.deepsc import DeepSC
from losses.deepsc_loss import DeepSCLoss, ms_ssim_loss
from data.datasets import get_dataloader
from monitoring.codebook import (compute_codebook_utilization, print_codebook_utilization, write_codebook_tensorboard,)
from training.schedules import compute_schedule
from utils.experiment_io import (
    append_codebook_records,
    append_epoch_record,
    rq_epoch_metric_fields,
)
from utils.reproducibility import setup_seed
from utils.checkpoint_utils import (
    build_checkpoint_payload,
    load_checkpoint_payload,
    load_model_state_dict,
)


RQ_MONITOR_QUANTIZER_TYPES = frozenset({"rq_ema", "residual_simvq"})


def _uses_rq_monitoring(quantizer_type):
    return str(quantizer_type).lower() in RQ_MONITOR_QUANTIZER_TYPES


def build_optimizer_parameter_groups(
    model,
    base_learning_rate,
    codebook_projection_learning_rate,
    quantizer_type=None,
):
    """Build and validate disjoint optimizer groups.

    Every trainable ``.codebook.proj.`` parameter is placed in the dedicated
    codebook group.  Residual-SimVQ's frozen ``.codebook.embed.`` parameters
    are asserted to remain frozen and absent from every optimizer group.
    The returned audit dictionary keeps this contract independently testable.
    """
    quantizer_type = str(
        quantizer_type
        if quantizer_type is not None
        else getattr(model, "quantizer_type", "")
    ).lower()
    projection_params = []
    projection_names = []
    other_params = []
    other_names = []
    frozen_embedding_params = []
    frozen_embedding_names = []

    for name, parameter in model.named_parameters():
        is_codebook_embedding = ".codebook.embed." in f".{name}"
        is_projection = (
            ".codebook.proj." in f".{name}" or ".qbridge." in f".{name}"
        )
        if is_codebook_embedding:
            frozen_embedding_params.append(parameter)
            frozen_embedding_names.append(name)
            if quantizer_type == "residual_simvq" and parameter.requires_grad:
                raise RuntimeError(
                    "Residual-SimVQ base embeddings must remain frozen: " + name
                )
        if not parameter.requires_grad:
            continue
        if is_projection:
            projection_params.append(parameter)
            projection_names.append(name)
        else:
            other_params.append(parameter)
            other_names.append(name)

    groups = [{"params": other_params, "lr": float(base_learning_rate)}]
    if projection_params:
        groups.append(
            {
                "params": projection_params,
                "lr": float(codebook_projection_learning_rate),
            }
        )

    grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]
    trainable_ids = [
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    ]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise RuntimeError("Optimizer parameter groups contain duplicate parameters.")
    if set(grouped_ids) != set(trainable_ids):
        raise RuntimeError(
            "Optimizer parameter groups must contain every trainable parameter exactly once."
        )

    projection_ids = {id(parameter) for parameter in projection_params}
    expected_projection_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (
            ".codebook.proj." in f".{name}"
            or ".qbridge." in f".{name}"
        )
    }
    if projection_ids != expected_projection_ids:
        raise RuntimeError(
            "Not every trainable codebook projection is in the projection LR group."
        )
    grouped_id_set = set(grouped_ids)
    if any(id(parameter) in grouped_id_set for parameter in frozen_embedding_params):
        raise RuntimeError("Frozen codebook embeddings must not enter the optimizer.")
    if projection_params and groups[-1]["lr"] != float(
        codebook_projection_learning_rate
    ):
        raise RuntimeError("Codebook projection parameters have the wrong learning rate.")

    return groups, {
        "other_names": other_names,
        "projection_names": projection_names,
        "frozen_embedding_names": frozen_embedding_names,
    }


def projection_gradient_norms(model):
    """Return one L2 gradient norm per scale's shared codebook projection."""
    norms = []
    for quantizer in getattr(model, "vector_quantizers", []):
        codebook = getattr(quantizer, "codebook", None)
        projection = getattr(codebook, "proj", None)
        squared_norm = None
        if projection is not None:
            for parameter in projection.parameters():
                if parameter.grad is None:
                    continue
                value = parameter.grad.detach().float().pow(2).sum()
                squared_norm = value if squared_norm is None else squared_norm + value
        norms.append(
            None if squared_norm is None else float(torch.sqrt(squared_norm).item())
        )
    return norms


def _empty_rq_diagnostic_accumulator(rq_depth_list):
    return [
        {
            "batches": 0,
            "codebook_loss": 0.0,
            "has_codebook_loss": False,
            "commitment_loss": 0.0,
            "codebook_per_depth": [0.0] * int(depth),
            "has_codebook_per_depth": False,
            "commitment_per_depth": [0.0] * int(depth),
            "residual_norm_per_depth": [0.0] * int(depth),
            "usage_per_depth": [0.0] * int(depth),
            "perplexity_per_depth": [0.0] * int(depth),
            "aggregate_usage": 0.0,
            "aggregate_perplexity": 0.0,
            "dead_codes": 0,
            "restarted_codes": 0,
            "projection_grad_norm": 0.0,
            "projection_grad_steps": 0,
        }
        for depth in rq_depth_list
    ]


def _as_float_list(value):
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().cpu().reshape(-1)]
    return [float(item) for item in value]


def _accumulate_rq_diagnostics(accumulator, diagnostics):
    for scale_index, scale_diagnostics in enumerate(diagnostics or []):
        if not scale_diagnostics or scale_index >= len(accumulator):
            continue
        target = accumulator[scale_index]
        target["batches"] += 1
        codebook = scale_diagnostics.get("codebook_loss")
        if codebook is not None:
            if isinstance(codebook, torch.Tensor):
                codebook = codebook.detach().item()
            target["codebook_loss"] += float(codebook)
            target["has_codebook_loss"] = True
        commitment = scale_diagnostics.get("commitment_loss", 0.0)
        if isinstance(commitment, torch.Tensor):
            commitment = commitment.detach().item()
        target["commitment_loss"] += float(commitment)
        codebook_per_depth = _as_float_list(
            scale_diagnostics.get(
                "codebook_per_depth",
                scale_diagnostics.get("codebook_loss_per_depth", []),
            )
        )
        if codebook_per_depth:
            target["has_codebook_per_depth"] = True
            for depth_index, value in enumerate(
                codebook_per_depth[:len(target["codebook_per_depth"])]
            ):
                target["codebook_per_depth"][depth_index] += value
        for key in (
            "commitment_per_depth",
            "residual_norm_per_depth",
            "usage_per_depth",
            "perplexity_per_depth",
        ):
            values = _as_float_list(scale_diagnostics.get(key, []))
            for depth_index, value in enumerate(values[:len(target[key])]):
                target[key][depth_index] += value
        target["aggregate_usage"] += float(scale_diagnostics.get("aggregate_usage", 0.0))
        target["aggregate_perplexity"] += float(
            scale_diagnostics.get("aggregate_perplexity", 0.0)
        )
        target["dead_codes"] += int(scale_diagnostics.get("dead_codes", 0))
        target["restarted_codes"] += int(scale_diagnostics.get("restarted_codes", 0))


def _accumulate_projection_gradient_norms(accumulator, model):
    for scale_index, value in enumerate(projection_gradient_norms(model)):
        if value is None or scale_index >= len(accumulator):
            continue
        accumulator[scale_index]["projection_grad_norm"] += value
        accumulator[scale_index]["projection_grad_steps"] += 1


def _average_rq_diagnostics(accumulator):
    averaged = []
    for scale in accumulator:
        count = max(scale["batches"], 1)
        grad_count = scale["projection_grad_steps"]
        averaged.append({
            "codebook_loss": (
                scale["codebook_loss"] / count
                if scale["has_codebook_loss"]
                else None
            ),
            "commitment_loss": scale["commitment_loss"] / count,
            "codebook_per_depth": (
                [value / count for value in scale["codebook_per_depth"]]
                if scale["has_codebook_per_depth"]
                else [None] * len(scale["codebook_per_depth"])
            ),
            "commitment_per_depth": [value / count for value in scale["commitment_per_depth"]],
            "residual_norm_per_depth": [value / count for value in scale["residual_norm_per_depth"]],
            "usage_per_depth": [value / count for value in scale["usage_per_depth"]],
            "perplexity_per_depth": [value / count for value in scale["perplexity_per_depth"]],
            "aggregate_usage": scale["aggregate_usage"] / count,
            "aggregate_perplexity": scale["aggregate_perplexity"] / count,
            "dead_codes": scale["dead_codes"],
            "restarted_codes": scale["restarted_codes"],
            "projection_grad_norm": (
                scale["projection_grad_norm"] / grad_count
                if grad_count
                else None
            ),
        })
    return averaged


def _write_rq_diagnostics(writer, prefix, diagnostics, step):
    for scale_index, scale in enumerate(diagnostics):
        base = f"RQ/{prefix}/Scale{scale_index}"
        codebook_loss = scale.get("codebook_loss")
        if codebook_loss is not None and math.isfinite(float(codebook_loss)):
            writer.add_scalar(f"{base}/CodebookLoss", codebook_loss, step)
        writer.add_scalar(f"{base}/Commitment", scale["commitment_loss"], step)
        writer.add_scalar(f"{base}/AggregateUsage", scale["aggregate_usage"], step)
        writer.add_scalar(f"{base}/AggregatePerplexity", scale["aggregate_perplexity"], step)
        writer.add_scalar(f"{base}/DeadCodes", scale["dead_codes"], step)
        writer.add_scalar(f"{base}/RestartedCodes", scale["restarted_codes"], step)
        projection_grad_norm = scale.get("projection_grad_norm")
        if projection_grad_norm is not None and math.isfinite(
            float(projection_grad_norm)
        ):
            writer.add_scalar(
                f"{base}/ProjectionGradNorm", projection_grad_norm, step
            )
        for depth_index in range(len(scale["commitment_per_depth"])):
            depth_base = f"{base}/Depth{depth_index}"
            codebook_per_depth = scale.get("codebook_per_depth", [])
            if (
                depth_index < len(codebook_per_depth)
                and codebook_per_depth[depth_index] is not None
                and math.isfinite(float(codebook_per_depth[depth_index]))
            ):
                writer.add_scalar(
                    f"{depth_base}/CodebookLoss",
                    codebook_per_depth[depth_index],
                    step,
                )
            writer.add_scalar(
                f"{depth_base}/Commitment", scale["commitment_per_depth"][depth_index], step
            )
            writer.add_scalar(
                f"{depth_base}/ResidualRMS", scale["residual_norm_per_depth"][depth_index], step
            )
            writer.add_scalar(
                f"{depth_base}/Usage", scale["usage_per_depth"][depth_index], step
            )
            writer.add_scalar(
                f"{depth_base}/Perplexity", scale["perplexity_per_depth"][depth_index], step
            )


@torch.no_grad()
def _batch_quality_metrics(real_images, reconstructed_images):
    if real_images.device != reconstructed_images.device:
        real_images = real_images.to(reconstructed_images.device, non_blocking=True)
    real_01 = ((real_images + 1.0) / 2.0).clamp(0.0, 1.0)
    reconstructed_01 = ((reconstructed_images + 1.0) / 2.0).clamp(0.0, 1.0)
    normalized_mse = torch.mean((real_01 - reconstructed_01) ** 2)
    psnr = 100.0 if normalized_mse.item() == 0 else -10.0 * math.log10(normalized_mse.item())
    ms_ssim = 1.0 - float(ms_ssim_loss(reconstructed_images, real_images).item())
    return psnr, ms_ssim


def load_pretrained_weights(model, pretrained_path, device):
    """加载预训练权重，仅加载兼容的参数（忽略形状不匹配的键）"""
    if not pretrained_path or not os.path.exists(pretrained_path):
        print(f"[Warning] 预训练权重路径不存在: {pretrained_path}，将从头训练")
        return 0
    pretrained_state = load_model_state_dict(pretrained_path, device)
    model_state = model.state_dict()
    loaded = 0
    skipped = 0
    for key, value in pretrained_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(model_state)
    print(f"[Info] 从预训练权重加载: {loaded} 个参数匹配, {skipped} 个跳过")
    return loaded


def main():
    cfg = Config()
    cfg.validate()
    setup_seed(42)
    run_id = os.environ.get(
        "EXPERIMENT_RUN_ID",
        f"{cfg.EXPERIMENT_NAME}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    metrics_path = cfg.METRICS_PATH

    device = torch.device(cfg.DEVICE)
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
    print(f"Start training on logical device: {device}")
    print(f"[Info] CUDA_VISIBLE_DEVICES: {visible_devices}")
    if device.type == "cuda":
        logical_index = device.index if device.index is not None else torch.cuda.current_device()
        visible_device_ids = [
            value.strip() for value in visible_devices.split(",") if value.strip()
        ]
        mapped_physical_device = (
            visible_device_ids[logical_index]
            if visible_devices != "<not set>" and logical_index < len(visible_device_ids)
            else str(logical_index)
        )
        print(
            f"[Info] GPU mapping: logical cuda:{logical_index} -> "
            f"physical GPU {mapped_physical_device} "
            f"({torch.cuda.get_device_name(logical_index)})"
        )
    print(f"[Info] Experiment name: {cfg.EXPERIMENT_NAME}")
    print(f"[Info] Experiment stage: {cfg.EXPERIMENT_STAGE}")
    print(f"[Info] Experiment run ID: {run_id}")
    print(f"[Info] Epoch metrics file: {metrics_path}")
    print(f"[Info] Checkpoint directory: {cfg.CHECKPOINT_DIR}")
    print(f"[Info] Resume checkpoint: {cfg.RESUME_PATH}")
    pretrained_path = os.environ.get("SIMVQ_PRETRAINED_CHECKPOINT", "")
    if pretrained_path:
        print(f"[Info] 预训练权重路径: {pretrained_path}")

    accumulation_steps = cfg.TOTAL_BATCH_SIZE // cfg.MICRO_BATCH_SIZE
    if accumulation_steps < 1:
        accumulation_steps = 1

    print("=" * 40)
    print(f"  - 总Batch Size：{cfg.TOTAL_BATCH_SIZE}")
    print(f"  - 小Batch Size：{cfg.MICRO_BATCH_SIZE}")
    print(f"  - 梯度累积步数: {accumulation_steps}")
    print(f"  - U-Net层数: {cfg.UNET_DEPTH}")
    print(f"  - 下采样步幅: {cfg.DOWNSAMPLE_STRIDES}")
    print(f"  - 总下采样倍率: {cfg.architecture_summary()['total_downsample']}x")
    print(f"  - 估算训练源端BPP: {cfg.ESTIMATED_SOURCE_BPP:.6f}")
    print(f"  - 估算训练离散比特/图: {cfg.ESTIMATED_SOURCE_BITS_PER_IMAGE:.0f}")
    print(f"  - 估算测试源端BPP: {cfg.ESTIMATED_TEST_SOURCE_BPP:.6f}")
    print(f"  - 估算测试传输压缩率(LDPC1/2+BPSK): {cfg.ESTIMATED_TEST_TRANSMISSION_RATIO:.8f}")
    print(f"  - 每层特征维度: {cfg.EMBEDDING_DIM_LIST}")
    print(f"  - 每层码本大小: {cfg.NUM_EMBEDDINGS_LIST}")
    print(f"  - 量化器类型: {cfg.QUANTIZER_TYPE}")
    if _uses_rq_monitoring(cfg.QUANTIZER_TYPE):
        print(f"  - RQ深度: {cfg.RQ_DEPTH_LIST}")
    if cfg.QUANTIZER_TYPE == "rq_ema":
        print(f"  - EMA decay/restart/shared: {cfg.RQ_EMA_DECAY} / "
              f"{cfg.RQ_RESTART_UNUSED_CODES} / {cfg.RQ_SHARED_CODEBOOK}")
    elif cfg.QUANTIZER_TYPE == "residual_simvq":
        print(
            "  - Residual-SimVQ: shared projected codebook, "
            f"shared={cfg.RQ_SHARED_CODEBOOK}, projection_lr={cfg.CODEBOOK_PROJ_LR}"
        )
    print(f"  - 逐层量化轴: {cfg.QUANTIZER_AXIS_LIST}")
    print(f"  - CVQ codeword shape: {cfg.CVQ_CODEWORD_SHAPES}")
    print(f"  - Nested channel dropout alpha: {cfg.NESTED_CHANNEL_DROPOUT_ALPHA}")
    print(f"  - 模型并行: {cfg.MODEL_PARALLEL}, encoder={cfg.ENCODER_DEVICE}, decoder={cfg.DECODER_DEVICE}")
    if cfg.QUANTIZER_TYPE == "none":
        print("  - 无量化直通模式: Encoder 特征直接输入 Decoder；离散信道与码本监控关闭")
    if cfg.QUANTIZER_TYPE == "vitvq_nocompress":
        print(f"  - ViTvq QBridge: {cfg.VITVQ_QBRIDGE_TYPE}, emb_nograd={cfg.VITVQ_EMB_NOGRAD}")
    print(f"  - 归一化/激活: {cfg.NORM_TYPE} / {cfg.ACTIVATION}")
    print(f"  - 编码器/解码器残差块数: {cfg.ENCODER_RES_BLOCKS} / {cfg.DECODER_RES_BLOCKS}")
    print(f"  - 级联下采样: {cfg.USE_CASCADE_DOWNSAMPLE}")
    print(f"  - 上采样方式: {cfg.UPSAMPLE_MODE}")
    print(f"  - Bottleneck Attention: {cfg.USE_BOTTLENECK_ATTENTION}, blocks={cfg.BOTTLENECK_ATTENTION_BLOCKS}")
    print(f"  - SwinIR Enhance: {cfg.USE_SWINIR_ENHANCE}, blocks={cfg.SWINIR_ENHANCE_BLOCKS}")
    print(f"  - Swin Backbone: {cfg.USE_SWIN_BACKBONE}")
    print(f"  - 重建损失: MSE*{cfg.MSE_LOSS_WEIGHT} + MS-SSIM*{cfg.MS_SSIM_LOSS_WEIGHT} + LPIPS*{cfg.LPIPS_LOSS_WEIGHT}")
    print(f"  - 信道课程: epoch<{cfg.CHANNEL_PROB_START_EPOCH}:0, "
          f"{cfg.CHANNEL_PROB_START_EPOCH}-{cfg.CHANNEL_PROB_END_EPOCH}:线性升至1, "
          f">={cfg.CHANNEL_PROB_END_EPOCH}:1")
    print(f"  - VQ损失层权重(初始): {cfg.LAYER_LOSS_WEIGHTS_INIT}")
    print(f"  - VQ损失层权重(最终): {cfg.LAYER_LOSS_WEIGHTS_FINAL}")
    print(f"  - 跳跃连接Dropout(初始): {cfg.SKIP_DROPOUT_P_INIT}")
    print(f"  - 跳跃连接Dropout(最终): {cfg.SKIP_DROPOUT_P_FINAL}")
    print(f"  - 调度阶段: Phase1[0,{int(cfg.PHASE1_END*cfg.NUM_EPOCHS)}), "
          f"Phase2[{int(cfg.PHASE1_END*cfg.NUM_EPOCHS)},{int(cfg.PHASE2_END*cfg.NUM_EPOCHS)}), "
          f"Phase3[{int(cfg.PHASE2_END*cfg.NUM_EPOCHS)},{cfg.NUM_EPOCHS}]")
    print("=" * 40)

    log_dir = os.path.join(cfg.LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    # 模型初始化 (无 RAQ 参数)
    deepsc_model = DeepSC(
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        num_downsample_blocks=cfg.NUM_DOWNSAMPLE_BLOCKS,
        base_channels=cfg.BASE_CHANNELS,
        num_embeddings_list=cfg.NUM_EMBEDDINGS_LIST,
        embedding_dim_list=cfg.EMBEDDING_DIM_LIST,
        commitment_cost=cfg.COMMITMENT_COST,
        device=device,
        strides=cfg.DOWNSAMPLE_STRIDES,
        skip_dropout_p=cfg.SKIP_DROPOUT_P_INIT,
        channel_coding_rate_train=cfg.CHANNEL_CODING_RATE_TRAIN,
        channel_coding_rate_val=cfg.CHANNEL_CODING_RATE_VAL,
        block_length=cfg.BLOCK_LENGTH,
        snr_range_db=cfg.SNR_RANGE_DB,
        norm_type=cfg.NORM_TYPE,
        norm_groups=cfg.GROUP_NORM_GROUPS,
        activation=cfg.ACTIVATION,
        encoder_res_blocks=cfg.ENCODER_RES_BLOCKS,
        decoder_res_blocks=cfg.DECODER_RES_BLOCKS,
        upsample_mode=cfg.UPSAMPLE_MODE,
        use_cascade_downsample=cfg.USE_CASCADE_DOWNSAMPLE,
        use_bottleneck_attention=cfg.USE_BOTTLENECK_ATTENTION,
        bottleneck_attention_blocks=cfg.BOTTLENECK_ATTENTION_BLOCKS,
        use_swinir_enhance=cfg.USE_SWINIR_ENHANCE,
        swinir_enhance_blocks=cfg.SWINIR_ENHANCE_BLOCKS,
        quantizer_type=cfg.QUANTIZER_TYPE,
        quantizer_axis_list=cfg.QUANTIZER_AXIS_LIST,
        cvq_codeword_shapes=cfg.CVQ_CODEWORD_SHAPES,
        nested_channel_dropout_alpha=cfg.NESTED_CHANNEL_DROPOUT_ALPHA,
        vitvq_qbridge_type=cfg.VITVQ_QBRIDGE_TYPE,
        vitvq_emb_nograd=cfg.VITVQ_EMB_NOGRAD,
        rq_depth_list=cfg.RQ_DEPTH_LIST,
        rq_ema_decay=cfg.RQ_EMA_DECAY,
        rq_restart_unused_codes=cfg.RQ_RESTART_UNUSED_CODES,
        rq_shared_codebook=cfg.RQ_SHARED_CODEBOOK,
    ).to(device)

    # 加载预训练权重（如果指定）
    if pretrained_path:
        load_pretrained_weights(deepsc_model, pretrained_path, device)

    if cfg.MODEL_PARALLEL:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("SIMVQ_MODEL_PARALLEL=1 requires at least two visible CUDA devices.")
        deepsc_model.enable_model_parallel(cfg.ENCODER_DEVICE, cfg.DECODER_DEVICE)
        print(
            f"[Info] Model parallel enabled: encoder/quantizer/channel on {cfg.ENCODER_DEVICE}, "
            f"decoder/enhance on {cfg.DECODER_DEVICE}"
        )

    # BN 动量调整
    if accumulation_steps > 1:
        current_momentum = 0.1
        new_momentum = 1 - (1 - current_momentum) ** (1 / accumulation_steps)
        print(f"[Info] Adjusting BN momentum from {current_momentum} to {new_momentum:.5f}")
        for module in deepsc_model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.momentum = new_momentum

    deepsc_loss_fn = DeepSCLoss(
        layer_weights=cfg.LAYER_LOSS_WEIGHTS_INIT,
        mse_weight=cfg.MSE_LOSS_WEIGHT,
        ms_ssim_weight=cfg.MS_SSIM_LOSS_WEIGHT,
        lpips_weight=cfg.LPIPS_LOSS_WEIGHT,
        quantization_weight=(cfg.COMMITMENT_COST if cfg.QUANTIZER_TYPE == "rq_ema" else 1.0),
    ).to(deepsc_model.decoder_device if cfg.MODEL_PARALLEL else device)

    if cfg.QUANTIZER_TYPE == "rq_ema":
        trainable_ema_parameters = [
            name for name, parameter in deepsc_model.named_parameters()
            if "vector_quantizers" in name and "codebooks" in name and parameter.requires_grad
        ]
        if trainable_ema_parameters:
            raise RuntimeError(
                "EMA codebook parameters must not require gradients: "
                + ", ".join(trainable_ema_parameters)
            )

    optimizer_groups, optimizer_audit = build_optimizer_parameter_groups(
        deepsc_model,
        cfg.LEARNING_RATE_G,
        cfg.CODEBOOK_PROJ_LR,
        quantizer_type=cfg.QUANTIZER_TYPE,
    )
    optimizer_g = optim.Adam(optimizer_groups, betas=cfg.BETAS)
    print(
        "[Info] 优化器参数分组: "
        f"普通参数 {len(optimizer_audit['other_names'])} 个 "
        f"(lr={cfg.LEARNING_RATE_G}), "
        f"码本变换层 {len(optimizer_audit['projection_names'])} 个 "
        f"(lr={cfg.CODEBOOK_PROJ_LR})"
    )
    if cfg.QUANTIZER_TYPE == "residual_simvq":
        print(
            "[Info] Residual-SimVQ optimizer audit: projections="
            f"{optimizer_audit['projection_names']}, frozen_embeddings="
            f"{optimizer_audit['frozen_embedding_names']}"
        )
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=100, gamma=0.5)

    # 断点续训
    start_epoch = 0
    best_val_loss = float('inf')
    if cfg.RESUME and os.path.exists(cfg.RESUME_PATH):
        print(f"Loading checkpoint: {cfg.RESUME_PATH}")
        checkpoint, resume_state_dict, _ = load_checkpoint_payload(cfg.RESUME_PATH, device)
        deepsc_model.load_state_dict(resume_state_dict)
        optimizer_g.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler_g.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        rng_state = checkpoint.get('rng_state')
        if rng_state is not None:
            torch.set_rng_state(rng_state.cpu())
        cuda_rng_state = checkpoint.get('cuda_rng_state')
        if torch.cuda.is_available() and cuda_rng_state is not None:
            cuda_states = [s.cpu() if isinstance(s, torch.Tensor) else s for s in cuda_rng_state]
            num_current_gpus = torch.cuda.device_count()
            if len(cuda_states) > num_current_gpus:
                print(f"[Warning] checkpoint 保存了 {len(cuda_states)} 个 GPU 的 RNG 状态，"
                      f"但当前只有 {num_current_gpus} 个 GPU，仅恢复前 {num_current_gpus} 个")
                cuda_states = cuda_states[:num_current_gpus]
            torch.cuda.set_rng_state_all(cuda_states)
        print(f"--> 成功恢复检查点，从 Epoch {start_epoch} 继续。")

    # 数据加载
    train_dataloader = get_dataloader(
        root_dir=cfg.TRAIN_DATASET_PATH,
        batch_size=cfg.MICRO_BATCH_SIZE,
        shuffle=True,
        mode='train',
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    val_dataloader = get_dataloader(
        root_dir=cfg.VAL_DATASET_PATH,
        batch_size=cfg.MICRO_BATCH_SIZE,
        shuffle=False,
        mode='val',
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    global_step = start_epoch * len(train_dataloader)

    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        # === 调度：更新 Dropout 概率和损失权重 ===
        dropout_p, loss_weights, channel_prob, phase_desc = compute_schedule(epoch, cfg.NUM_EPOCHS, cfg)
        deepsc_model.semantic_decoder.set_skip_dropout_p(dropout_p)
        deepsc_model.set_channel_prob(channel_prob)
        deepsc_loss_fn.set_layer_weights(loss_weights)

        deepsc_model.train()

        total_recon_losses = 0
        total_vq_losses = 0
        train_rq_accumulator = _empty_rq_diagnostic_accumulator(cfg.RQ_DEPTH_LIST)
        channel_used_batches = 0
        channel_snr_sum = 0.0
        channel_snr_count = 0

        optimizer_g.zero_grad()
        steps_per_epoch = len(train_dataloader)

        for i, real_images in enumerate(train_dataloader):
            real_images = real_images.to(device, non_blocking=True)

            do_step = ((i + 1) % accumulation_steps == 0) or ((i + 1) == len(train_dataloader))

            out = deepsc_model.forward_train(real_images)

            recon_loss, vq_loss = deepsc_loss_fn(
                real_images,
                out["reconstructed_images"],
                out["vq_losses"]
            )

            loss = (recon_loss + vq_loss) / accumulation_steps
            loss.backward()

            total_recon_losses += recon_loss.item()
            total_vq_losses += vq_loss.item()
            if _uses_rq_monitoring(cfg.QUANTIZER_TYPE):
                _accumulate_rq_diagnostics(
                    train_rq_accumulator, out.get("quantizer_diagnostics", [])
                )

            current_snr = out.get("current_snr")
            if out.get("channel_used"):
                channel_used_batches += 1
            if current_snr is not None:
                channel_snr_sum += float(current_snr)
                channel_snr_count += 1
            snr_desc = "clean" if current_snr is None else f"{current_snr:.2f} dB"

            if do_step:
                if cfg.QUANTIZER_TYPE == "residual_simvq":
                    _accumulate_projection_gradient_norms(
                        train_rq_accumulator, deepsc_model
                    )
                torch.nn.utils.clip_grad_norm_(deepsc_model.parameters(), max_norm=1.0)
                optimizer_g.step()
                optimizer_g.zero_grad()

            if i % (accumulation_steps * 10) == 0:
                print(f"Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}], Step [{i + 1}/{steps_per_epoch}], "
                      f"Recon: {recon_loss.item():.4f}, VQ: {vq_loss.item():.4f}, "
                      f"ChannelProb: {channel_prob:.2f}, SNR: {snr_desc}")
                if current_snr is not None:
                    writer.add_scalar("Train/SNR", current_snr, global_step)
                writer.add_scalar("Train/Loss_Step", recon_loss.item() + vq_loss.item(), global_step)
                writer.add_scalar("Train/ChannelProb", channel_prob, global_step)
                train_psnr, train_ms_ssim = _batch_quality_metrics(
                    real_images, out["reconstructed_images"]
                )
                writer.add_scalar("Quality/Train/PSNR", train_psnr, global_step)
                writer.add_scalar("Quality/Train/MS-SSIM", train_ms_ssim, global_step)

            global_step += 1

        scheduler_g.step()

        avg_recon = total_recon_losses / steps_per_epoch
        avg_vq = total_vq_losses / steps_per_epoch
        avg_channel_snr = (
            channel_snr_sum / channel_snr_count if channel_snr_count else None
        )
        actual_channel_fraction = channel_used_batches / max(steps_per_epoch, 1)
        train_rq_diagnostics = _average_rq_diagnostics(train_rq_accumulator)

        print(f"[{phase_desc}] Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}], "
              f"Recon: {avg_recon:.4f}, VQ: {avg_vq:.4f}, "
              f"ChannelProb: {channel_prob:.2f}, "
              f"Dropout: {[f'{p:.2f}' for p in dropout_p]}, "
              f"LossW: {[f'{w:.1f}' for w in loss_weights]}")

        writer.add_scalar("Loss/Train/Recon", avg_recon, epoch)
        writer.add_scalar("Loss/Train/VQ", avg_vq, epoch)
        writer.add_scalar("Loss/Train/Total", avg_recon + avg_vq, epoch)
        writer.add_scalar("Schedule/ChannelProb", channel_prob, epoch)
        writer.add_scalar("Channel/Train/UsedFraction", actual_channel_fraction, epoch)
        if avg_channel_snr is not None:
            writer.add_scalar("Channel/Train/AverageSNR", avg_channel_snr, epoch)
        writer.add_scalar("Rate/BitsPerImage", cfg.ESTIMATED_SOURCE_BITS_PER_IMAGE, epoch)
        writer.add_scalar("Rate/SourceBPP", cfg.ESTIMATED_SOURCE_BPP, epoch)
        writer.add_scalar(
            "Rate/TransmissionRatio_LDPC12_BPSK_RGB",
            cfg.ESTIMATED_SOURCE_BPP / (cfg.CHANNEL_CODING_RATE_VAL * 1 * 3),
            epoch,
        )
        if _uses_rq_monitoring(cfg.QUANTIZER_TYPE):
            _write_rq_diagnostics(writer, "Train", train_rq_diagnostics, epoch)
        # 记录调度参数
        for li, p in enumerate(dropout_p):
            writer.add_scalar(f"Schedule/Dropout_L{li}", p, epoch)
        for li, w in enumerate(loss_weights):
            writer.add_scalar(f"Schedule/LossWeight_L{li}", w, epoch)

        # 验证
        deepsc_model.eval()
        val_loss_sum = 0
        val_vq_loss_sum = 0
        val_psnr_sum = 0.0
        val_ms_ssim_sum = 0.0
        val_rq_accumulator = _empty_rq_diagnostic_accumulator(cfg.RQ_DEPTH_LIST)
        with torch.no_grad():
            for real_images in val_dataloader:
                real_images = real_images.to(device, non_blocking=True)
                out = deepsc_model.forward_val(real_images)
                recon_loss_val, vq_loss_val = deepsc_loss_fn(
                    real_images,
                    out["reconstructed_images"],
                    out["vq_losses"]
                )
                val_loss_sum += recon_loss_val.item()
                val_vq_loss_sum += vq_loss_val.item()
                batch_psnr, batch_ms_ssim = _batch_quality_metrics(
                    real_images, out["reconstructed_images"]
                )
                val_psnr_sum += batch_psnr
                val_ms_ssim_sum += batch_ms_ssim
                if _uses_rq_monitoring(cfg.QUANTIZER_TYPE):
                    _accumulate_rq_diagnostics(
                        val_rq_accumulator, out.get("quantizer_diagnostics", [])
                    )

        avg_val_loss = val_loss_sum / len(val_dataloader)
        avg_val_vq = val_vq_loss_sum / len(val_dataloader)
        avg_val_psnr = val_psnr_sum / len(val_dataloader)
        avg_val_ms_ssim = val_ms_ssim_sum / len(val_dataloader)
        val_rq_diagnostics = _average_rq_diagnostics(val_rq_accumulator)
        print(
            f"[VAL] Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}], "
            f"Recon: {avg_val_loss:.4f}, {cfg.QUANTIZER_TYPE}: {avg_val_vq:.4f}, "
            f"PSNR: {avg_val_psnr:.3f}, MS-SSIM: {avg_val_ms_ssim:.4f}"
        )
        writer.add_scalar("Loss/Val/Recon", avg_val_loss, epoch)
        writer.add_scalar("Loss/Val/VQ", avg_val_vq, epoch)
        writer.add_scalar("Quality/Val/PSNR", avg_val_psnr, epoch)
        writer.add_scalar("Quality/Val/MS-SSIM", avg_val_ms_ssim, epoch)
        if _uses_rq_monitoring(cfg.QUANTIZER_TYPE):
            _write_rq_diagnostics(writer, "Val", val_rq_diagnostics, epoch)

        # === 每 N 个 epoch 统计一次码本利用率 ===
        codebook_monitor_interval = 10
        if cfg.QUANTIZER_TYPE == "none" and (epoch + 1) % codebook_monitor_interval == 0:
            print(f"\n[Codebook Utilization] Epoch {epoch + 1} - 无量化模式，跳过码本统计")
        elif (epoch + 1) % codebook_monitor_interval == 0:
            print(f"\n[Codebook Utilization] Epoch {epoch + 1} - 统计中...")
            cb_stats = compute_codebook_utilization(
                deepsc_model,
                val_dataloader,
                max_batches=20,
                device=device
            )
            if cfg.QUANTIZER_TYPE == "residual_simvq":
                for scale_index, scale_stats in enumerate(cb_stats["src"]):
                    scale_stats["projection_grad_norm"] = (
                        train_rq_diagnostics[scale_index][
                            "projection_grad_norm"
                        ]
                    )
            print_codebook_utilization(cb_stats, cfg.NUM_EMBEDDINGS_LIST)
            write_codebook_tensorboard(writer, cb_stats, epoch)
            append_codebook_records(
                cfg.CODEBOOK_METRICS_PATH,
                run_id,
                epoch + 1,
                cb_stats,
                cfg.NUM_EMBEDDINGS_LIST,
            )

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            best_checkpoint = build_checkpoint_payload(
                deepsc_model,
                cfg,
                epoch=epoch,
                best_val_loss=best_val_loss,
            )
            torch.save(
                best_checkpoint,
                os.path.join(cfg.CHECKPOINT_DIR, "best_vq_deepsc.pth"),
            )
            print(f"Saved Best Model with Val Loss: {best_val_loss:.4f}")

        epoch_record = {
            "run_id": run_id,
            "epoch": epoch + 1,
            "train_recon": f"{avg_recon:.8f}",
            "train_vq": f"{avg_vq:.8f}",
            "train_total": f"{avg_recon + avg_vq:.8f}",
            "val_recon": f"{avg_val_loss:.8f}",
            "val_vq": f"{avg_val_vq:.8f}",
            "val_psnr": f"{avg_val_psnr:.8f}",
            "val_ms_ssim": f"{avg_val_ms_ssim:.8f}",
            "best_val_recon": f"{best_val_loss:.8f}",
            "is_best": int(is_best),
            "phase": phase_desc,
            "channel_prob": f"{channel_prob:.6f}",
            "channel_usage_ratio": f"{actual_channel_fraction:.8f}",
            "mean_channel_snr": (
                "" if avg_channel_snr is None else f"{avg_channel_snr:.8f}"
            ),
            "source_bits_per_image": f"{cfg.ESTIMATED_SOURCE_BITS_PER_IMAGE:.0f}",
            "source_bpp": f"{cfg.ESTIMATED_SOURCE_BPP:.10f}",
            "transmission_ratio": (
                f"{cfg.ESTIMATED_SOURCE_BPP / (cfg.CHANNEL_CODING_RATE_VAL * 1 * 3):.10f}"
            ),
            "learning_rate": f"{optimizer_g.param_groups[0]['lr']:.10g}",
        }
        if _uses_rq_monitoring(cfg.QUANTIZER_TYPE):
            epoch_record.update(
                rq_epoch_metric_fields("train", train_rq_diagnostics)
            )
            epoch_record.update(
                rq_epoch_metric_fields("val", val_rq_diagnostics)
            )
        append_epoch_record(metrics_path, epoch_record)

        # Save resume state after updating the best metric so resumed runs use
        # the same best-model threshold that produced best_vq_deepsc.pth.
        checkpoint = build_checkpoint_payload(
            deepsc_model,
            cfg,
            epoch=epoch,
            optimizer_state_dict=optimizer_g.state_dict(),
            scheduler_state_dict=scheduler_g.state_dict(),
            best_val_loss=best_val_loss,
            rng_state=torch.get_rng_state(),
            cuda_rng_state=(
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        )
        torch.save(checkpoint, cfg.RESUME_PATH)

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
