import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from config import Config
from models.deepsc import DeepSC
from losses.deepsc_loss import DeepSCLoss
from data.datasets import get_dataloader
from monitoring.codebook import (compute_codebook_utilization, print_codebook_utilization, write_codebook_tensorboard,)
from training.schedules import compute_schedule
from utils.experiment_io import append_codebook_records, append_epoch_record
from utils.reproducibility import setup_seed
from utils.checkpoint_utils import load_model_state_dict


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
    print(f"  - 估算测试源端BPP: {cfg.ESTIMATED_TEST_SOURCE_BPP:.6f}")
    print(f"  - 估算测试传输压缩率(LDPC1/2+BPSK): {cfg.ESTIMATED_TEST_TRANSMISSION_RATIO:.8f}")
    print(f"  - 每层特征维度: {cfg.EMBEDDING_DIM_LIST}")
    print(f"  - 每层码本大小: {cfg.NUM_EMBEDDINGS_LIST}")
    print(f"  - 量化器类型: {cfg.QUANTIZER_TYPE}")
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

    log_dir = os.path.join(cfg.LOG_DIR, datetime.now().strftime("%Y%M%D-%H%M%S"))
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
    ).to(deepsc_model.decoder_device if cfg.MODEL_PARALLEL else device)

    # SimVQ projection and ViTvq QBridge transformations use the codebook LR.
    proj_params = []
    other_params = []
    for name, param in deepsc_model.named_parameters():
        if not param.requires_grad:
            continue
        if "codebook.proj" in name or ".qbridge." in name:
            proj_params.append(param)
        else:
            other_params.append(param)

    optimizer_groups = [{"params": other_params, "lr": cfg.LEARNING_RATE_G}]
    if proj_params:
        optimizer_groups.append({"params": proj_params, "lr": cfg.CODEBOOK_PROJ_LR})
    optimizer_g = optim.Adam(optimizer_groups, betas=cfg.BETAS)
    print(f"[Info] 优化器参数分组: 普通参数 {len(other_params)} 个 (lr={cfg.LEARNING_RATE_G}), "
          f"码本变换层 {len(proj_params)} 个 (lr={cfg.CODEBOOK_PROJ_LR})")
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=100, gamma=0.5)

    # 断点续训
    start_epoch = 0
    best_val_loss = float('inf')
    if cfg.RESUME and os.path.exists(cfg.RESUME_PATH):
        print(f"Loading checkpoint: {cfg.RESUME_PATH}")
        checkpoint = torch.load(cfg.RESUME_PATH, map_location=device)
        deepsc_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler_g.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        torch.set_rng_state(checkpoint['rng_state'].cpu())
        if torch.cuda.is_available() and checkpoint['cuda_rng_state'] is not None:
            cuda_states = [s.cpu() if isinstance(s, torch.Tensor) else s for s in checkpoint['cuda_rng_state']]
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

            current_snr = out.get("current_snr")
            snr_desc = "clean" if current_snr is None else f"{current_snr:.2f} dB"

            if do_step:
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

            global_step += 1

        scheduler_g.step()

        avg_recon = total_recon_losses / steps_per_epoch
        avg_vq = total_vq_losses / steps_per_epoch

        print(f"[{phase_desc}] Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}], "
              f"Recon: {avg_recon:.4f}, VQ: {avg_vq:.4f}, "
              f"ChannelProb: {channel_prob:.2f}, "
              f"Dropout: {[f'{p:.2f}' for p in dropout_p]}, "
              f"LossW: {[f'{w:.1f}' for w in loss_weights]}")

        writer.add_scalar("Loss/Train/Recon", avg_recon, epoch)
        writer.add_scalar("Loss/Train/VQ", avg_vq, epoch)
        writer.add_scalar("Loss/Train/Total", avg_recon + avg_vq, epoch)
        writer.add_scalar("Schedule/ChannelProb", channel_prob, epoch)
        # 记录调度参数
        for li, p in enumerate(dropout_p):
            writer.add_scalar(f"Schedule/Dropout_L{li}", p, epoch)
        for li, w in enumerate(loss_weights):
            writer.add_scalar(f"Schedule/LossWeight_L{li}", w, epoch)

        # 验证
        deepsc_model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for real_images in val_dataloader:
                real_images = real_images.to(device, non_blocking=True)
                out = deepsc_model.forward_val(real_images)
                recon_loss_val, _ = deepsc_loss_fn(
                    real_images,
                    out["reconstructed_images"],
                    out["vq_losses"]
                )
                val_loss_sum += recon_loss_val.item()

        avg_val_loss = val_loss_sum / len(val_dataloader)
        print(f"[VAL] Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}], Val Recon Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/Val/Recon", avg_val_loss, epoch)

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
            torch.save(deepsc_model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "best_vq_deepsc.pth"))
            print(f"Saved Best Model with Val Loss: {best_val_loss:.4f}")

        append_epoch_record(metrics_path, {
            "run_id": run_id,
            "epoch": epoch + 1,
            "train_recon": f"{avg_recon:.8f}",
            "train_vq": f"{avg_vq:.8f}",
            "val_recon": f"{avg_val_loss:.8f}",
            "best_val_recon": f"{best_val_loss:.8f}",
            "is_best": int(is_best),
            "phase": phase_desc,
            "channel_prob": f"{channel_prob:.6f}",
            "learning_rate": f"{optimizer_g.param_groups[0]['lr']:.10g}",
        })

        # Save resume state after updating the best metric so resumed runs use
        # the same best-model threshold that produced best_vq_deepsc.pth.
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': deepsc_model.state_dict(),
            'optimizer_state_dict': optimizer_g.state_dict(),
            'scheduler_state_dict': scheduler_g.state_dict(),
            'best_val_loss': best_val_loss,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
        torch.save(checkpoint, cfg.RESUME_PATH)

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
