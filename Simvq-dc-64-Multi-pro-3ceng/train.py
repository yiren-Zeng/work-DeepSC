import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import random
import numpy as np
from datetime import datetime
from config import Config
from models.deepsc import DeepSC
from losses.deepsc_loss import DeepSCLoss
from data.datasets import get_dataloader


def compute_schedule(epoch, num_epochs, cfg):
    """
    根据当前 epoch 计算 Dropout 概率和损失权重
    返回 (dropout_p_list, loss_weights, phase_desc)
    """
    phase1_end = int(cfg.PHASE1_END * num_epochs)  # 240
    phase2_end = int(cfg.PHASE2_END * num_epochs)  # 360

    if epoch < phase1_end:
        # 阶段1：深层强制拓荒期
        dropout_p = list(cfg.SKIP_DROPOUT_P_INIT)
        loss_weights = list(cfg.LAYER_LOSS_WEIGHTS_INIT)
        phase_desc = "Phase1-拓荒"
    elif epoch < phase2_end:
        # 阶段2：全层复苏与退火期（线性插值）
        progress = (epoch - phase1_end) / (phase2_end - phase1_end)
        dropout_p = [init * (1 - progress) + final * progress
                     for init, final in zip(cfg.SKIP_DROPOUT_P_INIT, cfg.SKIP_DROPOUT_P_FINAL)]
        loss_weights = [init + (final - init) * progress
                        for init, final in zip(cfg.LAYER_LOSS_WEIGHTS_INIT, cfg.LAYER_LOSS_WEIGHTS_FINAL)]
        phase_desc = f"Phase2-退火({progress:.0%})"
    else:
        # 阶段3：无损高保真微调期
        dropout_p = list(cfg.SKIP_DROPOUT_P_FINAL)
        loss_weights = list(cfg.LAYER_LOSS_WEIGHTS_FINAL)
        phase_desc = "Phase3-微调"

    return dropout_p, loss_weights, phase_desc


def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[Info] Random seed set to {seed}")


def main():
    cfg = Config()
    setup_seed(42)

    device = torch.device(cfg.DEVICE)
    print(f"Start training on {device}")

    accumulation_steps = cfg.TOTAL_BATCH_SIZE // cfg.MICRO_BATCH_SIZE
    if accumulation_steps < 1:
        accumulation_steps = 1

    print("=" * 40)
    print(f"  - 总Batch Size：{cfg.TOTAL_BATCH_SIZE}")
    print(f"  - 小Batch Size：{cfg.MICRO_BATCH_SIZE}")
    print(f"  - 梯度累积步数: {accumulation_steps}")
    print(f"  - 下采样步幅: {cfg.DOWNSAMPLE_STRIDES}")
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
        strides=cfg.DOWNSAMPLE_STRIDES
    ).to(device)

    # BN 动量调整
    if accumulation_steps > 1:
        current_momentum = 0.1
        new_momentum = 1 - (1 - current_momentum) ** (1 / accumulation_steps)
        print(f"[Info] Adjusting BN momentum from {current_momentum} to {new_momentum:.5f}")
        for module in deepsc_model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.momentum = new_momentum

    deepsc_loss_fn = DeepSCLoss(layer_weights=cfg.LAYER_LOSS_WEIGHTS_INIT).to(device)

    # 参数分组：SimVQ 码本投影层单独学习率
    proj_params = []
    other_params = []
    for name, param in deepsc_model.named_parameters():
        if not param.requires_grad:
            continue
        if "codebook.proj" in name:
            proj_params.append(param)
        else:
            other_params.append(param)

    optimizer_g = optim.Adam([
        {"params": other_params, "lr": cfg.LEARNING_RATE_G},
        {"params": proj_params,  "lr": cfg.CODEBOOK_PROJ_LR},
    ], betas=cfg.BETAS)
    print(f"[Info] 优化器参数分组: 普通参数 {len(other_params)} 个 (lr={cfg.LEARNING_RATE_G}), "
          f"码本投影层 {len(proj_params)} 个 (lr={cfg.CODEBOOK_PROJ_LR})")
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
        dropout_p, loss_weights, phase_desc = compute_schedule(epoch, cfg.NUM_EPOCHS, cfg)
        deepsc_model.semantic_decoder.set_skip_dropout_p(dropout_p)
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

            current_snr = out.get("current_snr", 0.0)

            if do_step:
                torch.nn.utils.clip_grad_norm_(deepsc_model.parameters(), max_norm=1.0)
                optimizer_g.step()
                optimizer_g.zero_grad()

            if i % (accumulation_steps * 10) == 0:
                print(f"Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}], Step [{i + 1}/{steps_per_epoch}], "
                      f"Recon: {recon_loss.item():.4f}, VQ: {vq_loss.item():.4f}, "
                      f"SNR: {current_snr:.2f} dB")
                writer.add_scalar("Train/SNR", current_snr, global_step)
                writer.add_scalar("Train/Loss_Step", recon_loss.item() + vq_loss.item(), global_step)

            global_step += 1

        scheduler_g.step()

        avg_recon = total_recon_losses / steps_per_epoch
        avg_vq = total_vq_losses / steps_per_epoch

        print(f"[{phase_desc}] Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}], "
              f"Recon: {avg_recon:.4f}, VQ: {avg_vq:.4f}, "
              f"Dropout: {[f'{p:.2f}' for p in dropout_p]}, "
              f"LossW: {[f'{w:.1f}' for w in loss_weights]}")

        writer.add_scalar("Loss/Train/Recon", avg_recon, epoch)
        writer.add_scalar("Loss/Train/VQ", avg_vq, epoch)
        writer.add_scalar("Loss/Train/Total", avg_recon + avg_vq, epoch)
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
        if (epoch + 1) % codebook_monitor_interval == 0:
            print(f"\n[Codebook Utilization] Epoch {epoch + 1} - 统计中...")
            cb_stats = deepsc_model.compute_codebook_utilization(
                val_dataloader,
                max_batches=20,
                device=device
            )
            # 打印到控制台
            DeepSC.print_codebook_utilization(cb_stats, cfg.NUM_EMBEDDINGS_LIST)

            # 记录到 TensorBoard
            for i in range(len(cb_stats['src'])):
                writer.add_scalar(f"Codebook/L{i}/ActiveRatio", cb_stats['src'][i]['active_ratio'], epoch)
                writer.add_scalar(f"Codebook/L{i}/Perplexity", cb_stats['src'][i]['perplexity'], epoch)
                writer.add_scalar(f"Codebook/L{i}/MinL2Dist", cb_stats['src'][i]['min_l2_dist'], epoch)
                writer.add_scalar(f"Codebook/L{i}/CollapseRatio", cb_stats['src'][i]['collapse_ratio'], epoch)

        # 保存模型
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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(deepsc_model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "best_vq_deepsc.pth"))
            print(f"Saved Best Model with Val Loss: {best_val_loss:.4f}")

        if (epoch + 1) % cfg.SAVE_INTERVAL == 0:
            torch.save(deepsc_model.state_dict(),
                       os.path.join(cfg.CHECKPOINT_DIR, f"vq_deepsc_epoch_{epoch + 1}.pth"))

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
