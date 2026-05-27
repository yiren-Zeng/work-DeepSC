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
    print(f"  - 跳跃连接Dropout概率: {cfg.SKIP_DROPOUT_P}")
    print(f"  - 无量化模块 + 无噪声训练（消融实验）")
    print("=" * 40)

    log_dir = os.path.join(cfg.LOG_DIR, datetime.now().strftime("%Y%M%D-%H%M%S"))
    writer = SummaryWriter(log_dir)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    # 模型初始化（无量化模块）
    deepsc_model = DeepSC(
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        num_downsample_blocks=cfg.NUM_DOWNSAMPLE_BLOCKS,
        base_channels=cfg.BASE_CHANNELS,
        embedding_dim_list=cfg.EMBEDDING_DIM_LIST,
        device=device
    ).to(device)

    # GN 统计
    bn_count = sum(1 for m in deepsc_model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    gn_count = sum(1 for m in deepsc_model.modules() if isinstance(m, torch.nn.GroupNorm))
    print(f"[Info] BN 层数: {bn_count}, GN 层数: {gn_count}")
    if bn_count > 0:
        print(f"[Warning] 仍存在 {bn_count} 个 BatchNorm2d 层未替换！")

    deepsc_loss_fn = DeepSCLoss().to(device)

    # 优化器（无码本投影层分组）
    optimizer_g = optim.Adam(
        deepsc_model.parameters(),
        lr=cfg.LEARNING_RATE_G,
        betas=cfg.BETAS
    )
    print(f"[Info] 优化器: 所有参数统一 lr={cfg.LEARNING_RATE_G}")
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
        deepsc_model.train()

        total_recon_losses = 0

        optimizer_g.zero_grad()
        steps_per_epoch = len(train_dataloader)

        for i, real_images in enumerate(train_dataloader):
            real_images = real_images.to(device, non_blocking=True)

            do_step = ((i + 1) % accumulation_steps == 0) or ((i + 1) == len(train_dataloader))

            out = deepsc_model.forward_train(real_images)

            recon_loss = deepsc_loss_fn(
                real_images,
                out["reconstructed_images"]
            )

            loss = recon_loss / accumulation_steps
            loss.backward()

            total_recon_losses += recon_loss.item()

            if do_step:
                torch.nn.utils.clip_grad_norm_(deepsc_model.parameters(), max_norm=1.0)
                optimizer_g.step()
                optimizer_g.zero_grad()

            if i % (accumulation_steps * 10) == 0:
                print(f"Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}], Step [{i + 1}/{steps_per_epoch}], "
                      f"Recon: {recon_loss.item():.4f}")
                writer.add_scalar("Train/Loss_Step", recon_loss.item(), global_step)

            global_step += 1

        scheduler_g.step()

        avg_recon = total_recon_losses / steps_per_epoch

        writer.add_scalar("Loss/Train/Recon", avg_recon, epoch)

        # 验证
        deepsc_model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for real_images in val_dataloader:
                real_images = real_images.to(device, non_blocking=True)
                out = deepsc_model.forward_val(real_images)
                recon_loss_val = deepsc_loss_fn(
                    real_images,
                    out["reconstructed_images"]
                )
                val_loss_sum += recon_loss_val.item()

        avg_val_loss = val_loss_sum / len(val_dataloader)
        print(f"[VAL] Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}], Val Recon Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/Val/Recon", avg_val_loss, epoch)

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
            torch.save(deepsc_model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "best_deepsc.pth"))
            print(f"Saved Best Model with Val Loss: {best_val_loss:.4f}")

        if (epoch + 1) % cfg.SAVE_INTERVAL == 0:
            torch.save(deepsc_model.state_dict(),
                       os.path.join(cfg.CHECKPOINT_DIR, f"deepsc_epoch_{epoch + 1}.pth"))

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
