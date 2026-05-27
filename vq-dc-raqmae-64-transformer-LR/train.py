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
from utils.math_utils import sample_trg



# === 1. 固定随机种子 (解决结果随机性问题) ===
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

    # 初始化配置与种子
    cfg = Config()
    setup_seed(42)

    device = torch.device(cfg.DEVICE)
    print(f"Start training on {device}")

    # 2. 计算梯度累积步数
    # Accumulation = Target / Micro
    accumulation_steps = cfg.TOTAL_BATCH_SIZE //cfg.MICRO_BATCH_SIZE
    if accumulation_steps < 1: accumulation_steps = 1

    print("=" * 40)
    print(f"  - 总Batch Size：{cfg.TOTAL_BATCH_SIZE}")
    print(f"  - 小Batch Size：{cfg.MICRO_BATCH_SIZE}")
    print(f"  - 梯度累积步数: {accumulation_steps}")
    print("=" * 40)

    # 3. 初始化 Tensorboard
    log_dir = os.path.join(cfg.LOG_DIR,datetime.now().strftime("%Y%M%D-%H%M%S"))
    writer = SummaryWriter(log_dir)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    # 4. 模型初始化
    deepsc_model = DeepSC(
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        num_downsample_blocks=cfg.NUM_DOWNSAMPLE_BLOCKS,
        base_channels=cfg.BASE_CHANNELS,
        num_embeddings_list=cfg.NUM_EMBEDDINGS_LIST,
        embedding_dim_list=cfg.EMBEDDING_DIM_LIST,
        commitment_cost=cfg.COMMITMENT_COST,
        raq_min_trg=cfg.RAQ_MIN_TRG,
        raq_max_trg=cfg.RAQ_MAX_TRG,
        device=device
    ).to(device)

    # === 【修改 1】动态调整 BN Momentum 以适配梯度累积 ===
    # 原理：让累积版跑 N 次的衰减量 = 基础版跑 1 次的衰减量
    # 公式：(1 - m_large) = (1 - m_small) ^ accumulation_steps
    if accumulation_steps > 1:
        current_momentum = 0.1  # PyTorch 默认值
        # 计算等效的 momentum
        new_momentum = 1 - (1 - current_momentum) ** (1 / accumulation_steps)
        print(f"[Info] Adjusting BN momentum from {current_momentum} to {new_momentum:.5f} for accumulation.")

        # 遍历模型所有层，修改 BN 的 momentum
        for module in deepsc_model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.momentum = new_momentum


    deepsc_loss_fn = DeepSCLoss().to(device)

    optimizer_g = optim.Adam(
        deepsc_model.parameters(),
        lr=cfg.LEARNING_RATE_G,
        betas=cfg.BETAS
    )

    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=100, gamma=0.5)

    # 5. 断点续训
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
        print(f"--> 成功恢复检查点: {cfg.RESUME_PATH}, 从 Epoch {start_epoch} 继续。")

    # 6. 数据加载
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

    # === 7. 训练主循环 ===
    for epoch in range(start_epoch, cfg.NUM_EPOCHS):

        deepsc_model.train()

        total_mae_losses = 0
        total_vq_raq_losses = 0

        optimizer_g.zero_grad() # 确保每个epoch开始前清空梯度
        steps_per_epoch = len(train_dataloader)

        for i, real_images in enumerate(train_dataloader):

            real_images = real_images.to(device, non_blocking=True)

            # =======================================================
            # 【新增逻辑】: 判断是不是累积周期的“第一步” (Start)
            # 作用：在周期的开头生成随机数，并存起来
            # 注意：这里是 i % steps == 0 (0, 4, 8...)
            # =======================================================
            is_accumulation_start = (i % accumulation_steps == 0)

            if is_accumulation_start:
                # 这是一个新周期，生成一个新的随机列表，并保存到外部变量 current_trg_list 中
                current_trg_list = []
                for _ in range(cfg.NUM_DOWNSAMPLE_BLOCKS):
                    k = sample_trg(cfg.RAQ_MIN_TRG, cfg.RAQ_MAX_TRG)
                    current_trg_list.append(k)

            # (如果不是第一步，current_trg_list 会自动沿用上一步循环里留下的值，这就实现了“锁定”)

            # =======================================================
            # 【原有逻辑】: 判断是不是累积周期的“最后一步” (End)
            # 作用：决定什么时候进行梯度更新 (Optimizer Step)
            # 注意：这里是 (i+1) % steps == 0 (3, 7, 11...)
            # 这行代码完美保留，不要动！
            # =======================================================
            do_step = ((i + 1) % accumulation_steps == 0) or ((i + 1) == len(train_dataloader))

            # =======================================================
            # 【修改逻辑】: 传入锁定的 current_trg_list
            # =======================================================
            # 此时，current_trg_list 无论是刚生成的(Start)，还是沿用的(中间)，都是同一个值
            # === 前向传播 (含信道噪声) ===
            out = deepsc_model.forward_train_raq(real_images, trg_list = current_trg_list)

            recon_loss, latent_loss = deepsc_loss_fn(
                real_images,
                out["reconstructed_images_src"],
                out["reconstructed_images_raq"],
                out["vq_losses_src"],
                out["vq_losses_raq"]
            )
            mae_losses = recon_loss
            vq_raq_losses = latent_loss

            # 这里的(mae_losses + vq_raq_losses)是当前Micro Batch的loss,进行梯度缩放
            loss = (mae_losses + vq_raq_losses) / accumulation_steps

            # 反向传播 (梯度开始累积)
            loss.backward()

            # ... 统计 loss ...
            total_mae_losses += mae_losses.item()
            total_vq_raq_losses += vq_raq_losses.item()

            # 获取当前 Batch 的 SNR
            current_snr = out.get("current_snr", 0.0)

            if do_step:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(deepsc_model.parameters(), max_norm=1.0)
                optimizer_g.step()
                optimizer_g.zero_grad() # 更新完才清空梯度

                # 同步 RAQ 码本
                if (global_step // accumulation_steps) % cfg.RAQ_SYNC_EVERY == 0:
                    deepsc_model.sync_raq_from_vq()

            # === 日志打印 ===
            if i % (accumulation_steps * 10) == 0 :
                print(f"Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}], Step [{i + 1}/{steps_per_epoch}], "
                      f"MAE: {mae_losses.item():.4f}, VQ: {vq_raq_losses.item():.4f}, "
                      f"SNR: {current_snr:.2f} dB")

                # 记录到 TensorBoard
                writer.add_scalar("Train/SNR", current_snr, global_step)
                writer.add_scalar("Train/Loss_Step", mae_losses.item() + vq_raq_losses.item(), global_step)

            # 更新全局步数
            global_step += 1

        scheduler_g.step()

        # === 8. 验证循环 ===
        avg_mae = total_mae_losses / steps_per_epoch
        avg_vq = total_vq_raq_losses / steps_per_epoch


        writer.add_scalar("Loss/Train/MAE", avg_mae, epoch)
        writer.add_scalar("Loss/Train/Total", avg_mae + avg_vq, epoch)

        deepsc_model.eval()
        MAE_Loss_Val = 0
        with torch.no_grad():
            for real_images in val_dataloader:
                real_images = real_images.to(device, non_blocking=True)
                # 验证时也走信道模拟
                out = deepsc_model.forward_val_raq(real_images)

                recon_loss, _ = deepsc_loss_fn(
                    real_images,
                    out["reconstructed_images_src"],
                    out["reconstructed_images_raq"],
                    out["vq_losses_src"],
                    out["vq_losses_raq"]
                )
                MAE_Loss_Val += recon_loss.item()

        avg_val_mae_losses = MAE_Loss_Val / len(val_dataloader)

        print(f"[VAL] Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}], Val MAE Loss: {avg_val_mae_losses:.4f}")
        writer.add_scalar("Loss/Val/MAE", avg_val_mae_losses, epoch)

        # 保存模型
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': deepsc_model.state_dict(),
            'optimizer_state_dict': optimizer_g.state_dict(),
            'scheduler_state_dict': scheduler_g.state_dict(),
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint, cfg.RESUME_PATH)

        if avg_val_mae_losses < best_val_loss:
            best_val_loss = avg_val_mae_losses
            torch.save(deepsc_model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "best_vq_deepsc.pth"))
            print(f"Saved Best Model with Val Loss: {best_val_loss:.4f}")

        if (epoch + 1) % cfg.SAVE_INTERVAL == 0:
            torch.save(deepsc_model.state_dict(),
                       os.path.join(cfg.CHECKPOINT_DIR, f"vq_deepsc_epoch_{epoch + 1}.pth"))

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()