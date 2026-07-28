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
    # Load configuration
    cfg = Config()
    setup_seed(42)

    # Device setup
    device = torch.device(cfg.DEVICE)

    # Create model instances
    deepsc_model = DeepSC(
        in_channels=cfg.IN_CHANNELS, # 输入通道数
        out_channels=cfg.OUT_CHANNELS, # 输出通道数
        num_downsample_blocks=cfg.NUM_DOWNSAMPLE_BLOCKS, # 下采样块数量
        base_channels=cfg.BASE_CHANNELS,  # 基础通道数
        num_embeddings_list=cfg.NUM_EMBEDDINGS_LIST, # 向量量化字典大小列表
        embedding_dim_list=cfg.EMBEDDING_DIM_LIST, # 向量量化维度列表
        commitment_cost=cfg.COMMITMENT_COST, # 向量量化承诺成本系数
        raq_min_trg=cfg.RAQ_MIN_TRG, # RAQ目标最小值
        raq_max_trg=cfg.RAQ_MAX_TRG, # RAQ目标最大值
        device=device # 设备
    ).to(device) # 将模型移动到指定设备

    # Loss functions
    deepsc_loss_fn = DeepSCLoss().to(device)  # VQDeepSC损失


    # 初始化DeepSC优化器
    optimizer_g = optim.Adam(
        deepsc_model.parameters(),  # 优化DeepSC参数
        lr=cfg.LEARNING_RATE_G,  # 生成器学习率
        betas=cfg.BETAS  # Adam优化器的beta参数
    )

    # 设置生成器的学习率调度器（StepLR）
    scheduler_g = optim.lr_scheduler.StepLR(
        optimizer_g,
        step_size=100,  # 每100个epoch调整一次
        gamma=0.5  # 学习率衰减为原来的一半
    )

    # 初始化变量
    start_epoch = 0
    best_val_loss = float('inf')
    global_step = 0
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)  # 创建一个目录用于保存模型检查点，如果这个目录已经存在，则不会报错，而是继续执行，如果是False，则会报错

    # === 2. 断点续训逻辑 ===
    if cfg.RESUME and os.path.exists(cfg.RESUME_PATH):
        print(f"[Info] Loading checkpoint from {cfg.RESUME_PATH}...")
        checkpoint = torch.load(cfg.RESUME_PATH, map_location=device)
        deepsc_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler_g.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        global_step = checkpoint.get('global_step', start_epoch * 1000)  # 粗略估计
        print(f"[Info] Resumed training from epoch {start_epoch}")


    # 数据加载
    train_dataloader = get_dataloader(
        root_dir=cfg.TRAIN_DATASET_PATH,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        mode='train',
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    # 添加验证数据加载器
    val_dataloader = get_dataloader(
        root_dir=cfg.VAL_DATASET_PATH,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        mode='val',
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    # 设置TensorBoard日志目录（带时间戳）
    log_dir = os.path.join(cfg.LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 创建SummaryWriter对象用于记录日志
    writer = SummaryWriter(log_dir)

    # 同步步数，为了定期将源码本的权重同步给 RAQ 模块（源码本是不断持续更新的）
    raq_sync_every= cfg.RAQ_SYNC_EVERY


    # 打印训练信息
    print(f"Starting training on {device}...")
    print(f"Total epochs: {cfg.NUM_EPOCHS}, Starting at: {start_epoch}")
    print(f"Batch size: {cfg.BATCH_SIZE}")


    # 训练循环
    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        # 设置模型为训练模式
        deepsc_model.train()

        # 初始化损失累计值
        total_mae_losses = 0
        total_vq_raq_losses = 0


        # 遍历数据加载器中的每个批次
        for i, real_images in enumerate(train_dataloader):  # 形状是[（B，C，H，W）,(B，C，H，W），...]
            real_images = real_images.to(device)  # 将图像数据移动到指定设备

            # ---------------------
            #  Train Generator (DeepSC)
            # ---------------------
            # 清空生成器梯度
            optimizer_g.zero_grad()

            # 通过VQDeepSC+RAQ重建图像（计算梯度）
            out = deepsc_model.forward_train_raq(real_images)

            # 计算生成器总损失及各分量损失
            recon_loss, latent_loss = deepsc_loss_fn(
                real_images,  # 原始图像
                out["reconstructed_images_src"],  # 原始码本重建图像
                out["reconstructed_images_raq"],  # 自适应码本重建图像
                out["vq_losses_src"],  # 原始码本向量量化损失
                out["vq_losses_raq"]  # 自适应码本向量量化损失
            )
            mae_losses = recon_loss
            vq_raq_losses = latent_loss

            loss = mae_losses + vq_raq_losses
            total_mae_losses += mae_losses.item()
            total_vq_raq_losses +=vq_raq_losses.item()

            # 反向传播计算梯度
            loss.backward()
            # 更新生成器参数
            optimizer_g.step()

            # RAQ 的 Transformer 需要定期获取最新的源 codebook 作为输入
            if (global_step % raq_sync_every) == 0:
                deepsc_model.sync_raq_from_vq()

            # 更新全局步数
            global_step += 1

            # 每50个加载数据就打印训练进度以及第50轮的训练损失
            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}], Step [{i + 1}/{len(train_dataloader)}], "
                      f"MAE Loss: {mae_losses.item():.4f} "
                      f"RAQ_VQ Loss: {vq_raq_losses.item():.4f} "
                )

        # 更新学习率
        scheduler_g.step()

        # 计算训练平均损失
        avg_mae_losses = total_mae_losses / len(train_dataloader)
        avg_vq_raq_losses = total_vq_raq_losses / len(train_dataloader)
        avg_total_losses = avg_mae_losses + avg_vq_raq_losses

        # 记录训练指标到TensorBoard
        writer.add_scalar("Loss/Train/MAE", avg_mae_losses, epoch)
        writer.add_scalar("Loss/Train/RAQ_VQ", avg_vq_raq_losses, epoch)
        writer.add_scalar("Loss/Train/Total", avg_total_losses, epoch)


        # ---------------------
        # Validation Phase
        # ---------------------
        deepsc_model.eval()
        MAE_Loss_Val = 0
        VQ_RAQ_Loss_Val = 0

        with torch.no_grad():
            for real_images in val_dataloader:

                real_images = real_images.to(device)

                out = deepsc_model.forward_val_raq(real_images)
                recon_loss, latent_loss = deepsc_loss_fn(
                    real_images,  # 原始图像
                    out["reconstructed_images_src"],  # 原始码本重建图像
                    out["reconstructed_images_raq"],  # 自适应码本重建图像
                    out["vq_losses_src"],  # 原始码本向量量化损失
                    out["vq_losses_raq"]  # 自适应码本向量量化损失
                )

                MAE_Loss_Val += recon_loss.item()
                VQ_RAQ_Loss_Val += latent_loss.item()

        # 计算平均验证损失
        avg_val_mae_losses = MAE_Loss_Val / len(val_dataloader)
        avg_val_vq_raq_losses = VQ_RAQ_Loss_Val / len(val_dataloader)

        # 记录验证指标到TensorBoard
        writer.add_scalar("Loss/Val/MAE", avg_val_mae_losses, epoch)
        writer.add_scalar("Loss/Val/VQ_RAQ", avg_val_vq_raq_losses, epoch)

        print(f"[VAL]Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}], Val MAE Loss: {avg_val_mae_losses:.4f} ")

        # 保存最佳模型
        if avg_val_mae_losses < best_val_loss:
            best_val_loss = avg_val_mae_losses
            # 保存最佳模型
            torch.save(deepsc_model.state_dict(),  # 保存VQDeepSC模型
                       os.path.join(cfg.CHECKPOINT_DIR, "best_deepsc.pth"))
            print(f"*** Saved NEW best model at epoch {epoch + 1} with Val MAE Loss: {best_val_loss:.4f} ***")

        # 定期保存模型检查点
        if (epoch + 1) % cfg.SAVE_INTERVAL == 0:
            # 保存VQDeepSC模型
            torch.save(deepsc_model.state_dict(),
                       os.path.join(cfg.CHECKPOINT_DIR, f"deepsc_epoch_{epoch + 1}.pth"))
            print(f"Saved checkpoint at epoch {epoch + 1}")

        # 保存断点状态 (每个 epoch 更新 last_checkpoint)
        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': deepsc_model.state_dict(),
            'optimizer_state_dict': optimizer_g.state_dict(),
            'scheduler_state_dict': scheduler_g.state_dict(),
            'best_val_loss': best_val_loss,
            'global_step': global_step
        }
        torch.save(checkpoint_state, cfg.RESUME_PATH)

    # 关闭TensorBoard写入器
    writer.close()
    # 训练完成提示
    print(f"Training complete. Best model Val MAE Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()




