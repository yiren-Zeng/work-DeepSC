import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from config import Config
from models.vq_deepsc import VQDeepSC
from losses.vq_deepsc_loss import VQDeepSCLoss
from data.datasets import get_dataloader


def train():
    # Load configuration
    cfg = Config()

    # Device setup
    device = torch.device(cfg.DEVICE)

    # Create model instances
    vq_deepsc_model = VQDeepSC(
        in_channels=cfg.IN_CHANNELS, # 输入通道数
        out_channels=cfg.OUT_CHANNELS, # 输出通道数
        num_downsample_blocks=cfg.NUM_DOWNSAMPLE_BLOCKS, # 下采样块数量
        base_channels=cfg.BASE_CHANNELS,  # 基础通道数
        num_embeddings_list=cfg.NUM_EMBEDDINGS_LIST, # 向量量化字典大小列表
        embedding_dim_list=cfg.EMBEDDING_DIM_LIST, # 向量量化维度列表
        commitment_cost=cfg.COMMITMENT_COST, # 向量量化承诺成本系数
        raq_min_trg=cfg.RAQ_MIN_TRG, # RAQ目标最小值
        raq_max_trg=cfg.RAQ_MAX_TRG, # RAQ目标最大值
        device=cfg.DEVICE # 设备
    ).to(device) # 将模型移动到指定设备

    # Loss functions
    vq_deepsc_loss_fn = VQDeepSCLoss()  # VQDeepSC损失


    # 初始化VQDeepSC优化器
    optimizer_g = optim.Adam(
        vq_deepsc_model.parameters(),  # 优化VQDeepSC参数
        lr=cfg.LEARNING_RATE_G,  # 生成器学习率
        betas=cfg.BETAS  # Adam优化器的beta参数
    )

    # 设置生成器的学习率调度器（StepLR）
    scheduler_g = optim.lr_scheduler.StepLR(
        optimizer_g,
        step_size=100,  # 每100个epoch调整一次
        gamma=0.5  # 学习率衰减为原来的一半
    )


    # Data loaders
    train_dataloader = get_dataloader(
        root_dir=cfg.TRAIN_DATASET_PATH,  # 训练数据集路径
        batch_size=cfg.BATCH_SIZE,  # 批处理大小
        shuffle=True,  # 打乱数据顺序
        mode='train'
    )

    # 添加验证数据加载器
    val_dataloader = get_dataloader(
        root_dir=cfg.VAL_DATASET_PATH,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,  # 验证集不需要打乱
        mode='val'
    )

    # 设置TensorBoard日志目录（带时间戳）
    log_dir = os.path.join(cfg.LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 创建SummaryWriter对象用于记录日志
    writer = SummaryWriter(log_dir)

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)  # 创建一个目录用于保存模型检查点，如果这个目录已经存在，则不会报错，而是继续执行，如果是False，则会报错

    raq_sync_every= cfg.RAQ_SYNC_EVERY

    # 初始化最佳模型跟踪变量
    best_val_loss = float('inf')  # 初始化为无穷大
    best_epoch = 0
    global_step = 0

    # 打印训练信息
    print(f"Starting training on {device}...")
    print(f"Total epochs: {cfg.NUM_EPOCHS}")
    print(f"Batch size: {cfg.BATCH_SIZE}")
    print(f"[Train] device={device} epochs={cfg.NUM_EPOCHS} batch={cfg.BATCH_SIZE}")

    # 训练循环
    for epoch in range(cfg.NUM_EPOCHS):
        # 设置模型为训练模式
        vq_deepsc_model.train()

        # 初始化损失累计值
        total_mae_losses = 0
        total_vq_raq_losses = 0


        # 遍历数据加载器中的每个批次
        for i, real_images in enumerate(train_dataloader):  # 形状是[（B，C，H，W）,(B，C，H，W），...]
            real_images = real_images.to(device)  # 将图像数据移动到指定设备

            # ---------------------
            #  Train Generator (VQ-DeepSC)
            # ---------------------
            # 清空生成器梯度
            optimizer_g.zero_grad()

            # 通过VQDeepSC+RAQ重建图像（计算梯度）
            out=vq_deepsc_model.forward_train_raq(real_images)

            # 计算生成器总损失及各分量损失
            recon_loss, latent_loss = vq_deepsc_loss_fn(
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

            if (global_step % raq_sync_every) == 0:
                vq_deepsc_model.sync_raq_from_vq()

            # 反向传播计算梯度
            loss.backward()
            # 更新生成器参数
            optimizer_g.step()
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
        vq_deepsc_model.eval()

        MAE_Loss_Val = 0

        with torch.no_grad():
            for real_images in val_dataloader:

                real_images = real_images.to(device)

                out=vq_deepsc_model.forward_val_raq(real_images)
                recon_loss, _ = vq_deepsc_loss_fn(
                    real_images,  # 原始图像
                    out["reconstructed_images_src"],  # 原始码本重建图像
                    out["reconstructed_images_raq"],  # 自适应码本重建图像
                    out["vq_losses_src"],  # 原始码本向量量化损失
                    out["vq_losses_raq"]  # 自适应码本向量量化损失
                )

                MAE_Loss_Val += recon_loss.item()


        # 计算平均验证损失
        avg_val_mae_losses = MAE_Loss_Val / len(val_dataloader)

        # 记录验证指标到TensorBoard
        writer.add_scalar("Loss/Val/MAE", avg_val_mae_losses, epoch)
        early_metric = avg_val_mae_losses
        print(f"[VAL]Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}], Val MAE Loss: {avg_val_mae_losses:.4f} ")

        # 检查是否是最佳模型
        if early_metric < best_val_loss:
            best_val_loss = early_metric
            best_epoch = epoch + 1

            # 保存最佳模型
            torch.save(vq_deepsc_model.state_dict(),  # 保存VQDeepSC模型
                       os.path.join(cfg.CHECKPOINT_DIR, "best_vq_deepsc.pth"))
            print(f"Saved best model at epoch {epoch + 1} with Val MAE Loss: {best_val_loss:.4f}")

        # 定期保存模型检查点
        if (epoch + 1) % cfg.SAVE_INTERVAL == 0:
            # 保存VQDeepSC模型
            torch.save(vq_deepsc_model.state_dict(),
                       os.path.join(cfg.CHECKPOINT_DIR, f"vq_deepsc_epoch_{epoch + 1}.pth"))
            print(f"Saved checkpoint at epoch {epoch + 1}")

    # 关闭TensorBoard写入器
    writer.close()
    # 训练完成提示
    print(f"Training complete. Best model at epoch {best_epoch} with Val MAE Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()




