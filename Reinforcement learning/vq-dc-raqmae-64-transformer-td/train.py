import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import random
import numpy as np
from datetime import datetime
from config import Config
from models.deepsc import DeepSC
from losses.deepsc_loss import VQDeepSCLoss
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
    # Load configuration
    cfg = Config()
    setup_seed(42)

    device = torch.device(cfg.DEVICE)
    print(f"Start training on {device}")

    # 2. 计算梯度累积步数
    # Accumulation = Target / Micro
    accumulation_steps = cfg.TOTAL_BATCH_SIZE // cfg.MICRO_BATCH_SIZE
    if accumulation_steps < 1: accumulation_steps = 1

    print("=" * 40)
    print(f"  - 总Batch Size：{cfg.TOTAL_BATCH_SIZE}")
    print(f"  - 小Batch Size：{cfg.MICRO_BATCH_SIZE}")
    print(f"  - 梯度累积步数: {accumulation_steps}")
    print("=" * 40)


    # 4. 模型初始化
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
        device=cfg.DEVICE # 设备
    ).to(device) # 将模型移动到指定设备

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

    # Loss functions
    vq_deepsc_loss_fn = VQDeepSCLoss().to(device)  # VQDeepSC损失


    # 初始化VQDeepSC优化器
    optimizer_g = optim.Adam(
        deepsc_model.parameters(),  # 优化VQDeepSC参数
        lr=cfg.LEARNING_RATE_G,  # 生成器学习率
        betas=cfg.BETAS  # Adam优化器的beta参数
    )


    # 设置生成器的学习率调度器（StepLR）
    scheduler_g = optim.lr_scheduler.StepLR(
        optimizer_g,
        step_size=100,  # 每100个epoch调整一次
        gamma=0.5  # 学习率衰减为原来的一半
    )


    # 5. 数据加载
    train_dataloader = get_dataloader(
        root_dir=cfg.TRAIN_DATASET_PATH,
        batch_size=cfg.MICRO_BATCH_SIZE,
        shuffle=True,
        mode='train',
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    # 添加验证数据加载器
    val_dataloader = get_dataloader(
        root_dir=cfg.VAL_DATASET_PATH,
        batch_size=cfg.MICRO_BATCH_SIZE,
        shuffle=False,
        mode='val',
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    # 6. 断点续训
    start_epoch = 0
    best_val_loss = float('inf') # 初始化最佳模型跟踪变量
    if cfg.RESUME and os.path.exists(cfg.RESUME_PATH):
        print(f"Loading checkpoint: {cfg.RESUME_PATH}")
        checkpoint = torch.load(cfg.RESUME_PATH, map_location=device)
        deepsc_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler_g.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"--> 成功恢复检查点: {cfg.RESUME_PATH}, 从 Epoch {start_epoch} 继续。")

    # 7. 初始化 Tensorboard
    log_dir = os.path.join(cfg.LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))
    # 创建SummaryWriter对象用于记录日志
    writer = SummaryWriter(log_dir)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)  # 创建一个目录用于保存模型检查点，如果这个目录已经存在，则不会报错，而是继续执行，如果是False，则会报错


    global_step = start_epoch * len(train_dataloader)

    best_epoch = 0

    # 训练循环
    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        # 设置模型为训练模式
        deepsc_model.train()

        # 初始化损失累计值
        total_mae_losses = 0
        total_vq_raq_losses = 0

        optimizer_g.zero_grad()  # 确保每个epoch开始前清空梯度
        steps_per_epoch = len(train_dataloader)

        # 遍历数据加载器中的每个批次
        for i, real_images in enumerate(train_dataloader):  # 形状是[（B，C，H，W）,(B，C，H，W），...]
            real_images = real_images.to(device, non_blocking=True)  # 将图像数据移动到指定设备

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
            out = deepsc_model.forward_train_raq(real_images, trg_list= current_trg_list)

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

            # Loss 缩放
            # 必须除以累积步数，保证梯度幅度和大Batch Size训练时一致
            loss = (mae_losses + vq_raq_losses) / accumulation_steps

            # 反向传播 (梯度开始累积)
            loss.backward()

            # ... 统计 loss ...
            total_mae_losses += mae_losses.item()
            total_vq_raq_losses +=vq_raq_losses.item()

            if do_step:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(deepsc_model.parameters(), max_norm=1.0)
                optimizer_g.step()
                optimizer_g.zero_grad()  # 更新完才清空梯度

                # 同步 RAQ 码本
                if (global_step // accumulation_steps) % cfg.RAQ_SYNC_EVERY == 0:
                    deepsc_model.sync_raq_from_vq()

            # === 日志打印 ===
            if i % (accumulation_steps * 10) == 0 :
                print(f"Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}], Step [{i + 1}/{steps_per_epoch}], "
                      f"MAE: {mae_losses.item():.4f}, VQ: {vq_raq_losses.item():.4f}")

                # 记录到 TensorBoard
                writer.add_scalar("Train/Loss_Step", mae_losses.item() + vq_raq_losses.item(), global_step)

            # 更新全局步数
            global_step += 1

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

        with torch.no_grad():
            for real_images in val_dataloader:
                real_images = real_images.to(device, non_blocking=True)

                out = deepsc_model.forward_val_raq(real_images)
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

        # 保存断点的模型
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': deepsc_model.state_dict(),
            'optimizer_state_dict': optimizer_g.state_dict(),
            'scheduler_state_dict': scheduler_g.state_dict(),
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint, cfg.RESUME_PATH)

        # 检查是否是最佳模型
        if early_metric < best_val_loss:
            best_val_loss = early_metric
            best_epoch = epoch + 1

            # 保存最佳模型
            torch.save(deepsc_model.state_dict(),  # 保存VQDeepSC模型
                       os.path.join(cfg.CHECKPOINT_DIR, "best_vq_deepsc.pth"))
            print(f"Saved best model at epoch {epoch + 1} with Val MAE Loss: {best_val_loss:.4f}")

        # 定期保存模型检查点
        if (epoch + 1) % cfg.SAVE_INTERVAL == 0:
            # 保存VQDeepSC模型
            torch.save(deepsc_model.state_dict(),
                       os.path.join(cfg.CHECKPOINT_DIR, f"vq_deepsc_epoch_{epoch + 1}.pth"))
            print(f"Saved checkpoint at epoch {epoch + 1}")

    # 关闭TensorBoard写入器
    writer.close()
    # 训练完成提示
    print(f"Training complete. Best model at epoch {best_epoch} with Val MAE Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()




