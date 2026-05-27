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

cfg = Config()
device = torch.device(cfg.DEVICE)
# ---------------- 你的网络（替换成你自己的） ----------------
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

# -----------------------------------------------------------

# 【核心：打印参数 + 所属层类型 + 所属代码变量】
print("=" * 80)
print(f"{'参数完整名':<35} {'参数形状':<18} {'所属层类型':<15} {'是否可优化'}")
print("=" * 80)

# 遍历所有参数 + 溯源所属模块
for name, param in deepsc_model.named_parameters():
    # 拆分参数名，找到对应的层对象
    module = deepsc_model
    for attr in name.split(".")[:-1]:  # 去掉 .weight/.bias
        module = getattr(module, attr)

    layer_type = module.__class__.__name__  # 层类型：Conv2d/Linear/BatchNorm2d
    shape = str(list(param.shape))
    trainable = param.requires_grad

    print(f"{name:<35} {shape:<18} {layer_type:<15} {trainable}")

print("=" * 80)