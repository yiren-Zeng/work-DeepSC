import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import random
from datetime import datetime
from config import Config
from models.deepsc import DeepSC
from losses.deepsc_loss import DeepSCLoss
from data.datasets import get_dataloader
from monitoring.codebook import (compute_codebook_utilization, print_codebook_utilization, write_codebook_tensorboard,)
from training.schedules import compute_schedule
from utils.experiment_io import append_codebook_records, append_epoch_record
from utils.reproducibility import setup_seed
from utils.checkpoint_utils import load_model_state_dict, load_state_dict_compatible
from utils.math_utils import sample_trg
from utils.raq_rvq import resolve_rvq_stage_k_lists, sample_total_codebook_bit_split


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


def load_source_codebook_anchor_weights(anchor_path, device):
    if not anchor_path:
        return None
    if not os.path.exists(anchor_path):
        raise FileNotFoundError(f"source codebook anchor checkpoint not found: {anchor_path}")
    state_dict = load_model_state_dict(anchor_path, device)
    anchors = []
    idx = 0
    while True:
        embed_key = f"vector_quantizers.{idx}.codebook.embed.weight"
        proj_weight_key = f"vector_quantizers.{idx}.codebook.proj.weight"
        proj_bias_key = f"vector_quantizers.{idx}.codebook.proj.bias"
        if embed_key not in state_dict:
            break
        if proj_weight_key not in state_dict:
            raise KeyError(f"missing {proj_weight_key} in source codebook anchor checkpoint")
        embed_weight = state_dict[embed_key].to(device)
        proj_weight = state_dict[proj_weight_key].to(device)
        projected = embed_weight.matmul(proj_weight.t())
        if proj_bias_key in state_dict:
            projected = projected + state_dict[proj_bias_key].to(device)
        anchors.append(projected.detach())
        idx += 1
    if not anchors:
        raise ValueError("No SimVQ source codebook weights found in anchor checkpoint.")
    print(f"[Info] Loaded {len(anchors)} source codebook anchor tensors from {anchor_path}")
    return anchors


def scheduled_src_codebook_repulsion_weight(epoch, cfg):
    target = float(cfg.SRC_CODEBOOK_REPULSION_WEIGHT)
    if target <= 0:
        return 0.0
    start = int(cfg.SRC_CODEBOOK_REPULSION_WARMUP_START_EPOCH)
    end = int(cfg.SRC_CODEBOOK_REPULSION_WARMUP_END_EPOCH)
    if epoch < start:
        return 0.0
    if epoch >= end or end == start:
        return target
    return target * float(epoch - start) / float(end - start)


def scheduled_raq_distill_weight(epoch, cfg):
    start_weight = float(cfg.RAQ_LATENT_DISTILL_WEIGHT)
    final_weight = cfg.RAQ_LATENT_DISTILL_FINAL_WEIGHT
    if final_weight is None:
        return start_weight
    final_weight = float(final_weight)
    start = int(cfg.RAQ_LATENT_DISTILL_DECAY_START_EPOCH)
    end = cfg.RAQ_LATENT_DISTILL_DECAY_END_EPOCH
    end = int(cfg.NUM_EPOCHS if end is None else end)
    if epoch < start:
        return start_weight
    if epoch >= end or end == start:
        return final_weight
    progress = float(epoch - start) / float(end - start)
    return start_weight + (final_weight - start_weight) * progress


def scheduled_linear_weight(epoch, num_epochs, start_weight, final_weight, start_epoch, end_epoch):
    start_weight = float(start_weight)
    if final_weight is None:
        return start_weight
    final_weight = float(final_weight)
    start = int(start_epoch)
    end = int(num_epochs if end_epoch is None else end_epoch)
    if epoch < start:
        return start_weight
    if epoch >= end or end == start:
        return final_weight
    progress = float(epoch - start) / float(end - start)
    return start_weight + (final_weight - start_weight) * progress


def scheduled_raq_jointlite_weights(epoch, cfg):
    return {
        "src_recon": scheduled_linear_weight(
            epoch,
            cfg.NUM_EPOCHS,
            cfg.RAQ_SRC_RECON_WEIGHT,
            cfg.RAQ_SRC_RECON_FINAL_WEIGHT,
            cfg.RAQ_JOINTLITE_DECAY_START_EPOCH,
            cfg.RAQ_JOINTLITE_DECAY_END_EPOCH,
        ),
        "src_vq": scheduled_linear_weight(
            epoch,
            cfg.NUM_EPOCHS,
            cfg.RAQ_SRC_VQ_WEIGHT,
            cfg.RAQ_SRC_VQ_FINAL_WEIGHT,
            cfg.RAQ_JOINTLITE_DECAY_START_EPOCH,
            cfg.RAQ_JOINTLITE_DECAY_END_EPOCH,
        ),
        "codebook_anchor": scheduled_linear_weight(
            epoch,
            cfg.NUM_EPOCHS,
            cfg.RAQ_CODEBOOK_ANCHOR_WEIGHT,
            cfg.RAQ_CODEBOOK_ANCHOR_FINAL_WEIGHT,
            cfg.RAQ_JOINTLITE_DECAY_START_EPOCH,
            cfg.RAQ_JOINTLITE_DECAY_END_EPOCH,
        ),
    }


def raq_curriculum_values_for_epoch(epoch, cfg):
    if not cfg.RAQ_USE_CURRICULUM:
        return None, "uniform"
    phase1_end = int(cfg.PHASE1_END * cfg.NUM_EPOCHS)
    phase2_end = int(cfg.PHASE2_END * cfg.NUM_EPOCHS)
    if epoch < phase1_end:
        plural_name = "RAQ_CURRICULUM_EARLY_LISTS"
        singular_name = "RAQ_CURRICULUM_EARLY_LIST"
        phase = "early"
    elif epoch < phase2_end:
        plural_name = "RAQ_CURRICULUM_MIDDLE_LISTS"
        singular_name = "RAQ_CURRICULUM_MIDDLE_LIST"
        phase = "middle"
    else:
        plural_name = "RAQ_CURRICULUM_LATE_LISTS"
        singular_name = "RAQ_CURRICULUM_LATE_LIST"
        phase = "late"

    values_by_layer = getattr(cfg, plural_name, None)
    if values_by_layer is None:
        shared_values = list(getattr(cfg, singular_name))
        values_by_layer = [
            list(shared_values) for _ in range(cfg.NUM_DOWNSAMPLE_BLOCKS)
        ]
    else:
        values_by_layer = [list(values) for values in values_by_layer]
    return values_by_layer, phase


def sample_raq_target_list_for_epoch(epoch, cfg):
    values_by_layer, phase = raq_curriculum_values_for_epoch(epoch, cfg)
    if values_by_layer is None:
        min_values = getattr(cfg, "RAQ_MIN_TRG_LIST", None)
        max_values = getattr(cfg, "RAQ_MAX_TRG_LIST", None)
        if min_values is None:
            min_values = [cfg.RAQ_MIN_TRG] * cfg.NUM_DOWNSAMPLE_BLOCKS
        if max_values is None:
            max_values = [cfg.RAQ_MAX_TRG] * cfg.NUM_DOWNSAMPLE_BLOCKS
        return [
            sample_trg(min_k, max_k)
            for min_k, max_k in zip(min_values, max_values)
        ], phase
    return [random.choice(values) for values in values_by_layer], phase


def sample_dynamic_raq_rvq_for_epoch(epoch, cfg):
    """Sample total K first, then an ordered equal-bit RVQ allocation."""
    target_list, phase = sample_raq_target_list_for_epoch(epoch, cfg)
    rvq_k_lists = [
        sample_total_codebook_bit_split(k_total)
        for k_total in target_list
    ]
    return target_list, rvq_k_lists, phase


RAQ_ONLY_BRANCHES = {"raq_warmup", "raq_finetune", "raq_channel"}
RAQ_JOINTLITE_BRANCHES = {"raq_jointlite", "raq_jointlite_channel"}


def uses_raq_only_loss(cfg):
    return cfg.USE_RAQ and cfg.TRAIN_BRANCH in RAQ_ONLY_BRANCHES


def uses_raq_jointlite_loss(cfg):
    return cfg.USE_RAQ and cfg.TRAIN_BRANCH in RAQ_JOINTLITE_BRANCHES


def set_module_trainable(module, trainable):
    for param in module.parameters():
        param.requires_grad = trainable


def set_raq_trainable(module, trainable):
    set_module_trainable(module, trainable)
    for submodule in module.modules():
        embed = getattr(submodule, "embed", None)
        if embed is not None and hasattr(embed, "weight"):
            embed.weight.requires_grad = False


def set_source_codebook_proj_trainable(model, trainable):
    quantizer_lists = [model.vector_quantizers]
    small_quantizers = getattr(model, "vector_quantizers_small", None)
    if small_quantizers is not None:
        quantizer_lists.append(small_quantizers)
    for quantizers in quantizer_lists:
        for quantizer in quantizers:
            codebook = getattr(quantizer, "codebook", None)
            proj = getattr(codebook, "proj", None)
            if proj is not None:
                set_module_trainable(proj, trainable)


def configure_trainable_parameters(model, cfg):
    branch = cfg.TRAIN_BRANCH
    if branch in {"joint", "src"}:
        print(f"[Info] Train branch: {branch} (default trainable parameters)")
        return

    set_module_trainable(model, False)
    if branch in RAQ_JOINTLITE_BRANCHES:
        set_raq_trainable(model.raqs, True)
        if getattr(model, "use_dynamic_raq_rvq", False):
            set_raq_trainable(model.raqs_rvq_stage2, True)
    else:
        set_module_trainable(model.raqs, True)
        if getattr(model, "use_dynamic_raq_rvq", False):
            set_module_trainable(model.raqs_rvq_stage2, True)

    if branch in {"raq_finetune", "raq_channel"}:
        if cfg.RAQ_TRAIN_ENCODER:
            set_module_trainable(model.semantic_encoder, True)
            set_module_trainable(model.bottleneck_attention, True)
        set_module_trainable(model.semantic_decoder, True)
        set_module_trainable(model.swinir_enhance, True)
    elif branch in RAQ_JOINTLITE_BRANCHES:
        set_module_trainable(model.semantic_encoder, True)
        set_module_trainable(model.bottleneck_attention, True)
        set_module_trainable(model.semantic_decoder, True)
        set_module_trainable(model.swinir_enhance, True)
        set_source_codebook_proj_trainable(model, True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[Info] Train branch: {branch}; trainable params "
        f"{trainable_params:,}/{total_params:,}"
    )


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
    print(f"CUDA_VISIBLE_DEVICES: {visible_devices}")

    # 在使用 CUDA/GPU 时，打印 PyTorch 里的逻辑 GPU 编号对应服务器上的物理 GPU 编号
    if device.type == "cuda":
        logical_index = device.index if device.index is not None else torch.cuda.current_device() # 得到当前 PyTorch 使用的 逻辑 GPU 编号
        visible_device_ids = [
            value.strip() for value in visible_devices.split(",") if value.strip() # .split(",")会把字符串按逗号分开,并且用列表形成，如"2,3"变成["2","3"]
        ]

        # 映射逻辑 GPU 到物理 GPU
        mapped_physical_device = (
            visible_device_ids[logical_index]
            if visible_devices != "<not set>" and logical_index < len(visible_device_ids)
            else str(logical_index)
        )
        print(
            f"GPU mapping: logical cuda:{logical_index} -> "
            f"physical GPU {mapped_physical_device}"
            f"({torch.cuda.get_device_name(logical_index)})"
        )
    
    print(f"[Info] Experiment name: {cfg.EXPERIMENT_NAME}")
    print(f"[Info] Experiment stage: {cfg.EXPERIMENT_STAGE}")
    print(f"[Info] Experiment run ID: {run_id}")
    print(f"[Info] Train branch: {cfg.TRAIN_BRANCH}")
    print(f"[Info] Epoch metrics file: {metrics_path}")
    print(f"[Info] Checkpoint directory: {cfg.CHECKPOINT_DIR}")
    print(f"[Info] Resume checkpoint: {cfg.RESUME_PATH}")
    pretrained_path = os.environ.get("SIMVQ_PRETRAINED_CHECKPOINT", "")
    allow_pretrained = os.environ.get("SIMVQ_ALLOW_PRETRAINED", "0") == "1"
    if pretrained_path and allow_pretrained:
        print(f"[Info] 预训练权重路径: {pretrained_path}")
    elif pretrained_path:
        print("[Info] SIMVQ_PRETRAINED_CHECKPOINT was set but ignored; this shiyan config trains from scratch.")

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
    print(f"  - 估算RAQ目标端BPP: {cfg.ESTIMATED_RAQ_TARGET_BPP:.6f}")
    print(f"  - 估算测试传输压缩率(LDPC1/2+BPSK): {cfg.ESTIMATED_TEST_TRANSMISSION_RATIO:.8f}")
    print(f"  - 每层特征维度: {cfg.EMBEDDING_DIM_LIST}")
    print(f"  - 每层码本大小: {cfg.NUM_EMBEDDINGS_LIST}")
    raq_ranges = (
        list(zip(cfg.RAQ_MIN_TRG_LIST, cfg.RAQ_MAX_TRG_LIST))
        if cfg.RAQ_MIN_TRG_LIST is not None and cfg.RAQ_MAX_TRG_LIST is not None
        else None
    )
    print(f"  - RAQ动态目标码本: {cfg.USE_RAQ}, "
          f"train K逐层范围={raq_ranges}, "
          f"eval K={cfg.RAQ_TARGET_LIST}, repulsion={cfg.RAQ_REPULSION_WEIGHT}")
    print(f"  - 动态RAQ-RVQ训练: {cfg.USE_DYNAMIC_RAQ_RVQ}, "
          f"stage2_zero={cfg.DYNAMIC_RAQ_RVQ_ZERO_CODEWORD}")
    print(f"  - RAQ课程采样: {cfg.RAQ_USE_CURRICULUM}, "
          f"early={cfg.RAQ_CURRICULUM_EARLY_LISTS}, "
          f"middle={cfg.RAQ_CURRICULUM_MIDDLE_LISTS}, "
          f"late={cfg.RAQ_CURRICULUM_LATE_LISTS}")
    print(f"  - 训练分支模式: {cfg.TRAIN_BRANCH}")
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
    print(f"  - RAQ隐空间蒸馏: weight={cfg.RAQ_LATENT_DISTILL_WEIGHT}")
    if cfg.RAQ_LATENT_DISTILL_FINAL_WEIGHT is not None:
        print(f"  - RAQ隐空间蒸馏衰减: final={cfg.RAQ_LATENT_DISTILL_FINAL_WEIGHT}, "
              f"epoch=[{cfg.RAQ_LATENT_DISTILL_DECAY_START_EPOCH},"
              f"{cfg.RAQ_LATENT_DISTILL_DECAY_END_EPOCH or cfg.NUM_EPOCHS}]")
    print(f"  - RAQ重建梯度模式: {cfg.RAQ_RECON_GRAD_MODE}, "
          f"RAQ阶段解冻encoder={cfg.RAQ_TRAIN_ENCODER}")
    print(f"  - RAQ生成器结构: {cfg.RAQ_GENERATOR_TYPE}")
    print(f"  - RAQ路由SRC码本: {cfg.RAQ_ROUTED_SRC_ENABLED}, "
          f"threshold<={cfg.RAQ_ROUTED_SRC_THRESHOLD}, "
          f"small={cfg.RAQ_ROUTED_SRC_SMALL_LIST}, "
          f"large={cfg.RAQ_ROUTED_SRC_LARGE_LIST}")
    if cfg.TRAIN_BRANCH in RAQ_JOINTLITE_BRANCHES:
        print(f"  - Joint-lite SRC recon anchor: {cfg.RAQ_SRC_RECON_WEIGHT} -> "
              f"{cfg.RAQ_SRC_RECON_FINAL_WEIGHT}")
        print(f"  - Joint-lite SRC VQ anchor: {cfg.RAQ_SRC_VQ_WEIGHT} -> "
              f"{cfg.RAQ_SRC_VQ_FINAL_WEIGHT}")
        print(f"  - Joint-lite source codebook anchor: {cfg.RAQ_CODEBOOK_ANCHOR_WEIGHT} -> "
              f"{cfg.RAQ_CODEBOOK_ANCHOR_FINAL_WEIGHT}")
        print(f"  - Joint-lite anchor checkpoint: {cfg.SRC_CODEBOOK_ANCHOR_CHECKPOINT}")
    print(f"  - SRC码本排斥: target_weight={cfg.SRC_CODEBOOK_REPULSION_WEIGHT}, "
          f"margin={cfg.SRC_CODEBOOK_REPULSION_MARGIN}, "
          f"normalize={cfg.SRC_CODEBOOK_REPULSION_NORMALIZE}, "
          f"warmup=[{cfg.SRC_CODEBOOK_REPULSION_WARMUP_START_EPOCH},"
          f"{cfg.SRC_CODEBOOK_REPULSION_WARMUP_END_EPOCH}]")
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

    # 模型初始化：保留原 quality_v2_B_larger_rate044_A_patch 主干，并可启用 RAQ 双支路
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
        use_raq=cfg.USE_RAQ,
        raq_target_list=cfg.RAQ_TARGET_LIST,
        raq_min_trg=cfg.RAQ_MIN_TRG,
        raq_max_trg=cfg.RAQ_MAX_TRG,
        raq_min_trg_list=cfg.RAQ_MIN_TRG_LIST,
        raq_max_trg_list=cfg.RAQ_MAX_TRG_LIST,
        raq_recon_grad_mode=cfg.RAQ_RECON_GRAD_MODE,
        raq_generator_type=cfg.RAQ_GENERATOR_TYPE,
        raq_routed_src_enabled=cfg.RAQ_ROUTED_SRC_ENABLED,
        raq_routed_src_small_list=cfg.RAQ_ROUTED_SRC_SMALL_LIST,
        raq_routed_src_large_list=cfg.RAQ_ROUTED_SRC_LARGE_LIST,
        raq_routed_src_threshold=cfg.RAQ_ROUTED_SRC_THRESHOLD,
        use_dynamic_raq_rvq=cfg.USE_DYNAMIC_RAQ_RVQ,
        dynamic_raq_rvq_zero_codeword=cfg.DYNAMIC_RAQ_RVQ_ZERO_CODEWORD,
    ).to(device)

    # 加载预训练权重默认禁用；本实验要求从零开始训练。
    if pretrained_path and allow_pretrained:
        load_pretrained_weights(deepsc_model, pretrained_path, device)

    source_codebook_anchor_list = None
    if uses_raq_jointlite_loss(cfg) and cfg.RAQ_CODEBOOK_ANCHOR_WEIGHT > 0:
        source_codebook_anchor_list = load_source_codebook_anchor_weights(
            cfg.SRC_CODEBOOK_ANCHOR_CHECKPOINT,
            device,
        )
        if len(source_codebook_anchor_list) != cfg.NUM_DOWNSAMPLE_BLOCKS:
            raise ValueError(
                "source codebook anchor layer count differs from current model "
                f"({len(source_codebook_anchor_list)} vs {cfg.NUM_DOWNSAMPLE_BLOCKS})"
            )

    if cfg.MODEL_PARALLEL:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("SIMVQ_MODEL_PARALLEL=1 requires at least two visible CUDA devices.")
        deepsc_model.enable_model_parallel(cfg.ENCODER_DEVICE, cfg.DECODER_DEVICE)
        print(
            f"[Info] Model parallel enabled: encoder/quantizer/channel on {cfg.ENCODER_DEVICE}, "
            f"decoder/enhance on {cfg.DECODER_DEVICE}"
        )

    configure_trainable_parameters(deepsc_model, cfg)

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
        raq_repulsion_weight=cfg.RAQ_REPULSION_WEIGHT if cfg.USE_RAQ else 0.0,
        raq_latent_distill_weight=cfg.RAQ_LATENT_DISTILL_WEIGHT if cfg.USE_RAQ else 0.0,
        raq_src_recon_weight=cfg.RAQ_SRC_RECON_WEIGHT if uses_raq_jointlite_loss(cfg) else 0.0,
        raq_src_vq_weight=cfg.RAQ_SRC_VQ_WEIGHT if uses_raq_jointlite_loss(cfg) else 0.0,
        raq_codebook_anchor_weight=cfg.RAQ_CODEBOOK_ANCHOR_WEIGHT if uses_raq_jointlite_loss(cfg) else 0.0,
        src_codebook_repulsion_weight=0.0,
        src_codebook_repulsion_margin=cfg.SRC_CODEBOOK_REPULSION_MARGIN,
        src_codebook_repulsion_normalize=cfg.SRC_CODEBOOK_REPULSION_NORMALIZE,
    ).to(deepsc_model.decoder_device if cfg.MODEL_PARALLEL else device)

    # 把模型参数分成两组，用不同学习率训练，然后创建 Adam 优化器和 StepLR 学习率调度器.
    proj_params = []
    other_params = []
    for name, param in deepsc_model.named_parameters():
        if not param.requires_grad:
            continue
        if "codebook.proj" in name or "trg_embed.proj" in name or ".qbridge." in name:
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
        # Load resume tensors on CPU first. Loading the full model+optimizer
        # checkpoint directly onto CUDA keeps a second GPU copy alive during
        # training and can consume the remaining memory headroom.
        checkpoint = torch.load(cfg.RESUME_PATH, map_location="cpu")
        load_state_dict_compatible(deepsc_model, checkpoint['model_state_dict'])
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
        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
        _, raq_sampling_phase = raq_curriculum_values_for_epoch(epoch, cfg)
        deepsc_model.semantic_decoder.set_skip_dropout_p(dropout_p)
        deepsc_model.set_channel_prob(channel_prob)
        deepsc_loss_fn.set_layer_weights(loss_weights)
        src_repulsion_weight = scheduled_src_codebook_repulsion_weight(epoch, cfg)
        deepsc_loss_fn.set_src_codebook_repulsion_weight(src_repulsion_weight)
        raq_distill_weight = scheduled_raq_distill_weight(epoch, cfg) if cfg.USE_RAQ else 0.0
        deepsc_loss_fn.set_raq_latent_distill_weight(raq_distill_weight)
        jointlite_weights = (
            scheduled_raq_jointlite_weights(epoch, cfg)
            if uses_raq_jointlite_loss(cfg)
            else {"src_recon": 0.0, "src_vq": 0.0, "codebook_anchor": 0.0}
        )
        deepsc_loss_fn.set_raq_jointlite_weights(
            jointlite_weights["src_recon"],
            jointlite_weights["src_vq"],
            jointlite_weights["codebook_anchor"],
        )

        deepsc_model.train()

        total_recon_losses = 0
        total_vq_losses = 0
        total_distill_losses = 0
        total_src_repulsion_losses = 0
        total_src_recon_anchor_losses = 0
        total_src_vq_anchor_losses = 0
        total_src_codebook_anchor_losses = 0

        optimizer_g.zero_grad()
        steps_per_epoch = len(train_dataloader)

        for i, real_images in enumerate(train_dataloader):
            real_images = real_images.to(device, non_blocking=True)

            do_step = ((i + 1) % accumulation_steps == 0) or ((i + 1) == len(train_dataloader))
            if cfg.USE_RAQ and i % accumulation_steps == 0:
                if cfg.USE_DYNAMIC_RAQ_RVQ:
                    current_raq_trg_list, current_rvq_k_lists, raq_sampling_phase = (
                        sample_dynamic_raq_rvq_for_epoch(epoch, cfg)
                    )
                else:
                    current_raq_trg_list, raq_sampling_phase = sample_raq_target_list_for_epoch(epoch, cfg)
                    current_rvq_k_lists = None

            out = deepsc_model.forward_train(
                real_images,
                raq_trg_list=current_raq_trg_list if cfg.USE_RAQ else None,
                raq_rvq_k_lists=(
                    current_rvq_k_lists if cfg.USE_DYNAMIC_RAQ_RVQ else None
                ),
            )

            if uses_raq_jointlite_loss(cfg):
                recon_loss, vq_loss, loss_details = deepsc_loss_fn.forward_raq_jointlite(
                    real_images,
                    out["reconstructed_images_src"],
                    out["vq_losses_src"],
                    out["reconstructed_images_raq"],
                    out["vq_losses_raq"],
                    out["W_trg_list"],
                    out["z_q_src_list"],
                    out["z_q_raq_list"],
                    out["source_codebooks_list"],
                    source_codebook_anchor_list,
                    return_details=True,
                )
            elif uses_raq_only_loss(cfg):
                recon_loss, vq_loss, loss_details = deepsc_loss_fn.forward_raq_only(
                    real_images,
                    out["reconstructed_images_raq"],
                    out["vq_losses_raq"],
                    out["W_trg_list"],
                    out["z_q_src_list"],
                    out["z_q_raq_list"],
                    return_details=True,
                )
            elif cfg.USE_RAQ:
                recon_loss, vq_loss, loss_details = deepsc_loss_fn(
                    real_images,
                    out["reconstructed_images_src"],
                    out["vq_losses_src"],
                    out["reconstructed_images_raq"],
                    out["vq_losses_raq"],
                    out["W_trg_list"],
                    out["z_q_src_list"],
                    out["z_q_raq_list"],
                    out["source_codebooks_list"],
                    return_details=True,
                )
            else:
                recon_loss, vq_loss, loss_details = deepsc_loss_fn(
                    real_images,
                    out["reconstructed_images"],
                    out["vq_losses"],
                    return_details=True,
                )

            loss = (recon_loss + vq_loss) / accumulation_steps
            loss.backward()

            total_recon_losses += recon_loss.item()
            total_vq_losses += vq_loss.item()
            total_distill_losses += loss_details["latent_distill_loss"].item()
            total_src_repulsion_losses += loss_details["src_codebook_repulsion_loss"].item()
            total_src_recon_anchor_losses += loss_details["src_recon_anchor_loss"].item()
            total_src_vq_anchor_losses += loss_details["src_vq_anchor_loss"].item()
            total_src_codebook_anchor_losses += loss_details["src_codebook_anchor_loss"].item()

            current_snr = out.get("current_snr")
            snr_desc = "clean" if current_snr is None else f"{current_snr:.2f} dB"

            if do_step:
                torch.nn.utils.clip_grad_norm_(deepsc_model.parameters(), max_norm=1.0)
                optimizer_g.step()
                optimizer_g.zero_grad()

            if i % (accumulation_steps * 10) == 0:
                print(f"Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}], Step [{i + 1}/{steps_per_epoch}], "
                      f"Recon: {recon_loss.item():.4f}, Aux: {vq_loss.item():.4f}, "
                      f"Distill: {loss_details['latent_distill_loss'].item():.4f}, "
                      f"DistillW: {raq_distill_weight:.6g}, "
                      f"SrcRep: {loss_details['src_codebook_repulsion_loss'].item():.6f}, "
                      f"SrcRepW: {src_repulsion_weight:.6g}, "
                      f"SrcAnchor: {loss_details['src_recon_anchor_loss'].item():.4f}/"
                      f"{loss_details['src_vq_anchor_loss'].item():.4f}/"
                      f"{loss_details['src_codebook_anchor_loss'].item():.4f}, "
                      f"ChannelProb: {channel_prob:.2f}, SNR: {snr_desc}")
                if cfg.USE_RAQ:
                    print(f"  RAQ target K this accumulation: {out['raq_target_list']} "
                          f"(sampling={raq_sampling_phase})")
                    if cfg.USE_DYNAMIC_RAQ_RVQ:
                        print(f"  Dynamic RVQ stage K: {out['rvq_k_lists']}")
                    if out.get("source_route_list") is not None:
                        print(f"  RAQ source route this accumulation: {out['source_route_list']}")
                if current_snr is not None:
                    writer.add_scalar("Train/SNR", current_snr, global_step)
                writer.add_scalar("Train/Loss_Step", recon_loss.item() + vq_loss.item(), global_step)
                writer.add_scalar("Train/LatentDistill_Step", loss_details["latent_distill_loss"].item(), global_step)
                writer.add_scalar("Train/LatentDistillRaw_Step", loss_details["latent_distill_raw_loss"].item(), global_step)
                writer.add_scalar("Train/SrcCodebookRepulsion_Step", loss_details["src_codebook_repulsion_loss"].item(), global_step)
                writer.add_scalar("Train/SrcCodebookRepulsionRaw_Step", loss_details["src_codebook_repulsion_raw_loss"].item(), global_step)
                writer.add_scalar("Train/SrcCodebookRepulsionWeight", src_repulsion_weight, global_step)
                writer.add_scalar("Train/SrcReconAnchor_Step", loss_details["src_recon_anchor_loss"].item(), global_step)
                writer.add_scalar("Train/SrcVQAnchor_Step", loss_details["src_vq_anchor_loss"].item(), global_step)
                writer.add_scalar("Train/SrcCodebookAnchor_Step", loss_details["src_codebook_anchor_loss"].item(), global_step)
                writer.add_scalar("Train/ChannelProb", channel_prob, global_step)

            global_step += 1

        scheduler_g.step()

        avg_recon = total_recon_losses / steps_per_epoch
        avg_vq = total_vq_losses / steps_per_epoch
        avg_distill = total_distill_losses / steps_per_epoch
        avg_src_repulsion = total_src_repulsion_losses / steps_per_epoch
        avg_src_recon_anchor = total_src_recon_anchor_losses / steps_per_epoch
        avg_src_vq_anchor = total_src_vq_anchor_losses / steps_per_epoch
        avg_src_codebook_anchor = total_src_codebook_anchor_losses / steps_per_epoch

        print(f"[{phase_desc}] Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}], "
              f"Recon: {avg_recon:.4f}, Aux: {avg_vq:.4f}, Distill: {avg_distill:.4f}, "
              f"DistillW: {raq_distill_weight:.6g}, "
              f"SrcRep: {avg_src_repulsion:.6f}, SrcRepW: {src_repulsion_weight:.6g}, "
              f"SrcAnchor: {avg_src_recon_anchor:.4f}/{avg_src_vq_anchor:.4f}/"
              f"{avg_src_codebook_anchor:.4f}, "
              f"ChannelProb: {channel_prob:.2f}, RAQSampling: {raq_sampling_phase}, "
              f"Dropout: {[f'{p:.2f}' for p in dropout_p]}, "
              f"LossW: {[f'{w:.1f}' for w in loss_weights]}")

        writer.add_scalar("Loss/Train/Recon", avg_recon, epoch)
        writer.add_scalar("Loss/Train/VQ", avg_vq, epoch)
        writer.add_scalar("Loss/Train/LatentDistill", avg_distill, epoch)
        writer.add_scalar("Schedule/RAQLatentDistillWeight", raq_distill_weight, epoch)
        writer.add_scalar("Loss/Train/SrcCodebookRepulsion", avg_src_repulsion, epoch)
        writer.add_scalar("Schedule/SrcCodebookRepulsionWeight", src_repulsion_weight, epoch)
        writer.add_scalar("Loss/Train/SrcReconAnchor", avg_src_recon_anchor, epoch)
        writer.add_scalar("Loss/Train/SrcVQAnchor", avg_src_vq_anchor, epoch)
        writer.add_scalar("Loss/Train/SrcCodebookAnchor", avg_src_codebook_anchor, epoch)
        writer.add_scalar("Schedule/JointLiteSrcReconWeight", jointlite_weights["src_recon"], epoch)
        writer.add_scalar("Schedule/JointLiteSrcVQWeight", jointlite_weights["src_vq"], epoch)
        writer.add_scalar("Schedule/JointLiteCodebookAnchorWeight", jointlite_weights["codebook_anchor"], epoch)
        writer.add_scalar("Loss/Train/Total", avg_recon + avg_vq, epoch)
        writer.add_scalar("Schedule/ChannelProb", channel_prob, epoch)
        writer.add_text("Schedule/RAQSamplingPhase", raq_sampling_phase, epoch)
        # 记录调度参数
        for li, p in enumerate(dropout_p):
            writer.add_scalar(f"Schedule/Dropout_L{li}", p, epoch)
        for li, w in enumerate(loss_weights):
            writer.add_scalar(f"Schedule/LossWeight_L{li}", w, epoch)

        # 验证
        deepsc_model.eval()
        val_loss_sum = 0
        val_distill_sum = 0
        val_src_repulsion_sum = 0
        val_src_recon_anchor_sum = 0
        val_src_vq_anchor_sum = 0
        val_src_codebook_anchor_sum = 0
        with torch.no_grad():
            val_raq_target_list = (
                list(cfg.RAQ_TARGET_LIST) if cfg.USE_DYNAMIC_RAQ_RVQ else None
            )
            val_rvq_k_lists = (
                resolve_rvq_stage_k_lists(val_raq_target_list)
                if cfg.USE_DYNAMIC_RAQ_RVQ else None
            )
            for real_images in val_dataloader:
                real_images = real_images.to(device, non_blocking=True)
                out = deepsc_model.forward_val(
                    real_images,
                    raq_trg_list=val_raq_target_list,
                    raq_rvq_k_lists=val_rvq_k_lists,
                )
                if uses_raq_jointlite_loss(cfg):
                    recon_loss_val, _, val_details = deepsc_loss_fn.forward_raq_jointlite(
                        real_images,
                        out["reconstructed_images_src"],
                        out["vq_losses_src"],
                        out["reconstructed_images_raq"],
                        out["vq_losses_raq"],
                        out["W_trg_list"],
                        out["z_q_src_list"],
                        out["z_q_raq_list"],
                        out["source_codebooks_list"],
                        source_codebook_anchor_list,
                        return_details=True,
                    )
                elif uses_raq_only_loss(cfg):
                    recon_loss_val, _, val_details = deepsc_loss_fn.forward_raq_only(
                        real_images,
                        out["reconstructed_images_raq"],
                        out["vq_losses_raq"],
                        out["W_trg_list"],
                        out["z_q_src_list"],
                        out["z_q_raq_list"],
                        return_details=True,
                    )
                elif cfg.USE_RAQ:
                    recon_loss_val, _, val_details = deepsc_loss_fn(
                        real_images,
                        out["reconstructed_images_src"],
                        out["vq_losses_src"],
                        out["reconstructed_images_raq"],
                        out["vq_losses_raq"],
                        out["W_trg_list"],
                        out["z_q_src_list"],
                        out["z_q_raq_list"],
                        out["source_codebooks_list"],
                        return_details=True,
                    )
                else:
                    recon_loss_val, _, val_details = deepsc_loss_fn(
                        real_images,
                        out["reconstructed_images"],
                        out["vq_losses"],
                        return_details=True,
                    )
                val_loss_sum += recon_loss_val.item()
                val_distill_sum += val_details["latent_distill_loss"].item()
                val_src_repulsion_sum += val_details["src_codebook_repulsion_loss"].item()
                val_src_recon_anchor_sum += val_details["src_recon_anchor_loss"].item()
                val_src_vq_anchor_sum += val_details["src_vq_anchor_loss"].item()
                val_src_codebook_anchor_sum += val_details["src_codebook_anchor_loss"].item()

        avg_val_loss = val_loss_sum / len(val_dataloader)
        avg_val_distill = val_distill_sum / len(val_dataloader)
        avg_val_src_repulsion = val_src_repulsion_sum / len(val_dataloader)
        avg_val_src_recon_anchor = val_src_recon_anchor_sum / len(val_dataloader)
        avg_val_src_vq_anchor = val_src_vq_anchor_sum / len(val_dataloader)
        avg_val_src_codebook_anchor = val_src_codebook_anchor_sum / len(val_dataloader)
        print(f"[VAL] Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}], Val Recon Loss: {avg_val_loss:.4f}, "
              f"Val Distill: {avg_val_distill:.4f}, Val SrcRep: {avg_val_src_repulsion:.6f}, "
              f"Val SrcAnchor: {avg_val_src_recon_anchor:.4f}/{avg_val_src_vq_anchor:.4f}/"
              f"{avg_val_src_codebook_anchor:.4f}")
        writer.add_scalar("Loss/Val/Recon", avg_val_loss, epoch)
        writer.add_scalar("Loss/Val/LatentDistill", avg_val_distill, epoch)
        writer.add_scalar("Loss/Val/SrcCodebookRepulsion", avg_val_src_repulsion, epoch)
        writer.add_scalar("Loss/Val/SrcReconAnchor", avg_val_src_recon_anchor, epoch)
        writer.add_scalar("Loss/Val/SrcVQAnchor", avg_val_src_vq_anchor, epoch)
        writer.add_scalar("Loss/Val/SrcCodebookAnchor", avg_val_src_codebook_anchor, epoch)

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
            "train_distill": f"{avg_distill:.8f}",
            "train_src_repulsion": f"{avg_src_repulsion:.8f}",
            "val_recon": f"{avg_val_loss:.8f}",
            "val_distill": f"{avg_val_distill:.8f}",
            "val_src_repulsion": f"{avg_val_src_repulsion:.8f}",
            "src_repulsion_weight": f"{src_repulsion_weight:.10g}",
            "raq_distill_weight": f"{raq_distill_weight:.10g}",
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
