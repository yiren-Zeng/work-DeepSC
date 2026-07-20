"""Utilities for the single, immutable ``[2048, 2048]`` SRC teacher.

The variable-rate training path intentionally keeps the teacher as a separate
``DeepSC`` instance.  Nothing in this module ever aliases teacher parameters
into the student optimizer; student initialization is performed with copied
state dictionaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import torch

from models.deepsc import DeepSC
from utils.checkpoint_utils import extract_state_dict, load_state_dict_compatible


def build_source_teacher(cfg, device: torch.device) -> DeepSC:
    """Build the source-only architecture used by Stage 1 and by distillation."""
    model = DeepSC(
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        num_downsample_blocks=cfg.NUM_DOWNSAMPLE_BLOCKS,
        base_channels=cfg.BASE_CHANNELS,
        num_embeddings_list=cfg.NUM_EMBEDDINGS_LIST,
        embedding_dim_list=cfg.EMBEDDING_DIM_LIST,
        commitment_cost=cfg.COMMITMENT_COST,
        device=device,
        strides=cfg.DOWNSAMPLE_STRIDES,
        skip_dropout_p=[0.0] * max(0, cfg.NUM_DOWNSAMPLE_BLOCKS - 1),
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
        use_swinir_enhance=False,
        quantizer_type="simvq",
        quantizer_axis_list=["patch"] * cfg.NUM_DOWNSAMPLE_BLOCKS,
        cvq_codeword_shapes=[None] * cfg.NUM_DOWNSAMPLE_BLOCKS,
        use_raq=False,
    ).to(device)
    model.set_channel_prob(0.0)
    return model


def load_checkpoint_state(path: str | Path, map_location="cpu") -> Dict[str, torch.Tensor]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=map_location)
    return extract_state_dict(checkpoint)


def load_frozen_teacher(cfg, checkpoint_path: str | Path, device: torch.device) -> DeepSC:
    """Load exactly one full teacher and make its immutability explicit."""
    teacher = build_source_teacher(cfg, device)
    state_dict = load_checkpoint_state(checkpoint_path, map_location="cpu")
    load_state_dict_compatible(teacher, state_dict, strict=True)
    freeze_teacher(teacher)
    return teacher


def freeze_teacher(teacher: DeepSC) -> DeepSC:
    teacher.eval()
    teacher.set_channel_prob(0.0)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return teacher


@torch.no_grad()
def teacher_forward(teacher: DeepSC, images: torch.Tensor) -> Dict[str, object]:
    """Return every clean maximum-rate target required by RAQ training."""
    teacher.eval()
    images = teacher._to_encoder_device(images)
    encoder_features = teacher.semantic_encoder(images)
    encoder_features[-1] = teacher.bottleneck_attention(encoder_features[-1])

    quantized_features = []
    raw_quantized_features = []
    vq_losses = []
    indices = []
    source_codebooks = []
    for feature, quantizer in zip(encoder_features, teacher.vector_quantizers):
        vq_loss, quantized, encoding_idx, quantized_raw = quantizer(
            feature, return_raw=True
        )
        quantized_features.append(quantized)
        raw_quantized_features.append(quantized_raw)
        vq_losses.append(vq_loss)
        indices.append(encoding_idx)
        source_codebooks.append(quantizer.transformed_weight())

    reconstruction = teacher._decode_features(quantized_features)
    return {
        "reconstructed_images": reconstruction.detach(),
        "reconstruction": reconstruction.detach(),
        "encoder_features": [feature.detach() for feature in encoder_features],
        "quantized_features": [feature.detach() for feature in raw_quantized_features],
        # The variable-rate loss aligns the student's raw selected-code
        # features with this immutable SRC target.  Keep explicit aliases so
        # no caller can accidentally fall back to a detached student branch.
        "raw_quantized_features": [
            feature.detach() for feature in raw_quantized_features
        ],
        "student_features": [feature.detach() for feature in raw_quantized_features],
        "vq_losses": [loss.detach() for loss in vq_losses],
        "indices": [index.detach() for index in indices],
        "source_codebooks": [codebook.detach() for codebook in source_codebooks],
    }


def copy_teacher_into_student(student, teacher: DeepSC) -> None:
    """Initialize a student backbone/codebook bank without sharing tensors."""
    module_names: Iterable[str] = (
        "semantic_encoder",
        "semantic_decoder",
        "bottleneck_attention",
        "vector_quantizers",
    )
    for name in module_names:
        student_module = getattr(student, name)
        teacher_module = getattr(teacher, name)
        student_module.load_state_dict(teacher_module.state_dict(), strict=True)
    if hasattr(student, "freeze_source_codebooks"):
        student.freeze_source_codebooks()


def assert_teacher_has_no_grad(teacher: DeepSC) -> None:
    offenders = [
        name
        for name, parameter in teacher.named_parameters()
        if parameter.requires_grad or parameter.grad is not None
    ]
    if offenders:
        raise RuntimeError(
            "Frozen teacher received trainable state/gradients: " + ", ".join(offenders[:8])
        )
