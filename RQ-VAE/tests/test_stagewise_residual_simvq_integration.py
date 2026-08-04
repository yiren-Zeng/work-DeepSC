"""Integration contracts for independent per-depth Residual-SimVQ."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from config import _source_bpp
from models.deepsc import DeepSC
from train import (
    build_optimizer_parameter_groups,
    projection_gradient_norms,
)
from utils.bit_utils import bits_to_indices, indices_to_bits
from utils.checkpoint_utils import (
    build_checkpoint_payload,
    build_model_from_checkpoint,
)


torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = (
    ROOT
    / "scripts/train/current/"
    "run_stagewise_residual_simvq_k8x2-2x2_rate047.sh"
)
NOCHANNEL_SCRIPT = (
    ROOT
    / "scripts/eval/"
    "test_stagewise_residual_simvq_k8x2-2x2_rate047_nochannel.sh"
)
LDPC_SCRIPT = (
    ROOT
    / "scripts/eval/"
    "test_stagewise_residual_simvq_k8x2-2x2_rate047_ldpc_bpsk.sh"
)
EXPERIMENT_NAME = (
    "quality_v2_B_larger_rate047_stagewise_residual_simvq_"
    "unet2_ds8x2_k8x2-2x2_d2-2"
)


def _small_model():
    return DeepSC(
        in_channels=3,
        out_channels=3,
        num_downsample_blocks=2,
        base_channels=4,
        num_embeddings_list=[8, 2],
        embedding_dim_list=[8, 16],
        commitment_cost=0.25,
        device=torch.device("cpu"),
        strides=[2, 2],
        skip_dropout_p=[0.0],
        norm_type="group",
        norm_groups=4,
        activation="silu",
        encoder_res_blocks=0,
        decoder_res_blocks=0,
        upsample_mode="nearest",
        use_cascade_downsample=False,
        use_bottleneck_attention=False,
        quantizer_type="stagewise_residual_simvq",
        quantizer_axis_list=["patch", "patch"],
        cvq_codeword_shapes=[None, None],
        rq_depth_list=[2, 2],
        rq_shared_codebook=False,
        rq_codebook_size_lists=[[8, 2], [2, 2]],
    )


def test_stagewise_deepsc_encode_decode_contract_and_ranges():
    torch.manual_seed(301)
    model = _small_model().eval()
    image = torch.randn(1, 3, 16, 16)

    with torch.no_grad():
        encoded = model.forward_test(image)
        reconstructed = model.reconstruct_from_indices(
            encoded["indices"], feature_shapes=encoded["feature_shapes"]
        )
        bit_stream, token_shapes, codebook_sizes, stats = indices_to_bits(
            encoded["indices"],
            [[8, 2], [2, 2]],
            return_stats=True,
        )
        restored_indices = bits_to_indices(
            bit_stream,
            token_shapes,
            codebook_sizes,
        )
        restored_reconstruction = model.reconstruct_from_indices(
            restored_indices,
            feature_shapes=encoded["feature_shapes"],
        )

    assert model.quantizer_type == "stagewise_residual_simvq"
    assert model.rq_codebook_size_lists == [[8, 2], [2, 2]]
    assert [tuple(indices.shape) for indices in encoded["indices"]] == [
        (1, 8, 8, 2),
        (1, 4, 4, 2),
    ]
    assert int(encoded["indices"][0][..., 0].max()) < 8
    assert int(encoded["indices"][0][..., 1].max()) < 2
    assert int(encoded["indices"][1].max()) < 2
    assert reconstructed.shape == image.shape
    assert stats["total_bits"] == 288
    assert all(
        torch.equal(original.cpu().squeeze(0), restored)
        for original, restored in zip(encoded["indices"], restored_indices)
    )
    assert torch.equal(reconstructed, restored_reconstruction)
    for quantizer in model.vector_quantizers:
        assert quantizer.codebooks[0] is not quantizer.codebooks[1]


def test_stagewise_optimizer_groups_every_projection_and_training_step():
    torch.manual_seed(302)
    model = _small_model().train()
    model.set_channel_prob(0.0)
    groups, audit = build_optimizer_parameter_groups(
        model,
        base_learning_rate=5e-5,
        codebook_projection_learning_rate=2e-4,
        quantizer_type="stagewise_residual_simvq",
    )
    optimizer = torch.optim.Adam(groups)
    image = torch.randn(2, 3, 16, 16)

    output = model.forward_train(image)
    loss = output["reconstructed_images"].square().mean() + sum(
        output["vq_losses"]
    )
    loss.backward()

    assert len(audit["projection_names"]) == 4
    assert len(audit["frozen_embedding_names"]) == 4
    assert all(".codebooks." in name for name in audit["projection_names"])
    assert all(name.endswith(".proj.weight") for name in audit["projection_names"])
    assert all(
        name.endswith(".embed.weight")
        for name in audit["frozen_embedding_names"]
    )
    gradient_norms = projection_gradient_norms(model)
    assert len(gradient_norms) == 2
    assert all(value is not None and value > 0 for value in gradient_norms)
    optimizer.step()


def test_stagewise_training_with_channel_is_finite():
    torch.manual_seed(304)
    model = _small_model().train()
    model.set_channel_prob(1.0)
    image = torch.randn(1, 3, 16, 16)

    output = model.forward_train(image)
    loss = output["reconstructed_images"].square().mean() + sum(
        output["vq_losses"]
    )
    loss.backward()

    assert output["channel_used"] is True
    assert output["reconstructed_images"].shape == image.shape
    assert torch.isfinite(loss)
    for scale, diagnostics in enumerate(output["quantizer_diagnostics"]):
        expected_sizes = [[8, 2], [2, 2]][scale]
        actual_sizes = [
            int(counts.numel())
            for counts in diagnostics["usage_counts_per_depth"]
        ]
        assert actual_sizes == expected_sizes


def test_stagewise_source_rate_matches_shared_rate047():
    stagewise_bpp = _source_bpp(
        [8, 2],
        [8, 2],
        ["patch", "patch"],
        [256, 512],
        (256, 256),
        [2, 2],
        [[8, 2], [2, 2]],
    )
    shared_bpp = _source_bpp(
        [8, 2],
        [4, 2],
        ["patch", "patch"],
        [256, 512],
        (256, 256),
        [2, 2],
    )

    assert stagewise_bpp == shared_bpp == 0.0703125
    assert stagewise_bpp * 256 * 256 == 4608
    assert stagewise_bpp / (0.5 * 1 * 3) == 0.046875


def test_stagewise_real_checkpoint_roundtrip():
    torch.manual_seed(303)
    model = _small_model().eval()
    config = SimpleNamespace(
        validate=lambda: None,
        EXPERIMENT_NAME="stagewise_roundtrip",
        EXPERIMENT_FAMILY="stagewise_roundtrip",
        EXPERIMENT_STAGE="B",
        QUANTIZER_TYPE="stagewise_residual_simvq",
        NUM_DOWNSAMPLE_BLOCKS=2,
        UNET_DEPTH=2,
        NUM_EMBEDDINGS_LIST=[8, 2],
        EMBEDDING_DIM_LIST=[8, 16],
        RQ_DEPTH_LIST=[2, 2],
        RQ_CODEBOOK_SIZE_LISTS=[[8, 2], [2, 2]],
        RQ_SHARED_CODEBOOK=False,
        RQ_EMA_DECAY=0.99,
        RQ_RESTART_UNUSED_CODES=False,
        IN_CHANNELS=3,
        OUT_CHANNELS=3,
        BASE_CHANNELS=4,
        DOWNSAMPLE_STRIDES=[2, 2],
        QUANTIZER_AXIS_LIST=["patch", "patch"],
        CVQ_CODEWORD_SHAPES=[None, None],
        COMMITMENT_COST=0.25,
        NORM_TYPE="group",
        GROUP_NORM_GROUPS=4,
        ACTIVATION="silu",
        ENCODER_RES_BLOCKS=0,
        DECODER_RES_BLOCKS=0,
        UPSAMPLE_MODE="nearest",
        USE_CASCADE_DOWNSAMPLE=False,
        USE_BOTTLENECK_ATTENTION=False,
        BOTTLENECK_ATTENTION_BLOCKS=1,
        USE_SWINIR_ENHANCE=False,
        SWINIR_ENHANCE_BLOCKS=4,
        SKIP_DROPOUT_P_INIT=[0.0],
        SKIP_DROPOUT_P_FINAL=[0.0],
        LAYER_LOSS_WEIGHTS_INIT=[0.25, 0.50],
        LAYER_LOSS_WEIGHTS_FINAL=[0.25, 0.25],
        MSE_LOSS_WEIGHT=1.0,
        MS_SSIM_LOSS_WEIGHT=0.0,
        LPIPS_LOSS_WEIGHT=0.0,
        CHANNEL_CODING_RATE_TRAIN=0.5,
        CHANNEL_CODING_RATE_VAL=0.5,
        BLOCK_LENGTH=256,
        SNR_RANGE_DB=[0, 15],
        TRAIN_IMAGE_SIZE=(16, 16),
        TEST_IMAGE_SIZE=(16, 16),
        ESTIMATED_SOURCE_BITS_PER_IMAGE=18,
        ESTIMATED_SOURCE_BPP=0.0703125,
        ESTIMATED_TEST_SOURCE_BPP=0.0703125,
        ESTIMATED_TEST_TRANSMISSION_RATIO=0.046875,
        MODEL_PARALLEL=False,
    )
    payload = build_checkpoint_payload(model, config, epoch=0)

    with tempfile.TemporaryDirectory(prefix="stagewise_checkpoint_") as temp_dir:
        checkpoint_path = Path(temp_dir) / "model.pth"
        torch.save(payload, checkpoint_path)
        restored, inferred = build_model_from_checkpoint(
            checkpoint_path, config, torch.device("cpu")
        )

    assert inferred["quantizer_type"] == "stagewise_residual_simvq"
    assert inferred["rq_codebook_size_lists"] == [[8, 2], [2, 2]]
    assert restored.rq_codebook_size_lists == [[8, 2], [2, 2]]
    expected_state = model.state_dict()
    actual_state = restored.state_dict()
    assert tuple(actual_state) == tuple(expected_state)
    assert all(
        torch.equal(actual_state[key], expected_state[key])
        for key in expected_state
    )


def test_stagewise_config_contract_in_fresh_process():
    environment = os.environ.copy()
    environment.update(
        {
            "SIMVQ_EXPERIMENT_STAGE": "B",
            "SIMVQ_EXP_FAMILY": "quality_v2_B_larger_rate047",
            "SIMVQ_UNET_DEPTH": "2",
            "SIMVQ_DOWNSAMPLE_STRIDES": "8,2",
            "SIMVQ_BASE_CHANNELS": "128",
            "SIMVQ_ENCODER_RES_BLOCKS": "6",
            "SIMVQ_DECODER_RES_BLOCKS": "6",
            "SIMVQ_QUANTIZER_TYPE": "stagewise_residual_simvq",
            "SIMVQ_QUANTIZER_AXIS_LIST": "patch,patch",
            "SIMVQ_CVQ_CODEWORD_SHAPES": "patch,patch",
            "SIMVQ_NUM_EMBEDDINGS_LIST": "8,2",
            "SIMVQ_RQ_DEPTH_LIST": "2,2",
            "SIMVQ_RQ_CODEBOOK_SIZES": "8,2;2,2",
            "SIMVQ_RQ_SHARED_CODEBOOK": "0",
            "SIMVQ_LAYER_LOSS_WEIGHTS_INIT": "0.25,0.50",
            "SIMVQ_LAYER_LOSS_WEIGHTS_FINAL": "0.25,0.25",
            "SIMVQ_LPIPS_WEIGHT": "0",
        }
    )
    code = (
        "from config import Config;"
        "Config.validate();"
        "print({'name':Config.EXPERIMENT_NAME,"
        "'flat':Config.NUM_EMBEDDINGS_LIST,"
        "'nested':Config.RQ_CODEBOOK_SIZE_LISTS,"
        "'bits':Config.ESTIMATED_SOURCE_BITS_PER_IMAGE,"
        "'ratio':Config.ESTIMATED_TEST_TRANSMISSION_RATIO})"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert EXPERIMENT_NAME in completed.stdout
    assert "'flat': [8, 2]" in completed.stdout
    assert "'nested': [[8, 2], [2, 2]]" in completed.stdout
    assert "'bits': 4608" in completed.stdout
    assert "'ratio': 0.046875" in completed.stdout


def test_stagewise_scripts_are_fresh_isolated_and_syntax_valid():
    for path in (TRAIN_SCRIPT, NOCHANNEL_SCRIPT, LDPC_SCRIPT):
        assert path.is_file()
        assert os.access(path, os.X_OK)
        content = path.read_text(encoding="utf-8")
        assert "cd /workspace/yi/work/RQ-VAE" in content
        assert 'SIMVQ_QUANTIZER_TYPE="stagewise_residual_simvq"' in content
        assert 'SIMVQ_RQ_CODEBOOK_SIZES="8,2;2,2"' in content
        assert 'SIMVQ_RQ_SHARED_CODEBOOK="0"' in content
        assert EXPERIMENT_NAME in content
        syntax = subprocess.run(
            ["bash", "-n", str(path)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert syntax.returncode == 0, syntax.stderr

    train_content = TRAIN_SCRIPT.read_text(encoding="utf-8")
    assert 'SIMVQ_RESUME="0"' in train_content
    assert "unset SIMVQ_PRETRAINED_CHECKPOINT" in train_content
    assert "4608 bits/image = 0.0703125 bpp" in train_content
    assert "train.py" in train_content

    no_channel_content = NOCHANNEL_SCRIPT.read_text(encoding="utf-8")
    assert "--no-channel" in no_channel_content
    ldpc_content = LDPC_SCRIPT.read_text(encoding="utf-8")
    assert "--snrs 0 3 6 9 12" in ldpc_content
    assert "--modulation bpsk" in ldpc_content
