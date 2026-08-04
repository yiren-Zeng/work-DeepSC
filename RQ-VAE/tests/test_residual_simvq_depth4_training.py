"""Contracts for the independently trained K=[4,2], depth=[4,4] variant."""

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from models.deepsc import DeepSC
from models.residual_simvq_quantizer import ResidualSimVQQuantizer
from train import (
    _accumulate_rq_diagnostics,
    _average_rq_diagnostics,
    _empty_rq_diagnostic_accumulator,
)


torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = "/workspace/yi/work/Simvq-dc-64-Multi-pro-2ceng"
EXPERIMENT_NAME = (
    "quality_v2_B_larger_rate094_residual_simvq_"
    "unet2_ds8x2_k4-2_d4-4"
)
TRAIN_SCRIPT = (
    ROOT
    / "scripts/train/current/run_residual_simvq_k4-2_d4-4_rate094.sh"
)
RESUME_SCRIPT = (
    ROOT
    / "scripts/train/current/resume_residual_simvq_k4-2_d4-4_rate094.sh"
)
NOCHANNEL_SCRIPT = (
    ROOT
    / "scripts/eval/test_residual_simvq_k4-2_d4-4_rate094_nochannel.sh"
)
LDPC_SCRIPT = (
    ROOT
    / "scripts/eval/test_residual_simvq_k4-2_d4-4_rate094_ldpc_bpsk.sh"
)


def _small_depth4_model():
    return DeepSC(
        in_channels=3,
        out_channels=3,
        num_downsample_blocks=2,
        base_channels=4,
        num_embeddings_list=[4, 2],
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
        quantizer_type="residual_simvq",
        quantizer_axis_list=["patch", "patch"],
        cvq_codeword_shapes=[None, None],
        rq_depth_list=[4, 4],
        rq_ema_decay=0.99,
        rq_restart_unused_codes=False,
        rq_shared_codebook=True,
    )


def test_depth4_exact_loss_uses_all_q_and_commitment_stages():
    quantizer = ResidualSimVQQuantizer(
        num_embeddings=2,
        embedding_dim=1,
        commitment_cost=0.25,
        rq_depth=4,
    ).eval()
    with torch.no_grad():
        quantizer.codebook.embed.weight.copy_(torch.tensor([[0.0], [1.0]]))
        quantizer.codebook.proj.weight.fill_(1.0)

    inputs = torch.tensor([[[[0.0, 0.4, 1.0, 1.8]]]])
    loss, quantized, indices = quantizer(inputs)
    diagnostics = quantizer.get_last_diagnostics()

    expected_indices = torch.tensor(
        [[[[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]]]]
    )
    expected_quantized = torch.tensor([[[[0.0, 0.0, 1.0, 2.0]]]])
    expected_per_depth = torch.tensor([0.20, 0.05, 0.05, 0.05])
    expected_component = expected_per_depth.mean()

    assert tuple(indices.shape) == (1, 1, 4, 4)
    assert torch.equal(indices, expected_indices)
    assert torch.allclose(quantized, expected_quantized)
    assert torch.allclose(
        quantizer.get_quantized_features(indices), expected_quantized
    )
    assert torch.allclose(
        diagnostics["codebook_per_depth"], expected_per_depth
    )
    assert torch.allclose(
        diagnostics["commitment_per_depth"], expected_per_depth
    )
    assert torch.allclose(
        loss, expected_component + 0.25 * expected_component
    )
    assert all(
        codebook is quantizer.codebook for codebook in quantizer.codebooks
    )


def test_depth4_two_scale_forward_decode_gradients_and_monitoring():
    torch.manual_seed(61)
    model = _small_depth4_model()
    model.channel_prob = 0.0
    image = torch.randn(1, 3, 16, 16, requires_grad=True)

    output = model.forward_train(image)
    loss = output["reconstructed_images"].square().mean() + sum(
        output["vq_losses"]
    )
    loss.backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    for quantizer in model.vector_quantizers:
        assert quantizer.codebook.proj.weight.grad is not None
        assert torch.isfinite(quantizer.codebook.proj.weight.grad).all()
        assert quantizer.codebook.embed.weight.grad is None

    with torch.no_grad():
        encoded = model.eval().forward_test(image.detach())
        assert [tuple(indices.shape) for indices in encoded["indices"]] == [
            (1, 8, 8, 4),
            (1, 4, 4, 4),
        ]
        reconstructed = model.reconstruct_from_indices(
            encoded["indices"], feature_shapes=encoded["feature_shapes"]
        )
    assert reconstructed.shape == image.shape

    accumulator = _empty_rq_diagnostic_accumulator([4, 4])
    _accumulate_rq_diagnostics(
        accumulator,
        [
            quantizer.get_last_diagnostics()
            for quantizer in model.vector_quantizers
        ],
    )
    averaged = _average_rq_diagnostics(accumulator)
    assert len(averaged) == 2
    for scale in averaged:
        assert len(scale["codebook_per_depth"]) == 4
        assert len(scale["commitment_per_depth"]) == 4
        assert len(scale["residual_norm_per_depth"]) == 4
        assert len(scale["usage_per_depth"]) == 4
        assert len(scale["perplexity_per_depth"]) == 4


def test_depth4_production_config_name_and_rate_are_exact():
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SIMVQ_")
    }
    environment.update(
        {
            "SIMVQ_EXPERIMENT_STAGE": "B",
            "SIMVQ_EXP_FAMILY": "quality_v2_B_larger_rate094",
            "SIMVQ_UNET_DEPTH": "2",
            "SIMVQ_DOWNSAMPLE_STRIDES": "8,2",
            "SIMVQ_BASE_CHANNELS": "128",
            "SIMVQ_ENCODER_RES_BLOCKS": "6",
            "SIMVQ_DECODER_RES_BLOCKS": "6",
            "SIMVQ_QUANTIZER_TYPE": "residual_simvq",
            "SIMVQ_QUANTIZER_AXIS_LIST": "patch,patch",
            "SIMVQ_CVQ_CODEWORD_SHAPES": "patch,patch",
            "SIMVQ_NUM_EMBEDDINGS_LIST": "4,2",
            "SIMVQ_RQ_DEPTH_LIST": "4,4",
            "SIMVQ_RQ_SHARED_CODEBOOK": "1",
            "SIMVQ_RQ_RESTART_UNUSED_CODES": "0",
            "SIMVQ_LAYER_LOSS_WEIGHTS_INIT": "0.25,0.50",
            "SIMVQ_LAYER_LOSS_WEIGHTS_FINAL": "0.25,0.25",
            "SIMVQ_LPIPS_WEIGHT": "0",
        }
    )
    code = (
        "import json; from config import Config; Config.validate(); "
        "print(json.dumps({"
        "'name':Config.EXPERIMENT_NAME,"
        "'bits':Config.ESTIMATED_SOURCE_BITS_PER_IMAGE,"
        "'bpp':Config.ESTIMATED_SOURCE_BPP,"
        "'ratio':Config.ESTIMATED_TEST_TRANSMISSION_RATIO,"
        "'dims':Config.EMBEDDING_DIM_LIST,"
        "'depths':Config.RQ_DEPTH_LIST,"
        "'weights_init':Config.LAYER_LOSS_WEIGHTS_INIT,"
        "'weights_final':Config.LAYER_LOSS_WEIGHTS_FINAL,"
        "'epochs':Config.NUM_EPOCHS"
        "}))"
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
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == {
        "name": EXPERIMENT_NAME,
        "bits": 9216.0,
        "bpp": 0.140625,
        "ratio": 0.09375,
        "dims": [256, 512],
        "depths": [4, 4],
        "weights_init": [0.25, 0.5],
        "weights_final": [0.25, 0.25],
        "epochs": 200,
    }


def test_depth4_scripts_are_isolated_fresh_and_executable():
    scripts = (TRAIN_SCRIPT, NOCHANNEL_SCRIPT, LDPC_SCRIPT)
    for script in scripts:
        assert script.is_file()
        assert os.access(script, os.X_OK)
        content = script.read_text(encoding="utf-8")
        assert "cd /workspace/yi/work/RQ-VAE" in content
        assert OLD_ROOT not in content
        assert 'SIMVQ_EXP_FAMILY="quality_v2_B_larger_rate094"' in content
        assert 'SIMVQ_QUANTIZER_TYPE="residual_simvq"' in content
        assert 'SIMVQ_NUM_EMBEDDINGS_LIST="4,2"' in content
        assert 'SIMVQ_RQ_DEPTH_LIST="4,4"' in content
        assert 'SIMVQ_RQ_SHARED_CODEBOOK="1"' in content
        assert 'SIMVQ_LAYER_LOSS_WEIGHTS_INIT="0.25,0.50"' in content
        assert 'SIMVQ_LAYER_LOSS_WEIGHTS_FINAL="0.25,0.25"' in content
        assert 'GPU_ID="${GPU_ID:-1}"' in content
        assert 'SIMVQ_RESUME="0"' in content
        assert "unset SIMVQ_PRETRAINED_CHECKPOINT" in content
        assert EXPERIMENT_NAME in content
        syntax = subprocess.run(
            ["bash", "-n", str(script)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert syntax.returncode == 0, syntax.stderr

    train_content = TRAIN_SCRIPT.read_text(encoding="utf-8")
    assert 'SIMVQ_TOTAL_BATCH_SIZE="${SIMVQ_TOTAL_BATCH_SIZE:-24}"' in train_content
    assert 'SIMVQ_MICRO_BATCH_SIZE="${SIMVQ_MICRO_BATCH_SIZE:-24}"' in train_content
    assert 'SIMVQ_NUM_EPOCHS="200"' in train_content
    assert "9216 bits/image = 0.140625 bpp" in train_content
    assert "0.09375000" in train_content

    no_channel_content = NOCHANNEL_SCRIPT.read_text(encoding="utf-8")
    assert "--no-channel" in no_channel_content
    assert "residual_simvq_k4-2_d4-4_rate094_nochannel.json" in no_channel_content

    ldpc_content = LDPC_SCRIPT.read_text(encoding="utf-8")
    assert "--snrs 0 3 6 9 12" in ldpc_content
    assert "--modulation bpsk" in ldpc_content
    assert "residual_simvq_k4-2_d4-4_rate094_ldpc_bpsk.json" in ldpc_content


def test_depth4_resume_script_is_fail_closed_and_preserves_training_config():
    assert RESUME_SCRIPT.is_file()
    assert os.access(RESUME_SCRIPT, os.X_OK)
    content = RESUME_SCRIPT.read_text(encoding="utf-8")
    assert "cd /workspace/yi/work/RQ-VAE" in content
    assert OLD_ROOT not in content
    assert 'SIMVQ_EXP_FAMILY="quality_v2_B_larger_rate094"' in content
    assert 'SIMVQ_QUANTIZER_TYPE="residual_simvq"' in content
    assert 'SIMVQ_NUM_EMBEDDINGS_LIST="4,2"' in content
    assert 'SIMVQ_RQ_DEPTH_LIST="4,4"' in content
    assert 'SIMVQ_RESUME="1"' in content
    assert "unset SIMVQ_PRETRAINED_CHECKPOINT" in content
    assert 'GPU_ID="${GPU_ID:-1}"' in content
    assert 'SIMVQ_TOTAL_BATCH_SIZE="${SIMVQ_TOTAL_BATCH_SIZE:-24}"' in content
    assert 'SIMVQ_MICRO_BATCH_SIZE="${SIMVQ_MICRO_BATCH_SIZE:-24}"' in content
    assert 'SIMVQ_NUM_EPOCHS="200"' in content
    assert 'if [ ! -s "$CHECKPOINT_PATH" ]; then' in content
    assert EXPERIMENT_NAME in content
    syntax = subprocess.run(
        ["bash", "-n", str(RESUME_SCRIPT)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr
