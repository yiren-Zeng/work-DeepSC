import os
import subprocess
from pathlib import Path

import torch

from evaluation.residual_simvq_depth_extension import (
    extend_residual_simvq_depth_for_eval,
    set_residual_simvq_depth_for_eval,
    truncate_residual_indices,
)
from models.deepsc import DeepSC
from utils.bit_utils import count_index_bits


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/eval/"
    "test_residual_simvq_k4-2_d2-2_rate047_depth1to4_ldpc_bpsk.sh"
)


def _make_model(quantizer_type="residual_simvq"):
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
        quantizer_type=quantizer_type,
        quantizer_axis_list=["patch", "patch"],
        cvq_codeword_shapes=[None, None],
        rq_depth_list=[2, 2],
        rq_ema_decay=0.99,
        rq_restart_unused_codes=False,
        rq_shared_codebook=True,
    ).eval()


def _assert_raises(error_type, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def test_residual_extension_changes_only_logical_depth():
    model = _make_model()
    parameter_ids = {id(value) for value in model.parameters()}
    buffer_ids = {id(value) for value in model.buffers()}
    state_keys = tuple(model.state_dict())
    state_values = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    report = extend_residual_simvq_depth_for_eval(model, [4, 4])

    assert report["loaded_rq_depth_list"] == [2, 2]
    assert report["runtime_rq_depth_list"] == [4, 4]
    assert report["added_registered_parameters"] == 0
    assert report["added_registered_buffers"] == 0
    assert report["added_state_dict_keys"] == 0
    assert report["same_projected_codebook_object_per_scale"] == [True, True]
    assert model.rq_depth_list == [4, 4]
    assert {id(value) for value in model.parameters()} == parameter_ids
    assert {id(value) for value in model.buffers()} == buffer_ids
    assert tuple(model.state_dict()) == state_keys
    for name, value in model.state_dict().items():
        assert torch.equal(value, state_values[name])
    for quantizer in model.vector_quantizers:
        assert quantizer.rq_depth == 4
        assert len(quantizer.codebooks) == 4
        assert all(
            codebook is quantizer.codebook for codebook in quantizer.codebooks
        )


def test_residual_depth4_prefix_preserves_trained_depth2_exactly():
    torch.manual_seed(131)
    model = _make_model()
    image = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        trained = model.forward_test(image)
        trained_reconstruction = model.reconstruct_from_indices(
            trained["indices"], feature_shapes=trained["feature_shapes"]
        )
        extend_residual_simvq_depth_for_eval(model, 4)
        extended = model.forward_test(image)

    assert [tuple(value.shape) for value in extended["indices"]] == [
        (1, 8, 8, 4),
        (1, 4, 4, 4),
    ]
    for trained_indices, extended_indices in zip(
        trained["indices"], extended["indices"]
    ):
        assert torch.equal(trained_indices, extended_indices[..., :2])

    prefixes = truncate_residual_indices(extended["indices"], 2)
    set_residual_simvq_depth_for_eval(model, 2)
    with torch.no_grad():
        prefix_reconstruction = model.reconstruct_from_indices(
            prefixes, feature_shapes=extended["feature_shapes"]
        )
    assert torch.allclose(
        prefix_reconstruction,
        trained_reconstruction,
        atol=1e-6,
        rtol=1e-5,
    )

    set_residual_simvq_depth_for_eval(model, 4)
    with torch.no_grad():
        depth4_reconstruction = model.reconstruct_from_indices(
            extended["indices"], feature_shapes=extended["feature_shapes"]
        )
    assert depth4_reconstruction.shape == image.shape
    assert torch.isfinite(depth4_reconstruction).all()


def test_residual_depth1_to4_fixed_rate_contract():
    torch.manual_seed(132)
    expected_bits = [2304, 4608, 6912, 9216]
    for depth, expected in enumerate(expected_bits, start=1):
        indices = [
            torch.randint(0, 4, (1, 32, 32, depth), dtype=torch.long),
            torch.randint(0, 2, (1, 16, 16, depth), dtype=torch.long),
        ]
        stats = count_index_bits(indices, [4, 2])
        assert stats["per_scale_bits"] == [2048 * depth, 256 * depth]
        assert stats["total_bits"] == expected
        assert expected % 128 == 0
        assert expected / (256 * 256) == 0.03515625 * depth
        assert (expected / (256 * 256)) / (0.5 * 1 * 3) == (
            0.0234375 * depth
        )


def test_residual_depth_extension_guards_and_instance_isolation():
    training_model = _make_model().train()
    _assert_raises(
        RuntimeError,
        extend_residual_simvq_depth_for_eval,
        training_model,
        4,
    )

    simvq_model = _make_model("simvq")
    _assert_raises(
        ValueError, extend_residual_simvq_depth_for_eval, simvq_model, 4
    )

    model = _make_model()
    untouched = _make_model()
    _assert_raises(ValueError, extend_residual_simvq_depth_for_eval, model, 1)
    _assert_raises(
        ValueError, extend_residual_simvq_depth_for_eval, model, [4]
    )
    _assert_raises(ValueError, set_residual_simvq_depth_for_eval, model, 0)
    extend_residual_simvq_depth_for_eval(model, 4)
    assert model.rq_depth_list == [4, 4]
    assert untouched.rq_depth_list == [2, 2]


def test_residual_depth1to4_shell_script_is_isolated_and_executable():
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK)
    content = SCRIPT.read_text(encoding="utf-8")
    assert "cd /workspace/yi/work/RQ-VAE" in content
    assert 'SIMVQ_QUANTIZER_TYPE="residual_simvq"' in content
    assert 'SIMVQ_NUM_EMBEDDINGS_LIST="4,2"' in content
    assert 'SIMVQ_RQ_DEPTH_LIST="2,2"' in content
    assert 'GPU_ID="${GPU_ID:-1}"' in content
    assert "test_residual_simvq_depth1to4.py" in content
    assert "--depths 1 2 3 4" in content
    assert "--channel-depths 1 2 3 4" in content
    assert "--snrs 0 3 6 9 12" in content
    assert "--modulation bpsk" in content
    completed = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
