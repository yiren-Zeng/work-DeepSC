import io
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

from models.deepsc import DeepSC
from utils.checkpoint_utils import (
    build_checkpoint_payload,
    extract_checkpoint_metadata,
    extract_state_dict,
    infer_codebook_config,
)


ROOT = Path(__file__).resolve().parents[1]
NEW_SCRIPTS = [
    ROOT / "scripts/train/current/run_rq_ema_k4-2_d2-2_rate047.sh",
    ROOT / "scripts/eval/test_rq_ema_k4-2_d2-2_rate047.sh",
    ROOT / "scripts/eval/test_rq_ema_k4-2_d2-2_rate047_nochannel.sh",
    ROOT
    / "scripts/eval/test_rq_ema_k4-2_d2-2_rate047_adaptive_topk_ldpc_bpsk.sh",
    ROOT
    / "scripts/eval/test_rq_ema_k4-2_d2-2_extend_d1-4_nochannel.sh",
    ROOT
    / "scripts/eval/test_rq_ema_k4-2_d2-2_extend_d4_ldpc_bpsk.sh",
]
OLD_ROOT = "/workspace/yi/work/Simvq-dc-64-Multi-pro-2ceng"


def _small_model(quantizer_type="rq_ema"):
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
        rq_restart_unused_codes=True,
        rq_shared_codebook=True,
    ).eval()


def _metadata_config():
    return SimpleNamespace(
        EXPERIMENT_NAME="test_rq_ema_unet2_ds2x2_k4-2_d2-2",
        EXPERIMENT_FAMILY="test_rq_ema",
        EXPERIMENT_STAGE="B",
        QUANTIZER_TYPE="rq_ema",
        NUM_DOWNSAMPLE_BLOCKS=2,
        UNET_DEPTH=2,
        NUM_EMBEDDINGS_LIST=[4, 2],
        EMBEDDING_DIM_LIST=[8, 16],
        RQ_DEPTH_LIST=[2, 2],
        RQ_EMA_DECAY=0.99,
        RQ_RESTART_UNUSED_CODES=True,
        RQ_SHARED_CODEBOOK=True,
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
        BOTTLENECK_ATTENTION_BLOCKS=0,
        USE_SWINIR_ENHANCE=False,
        SWINIR_ENHANCE_BLOCKS=0,
        SKIP_DROPOUT_P_INIT=[0.0],
        CHANNEL_CODING_RATE_TRAIN=0.5,
        CHANNEL_CODING_RATE_VAL=0.5,
        BLOCK_LENGTH=256,
        SNR_RANGE_DB=[0, 15],
        TRAIN_IMAGE_SIZE=(256, 256),
        TEST_IMAGE_SIZE=(256, 256),
        ESTIMATED_SOURCE_BITS_PER_IMAGE=4608,
        ESTIMATED_SOURCE_BPP=0.0703125,
        ESTIMATED_TEST_SOURCE_BPP=0.0703125,
        ESTIMATED_TEST_TRANSMISSION_RATIO=0.046875,
    )


def test_new_shell_scripts_are_valid_executable_and_confined_to_rq_vae():
    for script in NEW_SCRIPTS:
        assert script.is_file()
        assert os.access(script, os.X_OK)
        content = script.read_text(encoding="utf-8")
        assert "cd /workspace/yi/work/RQ-VAE" in content
        assert OLD_ROOT not in content
        completed = subprocess.run(
            ["bash", "-n", str(script)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr


def test_checkpoint_metadata_survives_serialization_and_drives_rq_inference():
    model = _small_model()
    config = _metadata_config()
    payload = build_checkpoint_payload(model, config, epoch=7, best_val_loss=0.125)

    stream = io.BytesIO()
    torch.save(payload, stream)
    stream.seek(0)
    restored = torch.load(stream, map_location="cpu", weights_only=False)
    metadata = extract_checkpoint_metadata(restored)
    state_dict = extract_state_dict(restored)
    inferred = infer_codebook_config(state_dict, config, metadata=metadata)

    assert restored["epoch"] == 7
    assert metadata["quantizer_type"] == "rq_ema"
    assert metadata["num_embeddings_list"] == [4, 2]
    assert metadata["rq_depth_list"] == [2, 2]
    assert metadata["rate"]["source_bits_per_image"] == 4608
    assert inferred["quantizer_type"] == "rq_ema"
    assert inferred["num_embeddings_list"] == [4, 2]
    assert inferred["embedding_dim_list"] == [8, 16]
    assert inferred["rq_depth_list"] == [2, 2]
    assert inferred["rq_shared_codebook"] is True

    reloaded_model = _small_model()
    reloaded_model.load_state_dict(state_dict)
    assert reloaded_model.vector_quantizers[0].codebooks[0] is reloaded_model.vector_quantizers[0].codebooks[1]
    assert reloaded_model.vector_quantizers[1].codebooks[0] is reloaded_model.vector_quantizers[1].codebooks[1]
    for expected, actual in zip(model.state_dict().values(), reloaded_model.state_dict().values()):
        assert torch.equal(expected, actual)


def test_bare_rq_checkpoint_inference_excludes_padding_and_shared_depth_duplicates():
    model = _small_model()
    inferred = infer_codebook_config(model.state_dict())

    assert inferred["quantizer_type"] == "rq_ema"
    assert inferred["num_downsample_blocks"] == 2
    assert inferred["num_embeddings_list"] == [4, 2]
    assert inferred["embedding_dim_list"] == [8, 16]
    assert inferred["rq_depth_list"] == [2, 2]
    assert inferred["rq_shared_codebook"] is True


def test_legacy_bare_simvq_checkpoint_inference_remains_compatible():
    model = _small_model("simvq")
    inferred = infer_codebook_config(model.state_dict())

    assert inferred["quantizer_type"] == "simvq"
    assert inferred["num_downsample_blocks"] == 2
    assert inferred["num_embeddings_list"] == [4, 2]
    assert inferred["embedding_dim_list"] == [8, 16]
