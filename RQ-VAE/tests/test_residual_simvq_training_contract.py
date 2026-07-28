"""Training, monitoring, persistence, and script contracts for Residual-SimVQ."""

import csv
import io
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from config import _source_bpp
from models.deepsc import DeepSC
from monitoring.codebook import (
    compute_codebook_utilization,
    write_codebook_tensorboard,
)
from train import (
    _accumulate_projection_gradient_norms,
    _accumulate_rq_diagnostics,
    _average_rq_diagnostics,
    _empty_rq_diagnostic_accumulator,
    build_optimizer_parameter_groups,
    projection_gradient_norms,
)
from training.schedules import compute_schedule
from utils.checkpoint_utils import (
    build_checkpoint_payload,
    extract_checkpoint_metadata,
    extract_state_dict,
    infer_codebook_config,
)
from utils.experiment_io import (
    append_codebook_records,
    append_epoch_record,
    rq_epoch_metric_fields,
)


torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = "/workspace/yi/work/Simvq-dc-64-Multi-pro-2ceng"
TRAIN_SCRIPT = (
    ROOT
    / "scripts/train/current/run_residual_simvq_k4-2_d2-2_rate047.sh"
)
NOCHANNEL_SCRIPT = (
    ROOT
    / "scripts/eval/test_residual_simvq_k4-2_d2-2_rate047_nochannel.sh"
)
LDPC_SCRIPT = (
    ROOT
    / "scripts/eval/test_residual_simvq_k4-2_d2-2_rate047_ldpc_bpsk.sh"
)
RATE094_TRAIN_SCRIPT = (
    ROOT
    / "scripts/train/current/run_residual_simvq_k16-4_d2-2_rate094.sh"
)
RATE094_NOCHANNEL_SCRIPT = (
    ROOT
    / "scripts/eval/test_residual_simvq_k16-4_d2-2_rate094_nochannel.sh"
)
RATE094_LDPC_SCRIPT = (
    ROOT
    / "scripts/eval/test_residual_simvq_k16-4_d2-2_rate094_ldpc_bpsk.sh"
)


def _small_model():
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
        rq_depth_list=[2, 2],
        rq_ema_decay=0.99,
        rq_restart_unused_codes=False,
        rq_shared_codebook=True,
    )


def _set_unit_projection_gradients(model):
    for quantizer in model.vector_quantizers:
        parameter = quantizer.codebook.proj.weight
        parameter.grad = torch.ones_like(parameter)


def _monitoring_results():
    torch.manual_seed(51)
    model = _small_model()
    _set_unit_projection_gradients(model)
    dataloader = [torch.randn(2, 3, 16, 16)]
    return compute_codebook_utilization(
        model, dataloader, max_batches=1, device="cpu"
    )


def test_two_scale_nochannel_indices_reconstruct_clean_quantized_features():
    torch.manual_seed(50)
    model = _small_model().eval()
    image = torch.randn(1, 3, 16, 16)

    with torch.no_grad():
        encoded = model.forward_test(image)
        reconstructed = model.reconstruct_from_indices(
            encoded["indices"], feature_shapes=encoded["feature_shapes"]
        )
        features = model.semantic_encoder(image)
        features[-1] = model.bottleneck_attention(features[-1])
        clean_quantized = [
            quantizer(feature)[1]
            for quantizer, feature in zip(model.vector_quantizers, features)
        ]
        expected = model.swinir_enhance(
            model.semantic_decoder(clean_quantized)
        )

    assert [tuple(indices.shape) for indices in encoded["indices"]] == [
        (1, 8, 8, 2),
        (1, 4, 4, 2),
    ]
    assert reconstructed.shape == image.shape
    assert torch.allclose(reconstructed, expected, atol=1e-6, rtol=1e-5)


def test_residual_bhwd_indices_pass_through_finite_channel_and_decode():
    torch.manual_seed(50)
    model = _small_model().eval()
    encoded = model.forward_test(torch.randn(1, 3, 16, 16))
    recovered = []
    for indices, codebook_size, quantizer in zip(
        encoded["indices"], [4, 2], model.vector_quantizers
    ):
        corrupted, ber = model.channel.apply_channel_noise(
            indices,
            codebook_size,
            snr_db=torch.tensor(0.0),
            rc=0.5,
            mod_bits=1,
        )
        assert corrupted.shape == indices.shape
        assert corrupted.dtype == torch.long
        assert int(corrupted.min()) >= 0
        assert int(corrupted.max()) < codebook_size
        assert torch.is_tensor(ber) and ber.ndim == 0
        recovered.append(quantizer.get_quantized_features(corrupted))

    assert [tuple(feature.shape) for feature in recovered] == [
        (1, 8, 8, 8),
        (1, 16, 4, 4),
    ]


def test_residual_scale_and_channel_schedule_matches_all_boundaries():
    class ScheduleConfig:
        QUANTIZER_TYPE = "residual_simvq"
        UNET_DEPTH = 2
        PHASE1_END = 0.1
        PHASE2_END = 0.4
        SKIP_DROPOUT_P_INIT = [0.1]
        SKIP_DROPOUT_P_FINAL = [0.0]
        LAYER_LOSS_WEIGHTS_INIT = [0.25, 0.50]
        LAYER_LOSS_WEIGHTS_FINAL = [0.25, 0.25]
        CHANNEL_PROB_START_EPOCH = 80
        CHANNEL_PROB_END_EPOCH = 120

    expected = {
        0: ([0.1], [0.25, 0.50], 0.0),
        20: ([0.1], [0.25, 0.50], 0.0),
        50: ([0.05], [0.25, 0.375], 0.0),
        80: ([0.0], [0.25, 0.25], 0.0),
        100: ([0.0], [0.25, 0.25], 0.5),
        120: ([0.0], [0.25, 0.25], 1.0),
        199: ([0.0], [0.25, 0.25], 1.0),
    }
    for epoch, (dropout, weights, channel_prob) in expected.items():
        actual_dropout, actual_weights, actual_channel, _ = compute_schedule(
            epoch, 200, ScheduleConfig
        )
        assert actual_dropout == dropout
        assert actual_weights == weights
        assert actual_channel == channel_prob


def test_optimizer_groups_every_projection_at_2e4_and_excludes_frozen_embed():
    model = _small_model()
    groups, audit = build_optimizer_parameter_groups(
        model,
        base_learning_rate=5e-5,
        codebook_projection_learning_rate=2e-4,
        quantizer_type="residual_simvq",
    )

    assert [group["lr"] for group in groups] == [5e-5, 2e-4]
    assert len(audit["projection_names"]) == 2
    assert len(audit["frozen_embedding_names"]) == 2
    assert all(
        name.endswith(".codebook.proj.weight")
        for name in audit["projection_names"]
    )
    assert all(
        name.endswith(".codebook.embed.weight")
        for name in audit["frozen_embedding_names"]
    )

    projection_ids = {id(parameter) for parameter in groups[1]["params"]}
    frozen_ids = {
        id(quantizer.codebook.embed.weight)
        for quantizer in model.vector_quantizers
    }
    expected_projection_ids = {
        id(quantizer.codebook.proj.weight)
        for quantizer in model.vector_quantizers
    }
    assert projection_ids == expected_projection_ids
    assert projection_ids.isdisjoint(frozen_ids)
    assert all(
        not quantizer.codebook.embed.weight.requires_grad
        for quantizer in model.vector_quantizers
    )


def test_optimizer_runtime_audit_rejects_trainable_residual_embed():
    model = _small_model()
    model.vector_quantizers[0].codebook.embed.weight.requires_grad_(True)
    try:
        build_optimizer_parameter_groups(
            model,
            base_learning_rate=5e-5,
            codebook_projection_learning_rate=2e-4,
            quantizer_type="residual_simvq",
        )
    except RuntimeError as error:
        assert "must remain frozen" in str(error)
    else:
        raise AssertionError("trainable Residual-SimVQ embedding was accepted")


def test_projection_gradient_norm_and_epoch_diagnostic_averaging():
    model = _small_model()
    _set_unit_projection_gradients(model)
    # Projection matrices are 8x8 and 16x16 for the two scales.
    assert projection_gradient_norms(model) == [8.0, 16.0]

    accumulator = _empty_rq_diagnostic_accumulator([2, 2])
    diagnostics = []
    for quantizer in model.vector_quantizers:
        feature = torch.randn(
            1, quantizer.embedding_dim, 2, 2
        )
        quantizer(feature)
        diagnostics.append(quantizer.get_last_diagnostics())
    _accumulate_rq_diagnostics(accumulator, diagnostics)
    _accumulate_projection_gradient_norms(accumulator, model)
    averaged = _average_rq_diagnostics(accumulator)

    assert len(averaged) == 2
    assert [scale["projection_grad_norm"] for scale in averaged] == [
        8.0,
        16.0,
    ]
    for scale in averaged:
        assert scale["codebook_loss"] is not None
        assert math.isfinite(scale["codebook_loss"])
        assert math.isfinite(scale["commitment_loss"])
        assert len(scale["codebook_per_depth"]) == 2
        assert len(scale["commitment_per_depth"]) == 2
        assert len(scale["residual_norm_per_depth"]) == 2
        assert len(scale["usage_per_depth"]) == 2
        assert len(scale["perplexity_per_depth"]) == 2


def test_codebook_monitor_records_q_c_residual_usage_distance_and_grad_norm():
    results = _monitoring_results()

    assert results["quantizer_type"] == "residual_simvq"
    assert len(results["src"]) == 2
    assert len(results["rq_scales"]) == 2
    for scale_index, stats in enumerate(results["src"]):
        assert stats["rq_depth"] == 2
        assert len(stats["per_depth"]) == 2
        assert len(stats["codebook_per_depth"]) == 2
        assert len(stats["commitment_per_depth"]) == 2
        assert len(stats["residual_norm_per_depth"]) == 2
        assert len(stats["usage_per_depth"]) == 2
        assert len(stats["perplexity_per_depth"]) == 2
        assert math.isfinite(stats["codebook_loss"])
        assert math.isfinite(stats["commitment_loss"])
        assert math.isfinite(stats["residual_norm"])
        assert 0.0 <= stats["active_ratio"] <= 1.0
        assert stats["active_count"] + stats["dead_count"] == (4, 2)[
            scale_index
        ]
        assert stats["usage_counts"].numel() == (4, 2)[scale_index]
        assert 0.0 <= stats["collapse_ratio"] <= 1.0
        assert stats["distance_reference_count"] == (4, 2)[scale_index]
        assert stats["distance_stats_exact"] is True
        assert stats["projection_grad_norm"] == (8.0, 16.0)[scale_index]
        for depth, depth_stats in enumerate(stats["per_depth"]):
            assert depth_stats["depth"] == depth
            assert math.isfinite(depth_stats["codebook_loss"])
            assert math.isfinite(depth_stats["commitment_loss"])
            assert math.isfinite(depth_stats["residual_norm"])
            assert 0.0 <= depth_stats["active_ratio"] <= 1.0
            assert depth_stats["perplexity"] >= 1.0
            assert depth_stats["restarted_codes"] == 0


def test_tensorboard_and_codebook_csv_persist_residual_monitoring_fields():
    results = _monitoring_results()

    class RecordingWriter:
        def __init__(self):
            self.tags = {}

        def add_scalar(self, tag, value, step):
            self.tags[tag] = (float(value), int(step))

    writer = RecordingWriter()
    write_codebook_tensorboard(writer, results, epoch=9)
    required_tags = {
        "Codebook/L0/CodebookLoss",
        "Codebook/L0/CommitmentLoss",
        "Codebook/L0/ProjectionGradNorm",
        "Codebook/L0/MinL2Dist",
        "Codebook/L0/CollapseRatio",
        "Codebook/L0/Depth0/CodebookLoss",
        "Codebook/L0/Depth0/Commitment",
        "Codebook/L0/Depth0/ResidualNorm",
        "Codebook/L0/Depth0/ActiveRatio",
        "Codebook/L0/Depth0/Perplexity",
    }
    assert required_tags.issubset(writer.tags)

    with tempfile.TemporaryDirectory(
        prefix="residual_simvq_csv_", dir=ROOT / "tests"
    ) as temp_dir:
        csv_path = Path(temp_dir) / "codebook.csv"
        append_codebook_records(
            str(csv_path), "contract-run", 10, results, [4, 2]
        )
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

    assert len(rows) == 6
    assert {row["scope"] for row in rows} == {"depth", "aggregate"}
    assert all(row["codebook_loss"] for row in rows)
    assert all(row["commitment"] for row in rows)
    assert all(row["residual_norm"] for row in rows)
    aggregate_rows = [row for row in rows if row["scope"] == "aggregate"]
    assert all(row["projection_grad_norm"] for row in aggregate_rows)
    assert all(row["min_l2_dist"] for row in aggregate_rows)
    assert all(row["collapse_ratio"] for row in aggregate_rows)


def test_epoch_csv_has_explicit_scale_and_depth_q_c_monitoring_columns():
    model = _small_model()
    _set_unit_projection_gradients(model)
    accumulator = _empty_rq_diagnostic_accumulator([2, 2])
    diagnostics = []
    for quantizer in model.vector_quantizers:
        quantizer(torch.randn(1, quantizer.embedding_dim, 2, 2))
        diagnostics.append(quantizer.get_last_diagnostics())
    _accumulate_rq_diagnostics(accumulator, diagnostics)
    _accumulate_projection_gradient_norms(accumulator, model)
    averaged = _average_rq_diagnostics(accumulator)
    fields = rq_epoch_metric_fields("train", averaged)

    assert json.loads(fields["train_rq_codebook_loss_per_scale"])
    assert json.loads(fields["train_rq_commitment_loss_per_scale"])
    assert [len(values) for values in json.loads(
        fields["train_rq_codebook_per_depth"]
    )] == [2, 2]
    assert [len(values) for values in json.loads(
        fields["train_rq_commitment_per_depth"]
    )] == [2, 2]
    assert [len(values) for values in json.loads(
        fields["train_rq_residual_norm_per_depth"]
    )] == [2, 2]
    assert json.loads(
        fields["train_rq_projection_grad_norm_per_scale"]
    ) == [8.0, 16.0]

    with tempfile.TemporaryDirectory(
        prefix="residual_simvq_epoch_", dir=ROOT / "tests"
    ) as temp_dir:
        csv_path = Path(temp_dir) / "epoch.csv"
        append_epoch_record(
            str(csv_path), {"run_id": "contract-run", "epoch": 1, **fields}
        )
        with csv_path.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
    assert row["train_rq_codebook_loss_per_scale"] == fields[
        "train_rq_codebook_loss_per_scale"
    ]
    assert row["train_rq_commitment_per_depth"] == fields[
        "train_rq_commitment_per_depth"
    ]
    assert row["train_rq_projection_grad_norm_per_scale"] == "[8.0,16.0]"


def test_production_config_is_rate047_residual_family_with_original_scale_weights():
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SIMVQ_")
    }
    environment.update(
        {
            "SIMVQ_EXPERIMENT_STAGE": "B",
            "SIMVQ_EXP_FAMILY": "quality_v2_B_larger_rate047",
            "SIMVQ_UNET_DEPTH": "2",
            "SIMVQ_DOWNSAMPLE_STRIDES": "8,2",
            "SIMVQ_BASE_CHANNELS": "128",
            "SIMVQ_ENCODER_RES_BLOCKS": "6",
            "SIMVQ_DECODER_RES_BLOCKS": "6",
            "SIMVQ_QUANTIZER_TYPE": "residual_simvq",
            "SIMVQ_QUANTIZER_AXIS_LIST": "patch,patch",
            "SIMVQ_CVQ_CODEWORD_SHAPES": "patch,patch",
            "SIMVQ_NUM_EMBEDDINGS_LIST": "4,2",
            "SIMVQ_RQ_DEPTH_LIST": "2,2",
            "SIMVQ_RQ_SHARED_CODEBOOK": "1",
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
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload == {
        "name": (
            "quality_v2_B_larger_rate047_residual_simvq_"
            "unet2_ds8x2_k4-2_d2-2"
        ),
        "bits": 4608.0,
        "bpp": 0.0703125,
        "ratio": 0.046875,
        "dims": [256, 512],
        "weights_init": [0.25, 0.5],
        "weights_final": [0.25, 0.25],
        "epochs": 200,
    }
    assert _source_bpp(
        [8, 2],
        [16, 4],
        image_size=(256, 256),
        rq_depth_list=[1, 1],
    ) == 0.0703125


def test_rate094_k16_4_config_and_scripts_are_isolated():
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
            "SIMVQ_NUM_EMBEDDINGS_LIST": "16,4",
            "SIMVQ_RQ_DEPTH_LIST": "2,2",
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
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload == {
        "name": (
            "quality_v2_B_larger_rate094_residual_simvq_"
            "unet2_ds8x2_k16-4_d2-2"
        ),
        "bits": 9216.0,
        "bpp": 0.140625,
        "ratio": 0.09375,
        "dims": [256, 512],
        "depths": [2, 2],
        "weights_init": [0.25, 0.5],
        "weights_final": [0.25, 0.25],
        "epochs": 200,
    }

    scripts = (
        RATE094_TRAIN_SCRIPT,
        RATE094_NOCHANNEL_SCRIPT,
        RATE094_LDPC_SCRIPT,
    )
    for script in scripts:
        assert script.is_file()
        assert os.access(script, os.X_OK)
        content = script.read_text(encoding="utf-8")
        assert "cd /workspace/yi/work/RQ-VAE" in content
        assert OLD_ROOT not in content
        assert 'SIMVQ_EXP_FAMILY="quality_v2_B_larger_rate094"' in content
        assert 'SIMVQ_QUANTIZER_TYPE="residual_simvq"' in content
        assert 'SIMVQ_NUM_EMBEDDINGS_LIST="16,4"' in content
        assert 'SIMVQ_RQ_DEPTH_LIST="2,2"' in content
        assert 'SIMVQ_RQ_SHARED_CODEBOOK="1"' in content
        assert 'SIMVQ_LAYER_LOSS_WEIGHTS_INIT="0.25,0.50"' in content
        assert 'SIMVQ_LAYER_LOSS_WEIGHTS_FINAL="0.25,0.25"' in content
        assert 'GPU_ID="${GPU_ID:-2}"' in content
        assert 'SIMVQ_RESUME="0"' in content
        assert "unset SIMVQ_PRETRAINED_CHECKPOINT" in content
        assert (
            "quality_v2_B_larger_rate094_residual_simvq_"
            "unet2_ds8x2_k16-4_d2-2"
        ) in content
        syntax = subprocess.run(
            ["bash", "-n", str(script)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert syntax.returncode == 0, syntax.stderr

    train_content = RATE094_TRAIN_SCRIPT.read_text(encoding="utf-8")
    assert 'SIMVQ_TOTAL_BATCH_SIZE="${SIMVQ_TOTAL_BATCH_SIZE:-24}"' in train_content
    assert 'SIMVQ_MICRO_BATCH_SIZE="${SIMVQ_MICRO_BATCH_SIZE:-24}"' in train_content
    assert 'SIMVQ_NUM_EPOCHS="200"' in train_content
    assert "9216 bits/image = 0.140625 bpp" in train_content
    assert "0.09375000" in train_content

    no_channel_content = RATE094_NOCHANNEL_SCRIPT.read_text(encoding="utf-8")
    assert "test_real.py" in no_channel_content
    assert "--no-channel" in no_channel_content
    assert "residual_simvq_k16-4_d2-2_rate094_nochannel.json" in no_channel_content

    ldpc_content = RATE094_LDPC_SCRIPT.read_text(encoding="utf-8")
    assert "test_real.py" in ldpc_content
    assert "--snrs 0 3 6 9 12" in ldpc_content
    assert "--modulation bpsk" in ldpc_content
    assert "residual_simvq_k16-4_d2-2_rate094_ldpc_bpsk.json" in ldpc_content


def test_residual_checkpoint_is_self_describing_and_roundtrips_weights():
    model = _small_model().eval()
    config = SimpleNamespace(
        EXPERIMENT_NAME="contract_residual_simvq",
        EXPERIMENT_FAMILY="contract",
        EXPERIMENT_STAGE="B",
        QUANTIZER_TYPE="residual_simvq",
        NUM_DOWNSAMPLE_BLOCKS=2,
        UNET_DEPTH=2,
        NUM_EMBEDDINGS_LIST=[4, 2],
        EMBEDDING_DIM_LIST=[8, 16],
        RQ_DEPTH_LIST=[2, 2],
        RQ_EMA_DECAY=0.99,
        RQ_RESTART_UNUSED_CODES=False,
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
        SKIP_DROPOUT_P_INIT=[0.1],
        SKIP_DROPOUT_P_FINAL=[0.0],
        LAYER_LOSS_WEIGHTS_INIT=[0.25, 0.5],
        LAYER_LOSS_WEIGHTS_FINAL=[0.25, 0.25],
        MSE_LOSS_WEIGHT=1.0,
        MS_SSIM_LOSS_WEIGHT=0.0,
        LPIPS_LOSS_WEIGHT=0.0,
        PHASE1_END=0.1,
        PHASE2_END=0.4,
        CHANNEL_PROB_START_EPOCH=80,
        CHANNEL_PROB_END_EPOCH=120,
        CHANNEL_CODING_RATE_TRAIN=0.5,
        CHANNEL_CODING_RATE_VAL=0.5,
        BLOCK_LENGTH=256,
        SNR_RANGE_DB=[0, 15],
        LEARNING_RATE_G=5e-5,
        CODEBOOK_PROJ_LR=2e-4,
        BETAS=(0.5, 0.999),
        TRAIN_IMAGE_SIZE=(256, 256),
        TEST_IMAGE_SIZE=(256, 256),
        ESTIMATED_SOURCE_BITS_PER_IMAGE=4608,
        ESTIMATED_SOURCE_BPP=0.0703125,
        ESTIMATED_TEST_SOURCE_BPP=0.0703125,
        ESTIMATED_TEST_TRANSMISSION_RATIO=0.046875,
    )
    payload = build_checkpoint_payload(
        model, config, epoch=3, best_val_loss=0.125
    )
    stream = io.BytesIO()
    torch.save(payload, stream)
    stream.seek(0)
    restored = torch.load(stream, map_location="cpu", weights_only=False)
    metadata = extract_checkpoint_metadata(restored)
    state_dict = extract_state_dict(restored)
    inferred = infer_codebook_config(state_dict, config, metadata=metadata)

    assert metadata["quantizer_type"] == "residual_simvq"
    assert metadata["num_embeddings_list"] == [4, 2]
    assert metadata["rq_depth_list"] == [2, 2]
    assert metadata["projected_embedding"] == {
        "enabled": True,
        "base_embedding_frozen": True,
        "projection_type": "linear",
        "projection_bias": False,
        "shared_across_rq_depth": True,
    }
    assert metadata["loss"]["layer_weights_init"] == [0.25, 0.5]
    assert metadata["loss"]["layer_weights_final"] == [0.25, 0.25]
    assert metadata["schedule"]["channel_prob_start_epoch"] == 80
    assert metadata["schedule"]["channel_prob_end_epoch"] == 120
    assert metadata["optimizer"]["learning_rate"] == 5e-5
    assert metadata["optimizer"]["codebook_projection_learning_rate"] == 2e-4
    assert metadata["optimizer"]["scheduler"] == {
        "type": "StepLR",
        "step_size": 100,
        "gamma": 0.5,
    }
    assert inferred["quantizer_type"] == "residual_simvq"
    assert inferred["num_embeddings_list"] == [4, 2]
    assert inferred["embedding_dim_list"] == [8, 16]
    assert inferred["rq_depth_list"] == [2, 2]

    reloaded = _small_model().eval()
    reloaded.load_state_dict(state_dict)
    assert reloaded.vector_quantizers[0].codebooks[0] is (
        reloaded.vector_quantizers[0].codebooks[1]
    )
    for expected, actual in zip(
        model.state_dict().values(), reloaded.state_dict().values()
    ):
        assert torch.equal(expected, actual)


def test_residual_train_and_eval_scripts_are_isolated_and_locked():
    scripts = (TRAIN_SCRIPT, NOCHANNEL_SCRIPT, LDPC_SCRIPT)
    for script in scripts:
        assert script.is_file()
        assert os.access(script, os.X_OK)
        content = script.read_text(encoding="utf-8")
        assert "cd /workspace/yi/work/RQ-VAE" in content
        assert OLD_ROOT not in content
        assert 'SIMVQ_EXP_FAMILY="quality_v2_B_larger_rate047"' in content
        assert 'SIMVQ_QUANTIZER_TYPE="residual_simvq"' in content
        assert 'SIMVQ_NUM_EMBEDDINGS_LIST="4,2"' in content
        assert 'SIMVQ_RQ_DEPTH_LIST="2,2"' in content
        assert 'SIMVQ_RQ_SHARED_CODEBOOK="1"' in content
        assert 'SIMVQ_LAYER_LOSS_WEIGHTS_INIT="0.25,0.50"' in content
        assert 'SIMVQ_LAYER_LOSS_WEIGHTS_FINAL="0.25,0.25"' in content
        assert 'GPU_ID="${GPU_ID:-1}"' in content
        assert 'SIMVQ_RESUME="0"' in content
        assert "unset SIMVQ_PRETRAINED_CHECKPOINT" in content
        assert (
            "quality_v2_B_larger_rate047_residual_simvq_"
            "unet2_ds8x2_k4-2_d2-2"
        ) in content
        completed = subprocess.run(
            ["bash", "-n", str(script)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr

    train_content = TRAIN_SCRIPT.read_text(encoding="utf-8")
    assert 'SIMVQ_EXPERIMENT_STAGE="B"' in train_content
    assert 'SIMVQ_DOWNSAMPLE_STRIDES="8,2"' in train_content
    assert 'SIMVQ_BASE_CHANNELS="128"' in train_content
    assert 'SIMVQ_ENCODER_RES_BLOCKS="6"' in train_content
    assert 'SIMVQ_DECODER_RES_BLOCKS="6"' in train_content
    assert 'SIMVQ_TOTAL_BATCH_SIZE="${SIMVQ_TOTAL_BATCH_SIZE:-24}"' in train_content
    assert 'SIMVQ_MICRO_BATCH_SIZE="${SIMVQ_MICRO_BATCH_SIZE:-24}"' in train_content
    assert 'SIMVQ_NUM_EPOCHS="200"' in train_content

    no_channel_content = NOCHANNEL_SCRIPT.read_text(encoding="utf-8")
    assert "test_real.py" in no_channel_content
    assert "--no-channel" in no_channel_content
    assert "--json-output" in no_channel_content

    ldpc_content = LDPC_SCRIPT.read_text(encoding="utf-8")
    assert "test_real.py" in ldpc_content
    assert "--snrs 0 3 6 9 12" in ldpc_content
    assert "--modulation bpsk" in ldpc_content
    assert "--json-output" in ldpc_content
