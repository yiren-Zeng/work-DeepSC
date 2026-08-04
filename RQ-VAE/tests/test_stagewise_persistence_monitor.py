"""Persistence and monitoring contracts for stagewise Residual-SimVQ."""

import csv
import json
import math
from types import SimpleNamespace

import torch
import torch.nn as nn

import utils.checkpoint_utils as checkpoint_utils
from monitoring.codebook import compute_codebook_utilization
from utils.checkpoint_utils import (
    build_checkpoint_metadata,
    infer_codebook_config,
)
from utils.experiment_io import append_codebook_records


def _config():
    return SimpleNamespace(
        EXPERIMENT_NAME="stagewise_contract",
        EXPERIMENT_FAMILY="stagewise_contract",
        EXPERIMENT_STAGE="B",
        QUANTIZER_TYPE="stagewise_residual_simvq",
        NUM_DOWNSAMPLE_BLOCKS=2,
        UNET_DEPTH=2,
        # The compatibility field contains the first depth's K per scale.
        NUM_EMBEDDINGS_LIST=[8, 2],
        RQ_CODEBOOK_SIZE_LISTS=[[8, 2], [2, 2]],
        EMBEDDING_DIM_LIST=[8, 16],
        RQ_DEPTH_LIST=[2, 2],
        RQ_SHARED_CODEBOOK=False,
        IN_CHANNELS=3,
        OUT_CHANNELS=3,
        BASE_CHANNELS=4,
        DOWNSAMPLE_STRIDES=[2, 2],
        QUANTIZER_AXIS_LIST=["patch", "patch"],
        CVQ_CODEWORD_SHAPES=[None, None],
        COMMITMENT_COST=0.25,
        TRAIN_IMAGE_SIZE=(16, 16),
        TEST_IMAGE_SIZE=(16, 16),
    )


def test_stagewise_checkpoint_metadata_and_inference_keep_nested_sizes():
    config = _config()
    metadata = build_checkpoint_metadata(config)

    assert metadata["quantizer_type"] == "stagewise_residual_simvq"
    assert metadata["rq_codebook_size_lists"] == [[8, 2], [2, 2]]
    assert metadata["num_embeddings_list"] == [8, 2]
    assert metadata["rq_depth_list"] == [2, 2]
    assert metadata["rq_shared_codebook"] is False
    assert metadata["projected_embedding"]["enabled"] is True
    assert metadata["projected_embedding"]["shared_across_rq_depth"] is False

    inferred = infer_codebook_config({}, config, metadata=metadata)
    assert inferred["quantizer_type"] == "stagewise_residual_simvq"
    assert inferred["rq_codebook_size_lists"] == [[8, 2], [2, 2]]
    assert inferred["num_embeddings_list"] == [8, 2]
    assert inferred["rq_depth_list"] == [2, 2]
    assert inferred["rq_shared_codebook"] is False


def test_bare_stagewise_state_infers_each_depth_codebook_size():
    state_dict = {
        "vector_quantizers.0.codebooks.0.embed.weight": torch.randn(8, 8),
        "vector_quantizers.0.codebooks.0.proj.weight": torch.randn(8, 8),
        "vector_quantizers.0.codebooks.1.embed.weight": torch.randn(2, 8),
        "vector_quantizers.0.codebooks.1.proj.weight": torch.randn(8, 8),
        "vector_quantizers.1.codebooks.0.embed.weight": torch.randn(2, 16),
        "vector_quantizers.1.codebooks.0.proj.weight": torch.randn(16, 16),
        "vector_quantizers.1.codebooks.1.embed.weight": torch.randn(2, 16),
        "vector_quantizers.1.codebooks.1.proj.weight": torch.randn(16, 16),
    }

    inferred = infer_codebook_config(state_dict)
    assert inferred["quantizer_type"] == "stagewise_residual_simvq"
    assert inferred["rq_codebook_size_lists"] == [[8, 2], [2, 2]]
    assert inferred["num_embeddings_list"] == [8, 2]
    assert inferred["embedding_dim_list"] == [8, 16]
    assert inferred["rq_depth_list"] == [2, 2]
    assert inferred["rq_shared_codebook"] is False


def test_checkpoint_model_builder_passes_nested_stagewise_sizes(
    tmp_path, monkeypatch
):
    config = _config()
    config.validate = lambda: None
    metadata = build_checkpoint_metadata(config)
    checkpoint_path = tmp_path / "stagewise.pth"
    torch.save(
        {"model_state_dict": {}, "model_metadata": metadata},
        checkpoint_path,
    )

    class RecordingDeepSC(nn.Module):
        def __init__(self, rq_codebook_size_lists):
            super().__init__()
            self.rq_codebook_size_lists = rq_codebook_size_lists

    monkeypatch.setattr(checkpoint_utils, "DeepSC", RecordingDeepSC)
    model, inferred = checkpoint_utils.build_model_from_checkpoint(
        checkpoint_path, config, torch.device("cpu")
    )

    assert model.rq_codebook_size_lists == [[8, 2], [2, 2]]
    assert inferred["rq_codebook_size_lists"] == [[8, 2], [2, 2]]


class _ProjectedCodebook(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        self.proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def projected_weight(self):
        return self.proj(self.embed.weight)


class StagewiseResidualSimVQQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_embeddings_per_depth = (8, 2)
        self.rq_depth = 2
        self.codebooks = nn.ModuleList(
            [_ProjectedCodebook(8, 3), _ProjectedCodebook(2, 3)]
        )

    def forward(self, inputs):
        batch, _, height, width = inputs.shape
        count = batch * height * width
        depth0 = torch.arange(count, device=inputs.device).reshape(
            batch, height, width
        ) % 8
        depth1 = torch.arange(count, device=inputs.device).reshape(
            batch, height, width
        ) % 2
        indices = torch.stack([depth0, depth1], dim=-1)
        return inputs.new_zeros(()), inputs, indices


class _Encoder(nn.Module):
    def forward(self, images):
        return [images]


class _StagewiseMonitorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantizer_type = "stagewise_residual_simvq"
        self.num_embeddings_list = [8]
        self.rq_codebook_size_lists = [[8, 2]]
        self.semantic_encoder = _Encoder()
        self.vector_quantizers = nn.ModuleList(
            [StagewiseResidualSimVQQuantizer()]
        )
        self.device = torch.device("cpu")


def test_stagewise_monitor_and_csv_use_each_depth_size(tmp_path):
    model = _StagewiseMonitorModel()
    for codebook in model.vector_quantizers[0].codebooks:
        codebook.proj.weight.grad = torch.ones_like(codebook.proj.weight)

    results = compute_codebook_utilization(
        model,
        [torch.randn(1, 3, 2, 2)],
        max_batches=1,
        device="cpu",
    )
    stats = results["src"][0]

    assert results["rq_codebook_size_lists"] == [[8, 2]]
    assert [row["codebook_size"] for row in stats["per_depth"]] == [8, 2]
    assert [
        row["usage_counts"].numel() for row in stats["per_depth"]
    ] == [8, 2]
    assert stats["codebook_size"] == 10
    assert stats["usage_counts"].numel() == 10
    assert stats["active_count"] == 6
    assert stats["dead_count"] == 4
    assert stats["active_ratio"] == 0.6
    assert math.isclose(
        stats["projection_grad_norm"], 18.0 ** 0.5, rel_tol=1e-6
    )

    csv_path = tmp_path / "codebook.csv"
    # Production callers retain the flat first-depth list.  The monitor result
    # carries the nested sizes needed by the stagewise CSV rows.
    append_codebook_records(
        str(csv_path), "stagewise-run", 1, results, [8]
    )
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    depth_rows = [row for row in rows if row["scope"] == "depth"]
    aggregate_row = next(row for row in rows if row["scope"] == "aggregate")
    assert [int(row["codebook_size"]) for row in depth_rows] == [8, 2]
    assert [
        len(json.loads(row["usage_counts"]))
        for row in depth_rows
    ] == [8, 2]
    assert int(aggregate_row["codebook_size"]) == 10
    assert len(json.loads(aggregate_row["usage_counts"])) == 10
