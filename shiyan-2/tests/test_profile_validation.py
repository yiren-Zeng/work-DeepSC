import csv

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluation.profile_validation import (
    all_profiles,
    codebook_distance_stats,
    resolve_profiles,
    validate_profiles,
)


class DummyLPIPS(torch.nn.Module):
    def forward(self, first, second):
        return (first - second).abs().flatten(1).mean(1, keepdim=True)


class DummyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward_profile(self, images, profile, use_channel=False, generate_hierarchy=False):
        profile = tuple(profile)
        self.calls.append(profile)
        error = 0.02 if profile == (2, 2) else 0.08
        reconstruction = images + error
        indices = []
        codebooks = []
        for layer, k in enumerate(profile):
            base = torch.arange(images.shape[0] * 4, device=images.device).view(
                images.shape[0], 2, 2
            )
            indices.append((base + layer).remainder(k))
            codebooks.append(
                torch.stack(
                    (
                        torch.arange(k, device=images.device, dtype=images.dtype),
                        torch.full((k,), float(layer), device=images.device),
                    ),
                    dim=1,
                )
            )
        return {
            "reconstructed_images": reconstruction,
            "indices": indices,
            "codebooks": codebooks,
            "profile": profile,
        }


class DummyTeacher(torch.nn.Module):
    pass


class DummyWriter:
    def __init__(self):
        self.tags = []

    def add_scalar(self, tag, value, step):
        self.tags.append((tag, float(value), int(step)))


def _loader():
    images = torch.zeros(3, 3, 16, 16)
    return DataLoader(TensorDataset(images), batch_size=2, shuffle=False)


def test_profile_resolution_supports_all_121_and_rejects_duplicates():
    assert len(all_profiles()) == 121
    assert resolve_profiles("2048x2048;16x2") == ((2048, 2048), (16, 2))
    assert resolve_profiles("all") == all_profiles()
    with pytest.raises(ValueError, match="duplicates"):
        resolve_profiles("16x2;16x2")


def test_fixed_profiles_metrics_guard_csv_and_tensorboard(tmp_path):
    student = DummyStudent()
    student.train()
    teacher = DummyTeacher()
    teacher_calls = []

    def teacher_forward_fn(_teacher, images):
        teacher_calls.append(int(images.shape[0]))
        # Match the maximum-profile student, making the teacher guard exact.
        return {"reconstructed_images": images + 0.02}

    writer = DummyWriter()
    csv_path = tmp_path / "profiles.csv"
    per_profile_dir = tmp_path / "individual"
    result = validate_profiles(
        student,
        teacher,
        _loader(),
        [(2, 2), (4, 2)],
        "cpu",
        teacher_forward_fn=teacher_forward_fn,
        lpips_model=DummyLPIPS(),
        max_profile=(2, 2),
        max_teacher_psnr_drop_db=0.01,
        require_teacher_guard=True,
        src_reference_psnr={"4x2": 25.0},
        profile_weights={"2x2": 2.0, "4x2": 1.0},
        worst_profile_weight=0.5,
        writer=writer,
        global_step=7,
        csv_path=csv_path,
        per_profile_csv_dir=per_profile_dir,
        max_distance_elements=8,
    )

    # Teacher inference is once per batch, not once per student profile.
    assert teacher_calls == [2, 1]
    assert student.calls == [(2, 2), (4, 2), (2, 2), (4, 2)]
    assert student.training is True
    assert result["eligible"] is True
    assert result["is_guard_satisfied"] is True
    assert result["teacher_psnr_drop_db"] == pytest.approx(0.0, abs=1e-6)
    assert result["worst_profile"] == "4x2"
    assert result["per_profile"]["4x2"]["src_psnr_gap_db"] == pytest.approx(
        25.0 - result["per_profile"]["4x2"]["psnr"]
    )
    assert len(result["per_profile"]["2x2"]["layers"]) == 2
    assert result["per_profile"]["2x2"]["layers"][0]["dead_code_count"] == 0

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["profile"] for row in rows] == ["2x2", "4x2"]
    assert (per_profile_dir / "2x2.csv").is_file()
    assert (per_profile_dir / "4x2.csv").is_file()
    tags = {tag for tag, _, _ in writer.tags}
    assert "VariableRateValidation/2x2/psnr" in tags
    assert "VariableRateValidation/4x2/layer1/perplexity" in tags


def test_guard_can_reject_checkpoint():
    student = DummyStudent()
    teacher_outputs = [
        {"reconstructed_images": torch.zeros(2, 3, 16, 16)},
        {"reconstructed_images": torch.zeros(1, 3, 16, 16)},
    ]
    result = validate_profiles(
        student,
        None,
        _loader(),
        [(2, 2)],
        "cpu",
        teacher_outputs=teacher_outputs,
        lpips_model=DummyLPIPS(),
        max_profile=(2, 2),
        max_teacher_psnr_drop_db=0.1,
        require_teacher_guard=True,
        max_distance_elements=8,
    )
    assert result["is_guard_satisfied"] is False
    assert result["eligible"] is False
    assert result["teacher_psnr_drop_db"] > 0.1


def test_distance_stats_are_exact_with_bounded_row_chunks():
    codebook = torch.arange(64, dtype=torch.float32).unsqueeze(1)
    stats = codebook_distance_stats(codebook, max_distance_elements=64)
    assert stats["distance_stats_exact"] is True
    assert stats["distance_reference_count"] == 64
    assert stats["min_l2_distance"] == pytest.approx(1.0)
    assert stats["collapse_count"] == 0
