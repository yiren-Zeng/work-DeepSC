"""Fixed-profile validation for the single-teacher variable-rate RAQ model.

This module deliberately does not mutate ``model.raq_target_list``.  A profile
is passed explicitly to every student forward, so validation never falls back
to the legacy independently-randomized target-K path.
"""

from __future__ import annotations

import csv
import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from utils.metrics import calculate_ms_ssim


SUPPORTED_CODEBOOK_SIZES: Tuple[int, ...] = tuple(2**power for power in range(1, 12))
DEFAULT_VALIDATION_PROFILES: Tuple[Tuple[int, int], ...] = (
    (2048, 2048),
    (2048, 16),
    (16, 2),
    (1024, 256),
    (512, 64),
    (64, 16),
)
MAX_PROFILE = (2048, 2048)


def profile_key(profile: Sequence[int]) -> str:
    profile = normalize_profile(profile)
    return f"{profile[0]}x{profile[1]}"


def normalize_profile(profile: Any) -> Tuple[int, int]:
    """Parse one profile and enforce the supported two-layer power-of-two set."""
    if isinstance(profile, str):
        parts = [part.strip() for part in profile.lower().replace(",", "x").split("x")]
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"Invalid profile {profile!r}; expected K0xK1")
        profile = (int(parts[0]), int(parts[1]))
    if not isinstance(profile, Sequence) or isinstance(profile, (bytes, bytearray)):
        raise TypeError(f"Profile must be a two-item sequence, got {type(profile).__name__}")
    if len(profile) != 2:
        raise ValueError(f"Variable-rate RAQ requires two K values, got {profile!r}")
    normalized = (int(profile[0]), int(profile[1]))
    unsupported = [value for value in normalized if value not in SUPPORTED_CODEBOOK_SIZES]
    if unsupported:
        raise ValueError(
            f"Unsupported codebook size(s) {unsupported}; supported={SUPPORTED_CODEBOOK_SIZES}"
        )
    return normalized


def all_profiles() -> Tuple[Tuple[int, int], ...]:
    return tuple(
        (first, second)
        for first in SUPPORTED_CODEBOOK_SIZES
        for second in SUPPORTED_CODEBOOK_SIZES
    )


def resolve_profiles(profiles: Any = None) -> Tuple[Tuple[int, int], ...]:
    """Resolve defaults, ``all``, a semicolon string, or a profile sequence."""
    if profiles is None or profiles == "":
        return DEFAULT_VALIDATION_PROFILES
    if isinstance(profiles, str):
        if profiles.strip().lower() in {"all", "121"}:
            return all_profiles()
        profiles = [item for item in profiles.split(";") if item.strip()]
    normalized = tuple(normalize_profile(profile) for profile in profiles)
    if not normalized:
        raise ValueError("At least one validation profile is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Validation profiles must not contain duplicates")
    return normalized


def _extract_images(batch: Any) -> torch.Tensor:
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, Mapping):
        for key in ("images", "image", "inputs", "input", "pixel_values"):
            value = batch.get(key)
            if torch.is_tensor(value):
                return value
    if isinstance(batch, (tuple, list)) and batch and torch.is_tensor(batch[0]):
        return batch[0]
    raise TypeError("Dataloader batches must contain an image tensor")


def _extract_reconstruction(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, Mapping):
        for key in (
            "reconstructed_images",
            "reconstruction",
            "reconstructed_images_raq",
            "x_hat",
            "output",
        ):
            value = output.get(key)
            if torch.is_tensor(value):
                return value
    if isinstance(output, (tuple, list)):
        for value in output:
            if torch.is_tensor(value) and value.ndim >= 3:
                return value
    raise KeyError(
        "Forward output has no reconstruction tensor; expected one of "
        "reconstructed_images/reconstruction/reconstructed_images_raq/x_hat"
    )


def _extract_layer_values(output: Any, keys: Sequence[str]) -> Optional[list]:
    if not isinstance(output, Mapping):
        return None
    value = None
    for key in keys:
        if key in output and output[key] is not None:
            value = output[key]
            break
    if value is None:
        diagnostics = output.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            return _extract_layer_values(diagnostics, keys)
        return None
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (tuple, list)):
        # The variable-rate path is single-stage per scale.  Reject nested RVQ
        # diagnostics rather than silently combining a different experiment.
        if value and isinstance(value[0], (tuple, list)):
            raise ValueError("Nested residual-VQ diagnostics are not variable-rate RAQ profiles")
        return list(value)
    return None


def _accepted_kwargs(function: Callable, candidates: Dict[str, Any]) -> Dict[str, Any]:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return {}
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return candidates
    return {key: value for key, value in candidates.items() if key in signature.parameters}


def _student_forward(student, images: torch.Tensor, profile: Tuple[int, int]):
    candidates = {
        "use_channel": False,
        "generate_hierarchy": False,
        "return_diagnostics": True,
    }
    if hasattr(student, "forward_profile"):
        function = student.forward_profile
        kwargs = _accepted_kwargs(function, candidates)
        signature = inspect.signature(function)
        if "profile" in signature.parameters:
            return function(images, profile=profile, **kwargs)
        return function(images, profile, **kwargs)

    if hasattr(student, "forward_val"):
        function = student.forward_val
        kwargs = _accepted_kwargs(
            function,
            {**candidates, "profile": profile, "raq_trg_list": list(profile)},
        )
        if not ({"profile", "raq_trg_list"} & set(kwargs)):
            raise TypeError("student.forward_val does not accept an explicit fixed profile")
        return function(images, **kwargs)

    function = student.forward
    kwargs = _accepted_kwargs(
        function,
        {**candidates, "profile": profile, "target_profile": profile},
    )
    if not ({"profile", "target_profile"} & set(kwargs)):
        raise TypeError("Student has no forward_profile and forward() accepts no fixed profile")
    return function(images, **kwargs)


def _default_teacher_forward(teacher, images: torch.Tensor):
    try:
        from training.frozen_teacher import teacher_forward

        return teacher_forward(teacher, images)
    except (AttributeError, TypeError):
        if hasattr(teacher, "forward_src"):
            function = teacher.forward_src
            kwargs = _accepted_kwargs(function, {"profile": MAX_PROFILE, "use_channel": False})
            return function(images, **kwargs)
        if hasattr(teacher, "forward_val"):
            function = teacher.forward_val
            kwargs = _accepted_kwargs(function, {"raq_trg_list": list(MAX_PROFILE)})
            return function(images, **kwargs)
        return teacher(images)


def _call_teacher_forward(function: Callable, teacher, images: torch.Tensor):
    try:
        signature = inspect.signature(function)
        positional = [
            param
            for param in signature.parameters.values()
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and param.default is inspect.Parameter.empty
        ]
        if len(positional) <= 1:
            return function(images)
    except (TypeError, ValueError):
        pass
    return function(teacher, images)


_LPIPS_CACHE: Dict[Tuple[str, str], Any] = {}


def _build_lpips(device: torch.device, net: str = "vgg"):
    cache_key = (str(device), str(net))
    if cache_key in _LPIPS_CACHE:
        return _LPIPS_CACHE[cache_key]
    try:
        import lpips
    except ImportError as exc:  # pragma: no cover - deployment error path
        raise RuntimeError(
            "Real LPIPS validation requires the 'lpips' package; no pixel-loss fallback is used"
        ) from exc
    metric = lpips.LPIPS(net=net).to(device)
    metric.eval()
    for parameter in metric.parameters():
        parameter.requires_grad_(False)
    _LPIPS_CACHE[cache_key] = metric
    return metric


def _to_unit_range(images: torch.Tensor, input_range: str) -> torch.Tensor:
    if input_range == "-1,1":
        images = (images + 1.0) * 0.5
    elif input_range != "0,1":
        raise ValueError("input_range must be '-1,1' or '0,1'")
    return images.clamp(0.0, 1.0)


def _quality_sums(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    lpips_metric,
    input_range: str,
    reconstruction_loss_fn: Optional[Callable],
) -> Dict[str, float]:
    if target.shape != reconstruction.shape:
        raise ValueError(
            f"Reconstruction shape {tuple(reconstruction.shape)} != target {tuple(target.shape)}"
        )
    target_unit = _to_unit_range(target, input_range)
    reconstruction_unit = _to_unit_range(reconstruction, input_range)
    batch_size = int(target.shape[0])

    mse_per_image = (target_unit - reconstruction_unit).square().flatten(1).mean(1)
    psnr_per_image = torch.where(
        mse_per_image <= torch.finfo(mse_per_image.dtype).eps,
        torch.full_like(mse_per_image, 100.0),
        10.0 * torch.log10(1.0 / mse_per_image),
    )

    # The project implementation is scalar/one-image, so call it once per
    # sample instead of accidentally reporting only the first item in a batch.
    ms_ssim_sum = sum(
        float(calculate_ms_ssim(target_unit[index], reconstruction_unit[index]))
        for index in range(batch_size)
    )

    target_lpips = target_unit.mul(2.0).sub(1.0)
    reconstruction_lpips = reconstruction_unit.mul(2.0).sub(1.0)
    lpips_values = lpips_metric(reconstruction_lpips, target_lpips)
    if not torch.is_tensor(lpips_values):
        raise TypeError("LPIPS metric must return a tensor")

    if reconstruction_loss_fn is None:
        reconstruction_values = (target - reconstruction).square().flatten(1).mean(1)
        reconstruction_sum = float(reconstruction_values.sum().item())
    else:
        reconstruction_value = reconstruction_loss_fn(reconstruction, target)
        if not torch.is_tensor(reconstruction_value):
            reconstruction_value = target.new_tensor(float(reconstruction_value))
        reconstruction_sum = (
            float(reconstruction_value.sum().item())
            if reconstruction_value.ndim > 0 and reconstruction_value.numel() == batch_size
            else float(reconstruction_value.mean().item()) * batch_size
        )

    return {
        "count": batch_size,
        "psnr": float(psnr_per_image.sum().item()),
        "ms_ssim": ms_ssim_sum,
        "lpips": float(lpips_values.reshape(batch_size, -1).mean(1).sum().item()),
        "reconstruction_loss": reconstruction_sum,
    }


def _new_accumulator(profile: Tuple[int, int]) -> Dict[str, Any]:
    return {
        "profile": profile,
        "num_images": 0,
        "num_batches": 0,
        "psnr": 0.0,
        "ms_ssim": 0.0,
        "lpips": 0.0,
        "reconstruction_loss": 0.0,
        "usage_counts": [torch.zeros(k, dtype=torch.float64) for k in profile],
        "distance_stats": [None, None],
    }


def _accumulate_diagnostics(
    accumulator: Dict[str, Any],
    output: Any,
    *,
    collapse_threshold: float,
    max_distance_elements: int,
) -> None:
    profile = accumulator["profile"]
    indices = _extract_layer_values(
        output,
        ("indices", "encoding_indices", "indices_list", "raq_indices", "raq_indices_list"),
    )
    codebooks = _extract_layer_values(
        output,
        ("codebooks", "W_trg_list", "raq_codebooks", "codebooks_trg_list"),
    )
    if indices is None or len(indices) != len(profile):
        raise KeyError(
            "VariableRateDeepSC output must contain one index tensor per profile layer"
        )
    if codebooks is None or len(codebooks) != len(profile):
        raise KeyError(
            "VariableRateDeepSC output must contain one generated/bypassed codebook per layer"
        )

    for layer, (layer_indices, codebook, k) in enumerate(zip(indices, codebooks, profile)):
        if not torch.is_tensor(layer_indices) or not torch.is_tensor(codebook):
            raise TypeError("Layer indices and codebooks must be tensors")
        flat_indices = layer_indices.detach().reshape(-1).to(device="cpu", dtype=torch.long)
        if flat_indices.numel() and (
            int(flat_indices.min().item()) < 0 or int(flat_indices.max().item()) >= k
        ):
            raise ValueError(f"Layer {layer} indices fall outside [0, {k})")
        accumulator["usage_counts"][layer] += torch.bincount(
            flat_indices, minlength=k
        ).to(torch.float64)
        if codebook.ndim != 2 or int(codebook.shape[0]) != k:
            raise ValueError(
                f"Layer {layer} codebook shape must be [{k}, D], got {tuple(codebook.shape)}"
            )
        # A generated codebook is rate-conditioned, not image-conditioned.  Its
        # geometry therefore needs to be computed only on the first batch.  Do
        # it while the tensor is already on the accelerator, then release it;
        # retaining all 121 profile codebooks would waste substantial VRAM.
        if accumulator["distance_stats"][layer] is None:
            accumulator["distance_stats"][layer] = codebook_distance_stats(
                codebook.detach(),
                collapse_threshold=collapse_threshold,
                max_distance_elements=max_distance_elements,
            )


@torch.no_grad()
def codebook_distance_stats(
    codebook: torch.Tensor,
    *,
    collapse_threshold: float = 0.1,
    max_distance_elements: int = 1_048_576,
) -> Dict[str, Any]:
    """Exact nearest-neighbour stats without ever allocating a dense K x K tensor."""
    if codebook.ndim != 2:
        raise ValueError("codebook must have shape [K, D]")
    k = int(codebook.shape[0])
    if k < 2:
        return {
            "min_l2_distance": float("inf"),
            "collapse_count": 0,
            "collapse_ratio": 0.0,
            "distance_reference_count": k,
            "distance_stats_exact": True,
        }
    max_distance_elements = max(k, int(max_distance_elements))
    chunk_size = max(1, max_distance_elements // k)
    weight = codebook.float()
    norms = weight.square().sum(dim=1)
    nearest_parts = []
    for start in range(0, k, chunk_size):
        end = min(k, start + chunk_size)
        chunk = weight[start:end]
        squared = chunk.square().sum(dim=1, keepdim=True) + norms.unsqueeze(0)
        squared = squared - 2.0 * chunk.matmul(weight.t())
        rows = torch.arange(end - start, device=weight.device)
        cols = torch.arange(start, end, device=weight.device)
        squared[rows, cols] = float("inf")
        nearest_parts.append(squared.min(dim=1).values.clamp_min_(0).cpu())
    nearest = torch.cat(nearest_parts).sqrt_()
    collapse_count = int((nearest < float(collapse_threshold)).sum().item())
    return {
        "min_l2_distance": float(nearest.min().item()),
        "collapse_count": collapse_count,
        "collapse_ratio": collapse_count / k,
        "distance_reference_count": k,
        "distance_stats_exact": True,
    }


def _finalize_accumulator(
    accumulator: Dict[str, Any],
) -> Dict[str, Any]:
    count = int(accumulator["num_images"])
    if count == 0:
        raise ValueError("Validation dataloader produced no images")
    layers = []
    for layer, (usage_counts, distance, k) in enumerate(
        zip(
            accumulator["usage_counts"],
            accumulator["distance_stats"],
            accumulator["profile"],
        )
    ):
        if distance is None:
            raise RuntimeError(f"No codebook geometry was observed for layer {layer}")
        total = float(usage_counts.sum().item())
        active_count = int((usage_counts > 0).sum().item())
        if total:
            probabilities = usage_counts[usage_counts > 0] / total
            perplexity = float(torch.exp(-(probabilities * probabilities.log()).sum()).item())
        else:
            perplexity = 1.0
        layers.append(
            {
                "layer": layer,
                "codebook_size": int(k),
                "num_assignments": int(total),
                "active_count": active_count,
                "active_ratio": active_count / int(k),
                "perplexity": perplexity,
                "dead_code_count": int(k) - active_count,
                **distance,
            }
        )
    result = {
        "profile": list(accumulator["profile"]),
        "num_images": count,
        "num_batches": int(accumulator["num_batches"]),
        "psnr": accumulator["psnr"] / count,
        "ms_ssim": accumulator["ms_ssim"] / count,
        "lpips": accumulator["lpips"] / count,
        "reconstruction_loss": accumulator["reconstruction_loss"] / count,
        "layers": layers,
    }
    for metric in ("active_ratio", "perplexity", "dead_code_count", "collapse_ratio"):
        result[metric] = sum(float(layer[metric]) for layer in layers) / len(layers)
    result["min_l2_distance"] = min(layer["min_l2_distance"] for layer in layers)
    return result


def _normalize_reference_psnr(reference: Optional[Mapping]) -> Dict[str, float]:
    if reference is None:
        return {}
    normalized = {}
    for key, value in reference.items():
        normalized[profile_key(key)] = float(value)
    return normalized


def _normalize_profile_weights(
    profiles: Sequence[Tuple[int, int]], profile_weights: Optional[Mapping]
) -> Dict[str, float]:
    keys = [profile_key(profile) for profile in profiles]
    if profile_weights is None:
        return {key: 1.0 / len(keys) for key in keys}
    supplied = {profile_key(key): float(value) for key, value in profile_weights.items()}
    unknown = set(supplied) - set(keys)
    if unknown:
        raise ValueError(f"Profile weights include unvalidated profiles: {sorted(unknown)}")
    weights = {key: supplied.get(key, 1.0) for key in keys}
    if any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Profile weights must be non-negative with a positive sum")
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def _flatten_csv_row(key: str, result: Mapping, summary: Mapping, step: Optional[int]):
    row = {
        "step": "" if step is None else int(step),
        "profile": key,
        "k0": result["profile"][0],
        "k1": result["profile"][1],
        "num_images": result["num_images"],
        "psnr": result["psnr"],
        "ms_ssim": result["ms_ssim"],
        "lpips": result["lpips"],
        "reconstruction_loss": result["reconstruction_loss"],
        "active_ratio": result["active_ratio"],
        "perplexity": result["perplexity"],
        "dead_code_count": result["dead_code_count"],
        "collapse_ratio": result["collapse_ratio"],
        "min_l2_distance": result["min_l2_distance"],
        "src_reference_psnr": result.get("src_reference_psnr"),
        "src_psnr_gap_db": result.get("src_psnr_gap_db"),
        "teacher_psnr": summary.get("teacher_psnr"),
        "teacher_psnr_drop_db": summary.get("teacher_psnr_drop_db"),
        "score": summary["score"],
        "eligible": summary["eligible"],
    }
    for layer in result["layers"]:
        prefix = f"layer{layer['layer']}_"
        for field in (
            "codebook_size",
            "num_assignments",
            "active_count",
            "active_ratio",
            "perplexity",
            "dead_code_count",
            "collapse_count",
            "collapse_ratio",
            "min_l2_distance",
        ):
            row[prefix + field] = layer[field]
    return row


def _append_rows(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_profile_csvs(
    validation_result: Mapping,
    *,
    csv_path: Optional[str | Path] = None,
    per_profile_csv_dir: Optional[str | Path] = None,
    step: Optional[int] = None,
) -> None:
    """Append profile-specific rows to a combined CSV and/or separate files."""
    rows = [
        _flatten_csv_row(key, result, validation_result, step)
        for key, result in validation_result["per_profile"].items()
    ]
    if csv_path is not None:
        _append_rows(Path(csv_path), rows)
    if per_profile_csv_dir is not None:
        directory = Path(per_profile_csv_dir)
        for row in rows:
            _append_rows(directory / f"{row['profile']}.csv", [row])


def write_profile_tensorboard(writer, validation_result: Mapping, step: int) -> None:
    for key, result in validation_result["per_profile"].items():
        base = f"VariableRateValidation/{key}"
        for metric in (
            "psnr",
            "ms_ssim",
            "lpips",
            "reconstruction_loss",
            "active_ratio",
            "perplexity",
            "dead_code_count",
            "collapse_ratio",
            "min_l2_distance",
        ):
            writer.add_scalar(f"{base}/{metric}", result[metric], step)
        if result.get("src_psnr_gap_db") is not None:
            writer.add_scalar(f"{base}/src_psnr_gap_db", result["src_psnr_gap_db"], step)
        for layer in result["layers"]:
            layer_base = f"{base}/layer{layer['layer']}"
            for metric in (
                "active_ratio",
                "perplexity",
                "dead_code_count",
                "collapse_ratio",
                "min_l2_distance",
            ):
                writer.add_scalar(f"{layer_base}/{metric}", layer[metric], step)
    writer.add_scalar("VariableRateValidation/summary/score", validation_result["score"], step)
    writer.add_scalar(
        "VariableRateValidation/summary/weighted_mean_psnr",
        validation_result["weighted_mean_psnr"],
        step,
    )
    writer.add_scalar(
        "VariableRateValidation/summary/worst_psnr", validation_result["worst_psnr"], step
    )
    if validation_result.get("teacher_psnr_drop_db") is not None:
        writer.add_scalar(
            "VariableRateValidation/summary/max_profile_teacher_psnr_drop_db",
            validation_result["teacher_psnr_drop_db"],
            step,
        )


@torch.no_grad()
def validate_profiles(
    student,
    teacher,
    dataloader: Iterable,
    profiles: Any,
    device: torch.device | str,
    *,
    teacher_forward_fn: Optional[Callable] = None,
    teacher_outputs: Optional[Iterable | Mapping] = None,
    forward_fn: Optional[Callable] = None,
    lpips_model=None,
    lpips_net: str = "vgg",
    reconstruction_loss_fn: Optional[Callable] = None,
    input_range: str = "-1,1",
    max_batches: Optional[int] = None,
    src_reference_psnr: Optional[Mapping] = None,
    profile_weights: Optional[Mapping] = None,
    worst_profile_weight: float = 0.25,
    max_profile: Sequence[int] = MAX_PROFILE,
    max_teacher_psnr_drop_db: float = 0.3,
    require_teacher_guard: bool = False,
    collapse_threshold: float = 0.1,
    max_distance_elements: int = 1_048_576,
    writer=None,
    global_step: Optional[int] = None,
    csv_path: Optional[str | Path] = None,
    per_profile_csv_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Evaluate each requested profile and return checkpoint-selection data.

    ``score`` is a maximize objective:

    ``(1-worst_profile_weight) * weighted_mean_psnr + worst_profile_weight * worst_psnr``.

    A checkpoint is ``eligible`` only when the maximum-profile teacher drop
    guard passes (or when a teacher is optional and was not supplied).
    """
    resolved_profiles = resolve_profiles(profiles)
    max_profile = normalize_profile(max_profile)
    if max_profile not in resolved_profiles:
        raise ValueError("max_profile must be included in fixed-profile validation")
    if not 0.0 <= float(worst_profile_weight) <= 1.0:
        raise ValueError("worst_profile_weight must be in [0, 1]")
    if max_batches is not None and int(max_batches) <= 0:
        raise ValueError("max_batches must be positive when provided")
    if teacher is not None and teacher_outputs is not None:
        raise ValueError("Pass either teacher or teacher_outputs, not both")
    if require_teacher_guard and teacher is None and teacher_outputs is None:
        raise ValueError("Teacher output is required for the maximum-rate guard")

    device = torch.device(device)
    lpips_metric = lpips_model if lpips_model is not None else _build_lpips(device, lpips_net)
    if hasattr(lpips_metric, "to"):
        lpips_metric = lpips_metric.to(device)
    if hasattr(lpips_metric, "eval"):
        lpips_metric.eval()

    accumulators = {profile_key(profile): _new_accumulator(profile) for profile in resolved_profiles}
    teacher_accumulator = {"num_images": 0, "psnr": 0.0}
    teacher_iterator = None
    if teacher_outputs is not None and not isinstance(teacher_outputs, Mapping):
        teacher_iterator = iter(teacher_outputs)

    student_was_training = bool(getattr(student, "training", False))
    student.eval()
    if teacher is not None:
        teacher.eval()

    processed_batches = 0
    try:
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            images = _extract_images(batch).to(device, non_blocking=True)
            processed_batches += 1

            teacher_output = None
            if teacher is not None:
                teacher_output = _call_teacher_forward(
                    teacher_forward_fn or _default_teacher_forward, teacher, images
                )
            elif teacher_outputs is not None:
                if isinstance(teacher_outputs, Mapping):
                    teacher_output = teacher_outputs[batch_index]
                else:
                    try:
                        teacher_output = next(teacher_iterator)
                    except StopIteration as exc:
                        raise ValueError("teacher_outputs ended before the validation dataloader") from exc
            if teacher_output is not None:
                teacher_reconstruction = _extract_reconstruction(teacher_output).to(device)
                target_unit = _to_unit_range(images, input_range)
                teacher_unit = _to_unit_range(teacher_reconstruction, input_range)
                teacher_mse = (target_unit - teacher_unit).square().flatten(1).mean(1)
                teacher_psnr = torch.where(
                    teacher_mse <= torch.finfo(teacher_mse.dtype).eps,
                    torch.full_like(teacher_mse, 100.0),
                    10.0 * torch.log10(1.0 / teacher_mse),
                )
                teacher_accumulator["num_images"] += int(images.shape[0])
                teacher_accumulator["psnr"] += float(teacher_psnr.sum().item())

            for profile in resolved_profiles:
                key = profile_key(profile)
                output = (
                    forward_fn(student, images, profile)
                    if forward_fn is not None
                    else _student_forward(student, images, profile)
                )
                reconstruction = _extract_reconstruction(output).to(device)
                sums = _quality_sums(
                    images,
                    reconstruction,
                    lpips_metric,
                    input_range,
                    reconstruction_loss_fn,
                )
                accumulator = accumulators[key]
                accumulator["num_images"] += sums.pop("count")
                accumulator["num_batches"] += 1
                for metric, value in sums.items():
                    accumulator[metric] += value
                _accumulate_diagnostics(
                    accumulator,
                    output,
                    collapse_threshold=collapse_threshold,
                    max_distance_elements=max_distance_elements,
                )
    finally:
        student.train(student_was_training)

    if processed_batches == 0:
        raise ValueError("Validation dataloader produced no batches")

    per_profile = {
        key: _finalize_accumulator(accumulator)
        for key, accumulator in accumulators.items()
    }
    references = _normalize_reference_psnr(src_reference_psnr)
    for key, result in per_profile.items():
        reference = references.get(key)
        result["src_reference_psnr"] = reference
        result["src_psnr_gap_db"] = None if reference is None else reference - result["psnr"]

    weights = _normalize_profile_weights(resolved_profiles, profile_weights)
    weighted_mean_psnr = sum(per_profile[key]["psnr"] * weight for key, weight in weights.items())
    worst_key = min(per_profile, key=lambda key: per_profile[key]["psnr"])
    worst_psnr = per_profile[worst_key]["psnr"]
    score = (
        (1.0 - float(worst_profile_weight)) * weighted_mean_psnr
        + float(worst_profile_weight) * worst_psnr
    )

    teacher_psnr = None
    if teacher_accumulator["num_images"]:
        teacher_psnr = teacher_accumulator["psnr"] / teacher_accumulator["num_images"]
    max_key = profile_key(max_profile)
    teacher_drop = None if teacher_psnr is None else teacher_psnr - per_profile[max_key]["psnr"]
    is_guard_satisfied = (
        None if teacher_drop is None else teacher_drop <= float(max_teacher_psnr_drop_db)
    )
    eligible = is_guard_satisfied is not False and (
        not require_teacher_guard or is_guard_satisfied is True
    )

    result = {
        "profiles": [list(profile) for profile in resolved_profiles],
        "per_profile": per_profile,
        "profile_weights": weights,
        "weighted_mean_psnr": weighted_mean_psnr,
        "worst_profile": worst_key,
        "worst_psnr": worst_psnr,
        "worst_profile_weight": float(worst_profile_weight),
        "score": score,
        "teacher_psnr": teacher_psnr,
        "max_profile": list(max_profile),
        "max_profile_psnr": per_profile[max_key]["psnr"],
        "teacher_psnr_drop_db": teacher_drop,
        "max_teacher_psnr_drop_db": float(max_teacher_psnr_drop_db),
        "is_guard_satisfied": is_guard_satisfied,
        "eligible": bool(eligible),
        "guard": {
            "available": teacher_drop is not None,
            "required": bool(require_teacher_guard),
            "max_profile": list(max_profile),
            "teacher_psnr": teacher_psnr,
            "student_psnr": per_profile[max_key]["psnr"],
            "drop_db": teacher_drop,
            "threshold_db": float(max_teacher_psnr_drop_db),
            "passed": is_guard_satisfied,
        },
    }

    if writer is not None:
        if global_step is None:
            raise ValueError("global_step is required when a TensorBoard writer is supplied")
        write_profile_tensorboard(writer, result, int(global_step))
    if csv_path is not None or per_profile_csv_dir is not None:
        write_profile_csvs(
            result,
            csv_path=csv_path,
            per_profile_csv_dir=per_profile_csv_dir,
            step=global_step,
        )
    return result


# Descriptive alias for standalone callers.
evaluate_fixed_profiles = validate_profiles


__all__ = [
    "SUPPORTED_CODEBOOK_SIZES",
    "DEFAULT_VALIDATION_PROFILES",
    "MAX_PROFILE",
    "all_profiles",
    "normalize_profile",
    "profile_key",
    "resolve_profiles",
    "codebook_distance_stats",
    "validate_profiles",
    "evaluate_fixed_profiles",
    "write_profile_csvs",
    "write_profile_tensorboard",
]
