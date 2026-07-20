#!/usr/bin/env python3
"""Stage-aware evaluation for the single-teacher variable-rate RAQ pipeline.

The default path evaluates Stage-2 through Stage-5 ``VariableRateDeepSC``
checkpoints.  ``--src-teacher-only`` evaluates the Stage-1 ``DeepSC`` source
teacher at its sole [2048, 2048] profile.  ``--use-channel`` enables a
deterministic fixed-SNR Stage-5 quality test; it is intentionally incompatible
with the clean maximum-profile teacher guard.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from data.datasets import get_dataloader
from evaluation.profile_validation import (
    DEFAULT_VALIDATION_PROFILES,
    profile_key,
    resolve_profiles,
    validate_profiles,
)
from models.deepsc import DeepSC
from utils.checkpoint_utils import extract_state_dict, load_state_dict_compatible
from utils.reproducibility import setup_seed


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(
            "Variable-rate evaluation requires a metadata checkpoint dictionary, not a raw state_dict"
        )
    return checkpoint


def _model_config(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    """Read the new top-level schema, with narrow compatibility aliases."""
    value = checkpoint.get("model_config")
    if value is None and isinstance(checkpoint.get("metadata"), Mapping):
        value = checkpoint["metadata"].get("model_config")
    if value is None:
        value = checkpoint.get("variable_rate_model_config")
    if hasattr(value, "__dict__"):
        value = vars(value)
    if not isinstance(value, Mapping):
        raise KeyError(
            "Checkpoint has no model_config metadata. Re-save it with the variable-rate "
            "training/checkpoint utility; architecture inference is intentionally not guessed."
        )
    # Some writers wrap the constructor mapping for readability.
    for key in ("constructor", "constructor_kwargs", "kwargs"):
        if isinstance(value.get(key), Mapping):
            value = value[key]
            break
    return {str(key): item for key, item in value.items()}


def _checkpoint_config(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    value = checkpoint.get("config", {})
    if hasattr(value, "__dict__"):
        value = vars(value)
    return dict(value) if isinstance(value, Mapping) else {}


def _state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    for key in ("model_state_dict", "student_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            state_dict = value
            break
    else:
        state_dict = extract_state_dict(checkpoint)
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        state_dict = {str(key)[7:]: value for key, value in state_dict.items()}
    return state_dict


def _constructor_kwargs(model_class, config: Mapping[str, Any], device: torch.device):
    aliases = dict(config)
    if "source_num_embeddings" not in aliases and "num_embeddings_list" in aliases:
        aliases["source_num_embeddings"] = aliases["num_embeddings_list"]
    if "num_embeddings_list" not in aliases and "source_num_embeddings" in aliases:
        aliases["num_embeddings_list"] = aliases["source_num_embeddings"]
    if "embedding_dims" not in aliases and "embedding_dim_list" in aliases:
        aliases["embedding_dims"] = aliases["embedding_dim_list"]
    if "embedding_dim_list" not in aliases and "embedding_dims" in aliases:
        aliases["embedding_dim_list"] = aliases["embedding_dims"]
    if "strides" not in aliases and "downsample_strides" in aliases:
        aliases["strides"] = aliases["downsample_strides"]
    source_sizes = aliases.get("source_num_embeddings") or aliases.get("num_embeddings_list")
    if "num_downsample_blocks" not in aliases and source_sizes is not None:
        aliases["num_downsample_blocks"] = len(source_sizes)
    aliases["device"] = device

    signature = inspect.signature(model_class.__init__)
    accepts_extra = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_extra:
        return {key: value for key, value in aliases.items() if value is not None}
    accepted = set(signature.parameters) - {"self"}
    kwargs = {
        key: value for key, value in aliases.items() if key in accepted and value is not None
    }
    missing = [
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.default is inspect.Parameter.empty
        and parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and name not in kwargs
    ]
    if missing:
        raise KeyError(
            "model_config is missing required VariableRateDeepSC constructor fields: "
            + ", ".join(missing)
        )
    return kwargs


def build_student_from_checkpoint(
    checkpoint: Mapping[str, Any], device: torch.device
):
    from models.variable_rate_deepsc import VariableRateDeepSC

    config = _model_config(checkpoint)
    kwargs = _constructor_kwargs(VariableRateDeepSC, config, device)
    student = VariableRateDeepSC(**kwargs).to(device)
    load_state_dict_compatible(student, _state_dict(checkpoint), strict=True)
    student.eval()
    if hasattr(student, "set_channel_prob"):
        student.set_channel_prob(0.0)
    return student, config


def _teacher_constructor_config(
    variable_config: Mapping[str, Any], device: torch.device
) -> Dict[str, Any]:
    aliases = dict(variable_config)
    aliases["num_embeddings_list"] = aliases.get(
        "source_num_embeddings", aliases.get("num_embeddings_list")
    )
    aliases["embedding_dim_list"] = aliases.get(
        "embedding_dims", aliases.get("embedding_dim_list")
    )
    aliases["strides"] = aliases.get("strides", aliases.get("downsample_strides"))
    aliases["device"] = device
    aliases["use_raq"] = False
    aliases["quantizer_type"] = "simvq"
    aliases["use_swinir_enhance"] = False
    if aliases.get("num_downsample_blocks") is None and aliases.get("num_embeddings_list"):
        aliases["num_downsample_blocks"] = len(aliases["num_embeddings_list"])
    aliases.setdefault(
        "skip_dropout_p", [0.0] * max(0, int(aliases.get("num_downsample_blocks", 2)) - 1)
    )
    aliases.setdefault(
        "quantizer_axis_list", ["patch"] * int(aliases.get("num_downsample_blocks", 2))
    )
    aliases.setdefault(
        "cvq_codeword_shapes", [None] * int(aliases.get("num_downsample_blocks", 2))
    )
    signature = inspect.signature(DeepSC.__init__)
    accepted = set(signature.parameters) - {"self"}
    kwargs = {
        key: value for key, value in aliases.items() if key in accepted and value is not None
    }
    required = [
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.default is inspect.Parameter.empty
        and name not in kwargs
    ]
    if required:
        raise KeyError(
            "model_config cannot reconstruct the SRC teacher; missing: " + ", ".join(required)
        )
    return kwargs


def build_teacher_from_checkpoint(
    path: Path, variable_config: Mapping[str, Any], device: torch.device
) -> DeepSC:
    checkpoint = _load_checkpoint(path)
    teacher_config = dict(variable_config)
    try:
        own_config = _model_config(checkpoint)
    except KeyError:
        own_config = None
    if own_config:
        # A Stage-1 metadata checkpoint is authoritative for its backbone.
        teacher_config.update(own_config)
    teacher = DeepSC(**_teacher_constructor_config(teacher_config, device)).to(device)
    load_state_dict_compatible(teacher, _state_dict(checkpoint), strict=True)
    teacher.eval()
    teacher.set_channel_prob(0.0)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return teacher


CHECKPOINT_STAGES = (
    "src_teacher",
    "identity_warmup",
    "variable_rate",
    "joint_lite",
    "channel_finetune",
)


def _require_checkpoint_stage(
    checkpoint: Mapping[str, Any], expected_stage: Optional[str]
) -> Optional[str]:
    actual_stage = checkpoint.get("stage")
    if actual_stage is not None:
        actual_stage = str(actual_stage)
    if expected_stage is not None and actual_stage != expected_stage:
        raise ValueError(
            f"checkpoint stage is {actual_stage!r}; expected {expected_stage!r}"
        )
    return actual_stage


def _source_forward_for_validation(source_model, images: torch.Tensor, profile):
    """Adapt the Stage-1 teacher output to the profile diagnostics contract."""

    from training.frozen_teacher import teacher_forward

    output = dict(teacher_forward(source_model, images))
    source_codebooks = output.get("source_codebooks")
    if not isinstance(source_codebooks, list):
        raise KeyError("Stage-1 teacher output has no source_codebooks list")
    observed_profile = tuple(int(codebook.shape[0]) for codebook in source_codebooks)
    requested_profile = tuple(int(value) for value in profile)
    if requested_profile != observed_profile:
        raise ValueError(
            f"Stage-1 source profile is {observed_profile}, got {requested_profile}"
        )
    output["codebooks"] = source_codebooks
    output["profile"] = requested_profile
    return output


def _resolve_teacher_path(
    argument: Optional[str], checkpoint: Mapping[str, Any], student_path: Path
) -> Optional[Path]:
    raw = argument or checkpoint.get("teacher_checkpoint")
    if raw is None and isinstance(checkpoint.get("metadata"), Mapping):
        raw = checkpoint["metadata"].get("teacher_checkpoint")
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if path.is_absolute() or path.is_file():
        return path
    relative = student_path.parent / path
    return relative if relative.is_file() else path


def _config_get(config: Mapping[str, Any], *names: str):
    for name in names:
        if name in config and config[name] not in (None, ""):
            return config[name]
    return None


def _json_mapping(value: Optional[str], label: str) -> Optional[Dict[str, float]]:
    if not value:
        return None
    path = Path(value)
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else json.loads(value)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object mapping profile to a number")
    return {str(key): float(item) for key, item in payload.items()}


def _json_ready(value: Any):
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _print_results(results: Mapping[str, Any], title: str) -> None:
    print(f"\n{title}")
    print(f"{'Profile':>11}  {'PSNR':>8}  {'MS-SSIM':>8}  {'LPIPS':>8}  {'Active':>8}")
    for key, metrics in results["per_profile"].items():
        print(
            f"{key:>11}  {metrics['psnr']:8.4f}  {metrics['ms_ssim']:8.5f}  "
            f"{metrics['lpips']:8.5f}  {metrics['active_ratio']:8.3f}"
        )
    print(
        f"score={results['score']:.4f}, weighted_mean={results['weighted_mean_psnr']:.4f}, "
        f"worst={results['worst_profile']} ({results['worst_psnr']:.4f} dB)"
    )
    if results["teacher_psnr"] is not None:
        print(
            f"max-profile teacher drop={results['teacher_psnr_drop_db']:.4f} dB, "
            f"guard_passed={results['is_guard_satisfied']}, eligible={results['eligible']}"
        )
    else:
        print("teacher guard unavailable (evaluation explicitly allowed without a teacher)")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Stage metadata checkpoint")
    parser.add_argument(
        "--expected-stage",
        choices=CHECKPOINT_STAGES,
        help="Reject a checkpoint whose stage metadata does not match",
    )
    parser.add_argument(
        "--src-teacher-only",
        action="store_true",
        help="Evaluate a Stage-1 DeepSC checkpoint only at [2048,2048]",
    )
    parser.add_argument("--teacher-checkpoint", help="Override the one SRC [2048,2048] teacher")
    parser.add_argument(
        "--no-teacher",
        action="store_true",
        help="Evaluate metrics but disable the maximum-profile teacher guard",
    )
    parser.add_argument("--dataset", help="Image directory (defaults to checkpoint config)")
    parser.add_argument("--profiles", default=";".join(profile_key(p) for p in DEFAULT_VALIDATION_PROFILES))
    parser.add_argument("--all-profiles", action="store_true", help="Evaluate all 121 profiles")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-resize", help="Optional HxW resize understood by the dataset loader")
    parser.add_argument("--csv", required=True, help="Combined output CSV (one independent row/profile)")
    parser.add_argument("--per-profile-csv-dir", help="Also append one CSV file per profile")
    parser.add_argument("--json", required=True, help="Structured JSON output")
    parser.add_argument("--src-reference-psnr", help="JSON object or file: {'2048x16': 30.1}")
    parser.add_argument("--profile-weights", help="JSON object or file of checkpoint-score weights")
    parser.add_argument("--worst-profile-weight", type=float, default=0.25)
    parser.add_argument("--max-teacher-drop-db", type=float, default=0.3)
    parser.add_argument("--collapse-threshold", type=float, default=0.1)
    parser.add_argument("--lpips-net", default="vgg", choices=("vgg",))
    parser.add_argument(
        "--use-channel",
        action="store_true",
        help="Evaluate the student with its finite-blocklength index channel enabled",
    )
    parser.add_argument(
        "--snr-db",
        type=float,
        help="Fixed channel SNR in dB; required with --use-channel",
    )
    parser.add_argument(
        "--channel-coding-rate",
        type=float,
        help="Override the checkpoint validation coding rate",
    )
    parser.add_argument(
        "--mod-bits",
        type=int,
        choices=(1, 2, 4),
        help="Bits per channel symbol used by the finite-blocklength approximation",
    )
    parser.add_argument(
        "--channel-prob",
        type=float,
        default=1.0,
        help="Probability of applying the channel to a batch (default: 1)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_seed(args.seed)
    if args.test_resize:
        os.environ["SIMVQ_TEST_RESIZE"] = args.test_resize
    if args.src_teacher_only and args.all_profiles:
        raise ValueError("Stage-1 SRC teacher has only [2048,2048]; --all-profiles is invalid")
    if args.src_teacher_only and args.use_channel:
        raise ValueError("Stage-1 source evaluation is clean-only")
    if args.use_channel and not args.no_teacher:
        raise ValueError(
            "A noisy-channel run cannot use the clean maximum-profile teacher guard; "
            "pass --no-teacher"
        )
    if args.use_channel and args.snr_db is None:
        raise ValueError("--snr-db is required with --use-channel")
    if not args.use_channel and any(
        value is not None
        for value in (args.snr_db, args.channel_coding_rate, args.mod_bits)
    ):
        raise ValueError(
            "--snr-db/--channel-coding-rate/--mod-bits require --use-channel"
        )
    if not 0.0 < float(args.channel_prob) <= 1.0:
        raise ValueError("--channel-prob must be in (0, 1]")
    if args.channel_coding_rate is not None and not (
        0.0 < float(args.channel_coding_rate) <= 1.0
    ):
        raise ValueError("--channel-coding-rate must be in (0, 1]")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    student_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = _load_checkpoint(student_path)
    expected_stage = args.expected_stage
    if args.src_teacher_only:
        if expected_stage not in (None, "src_teacher"):
            raise ValueError("--src-teacher-only requires --expected-stage src_teacher")
        expected_stage = "src_teacher"
    checkpoint_stage = _require_checkpoint_stage(checkpoint, expected_stage)
    if args.src_teacher_only:
        model_config = _model_config(checkpoint)
        student = build_teacher_from_checkpoint(student_path, model_config, device)
    else:
        if checkpoint_stage == "src_teacher":
            raise ValueError(
                "Stage-1 DeepSC cannot be loaded as VariableRateDeepSC; "
                "pass --src-teacher-only"
            )
        student, model_config = build_student_from_checkpoint(checkpoint, device)
    saved_config = _checkpoint_config(checkpoint)

    dataset = args.dataset or _config_get(
        saved_config,
        "TEST_DATASET_PATH",
        "test_dataset_path",
        "VAL_DATASET_PATH",
        "val_dataset_path",
    )
    if not dataset:
        raise ValueError("--dataset is required because checkpoint config has no dataset path")
    dataloader = get_dataloader(
        root_dir=str(dataset),
        batch_size=args.batch_size,
        shuffle=False,
        mode="test",
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    source_max_profile = tuple(int(value) for value in model_config["source_num_embeddings"])
    teacher_path = None
    teacher = None
    forward_fn = None
    if args.src_teacher_only:
        profiles = (source_max_profile,)
        forward_fn = _source_forward_for_validation
        evaluation_mode = "src_teacher_clean"
    else:
        teacher_path = None if args.no_teacher else _resolve_teacher_path(
            args.teacher_checkpoint, checkpoint, student_path
        )
        if not args.no_teacher and teacher_path is None:
            raise ValueError(
                "No teacher checkpoint was supplied or recorded. Use --teacher-checkpoint, "
                "or explicitly pass --no-teacher to disable the guard."
            )
        teacher = (
            None
            if teacher_path is None
            else build_teacher_from_checkpoint(teacher_path, model_config, device)
        )
        profiles = resolve_profiles("all" if args.all_profiles else args.profiles)
        # Checkpoint selection always reports the source-rate anchor.  Preserve
        # an explicitly requested subset, but add its mandatory maximum-rate
        # profile when the caller omitted it.
        if source_max_profile not in profiles:
            profiles = (source_max_profile,) + profiles
        if args.use_channel:
            student.set_channel_prob(args.channel_prob)

            def channel_forward(model, images, profile):
                output = model.forward_profile(
                    images,
                    profile,
                    use_channel=True,
                    generate_hierarchy=False,
                    snr_db=args.snr_db,
                    channel_coding_rate=args.channel_coding_rate,
                    mod_bits=args.mod_bits,
                )
                if not output.get("channel_used", False):
                    raise RuntimeError(
                        "fixed-SNR evaluation requested a channel, but the model did not apply it"
                    )
                observed_snr = output.get("current_snr")
                if observed_snr is None or abs(float(observed_snr) - float(args.snr_db)) > 1e-8:
                    raise RuntimeError(
                        f"channel SNR mismatch: observed={observed_snr}, requested={args.snr_db}"
                    )
                return output

            forward_fn = channel_forward
            evaluation_mode = "student_fixed_snr_channel"
        else:
            evaluation_mode = "student_clean"

    results = validate_profiles(
        student,
        teacher,
        dataloader,
        profiles,
        device,
        max_batches=args.max_batches,
        src_reference_psnr=_json_mapping(args.src_reference_psnr, "src_reference_psnr"),
        profile_weights=_json_mapping(args.profile_weights, "profile_weights"),
        worst_profile_weight=args.worst_profile_weight,
        max_profile=source_max_profile,
        max_teacher_psnr_drop_db=args.max_teacher_drop_db,
        require_teacher_guard=not args.src_teacher_only and not args.no_teacher,
        collapse_threshold=args.collapse_threshold,
        forward_fn=forward_fn,
        lpips_net=args.lpips_net,
        csv_path=args.csv,
        per_profile_csv_dir=args.per_profile_csv_dir,
    )

    payload = {
        "checkpoint": str(student_path),
        "teacher_checkpoint": None if teacher_path is None else str(teacher_path),
        "checkpoint_stage": checkpoint_stage,
        "evaluation_mode": evaluation_mode,
        "channel": {
            "enabled": bool(args.use_channel),
            "snr_db": args.snr_db,
            "coding_rate": args.channel_coding_rate,
            "mod_bits": args.mod_bits,
            "probability": args.channel_prob if args.use_channel else 0.0,
        },
        "dataset": str(dataset),
        "max_batches": args.max_batches,
        "results": results,
    }
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _print_results(
        results,
        "Stage-1 SRC teacher validation"
        if args.src_teacher_only
        else (
            f"Fixed-profile channel validation at {args.snr_db:g} dB"
            if args.use_channel
            else "Fixed-profile variable-rate validation"
        ),
    )
    print(f"CSV: {Path(args.csv).resolve()}")
    print(f"JSON: {json_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
