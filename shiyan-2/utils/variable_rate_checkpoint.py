"""Metadata-rich checkpoints for the isolated variable-rate experiment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch


FORMAT_VERSION = 1


def _config_payload(config) -> Dict[str, Any]:
    if hasattr(config, "as_dict"):
        return dict(config.as_dict())
    if hasattr(config, "architecture_summary"):
        return dict(config.architecture_summary())
    payload = {}
    for name in dir(config):
        if not name.isupper():
            continue
        value = getattr(config, name)
        if isinstance(value, (str, int, float, bool, type(None), list, tuple, dict)):
            payload[name.lower()] = value
    return payload


def make_checkpoint(
    *,
    model,
    stage: str,
    epoch: int,
    config,
    optimizer=None,
    scheduler=None,
    scaler=None,
    sampler=None,
    global_step: int = 0,
    best_score: Optional[float] = None,
    teacher_checkpoint: Optional[str] = None,
    validation: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    extra_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "format": "single_teacher_variable_rate_raq",
        "format_version": FORMAT_VERSION,
        "stage": str(stage),
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "profile_sampler_state": sampler.state_dict() if sampler is not None else None,
        "global_step": int(global_step),
        "best_score": best_score,
        "teacher_checkpoint": str(teacher_checkpoint) if teacher_checkpoint else None,
        "config": _config_payload(config),
        "model_config": dict(model_config or {}),
        "validation": validation,
        "extra_state": dict(extra_state or {}),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def save_checkpoint(path: str | Path, **kwargs) -> None:
    """Write through a sibling temporary file so an interrupted save is recoverable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(make_checkpoint(**kwargs), temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        # Stage-1 compatibility with a plain state_dict is intentionally explicit.
        return {
            "format": "legacy_state_dict",
            "format_version": 0,
            "model_state_dict": checkpoint,
        }
    return checkpoint


def restore_rng(checkpoint: Dict[str, Any]) -> None:
    rng_state = checkpoint.get("rng_state")
    if rng_state is not None:
        torch.set_rng_state(rng_state.cpu())
    cuda_states = checkpoint.get("cuda_rng_state")
    if cuda_states is not None and torch.cuda.is_available():
        available = torch.cuda.device_count()
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states[:available]])
