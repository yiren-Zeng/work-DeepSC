"""PSNR-driven, stateless epsilon-greedy search for two-scale RAQ targets.

This is an additive evaluation entry point.  It deliberately leaves the
historical ``test_real.py`` and ``evaluation/quality.py`` paths unchanged.
The physical-rate constraint follows the existing project convention:

    channel-use ratio = modulation symbols / RGB source values

For the paper configuration this ratio must equal exactly 1/32.
"""

from __future__ import annotations

__test__ = False

import argparse
import csv
import json
import math
import random
import statistics
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
from itertools import islice, product
from pathlib import Path
from typing import Callable, Iterable, Sequence


Action = tuple[int, int]


@dataclass(frozen=True)
class ChannelProfile:
    key: str
    label: str
    ldpc_rate_numerator: int
    ldpc_rate_denominator: int
    modulation: str
    bits_per_symbol: int

    @property
    def ldpc_rate(self) -> float:
        return self.ldpc_rate_numerator / self.ldpc_rate_denominator

    def information_block_length(self, coded_block_length: int) -> int:
        numerator = coded_block_length * self.ldpc_rate_numerator
        if numerator % self.ldpc_rate_denominator:
            raise ValueError(
                f"LDPC n={coded_block_length} is incompatible with rate "
                f"{self.ldpc_rate_numerator}/{self.ldpc_rate_denominator}."
            )
        return numerator // self.ldpc_rate_denominator


CHANNEL_PROFILES = {
    "ldpc12_bpsk": ChannelProfile(
        "ldpc12_bpsk", "LDPC 1/2 + BPSK", 1, 2, "bpsk", 1
    ),
    "ldpc12_qpsk": ChannelProfile(
        "ldpc12_qpsk", "LDPC 1/2 + QPSK", 1, 2, "qpsk", 2
    ),
    "ldpc34_qpsk": ChannelProfile(
        "ldpc34_qpsk", "LDPC 3/4 + QPSK", 3, 4, "qpsk", 2
    ),
    "ldpc12_16qam": ChannelProfile(
        "ldpc12_16qam", "LDPC 1/2 + 16QAM", 1, 2, "16qam", 4
    ),
}


@dataclass(frozen=True)
class PhysicalLengths:
    action: Action
    bits_per_index: tuple[int, int]
    index_counts: tuple[int, int]
    source_values: int
    payload_bits: int
    ldpc_input_bits: int
    ldpc_padding_bits: int
    coded_bits: int
    modulation_padding_bits: int
    transmitted_bits: int
    channel_symbols: int

    @property
    def transmission_ratio(self) -> Fraction:
        return Fraction(self.channel_symbols, self.source_values)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["action"] = list(self.action)
        result["bits_per_index"] = list(self.bits_per_index)
        result["index_counts"] = list(self.index_counts)
        result["transmission_ratio"] = str(self.transmission_ratio)
        result["transmission_ratio_float"] = float(self.transmission_ratio)
        return result


@dataclass(frozen=True)
class EvaluationMetrics:
    action: Action
    snr: float
    seed: int
    ms_ssim: float
    psnr: float
    total_diagnostics: dict

    def to_dict(self) -> dict:
        return {
            "action": list(self.action),
            "snr": self.snr,
            "seed": self.seed,
            "ms_ssim": self.ms_ssim,
            "psnr": self.psnr,
            "total_diagnostics": self.total_diagnostics,
        }


def _is_power_of_two(value: int) -> bool:
    return value >= 1 and (value & (value - 1)) == 0


def _powers_of_two(minimum: int, maximum: int) -> list[int]:
    if minimum < 1 or maximum < minimum:
        raise ValueError(f"Invalid K range [{minimum}, {maximum}].")
    value = 1
    while value < minimum:
        value <<= 1
    values = []
    while value <= maximum:
        values.append(value)
        value <<= 1
    return values


def parse_ratio(value: str) -> Fraction:
    try:
        ratio = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid ratio {value!r}; use a value such as 1/32."
        ) from exc
    if ratio <= 0:
        raise argparse.ArgumentTypeError("The target ratio must be positive.")
    return ratio


def calculate_physical_lengths(
    action: Sequence[int],
    index_counts: Sequence[int],
    source_values: int,
    profile: ChannelProfile,
    ldpc_n: int,
) -> PhysicalLengths:
    """Calculate exact one-image lengths, including both padding boundaries."""
    if len(action) != 2 or len(index_counts) != 2:
        raise ValueError("This search entry point requires exactly two RAQ scales.")
    action_tuple = tuple(int(value) for value in action)
    if any(not _is_power_of_two(value) for value in action_tuple):
        raise ValueError(f"RAQ K values must be powers of two: {action_tuple}.")
    counts = tuple(int(value) for value in index_counts)
    if any(value <= 0 for value in counts) or source_values <= 0:
        raise ValueError("Index counts and source_values must be positive.")

    bits_per_index = tuple(value.bit_length() - 1 for value in action_tuple)
    payload_bits = sum(
        count * bits for count, bits in zip(counts, bits_per_index)
    )
    ldpc_k = profile.information_block_length(ldpc_n)
    num_blocks = (payload_bits + ldpc_k - 1) // ldpc_k
    ldpc_input_bits = num_blocks * ldpc_k
    coded_bits = num_blocks * ldpc_n
    modulation_padding_bits = (-coded_bits) % profile.bits_per_symbol
    transmitted_bits = coded_bits + modulation_padding_bits
    channel_symbols = transmitted_bits // profile.bits_per_symbol

    return PhysicalLengths(
        action=action_tuple,
        bits_per_index=bits_per_index,
        index_counts=counts,
        source_values=int(source_values),
        payload_bits=payload_bits,
        ldpc_input_bits=ldpc_input_bits,
        ldpc_padding_bits=ldpc_input_bits - payload_bits,
        coded_bits=coded_bits,
        modulation_padding_bits=modulation_padding_bits,
        transmitted_bits=transmitted_bits,
        channel_symbols=channel_symbols,
    )


def enumerate_exact_actions(
    index_counts: Sequence[int],
    source_values: int,
    profile: ChannelProfile,
    target_ratio: Fraction,
    min_k: int,
    max_k: int,
    ldpc_n: int = 256,
) -> tuple[list[Action], dict[Action, PhysicalLengths]]:
    """Enumerate power-of-two actions whose exact channel-use ratio matches."""
    valid_ks = _powers_of_two(min_k, max_k)
    actions: list[Action] = []
    ledger: dict[Action, PhysicalLengths] = {}
    for raw_action in product(valid_ks, repeat=2):
        action = (int(raw_action[0]), int(raw_action[1]))
        lengths = calculate_physical_lengths(
            action, index_counts, source_values, profile, ldpc_n
        )
        if lengths.transmission_ratio == target_ratio:
            actions.append(action)
            ledger[action] = lengths
    return actions, ledger


class EpsilonGreedyAgent:
    """Stateless epsilon-greedy bandit with an independent RNG."""

    def __init__(
        self,
        actions: Iterable[Action],
        eps_start: float = 0.4,
        eps_end: float = 0.05,
        decay: float = 30.0,
        seed: int = 42,
    ):
        self.actions = [tuple(action) for action in actions]
        if not self.actions:
            raise ValueError("The bandit needs at least one action.")
        if not 0 <= eps_end <= eps_start <= 1:
            raise ValueError("Require 0 <= eps_end <= eps_start <= 1.")
        if decay <= 0:
            raise ValueError("epsilon decay must be positive.")
        self.q = {action: 0.0 for action in self.actions}
        self.n = {action: 0 for action in self.actions}
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.decay = float(decay)
        self.t = 0
        self.rng = random.Random(seed)

    def epsilon(self) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -self.t / self.decay
        )

    def select(self) -> tuple[Action, str, float]:
        self.t += 1
        epsilon = self.epsilon()
        unexplored = [action for action in self.actions if self.n[action] == 0]
        if unexplored:
            return self.rng.choice(unexplored), "unexplored", epsilon
        if self.rng.random() < epsilon:
            return self.rng.choice(self.actions), "explore", epsilon
        best_q = max(self.q.values())
        tied = [action for action in self.actions if self.q[action] == best_q]
        return self.rng.choice(tied), "exploit", epsilon

    def update(self, action: Action, reward: float) -> None:
        action = tuple(action)
        if action not in self.q:
            raise KeyError(f"Unknown bandit action: {action}.")
        self.n[action] += 1
        self.q[action] += (float(reward) - self.q[action]) / self.n[action]

    def snapshot(self) -> list[dict]:
        return [
            {"action": list(action), "q": self.q[action], "n": self.n[action]}
            for action in self.actions
        ]


def run_epsilon_greedy(
    actions: Sequence[Action],
    evaluator: Callable[[Action, int], EvaluationMetrics],
    episodes: int,
    warmup_pulls: int,
    search_seed_base: int,
    eps_start: float,
    eps_end: float,
    eps_decay: float,
    agent_seed: int,
    fixed_channel_seed: int | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[EpsilonGreedyAgent, list[dict]]:
    """Run common-seed warmup followed by epsilon-greedy pulls.

    ``fixed_channel_seed`` reproduces the historical evaluator protocol: every
    action and every pull sees the exact same channel realization.  The
    evaluator cache then avoids recomputing repeated action/seed pairs.
    """
    required_warmup = len(actions) * warmup_pulls
    if warmup_pulls < 1:
        raise ValueError("warmup_pulls must be at least one.")
    if episodes < required_warmup:
        raise ValueError(
            f"episodes={episodes} is smaller than the {required_warmup} "
            "required common-seed warmup pulls."
        )

    agent = EpsilonGreedyAgent(
        actions,
        eps_start=eps_start,
        eps_end=eps_end,
        decay=eps_decay,
        seed=agent_seed,
    )
    action_visits = {action: 0 for action in agent.actions}
    trace: list[dict] = []

    def pull(action: Action, phase: str, decision: str, epsilon: float | None):
        visit_index = action_visits[action]
        seed = (
            int(fixed_channel_seed)
            if fixed_channel_seed is not None
            else search_seed_base + visit_index
        )
        metrics = evaluator(action, seed)
        # PSNR is the only bandit reward. MS-SSIM remains diagnostic-only.
        agent.update(action, metrics.psnr)
        action_visits[action] += 1
        record = {
            "episode": len(trace) + 1,
            "phase": phase,
            "decision": decision,
            "epsilon": epsilon,
            "action": list(action),
            "seed": seed,
            "reward_psnr": metrics.psnr,
            "diagnostic_ms_ssim": metrics.ms_ssim,
            "action_n": agent.n[action],
            "action_q": agent.q[action],
            "q_table": agent.snapshot(),
        }
        trace.append(record)
        if log is not None:
            epsilon_text = "-" if epsilon is None else f"{epsilon:.4f}"
            log(
                f"[pull {record['episode']:02d}/{episodes:02d}] "
                f"phase={phase:<7} mode={decision:<10} eps={epsilon_text:<6} "
                f"seed={seed} K={list(action)} PSNR={metrics.psnr:.4f} dB "
                f"Q={agent.q[action]:.4f}"
            )

    # Each action sees the same warmup seed set (common random numbers).
    for _ in range(warmup_pulls):
        for action in agent.actions:
            pull(action, "warmup", "forced", None)

    while len(trace) < episodes:
        action, decision, epsilon = agent.select()
        pull(action, "bandit", decision, epsilon)

    return agent, trace


def _mean_std_ci95(values: Sequence[float]) -> tuple[float, float, float]:
    if not values:
        raise ValueError("At least one value is required.")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    # Two-sided Student-t 95% critical values for the small Monte-Carlo sample
    # counts used by this script. The normal limit is sufficient above df=30.
    t95 = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    if len(values) > 1:
        critical = t95.get(len(values) - 1, 1.96)
        ci95 = critical * std / math.sqrt(len(values))
    else:
        ci95 = 0.0
    return mean, std, ci95


def summarize_evaluations(records: Sequence[EvaluationMetrics]) -> dict:
    if not records:
        raise ValueError("Cannot summarize an empty evaluation list.")
    psnr_mean, psnr_std, psnr_ci95 = _mean_std_ci95(
        [record.psnr for record in records]
    )
    ms_mean, ms_std, _ = _mean_std_ci95(
        [record.ms_ssim for record in records]
    )
    return {
        "action": list(records[0].action),
        "num_seeds": len(records),
        "seeds": [record.seed for record in records],
        "psnr_mean": psnr_mean,
        "psnr_std": psnr_std,
        "psnr_ci95": psnr_ci95,
        "psnr_ci95_method": (
            "two-sided Student-t half-width"
            if len(records) > 1
            else "not estimable from one fixed channel seed"
        ),
        "uncertainty_estimated": len(records) > 1,
        "ms_ssim_mean": ms_mean,
        "ms_ssim_std": ms_std,
        "per_seed": [record.to_dict() for record in records],
    }


def evaluate_common_seeds(
    actions: Sequence[Action],
    seeds: Sequence[int],
    evaluator: Callable[[Action, int], EvaluationMetrics],
) -> list[dict]:
    if not seeds:
        raise ValueError("At least one common evaluation seed is required.")
    summaries = []
    for action in actions:
        records = [evaluator(action, int(seed)) for seed in seeds]
        summaries.append(summarize_evaluations(records))
    return summaries


def _vectorized_qam16_modulate(bits):
    """Device-safe equivalent of the existing Gray-coded 16QAM modulator."""
    import torch

    if bits.numel() % 4:
        raise ValueError("16QAM requires a multiple of four transmitted bits.")
    grouped = bits.reshape(-1, 4)
    real = (2 * grouped[:, 0] - 1) * (3 - 2 * grouped[:, 1])
    imag = (2 * grouped[:, 2] - 1) * (3 - 2 * grouped[:, 3])
    scale = torch.sqrt(bits.new_tensor(10.0))
    return torch.complex(real, imag) / scale


@contextmanager
def _seeded_quality_runtime(quality_module, seed: int, fast_16qam: bool):
    """Scope seed and optional 16QAM overrides to this new entry point only."""
    original_reset = quality_module._reset_eval_seed
    original_qam16 = quality_module.qam16_modulate

    def reset_to_requested_seed():
        original_reset(int(seed))

    quality_module._reset_eval_seed = reset_to_requested_seed
    if fast_16qam:
        quality_module.qam16_modulate = _vectorized_qam16_modulate
    try:
        yield
    finally:
        quality_module._reset_eval_seed = original_reset
        quality_module.qam16_modulate = original_qam16


def _plain_total_diagnostics(total: dict) -> dict:
    keys = (
        "num_images",
        "source_pixels",
        "source_values",
        "num_indices",
        "payload_bits",
        "ldpc_input_bits",
        "ldpc_padding_bits",
        "coded_bits",
        "modulation_padding_bits",
        "transmitted_bits",
        "channel_symbols",
        "payload_bpp",
        "coded_bpp",
        "transmitted_bpp",
        "channel_uses_per_pixel",
        "transmission_ratio",
        "ber",
        "index_error_rate",
    )
    result = {}
    for key in keys:
        if key not in total:
            continue
        value = total[key]
        if value is None or isinstance(value, (str, bool, int, float)):
            result[key] = value
        elif hasattr(value, "item"):
            result[key] = value.item()
        else:
            result[key] = float(value)
    return result


def _evaluate_model_action(
    *,
    model,
    loader,
    action: Action,
    snr: float,
    seed: int,
    ldpc_code,
    device,
    profile: ChannelProfile,
    target_ratio: Fraction,
    quality_module,
) -> EvaluationMetrics:
    previous_target = list(model.raq_target_list)
    model.raq_target_list = list(action)
    try:
        with _seeded_quality_runtime(
            quality_module, seed, fast_16qam=(profile.modulation == "16qam")
        ):
            mean_ms_ssim, mean_psnr, diagnostics = (
                quality_module.evaluate_ldpc_channel(
                    model,
                    loader,
                    list(action),
                    snr,
                    ldpc_code,
                    device,
                    modulation=profile.modulation,
                    return_diagnostics=True,
                )
            )
    finally:
        model.raq_target_list = previous_target

    total = diagnostics["total"]
    actual_ratio = Fraction(
        int(total["channel_symbols"]), int(total["source_values"])
    )
    if actual_ratio != target_ratio:
        raise RuntimeError(
            f"Runtime physical ratio mismatch for K={list(action)}: "
            f"expected {target_ratio}, got {actual_ratio}."
        )
    return EvaluationMetrics(
        action=action,
        snr=float(snr),
        seed=int(seed),
        ms_ssim=float(mean_ms_ssim),
        psnr=float(mean_psnr),
        total_diagnostics=_plain_total_diagnostics(total),
    )


def _probe_layout(model, loader, device) -> tuple[tuple[int, int], int, list[int]]:
    import torch

    try:
        sample = next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError("The evaluation dataset is empty.") from exc
    if sample.ndim != 4 or sample.shape[0] != 1:
        raise RuntimeError(
            f"Expected test batch [1,C,H,W], got {tuple(sample.shape)}."
        )
    source_values = int(sample.numel())
    sample = sample.to(device)
    if hasattr(model, "_to_encoder_device"):
        sample = model._to_encoder_device(sample)
    model.eval()
    with torch.no_grad():
        features = model.semantic_encoder(sample)
    if len(features) != 2:
        raise RuntimeError(
            f"This entry point expects two encoder scales, got {len(features)}."
        )
    index_counts = tuple(
        int(feature.shape[0] * feature.shape[-2] * feature.shape[-1])
        for feature in features
    )
    image_shape = [int(value) for value in sample.shape]
    return index_counts, source_values, image_shape


def _format_snr(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _action_key(action: Sequence[int]) -> str:
    return f"{int(action[0])},{int(action[1])}"


def _validate_seed_partitions(args) -> None:
    if args.fixed_channel_seed is not None:
        if args.fixed_channel_seed < 0:
            raise ValueError("fixed-channel-seed must be non-negative.")
        return
    if not args.confirm_seeds or not args.report_seeds:
        raise ValueError("confirm-seeds and report-seeds must both be non-empty.")
    if len(set(args.confirm_seeds)) != len(args.confirm_seeds):
        raise ValueError("Confirmation seeds must be unique.")
    if len(set(args.report_seeds)) != len(args.report_seeds):
        raise ValueError("Report seeds must be unique.")
    if set(args.confirm_seeds) & set(args.report_seeds):
        raise ValueError("Confirmation and report seed sets must be disjoint.")
    possible_search_seeds = {
        args.search_seed_base + offset for offset in range(args.episodes + 1)
    }
    if possible_search_seeds & set(args.confirm_seeds):
        raise ValueError("Search and confirmation seed sets must be disjoint.")
    if possible_search_seeds & set(args.report_seeds):
        raise ValueError("Search and report seed sets must be disjoint.")


def _make_runtime_evaluator(
    *, model, loader, snr, ldpc_code, device, profile, target_ratio, quality_module
):
    cache: dict[tuple[Action, int], EvaluationMetrics] = {}

    def evaluate(action: Action, seed: int) -> EvaluationMetrics:
        key = (tuple(action), int(seed))
        if key not in cache:
            cache[key] = _evaluate_model_action(
                model=model,
                loader=loader,
                action=tuple(action),
                snr=snr,
                seed=seed,
                ldpc_code=ldpc_code,
                device=device,
                profile=profile,
                target_ratio=target_ratio,
                quality_module=quality_module,
            )
        return cache[key]

    return evaluate


def _write_json(path: str, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"JSON results saved to {output}")


def _write_csv(path: str, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "channel_profile",
        "ldpc_rate",
        "modulation",
        "snr_db",
        "target_ratio",
        "k1",
        "k2",
        "selected",
        "channel_symbols_per_image",
        "actual_ratio",
        "confirm_psnr_mean_db",
        "confirm_psnr_std_db",
        "confirm_psnr_ci95_db",
        "confirm_ms_ssim_mean",
        "report_psnr_mean_db",
        "report_psnr_std_db",
        "report_psnr_ci95_db",
        "report_ms_ssim_mean",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for profile_key, profile_result in payload["profiles"].items():
            for snr_key, snr_result in profile_result["snr_results"].items():
                best = tuple(snr_result["best_action"])
                report = snr_result["report"]
                for confirmation in snr_result["confirmation"]:
                    action = tuple(confirmation["action"])
                    physical = profile_result["action_ledger"][_action_key(action)]
                    selected = action == best
                    writer.writerow({
                        "channel_profile": profile_key,
                        "ldpc_rate": profile_result["ldpc_rate"],
                        "modulation": profile_result["modulation"],
                        "snr_db": snr_key,
                        "target_ratio": payload["target_ratio"],
                        "k1": action[0],
                        "k2": action[1],
                        "selected": selected,
                        "channel_symbols_per_image": physical["channel_symbols"],
                        "actual_ratio": physical["transmission_ratio"],
                        "confirm_psnr_mean_db": confirmation["psnr_mean"],
                        "confirm_psnr_std_db": confirmation["psnr_std"],
                        "confirm_psnr_ci95_db": confirmation["psnr_ci95"],
                        "confirm_ms_ssim_mean": confirmation["ms_ssim_mean"],
                        "report_psnr_mean_db": report["psnr_mean"] if selected else "",
                        "report_psnr_std_db": report["psnr_std"] if selected else "",
                        "report_psnr_ci95_db": report["psnr_ci95"] if selected else "",
                        "report_ms_ssim_mean": report["ms_ssim_mean"] if selected else "",
                    })
    print(f"CSV results saved to {output}")


def run(args) -> dict:
    _validate_seed_partitions(args)

    fixed_channel_seed = args.fixed_channel_seed
    if fixed_channel_seed is None:
        confirmation_seeds = list(args.confirm_seeds)
        report_seeds = list(args.report_seeds)
        channel_seed_mode = "partitioned_multi_seed"
    else:
        confirmation_seeds = [int(fixed_channel_seed)]
        report_seeds = [int(fixed_channel_seed)]
        channel_seed_mode = "fixed"

    # Runtime-heavy imports stay here so --help and unit tests need no TF/Sionna.
    import torch

    from communications.ldpc_coding import get_ldpc_code
    from config import Config
    from data.datasets import get_dataloader
    from evaluation import quality
    from utils.checkpoint_utils import build_model_from_checkpoint
    from utils.reproducibility import setup_seed

    cfg = Config()
    cfg.validate()
    setup_seed(args.agent_seed)
    device = torch.device(cfg.DEVICE)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model, inferred = build_model_from_checkpoint(str(checkpoint), cfg, device)
    if not getattr(model, "use_raq", False):
        raise RuntimeError("The selected checkpoint/configuration does not enable RAQ.")
    if getattr(model, "test_use_raq_rvq", False):
        raise RuntimeError("This search targets ordinary two-scale RAQ, not RAQ-RVQ.")

    base_loader = get_dataloader(
        root_dir=cfg.TEST_DATASET_PATH,
        batch_size=1,
        shuffle=False,
        mode="test",
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    dataset = base_loader.dataset
    if hasattr(dataset, "image_files"):
        dataset.image_files.sort()
        dataset_files = list(dataset.image_files)
    else:
        dataset_files = []
    dataset_size = len(dataset)
    if dataset_size != args.expected_images:
        raise RuntimeError(
            f"Expected {args.expected_images} evaluation images, found {dataset_size} "
            f"under {cfg.TEST_DATASET_PATH}."
        )

    if args.max_images:
        if not 1 <= args.max_images <= dataset_size:
            raise ValueError(
                f"max-images must be in [1,{dataset_size}], got {args.max_images}."
            )
        loader = list(islice(iter(base_loader), args.max_images))
        print(
            f"[Smoke mode] Evaluating only {args.max_images}/{dataset_size} images. "
            "Do not use this result in a paper."
        )
    else:
        loader = base_loader

    index_counts, source_values, image_shape = _probe_layout(model, loader, device)
    target_ratio = args.target_ratio
    profile_keys = (
        list(CHANNEL_PROFILES) if args.channel_profile == "all"
        else [args.channel_profile]
    )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "entry_point": "bandit_psnr_search.py",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "dataset": str(Path(cfg.TEST_DATASET_PATH).resolve()),
        "dataset_size": dataset_size,
        "evaluated_images": args.max_images or dataset_size,
        "dataset_files": dataset_files,
        "dataset_order": "sorted filename order",
        "selection_and_report_dataset_are_the_same": True,
        "methodology_note": (
            "RAQ selection and final reporting both use the requested Kodak set. "
            + (
                f"All stages reuse fixed channel seed {fixed_channel_seed} to match "
                "the historical test_real.py evaluation; this single realization "
                "does not estimate channel-noise variability."
                if fixed_channel_seed is not None
                else "Independent channel-noise seeds do not remove image-level "
                "selection bias."
            )
        ),
        "image_shape": image_shape,
        "index_counts_per_image": list(index_counts),
        "source_values_per_image": source_values,
        "source_codebooks": inferred["num_embeddings_list"],
        "raq_range": [model.raq_min_trg, model.raq_max_trg],
        "target_ratio": str(target_ratio),
        "reward": "mean_psnr_db",
        "snr_semantics": "normalized-symbol SNR (approximately Es/N0)",
        "qam16_modulator": (
            "device-safe vectorized implementation with the same Gray mapping as "
            "communications.modulation.qam16_modulate"
        ),
        "snrs_db": list(args.snrs),
        "bandit": {
            "algorithm": "stateless epsilon-greedy",
            "episodes": args.episodes,
            "warmup_pulls_per_action": args.warmup_pulls,
            "eps_start": args.eps_start,
            "eps_end": args.eps_end,
            "eps_decay": args.eps_decay,
            "agent_seed": args.agent_seed,
            "channel_seed_mode": channel_seed_mode,
            "fixed_channel_seed": fixed_channel_seed,
            "independent_report": fixed_channel_seed is None,
            "evaluation_cache_key": "(RAQ action, actual channel seed)",
            "search_seed_base": (
                None if fixed_channel_seed is not None else args.search_seed_base
            ),
            "confirm_seeds": confirmation_seeds,
            "report_seeds": report_seeds,
        },
        "profiles": {},
    }

    print("=" * 72)
    print("PSNR-driven stateless epsilon-greedy RAQ search")
    print(f"Checkpoint: {checkpoint}")
    print(f"Dataset: {cfg.TEST_DATASET_PATH} ({args.max_images or dataset_size} images)")
    print(f"Index counts: {list(index_counts)}, source values: {source_values}")
    print(f"Exact channel-use ratio: {target_ratio}")
    print("Reward: average PSNR only (MS-SSIM is diagnostic)")
    if fixed_channel_seed is not None:
        print(
            f"Channel seed: fixed at {fixed_channel_seed} for search, confirmation, "
            "and report (historical-comparison mode)"
        )
        print(
            "[Seed warning] A single fixed channel realization does not estimate "
            "Monte-Carlo variability."
        )
    else:
        print("Channel seed: partitioned multi-seed Monte-Carlo mode")
    print("=" * 72)
    print(
        "[Paper warning] Kodak is used for both action selection and reporting, "
        "as explicitly requested. Record this protocol as test-set selection."
    )

    initial_target = list(model.raq_target_list)
    try:
        for profile_index, profile_key in enumerate(profile_keys):
            profile = CHANNEL_PROFILES[profile_key]
            ldpc_k = profile.information_block_length(args.ldpc_n)
            ldpc_code = get_ldpc_code(ldpc_k, rate=profile.ldpc_rate)
            if int(ldpc_code["k"]) != ldpc_k or int(ldpc_code["n"]) != args.ldpc_n:
                raise RuntimeError(
                    f"Unexpected LDPC code dimensions: {ldpc_code['k']}/{ldpc_code['n']}."
                )
            actions, ledger = enumerate_exact_actions(
                index_counts=index_counts,
                source_values=source_values,
                profile=profile,
                target_ratio=target_ratio,
                min_k=model.raq_min_trg,
                max_k=model.raq_max_trg,
                ldpc_n=args.ldpc_n,
            )
            if not actions:
                raise RuntimeError(
                    f"No exact {target_ratio} actions for profile {profile.key}."
                )

            print("\n" + "#" * 72)
            print(f"Profile: {profile.label} ({profile.key})")
            print(f"LDPC: n={args.ldpc_n}, k={ldpc_k}, R={profile.ldpc_rate:g}")
            print(f"Modulation: {profile.modulation.upper()}")
            print(f"Exact actions ({len(actions)}):")
            for action in actions:
                lengths = ledger[action]
                print(
                    f"  K={list(action)} payload={lengths.payload_bits} "
                    f"coded={lengths.coded_bits} symbols={lengths.channel_symbols} "
                    f"ratio={lengths.transmission_ratio}"
                )

            profile_result = {
                "label": profile.label,
                "ldpc_n": args.ldpc_n,
                "ldpc_k": ldpc_k,
                "ldpc_rate": profile.ldpc_rate,
                "modulation": profile.modulation,
                "bits_per_symbol": profile.bits_per_symbol,
                "actions": [list(action) for action in actions],
                "action_ledger": {
                    _action_key(action): ledger[action].to_dict() for action in actions
                },
                "snr_results": {},
            }
            # Register before the SNR loop so long runs can checkpoint partial
            # results after every completed SNR.
            payload["profiles"][profile_key] = profile_result

            for snr_index, snr in enumerate(args.snrs):
                print("\n" + "-" * 72)
                print(f"Searching SNR={snr:g} dB with a fresh agent")
                evaluator = _make_runtime_evaluator(
                    model=model,
                    loader=loader,
                    snr=snr,
                    ldpc_code=ldpc_code,
                    device=device,
                    profile=profile,
                    target_ratio=target_ratio,
                    quality_module=quality,
                )
                agent, trace = run_epsilon_greedy(
                    actions=actions,
                    evaluator=evaluator,
                    episodes=args.episodes,
                    warmup_pulls=args.warmup_pulls,
                    search_seed_base=args.search_seed_base,
                    eps_start=args.eps_start,
                    eps_end=args.eps_end,
                    eps_decay=args.eps_decay,
                    agent_seed=args.agent_seed + profile_index * 1000 + snr_index,
                    fixed_channel_seed=fixed_channel_seed,
                    log=print,
                )
                bandit_best_action = max(
                    actions, key=lambda action: agent.q[action]
                )
                print(
                    f"[bandit] Q-table recommendation: K={list(bandit_best_action)}, "
                    f"Q={agent.q[bandit_best_action]:.4f} dB"
                )

                print("[confirm] Evaluating every exact-rate action on common seeds...")
                confirmation = evaluate_common_seeds(
                    actions, confirmation_seeds, evaluator
                )
                for summary in confirmation:
                    if summary["num_seeds"] == 1:
                        print(
                            f"  K={summary['action']} PSNR="
                            f"{summary['psnr_mean']:.4f} dB "
                            f"MS-SSIM={summary['ms_ssim_mean']:.4f} "
                            "(one fixed seed; uncertainty not estimated)"
                        )
                    else:
                        print(
                            f"  K={summary['action']} PSNR="
                            f"{summary['psnr_mean']:.4f}±{summary['psnr_std']:.4f} dB "
                            f"MS-SSIM={summary['ms_ssim_mean']:.4f}"
                        )
                best_confirmation = max(
                    confirmation,
                    key=lambda item: item["psnr_mean"],
                )
                best_action = tuple(best_confirmation["action"])

                if fixed_channel_seed is not None:
                    print(
                        f"[report] Selected K={list(best_action)}; reusing fixed "
                        f"channel seed {fixed_channel_seed} for historical comparison..."
                    )
                else:
                    print(
                        f"[report] Selected K={list(best_action)}; evaluating independent "
                        f"report seeds {report_seeds}..."
                    )
                report_records = [
                    evaluator(best_action, int(seed)) for seed in report_seeds
                ]
                report = summarize_evaluations(report_records)
                if report["num_seeds"] == 1:
                    print(
                        f"[result] {profile.label}, SNR={snr:g} dB: "
                        f"K={list(best_action)}, fixed-seed report PSNR="
                        f"{report['psnr_mean']:.4f} dB (CI not estimable)"
                    )
                else:
                    print(
                        f"[result] {profile.label}, SNR={snr:g} dB: "
                        f"K={list(best_action)}, report PSNR="
                        f"{report['psnr_mean']:.4f}±{report['psnr_std']:.4f} dB, "
                        f"95% CI half-width={report['psnr_ci95']:.4f} dB"
                    )

                profile_result["snr_results"][_format_snr(snr)] = {
                    "snr_db": snr,
                    "search_trace": trace,
                    "search_final_q_table": agent.snapshot(),
                    "bandit_best_action": list(bandit_best_action),
                    "confirmation": confirmation,
                    "selection_metric": "confirmation psnr_mean",
                    "best_action": list(best_action),
                    "report": report,
                }
                if args.json_output:
                    _write_json(args.json_output, payload)
                if args.csv_output:
                    _write_csv(args.csv_output, payload)
    finally:
        model.raq_target_list = initial_target

    print("\n" + "=" * 72)
    print("Final PSNR report summary")
    for profile_key, profile_result in payload["profiles"].items():
        for snr_key, result in profile_result["snr_results"].items():
            report = result["report"]
            uncertainty = (
                f"±{report['psnr_std']:.4f} dB"
                if report["uncertainty_estimated"]
                else " dB (fixed seed; uncertainty not estimated)"
            )
            print(
                f"{profile_key:<15} SNR={snr_key:>5} dB "
                f"K={result['best_action']} PSNR={report['psnr_mean']:.4f}"
                f"{uncertainty}"
            )
    print("=" * 72)

    if args.json_output:
        _write_json(args.json_output, payload)
    if args.csv_output:
        _write_csv(args.csv_output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search exact-rate two-scale RAQ targets with a stateless "
            "epsilon-greedy bandit and PSNR reward."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/shiyan_raq_src2048-2048_raq2-2048_curriculum_"
            "rate044_A_patch_ch256-512_unet2_ds8x2_k2048/best_vq_deepsc.pth"
        ),
    )
    parser.add_argument("--snrs", type=float, nargs="+", default=[0, 3, 6, 9, 12])
    parser.add_argument(
        "--channel-profile",
        choices=[*CHANNEL_PROFILES, "all"],
        default="ldpc12_bpsk",
        help="One of four paper channel profiles; 'all' runs all four sequentially.",
    )
    parser.add_argument("--target-ratio", type=parse_ratio, default=Fraction(1, 32))
    parser.add_argument("--ldpc-n", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--warmup-pulls", type=int, default=1)
    parser.add_argument("--eps-start", type=float, default=0.4)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=float, default=30.0)
    parser.add_argument("--agent-seed", type=int, default=42)
    parser.add_argument(
        "--fixed-channel-seed",
        type=int,
        default=None,
        help=(
            "Use one actual channel seed for every search/confirmation/report "
            "evaluation. This matches the historical fixed-seed evaluator; omit "
            "for partitioned multi-seed Monte-Carlo evaluation."
        ),
    )
    parser.add_argument("--search-seed-base", type=int, default=42000)
    parser.add_argument(
        "--confirm-seeds", type=int, nargs="+", default=[52000, 52001, 52002]
    )
    parser.add_argument(
        "--report-seeds",
        type=int,
        nargs="+",
        default=[62000, 62001, 62002, 62003, 62004],
    )
    parser.add_argument("--expected-images", type=int, default=24)
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Development smoke-test limit; zero means all 24 images.",
    )
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--csv-output", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
