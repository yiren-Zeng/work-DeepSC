"""PSNR bandit search for trained independent RAQ-RVQ codebooks.

This is an additive, evaluation-only entry point.  The checkpoint is frozen and
an ordered action contains one two-stage allocation per encoder scale::

    ((K_scale0_stage0, K_scale0_stage1), ...)

All actions are filtered by the exact physical channel-use ratio.  The
scale/stage payloads can either keep separate LDPC padding boundaries or be
concatenated into one physical payload, matching ``evaluation.quality``.
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

MODULATION_BITS_PER_SYMBOL = {
    "bpsk": 1,
    "qpsk": 2,
    "16qam": 4,
}


def parse_ratio(value: str) -> Fraction:
    try:
        ratio = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid ratio {value!r}; use a value such as 1/64."
        ) from exc
    if ratio <= 0:
        raise argparse.ArgumentTypeError("The target ratio must be positive.")
    return ratio


def parse_ldpc_rate(value: str) -> Fraction:
    """Parse an exact LDPC rate such as ``5/8`` or ``0.625``."""
    try:
        rate = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid LDPC rate {value!r}; use a value such as 5/8."
        ) from exc
    if not 0 < rate < 1:
        raise argparse.ArgumentTypeError("The LDPC rate must be in (0, 1).")
    return rate


def resolve_channel_profiles(args) -> list[tuple[str, ChannelProfile]]:
    """Resolve legacy named profiles or one freely configured LDPC channel."""
    rate_override = getattr(args, "ldpc_rate", None)
    modulation_override = getattr(args, "modulation", None)
    if rate_override is None and modulation_override is None:
        keys = (
            list(CHANNEL_PROFILES)
            if args.channel_profile == "all"
            else [args.channel_profile]
        )
        return [(key, CHANNEL_PROFILES[key]) for key in keys]

    if args.channel_profile == "all":
        raise ValueError(
            "--channel-profile all cannot be combined with --ldpc-rate or "
            "--modulation."
        )
    base = CHANNEL_PROFILES[args.channel_profile]
    rate = (
        Fraction(rate_override)
        if rate_override is not None
        else Fraction(base.ldpc_rate_numerator, base.ldpc_rate_denominator)
    )
    modulation = modulation_override or base.modulation
    key = f"ldpc{rate.numerator}over{rate.denominator}_{modulation}"
    profile = ChannelProfile(
        key=key,
        label=f"LDPC {rate} + {modulation.upper()}",
        ldpc_rate_numerator=rate.numerator,
        ldpc_rate_denominator=rate.denominator,
        modulation=modulation,
        bits_per_symbol=MODULATION_BITS_PER_SYMBOL[modulation],
    )
    return [(key, profile)]


class _BaseEpsilonGreedyAgent:
    """Stateless epsilon-greedy policy with an independent random generator."""

    def __init__(
        self,
        actions: Iterable[tuple],
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

    def select(self) -> tuple[tuple, str, float]:
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

    def update(self, action: tuple, reward: float) -> None:
        self.n[action] += 1
        self.q[action] += (float(reward) - self.q[action]) / self.n[action]


def _vectorized_qam16_modulate(bits):
    """Device-safe equivalent of the project's Gray-coded 16QAM modulator."""
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
    """Use one channel seed for an evaluation and restore the runtime afterward."""
    original_reset = quality_module._reset_eval_seed
    original_qam16 = quality_module.qam16_modulate

    def reset_to_requested_seed(*_args, **_kwargs):
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
        "num_images", "source_pixels", "source_values", "num_indices",
        "payload_bits", "ldpc_input_bits", "ldpc_padding_bits", "coded_bits",
        "modulation_padding_bits", "transmitted_bits", "channel_symbols",
        "payload_bpp", "coded_bpp", "transmitted_bpp",
        "channel_uses_per_pixel", "transmission_ratio", "ber",
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


Action = tuple[tuple[int, int], ...]
STREAM_PACKINGS = ("per_stage", "combined")


def normalize_stream_packing(stream_packing: str) -> str:
    packing = str(stream_packing).strip().lower()
    if packing not in STREAM_PACKINGS:
        raise ValueError(
            f"stream_packing must be one of {STREAM_PACKINGS}, got {stream_packing!r}."
        )
    return packing


def _is_power_of_two(value: int) -> bool:
    return value >= 1 and (value & (value - 1)) == 0


def _powers_of_two(minimum: int, maximum: int) -> list[int]:
    if minimum < 2 or maximum < minimum:
        raise ValueError(f"Invalid independent RAQ-RVQ K range [{minimum}, {maximum}].")
    value = 1
    while value < minimum:
        value <<= 1
    values = []
    while value <= maximum:
        values.append(value)
        value <<= 1
    return values


def normalize_action(action: Sequence[Sequence[int]]) -> Action:
    """Return a strict, hashable scale-major action with two stages per scale."""
    try:
        scales = tuple(tuple(int(value) for value in stage_ks) for stage_ks in action)
    except (TypeError, ValueError) as exc:
        raise TypeError("An independent RAQ-RVQ action must be a nested integer list.") from exc
    if not scales or any(len(stage_ks) != 2 for stage_ks in scales):
        raise ValueError(
            "An independent RAQ-RVQ action must contain at least one scale "
            "and exactly two K values per scale."
        )
    if any(not _is_power_of_two(value) or value < 2 for scale in scales for value in scale):
        raise ValueError(f"Every independent RAQ-RVQ K must be a power of two >= 2: {scales}.")
    return scales  # type: ignore[return-value]


def action_to_lists(action: Sequence[Sequence[int]]) -> list[list[int]]:
    normalized = normalize_action(action)
    return [list(stage_ks) for stage_ks in normalized]


def _action_key(action: Sequence[Sequence[int]]) -> str:
    normalized = normalize_action(action)
    return ";".join(f"{stage_ks[0]},{stage_ks[1]}" for stage_ks in normalized)


@dataclass(frozen=True)
class StreamPhysicalLengths:
    scale: int | None
    stage: int | None
    num_embeddings: int | None
    index_count: int
    bits_per_index: int | None
    payload_bits: int
    ldpc_input_bits: int
    ldpc_padding_bits: int
    coded_bits: int
    modulation_padding_bits: int
    transmitted_bits: int
    channel_symbols: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PhysicalLengths:
    action: Action
    stream_packing: str
    bits_per_index: tuple[tuple[int, int], ...]
    index_counts: tuple[int, ...]
    source_values: int
    streams: tuple[StreamPhysicalLengths, ...]
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
        result["action"] = action_to_lists(self.action)
        result["bits_per_index"] = [list(values) for values in self.bits_per_index]
        result["index_counts"] = list(self.index_counts)
        result["streams"] = [stream.to_dict() for stream in self.streams]
        result["transmission_ratio"] = str(self.transmission_ratio)
        result["transmission_ratio_float"] = float(self.transmission_ratio)
        return result


def calculate_physical_lengths(
    action: Sequence[Sequence[int]],
    index_counts: Sequence[int],
    source_values: int,
    profile: ChannelProfile,
    ldpc_n: int,
    stream_packing: str = "per_stage",
) -> PhysicalLengths:
    """Calculate physical lengths for separate or combined payload packing."""
    normalized = normalize_action(action)
    packing = normalize_stream_packing(stream_packing)
    counts = tuple(int(value) for value in index_counts)
    if len(counts) != len(normalized):
        raise ValueError(
            "Independent RAQ-RVQ action scale count must match index_counts: "
            f"{len(normalized)} != {len(counts)}."
        )
    if any(value <= 0 for value in counts) or int(source_values) <= 0:
        raise ValueError("Index counts and source_values must be positive.")
    if int(ldpc_n) <= 0:
        raise ValueError("ldpc_n must be positive.")

    ldpc_k = profile.information_block_length(int(ldpc_n))
    bits_nested = tuple(
        tuple(value.bit_length() - 1 for value in stage_ks)
        for stage_ks in normalized
    )
    payload_components = []
    for scale, (stage_ks, stage_bits, index_count) in enumerate(
        zip(normalized, bits_nested, counts)
    ):
        for stage, (num_embeddings, bits_per_index) in enumerate(
            zip(stage_ks, stage_bits)
        ):
            payload_components.append(
                (scale, stage, num_embeddings, index_count, bits_per_index)
            )

    streams = []
    if packing == "combined":
        payload_bits = sum(
            index_count * bits_per_index
            for _, _, _, index_count, bits_per_index in payload_components
        )
        blocks = (payload_bits + ldpc_k - 1) // ldpc_k
        ldpc_input_bits = blocks * ldpc_k
        coded_bits = blocks * int(ldpc_n)
        modulation_padding_bits = (-coded_bits) % profile.bits_per_symbol
        transmitted_bits = coded_bits + modulation_padding_bits
        streams.append(
            StreamPhysicalLengths(
                scale=None,
                stage=None,
                num_embeddings=None,
                index_count=sum(component[3] for component in payload_components),
                bits_per_index=None,
                payload_bits=payload_bits,
                ldpc_input_bits=ldpc_input_bits,
                ldpc_padding_bits=ldpc_input_bits - payload_bits,
                coded_bits=coded_bits,
                modulation_padding_bits=modulation_padding_bits,
                transmitted_bits=transmitted_bits,
                channel_symbols=transmitted_bits // profile.bits_per_symbol,
            )
        )
    else:
        for scale, stage, num_embeddings, index_count, bits_per_index in payload_components:
            payload_bits = index_count * bits_per_index
            blocks = (payload_bits + ldpc_k - 1) // ldpc_k
            ldpc_input_bits = blocks * ldpc_k
            coded_bits = blocks * int(ldpc_n)
            modulation_padding_bits = (-coded_bits) % profile.bits_per_symbol
            transmitted_bits = coded_bits + modulation_padding_bits
            streams.append(
                StreamPhysicalLengths(
                    scale=scale,
                    stage=stage,
                    num_embeddings=num_embeddings,
                    index_count=index_count,
                    bits_per_index=bits_per_index,
                    payload_bits=payload_bits,
                    ldpc_input_bits=ldpc_input_bits,
                    ldpc_padding_bits=ldpc_input_bits - payload_bits,
                    coded_bits=coded_bits,
                    modulation_padding_bits=modulation_padding_bits,
                    transmitted_bits=transmitted_bits,
                    channel_symbols=transmitted_bits // profile.bits_per_symbol,
                )
            )

    def total(name: str) -> int:
        return sum(int(getattr(stream, name)) for stream in streams)

    return PhysicalLengths(
        action=normalized,
        stream_packing=packing,
        bits_per_index=bits_nested,  # type: ignore[arg-type]
        index_counts=counts,  # type: ignore[arg-type]
        source_values=int(source_values),
        streams=tuple(streams),
        payload_bits=total("payload_bits"),
        ldpc_input_bits=total("ldpc_input_bits"),
        ldpc_padding_bits=total("ldpc_padding_bits"),
        coded_bits=total("coded_bits"),
        modulation_padding_bits=total("modulation_padding_bits"),
        transmitted_bits=total("transmitted_bits"),
        channel_symbols=total("channel_symbols"),
    )


def enumerate_exact_actions(
    index_counts: Sequence[int],
    source_values: int,
    profile: ChannelProfile,
    target_ratio: Fraction,
    min_k: int,
    max_k: int,
    ldpc_n: int = 256,
    stream_packing: str = "per_stage",
) -> tuple[list[Action], dict[Action, PhysicalLengths]]:
    """Enumerate ordered two-K-per-scale actions at one exact physical rate."""
    if target_ratio <= 0:
        raise ValueError("target_ratio must be positive.")
    packing = normalize_stream_packing(stream_packing)
    valid_ks = _powers_of_two(int(min_k), int(max_k))
    counts = tuple(int(value) for value in index_counts)
    num_scales = len(counts)
    if num_scales < 1:
        raise ValueError("Independent RAQ-RVQ search requires at least one scale.")
    actions = []
    ledger = {}
    for values in product(valid_ks, repeat=2 * num_scales):
        action: Action = tuple(
            (values[2 * scale], values[2 * scale + 1])
            for scale in range(num_scales)
        )
        lengths = calculate_physical_lengths(
            action,
            counts,
            source_values,
            profile,
            ldpc_n,
            stream_packing=packing,
        )
        if lengths.transmission_ratio == target_ratio:
            actions.append(action)
            ledger[action] = lengths
    return actions, ledger


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
            "action": action_to_lists(self.action),
            "snr": self.snr,
            "seed": self.seed,
            "ms_ssim": self.ms_ssim,
            "psnr": self.psnr,
            "total_diagnostics": self.total_diagnostics,
        }


class EpsilonGreedyAgent(_BaseEpsilonGreedyAgent):
    """Epsilon-greedy policy with nested-action serialization."""

    def __init__(self, actions: Iterable[Action], **kwargs):
        super().__init__([normalize_action(action) for action in actions], **kwargs)

    def snapshot(self) -> list[dict]:
        return [
            {"action": action_to_lists(action), "q": self.q[action], "n": self.n[action]}
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
    """Run common-seed warmup and then PSNR-driven epsilon-greedy pulls."""
    normalized_actions = [normalize_action(action) for action in actions]
    required_warmup = len(normalized_actions) * int(warmup_pulls)
    if warmup_pulls < 1:
        raise ValueError("warmup_pulls must be at least one.")
    if episodes < required_warmup:
        raise ValueError(
            f"episodes={episodes} is smaller than the {required_warmup} "
            "required common-seed warmup pulls."
        )
    agent = EpsilonGreedyAgent(
        normalized_actions,
        eps_start=eps_start,
        eps_end=eps_end,
        decay=eps_decay,
        seed=agent_seed,
    )
    action_visits = {action: 0 for action in agent.actions}
    trace = []

    def pull(action: Action, phase: str, decision: str, epsilon: float | None):
        visit = action_visits[action]
        seed = (
            int(fixed_channel_seed)
            if fixed_channel_seed is not None
            else int(search_seed_base) + visit
        )
        metrics = evaluator(action, seed)
        agent.update(action, metrics.psnr)
        action_visits[action] += 1
        record = {
            "episode": len(trace) + 1,
            "phase": phase,
            "decision": decision,
            "epsilon": epsilon,
            "action": action_to_lists(action),
            "seed": seed,
            "reward_psnr": metrics.psnr,
            "diagnostic_ms_ssim": metrics.ms_ssim,
            "action_n": agent.n[action],
            "action_q": agent.q[action],
            "q_table": agent.snapshot(),
        }
        trace.append(record)
        if log is not None:
            eps_text = "-" if epsilon is None else f"{epsilon:.4f}"
            log(
                f"[pull {record['episode']:03d}/{episodes:03d}] "
                f"phase={phase:<7} mode={decision:<10} eps={eps_text:<6} "
                f"seed={seed} K={record['action']} PSNR={metrics.psnr:.4f} dB "
                f"Q={agent.q[action]:.4f}"
            )

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
    t95 = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
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
    ms_mean, ms_std, _ = _mean_std_ci95([record.ms_ssim for record in records])
    return {
        "action": action_to_lists(records[0].action),
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
    return [
        summarize_evaluations([evaluator(action, int(seed)) for seed in seeds])
        for action in actions
    ]


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
    stream_packing: str = "per_stage",
) -> EvaluationMetrics:
    """Evaluate one nested action and restore model state even after failure."""
    normalized = normalize_action(action)
    packing = normalize_stream_packing(stream_packing)
    if not hasattr(model, "independent_raq_rvq_k_lists"):
        raise RuntimeError("The model has no independent RAQ-RVQ K layout.")
    previous = [list(stage_ks) for stage_ks in model.independent_raq_rvq_k_lists]
    model.independent_raq_rvq_k_lists = action_to_lists(normalized)
    try:
        with _seeded_quality_runtime(
            quality_module, int(seed), fast_16qam=(profile.modulation == "16qam")
        ):
            mean_ms_ssim, mean_psnr, diagnostics = (
                quality_module.evaluate_ldpc_channel(
                    model,
                    loader,
                    snr,
                    ldpc_code,
                    device,
                    modulation=profile.modulation,
                    return_diagnostics=True,
                    stream_packing=packing,
                )
            )
    finally:
        model.independent_raq_rvq_k_lists = previous

    if not diagnostics.get("rvq_enabled", False):
        raise RuntimeError("Runtime evaluator did not observe the nested RAQ-RVQ branch.")
    total = diagnostics["total"]
    actual_ratio = Fraction(
        int(total["channel_symbols"]), int(total["source_values"])
    )
    if actual_ratio != target_ratio:
        raise RuntimeError(
            f"Runtime physical ratio mismatch for K={action_to_lists(normalized)}: "
            f"expected {target_ratio}, got {actual_ratio}."
        )
    return EvaluationMetrics(
        action=normalized,
        snr=float(snr),
        seed=int(seed),
        ms_ssim=float(mean_ms_ssim),
        psnr=float(mean_psnr),
        total_diagnostics=_plain_total_diagnostics(total),
    )


def _make_runtime_evaluator(
    *,
    model,
    loader,
    snr,
    ldpc_code,
    device,
    profile,
    target_ratio,
    quality_module,
    stream_packing="per_stage",
):
    packing = normalize_stream_packing(stream_packing)
    cache: dict[tuple[Action, int], EvaluationMetrics] = {}

    def evaluate(action: Action, seed: int) -> EvaluationMetrics:
        normalized = normalize_action(action)
        key = (normalized, int(seed))
        if key not in cache:
            cache[key] = _evaluate_model_action(
                model=model,
                loader=loader,
                action=normalized,
                snr=snr,
                seed=seed,
                ldpc_code=ldpc_code,
                device=device,
                profile=profile,
                target_ratio=target_ratio,
                quality_module=quality_module,
                stream_packing=packing,
            )
        return cache[key]

    return evaluate


def _probe_layout(model, loader, device) -> tuple[tuple[int, ...], int, list[int]]:
    import torch

    try:
        sample = next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError("The evaluation dataset is empty.") from exc
    if sample.ndim != 4 or sample.shape[0] != 1:
        raise RuntimeError(f"Expected test batch [1,C,H,W], got {tuple(sample.shape)}.")
    source_values = int(sample.numel())
    sample = sample.to(device)
    if hasattr(model, "_to_encoder_device"):
        sample = model._to_encoder_device(sample)
    model.eval()
    with torch.no_grad():
        features = model.semantic_encoder(sample)
    if not features:
        raise RuntimeError("The encoder did not return any feature scales.")
    counts = tuple(
        int(feature.shape[0] * feature.shape[-2] * feature.shape[-1])
        for feature in features
    )
    return counts, source_values, [int(value) for value in sample.shape]  # type: ignore[return-value]


def _format_snr(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


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
    search_seeds = {
        int(args.search_seed_base) + offset for offset in range(int(args.episodes) + 1)
    }
    if search_seeds & set(args.confirm_seeds):
        raise ValueError("Search and confirmation seed sets must be disjoint.")
    if search_seeds & set(args.report_seeds):
        raise ValueError("Search and report seed sets must be disjoint.")


def _write_json(path: str, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"JSON results saved to {output}")


def _write_csv(path: str, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    num_scales = payload.get("num_scales")
    if num_scales is None:
        for profile_result in payload["profiles"].values():
            for snr_result in profile_result["snr_results"].values():
                num_scales = len(normalize_action(snr_result["best_action"]))
                break
            if num_scales is not None:
                break
    if num_scales is None or int(num_scales) < 1:
        raise ValueError("Cannot determine the number of action scales for CSV output.")
    k_fieldnames = [
        f"k_scale{scale}_stage{stage}"
        for scale in range(int(num_scales))
        for stage in range(2)
    ]
    fieldnames = [
        "channel_profile", "ldpc_rate", "modulation", "stream_packing",
        "snr_db", "target_ratio",
        *k_fieldnames,
        "selected", "channel_symbols_per_image", "actual_ratio",
        "confirm_psnr_mean_db", "confirm_psnr_std_db", "confirm_psnr_ci95_db",
        "confirm_ms_ssim_mean", "report_psnr_mean_db", "report_psnr_std_db",
        "report_psnr_ci95_db", "report_ms_ssim_mean",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for profile_key, profile_result in payload["profiles"].items():
            for snr_key, snr_result in profile_result["snr_results"].items():
                best = normalize_action(snr_result["best_action"])
                report = snr_result["report"]
                for confirmation in snr_result["confirmation"]:
                    action = normalize_action(confirmation["action"])
                    if len(action) != int(num_scales):
                        raise ValueError(
                            "CSV action scale count changed within one result payload."
                        )
                    physical = profile_result["action_ledger"][_action_key(action)]
                    selected = action == best
                    row = {
                        "channel_profile": profile_key,
                        "ldpc_rate": profile_result["ldpc_rate"],
                        "modulation": profile_result["modulation"],
                        "stream_packing": payload.get("stream_packing", "per_stage"),
                        "snr_db": snr_key,
                        "target_ratio": payload["target_ratio"],
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
                    }
                    row.update({
                        f"k_scale{scale}_stage{stage}": action[scale][stage]
                        for scale in range(int(num_scales))
                        for stage in range(2)
                    })
                    writer.writerow(row)
    print(f"CSV results saved to {output}")


def run(args) -> dict:
    _validate_seed_partitions(args)
    stream_packing = normalize_stream_packing(
        getattr(args, "stream_packing", "per_stage")
    )
    profiles = resolve_channel_profiles(args)
    for _, profile in profiles:
        profile.information_block_length(args.ldpc_n)
    if args.confirm_top_k < 1:
        raise ValueError("confirm-top-k must be at least one.")

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
    if not getattr(model, "independent_raq_rvq_k_lists", None):
        raise RuntimeError("The selected checkpoint/config has no independent RAQ-RVQ layout.")
    if int(getattr(model, "independent_raq_rvq_depth", 0)) != 2:
        raise RuntimeError("This search requires independent RAQ-RVQ depth=2.")

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
            raise ValueError(f"max-images must be in [1,{dataset_size}], got {args.max_images}.")
        loader = list(islice(iter(base_loader), args.max_images))
        print(
            f"[Smoke mode] Evaluating only {args.max_images}/{dataset_size} images. "
            "Do not use this result in a paper."
        )
    else:
        loader = base_loader

    index_counts, source_values, image_shape = _probe_layout(model, loader, device)
    num_scales = len(index_counts)
    configured_layout = [
        list(values) for values in model.independent_raq_rvq_k_lists
    ]
    if len(configured_layout) != num_scales:
        raise RuntimeError(
            "Configured independent RAQ-RVQ scale count does not match encoder "
            f"features: {len(configured_layout)} != {num_scales}."
        )
    fixed_seed = args.fixed_channel_seed
    confirmation_seeds = (
        [int(fixed_seed)] if fixed_seed is not None else list(args.confirm_seeds)
    )
    report_seeds = (
        [int(fixed_seed)] if fixed_seed is not None else list(args.report_seeds)
    )
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "entry_point": "bandit_independent_raq_rvq_psnr_search.py",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "dataset": str(Path(cfg.TEST_DATASET_PATH).resolve()),
        "dataset_size": dataset_size,
        "evaluated_images": args.max_images or dataset_size,
        "dataset_files": dataset_files,
        "dataset_order": "sorted filename order",
        "selection_and_report_dataset_are_the_same": True,
        "methodology_note": (
            "The requested evaluation calibrates and reports on the same dataset; "
            "treat the selected action as test-set tuning."
        ),
        "image_shape": image_shape,
        "index_counts_per_image": list(index_counts),
        "source_values_per_image": source_values,
        "transmission_ratio_definition": "channel_symbols / (C * H * W)",
        "source_codebooks": inferred["num_embeddings_list"],
        "num_scales": num_scales,
        "independent_raq_rvq_depth": 2,
        "action_layout": "[" + ",".join(
            f"[scale{scale}_stage0,scale{scale}_stage1]"
            for scale in range(num_scales)
        ) + "]",
        "k_range": [args.min_k, args.max_k],
        "target_ratio": str(args.target_ratio),
        "stream_packing": stream_packing,
        "reward": "mean_psnr_db",
        "snrs_db": list(args.snrs),
        "bandit": {
            "algorithm": "stateless epsilon-greedy",
            "episodes": args.episodes,
            "warmup_pulls_per_action": args.warmup_pulls,
            "eps_start": args.eps_start,
            "eps_end": args.eps_end,
            "eps_decay": args.eps_decay,
            "agent_seed": args.agent_seed,
            "channel_seed_mode": "fixed" if fixed_seed is not None else "partitioned_multi_seed",
            "fixed_channel_seed": fixed_seed,
            "search_seed_base": None if fixed_seed is not None else args.search_seed_base,
            "confirm_seeds": confirmation_seeds,
            "report_seeds": report_seeds,
            "confirm_top_k": args.confirm_top_k,
            "evaluation_cache_key": "(nested action, actual channel seed)",
        },
        "profiles": {},
    }

    print("=" * 76)
    print("PSNR-driven epsilon-greedy independent RAQ-RVQ search")
    print(f"Checkpoint: {checkpoint}")
    print(f"Index counts: {list(index_counts)}, source values: {source_values}")
    print(
        "Fixed exact transmission ratio "
        f"(channel symbols / RGB source values): {args.target_ratio}"
    )
    print(
        f"Action: {2 * num_scales} ordered and independently selected K values "
        f"across {num_scales} scale(s)"
    )
    print(f"Stream packing: {stream_packing}")
    print("=" * 76)

    initial_layout = [list(values) for values in configured_layout]
    try:
        for profile_index, (profile_key, profile) in enumerate(profiles):
            ldpc_k = profile.information_block_length(args.ldpc_n)
            ldpc_code = get_ldpc_code(ldpc_k, rate=profile.ldpc_rate)
            if int(ldpc_code["k"]) != ldpc_k or int(ldpc_code["n"]) != args.ldpc_n:
                raise RuntimeError(
                    f"Unexpected LDPC dimensions: {ldpc_code['k']}/{ldpc_code['n']}."
                )
            actions, ledger = enumerate_exact_actions(
                index_counts,
                source_values,
                profile,
                args.target_ratio,
                args.min_k,
                args.max_k,
                args.ldpc_n,
                stream_packing=stream_packing,
            )
            if not actions:
                raise RuntimeError(
                    f"No exact {args.target_ratio} actions for {profile.key}."
                )
            if args.confirm_top_k > len(actions):
                raise ValueError(
                    f"confirm-top-k={args.confirm_top_k} exceeds {len(actions)} actions."
                )
            print(
                f"\nProfile {profile.label}: LDPC k/n={ldpc_k}/{args.ldpc_n}, "
                f"{len(actions)} exact-rate actions"
            )
            profile_result = {
                "label": profile.label,
                "ldpc_n": args.ldpc_n,
                "ldpc_k": ldpc_k,
                "ldpc_rate": profile.ldpc_rate,
                "ldpc_rate_fraction": (
                    f"{profile.ldpc_rate_numerator}/"
                    f"{profile.ldpc_rate_denominator}"
                ),
                "modulation": profile.modulation,
                "bits_per_symbol": profile.bits_per_symbol,
                "stream_packing": stream_packing,
                "actions": [action_to_lists(action) for action in actions],
                "action_ledger": {
                    _action_key(action): ledger[action].to_dict() for action in actions
                },
                "snr_results": {},
            }
            payload["profiles"][profile_key] = profile_result

            for snr_index, snr in enumerate(args.snrs):
                print(f"\nSearching SNR={snr:g} dB with a fresh agent")
                evaluator = _make_runtime_evaluator(
                    model=model,
                    loader=loader,
                    snr=snr,
                    ldpc_code=ldpc_code,
                    device=device,
                    profile=profile,
                    target_ratio=args.target_ratio,
                    quality_module=quality,
                    stream_packing=stream_packing,
                )
                agent, trace = run_epsilon_greedy(
                    actions,
                    evaluator,
                    args.episodes,
                    args.warmup_pulls,
                    args.search_seed_base,
                    args.eps_start,
                    args.eps_end,
                    args.eps_decay,
                    args.agent_seed + profile_index * 1000 + snr_index,
                    fixed_channel_seed=fixed_seed,
                    log=print,
                )
                ranked = sorted(actions, key=lambda action: (-agent.q[action], action))
                bandit_best = ranked[0]
                candidates = ranked[: args.confirm_top_k]
                print(
                    f"[bandit] Q recommendation K={action_to_lists(bandit_best)}, "
                    f"Q={agent.q[bandit_best]:.4f} dB"
                )
                print(
                    f"[confirm] Evaluating Q top-{len(candidates)} on common seeds "
                    f"{confirmation_seeds}"
                )
                confirmation = evaluate_common_seeds(
                    candidates, confirmation_seeds, evaluator
                )
                best_confirmation = max(
                    confirmation, key=lambda item: item["psnr_mean"]
                )
                best_action = normalize_action(best_confirmation["action"])
                report_records = [
                    evaluator(best_action, int(seed)) for seed in report_seeds
                ]
                report = summarize_evaluations(report_records)
                print(
                    f"[result] SNR={snr:g} K={action_to_lists(best_action)} "
                    f"PSNR={report['psnr_mean']:.4f} dB"
                )
                profile_result["snr_results"][_format_snr(snr)] = {
                    "snr_db": snr,
                    "search_trace": trace,
                    "search_final_q_table": agent.snapshot(),
                    "bandit_best_action": action_to_lists(bandit_best),
                    "confirmation_candidates": [
                        action_to_lists(action) for action in candidates
                    ],
                    "confirmation": confirmation,
                    "selection_metric": "top-K confirmation psnr_mean",
                    "best_action": action_to_lists(best_action),
                    "report": report,
                }
                if args.json_output:
                    _write_json(args.json_output, payload)
                if args.csv_output:
                    _write_csv(args.csv_output, payload)
    finally:
        model.independent_raq_rvq_k_lists = initial_layout

    if args.json_output:
        _write_json(args.json_output, payload)
    if args.csv_output:
        _write_csv(args.csv_output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search ordered independent RAQ-RVQ allocations at one exact "
            "physical rate using PSNR-driven epsilon-greedy."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/shiyan_independent_raq_rvq_src64-64_trg2-64_d2_"
            "curriculum_rate094_A_patch_ch256-512_res6-6_randomval_"
            "unet2_ds8x2_k64/"
            "best_vq_deepsc.pth"
        ),
    )
    parser.add_argument("--snrs", type=float, nargs="+", default=[0, 3, 6, 9, 12])
    parser.add_argument(
        "--channel-profile",
        choices=[*CHANNEL_PROFILES, "all"],
        default="ldpc12_qpsk",
        help=(
            "Legacy preset. When --ldpc-rate and/or --modulation is supplied, "
            "the supplied value overrides this preset."
        ),
    )
    parser.add_argument(
        "--ldpc-rate",
        type=parse_ldpc_rate,
        default=None,
        help=(
            "Exact custom LDPC rate, for example 5/8. LDPC_N multiplied by "
            "this rate must be an integer."
        ),
    )
    parser.add_argument(
        "--modulation",
        choices=tuple(MODULATION_BITS_PER_SYMBOL),
        default=None,
        help="Custom modulation used with --ldpc-rate (default: preset modulation).",
    )
    parser.add_argument("--target-ratio", type=parse_ratio, default=Fraction(1, 64))
    parser.add_argument(
        "--stream-packing",
        choices=STREAM_PACKINGS,
        default="combined",
        help=(
            "Concatenate all payloads before one LDPC padding boundary "
            "(default), or use a separate stream for each scale/stage."
        ),
    )
    parser.add_argument("--ldpc-n", type=int, default=256)
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--warmup-pulls", type=int, default=1)
    parser.add_argument("--eps-start", type=float, default=0.4)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=float, default=30.0)
    parser.add_argument("--agent-seed", type=int, default=42)
    parser.add_argument("--fixed-channel-seed", type=int, default=None)
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
    parser.add_argument("--confirm-top-k", type=int, default=2)
    parser.add_argument("--expected-images", type=int, default=24)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--csv-output", default=None)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
