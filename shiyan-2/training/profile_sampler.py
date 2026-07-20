"""Atomic, coverage-aware profile sampling for variable-rate RAQ training."""

from __future__ import annotations

import random
from collections import Counter
from typing import Iterable, Mapping, Sequence

from config_variable_rate import SUPPORTED_K_VALUES, parse_profile


Profile = tuple[int, int]


def _profile_key(profile: Profile) -> str:
    return f"{profile[0]}x{profile[1]}"


class ProfileSampler:
    """Sandwich sampler whose unit is a complete two-layer profile.

    Each call contains the maximum profile, the configured minimum profile, and
    ``num_random`` intermediate profiles.  Intermediate choices are drawn from
    the least-seen count bucket, with randomness only used to break ties.  This
    gives deterministic coverage guarantees without independently sampling K0
    and K1.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        profiles: Iterable[Sequence[int]],
        *,
        num_random: int = 1,
        min_profile: Sequence[int] = (2, 2),
        max_profile: Sequence[int] = (2048, 2048),
        supported_k: Sequence[int] = SUPPORTED_K_VALUES,
        seed: int = 3407,
    ) -> None:
        self.supported_k = tuple(int(k) for k in supported_k)
        if len(self.supported_k) == 0 or len(set(self.supported_k)) != len(self.supported_k):
            raise ValueError("supported_k must contain distinct values")
        if tuple(sorted(self.supported_k)) != self.supported_k:
            raise ValueError("supported_k must be strictly increasing")
        if num_random < 0:
            raise ValueError("num_random must be non-negative")

        parsed_profiles = [
            parse_profile(profile, supported_k=self.supported_k, name="profiles")
            for profile in profiles
        ]
        if not parsed_profiles:
            raise ValueError("profiles must contain at least one profile")
        if len(set(parsed_profiles)) != len(parsed_profiles):
            raise ValueError("profiles contains duplicates")

        self.min_profile = parse_profile(
            min_profile, supported_k=self.supported_k, name="min_profile"
        )
        self.max_profile = parse_profile(
            max_profile, supported_k=self.supported_k, name="max_profile"
        )
        # Boundaries are mandatory sandwich samples even when a custom target
        # list omitted them.  Preserve the caller's order for reproducibility.
        ordered_profiles = list(parsed_profiles)
        for boundary in (self.max_profile, self.min_profile):
            if boundary not in ordered_profiles:
                ordered_profiles.append(boundary)

        self.profiles: tuple[Profile, ...] = tuple(ordered_profiles)
        self.intermediate_profiles: tuple[Profile, ...] = tuple(
            profile
            for profile in self.profiles
            if profile not in {self.min_profile, self.max_profile}
        )
        self.num_random = int(num_random)
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self._counts: Counter[Profile] = Counter({profile: 0 for profile in self.profiles})
        self._cycles = 0

    @property
    def counts(self) -> dict[Profile, int]:
        """Return a copy so callers cannot corrupt checkpointed coverage state."""

        return {profile: int(self._counts[profile]) for profile in self.profiles}

    @property
    def cycles(self) -> int:
        return self._cycles

    @property
    def total_profile_samples(self) -> int:
        return sum(self._counts.values())

    def _least_sampled_choice(self, candidates: Sequence[Profile]) -> Profile:
        minimum_count = min(self._counts[profile] for profile in candidates)
        tied = [profile for profile in candidates if self._counts[profile] == minimum_count]
        return self._rng.choice(tied)

    def sample_profiles(self, *, update_counts: bool = True) -> list[Profile]:
        """Return one sandwich cycle in max/min/under-sampled order."""

        selected: list[Profile] = []
        for boundary in (self.max_profile, self.min_profile):
            if boundary not in selected:
                selected.append(boundary)

        available = list(self.intermediate_profiles)
        random_count = min(self.num_random, len(available))
        for _ in range(random_count):
            profile = self._least_sampled_choice(available)
            selected.append(profile)
            available.remove(profile)

        if update_counts:
            self.record_profiles(selected)
            self._cycles += 1
        return selected

    # Convenient aliases for trainer code.
    sample = sample_profiles
    __call__ = sample_profiles

    def record_profiles(self, profiles: Iterable[Sequence[int]]) -> None:
        """Record externally selected profiles (e.g. distributed synchronization)."""

        known = set(self.profiles)
        for raw_profile in profiles:
            profile = parse_profile(
                raw_profile, supported_k=self.supported_k, name="recorded profile"
            )
            if profile not in known:
                raise ValueError(f"cannot record profile outside sampler universe: {profile}")
            self._counts[profile] += 1

    def state_dict(self) -> dict[str, object]:
        """Return a torch-save-friendly state including the exact RNG state."""

        return {
            "version": self.STATE_VERSION,
            "profiles": [list(profile) for profile in self.profiles],
            "supported_k": list(self.supported_k),
            "min_profile": list(self.min_profile),
            "max_profile": list(self.max_profile),
            "num_random": self.num_random,
            "seed": self.seed,
            "cycles": self._cycles,
            "counts": {
                _profile_key(profile): int(self._counts[profile])
                for profile in self.profiles
            },
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state_dict: Mapping[str, object], *, strict: bool = True) -> None:
        """Restore counts and tie-breaking state.

        Strict mode rejects profile-space or sandwich-setting drift, preventing a
        resumed run from silently using incompatible coverage metadata.
        """

        version = int(state_dict.get("version", -1))
        if version != self.STATE_VERSION:
            raise ValueError(
                f"unsupported ProfileSampler state version {version}; "
                f"expected {self.STATE_VERSION}"
            )

        saved_profiles = tuple(
            parse_profile(profile, supported_k=self.supported_k, name="saved profile")
            for profile in state_dict.get("profiles", [])
        )
        if strict and saved_profiles != self.profiles:
            raise ValueError("saved sampler profiles do not match current profiles")
        for field, current in (
            ("min_profile", self.min_profile),
            ("max_profile", self.max_profile),
        ):
            saved = parse_profile(
                state_dict.get(field, current),
                supported_k=self.supported_k,
                name=f"saved {field}",
            )
            if strict and saved != current:
                raise ValueError(f"saved {field} does not match current sampler")
        if strict and int(state_dict.get("num_random", self.num_random)) != self.num_random:
            raise ValueError("saved num_random does not match current sampler")

        raw_counts = state_dict.get("counts")
        if not isinstance(raw_counts, Mapping):
            raise ValueError("saved sampler counts must be a mapping")
        loaded_counts: Counter[Profile] = Counter()
        for profile in self.profiles:
            key = _profile_key(profile)
            value = int(raw_counts.get(key, 0))
            if value < 0:
                raise ValueError(f"negative sample count for {key}")
            loaded_counts[profile] = value
        if strict:
            unexpected = set(str(key) for key in raw_counts) - {
                _profile_key(profile) for profile in self.profiles
            }
            if unexpected:
                raise ValueError(f"unexpected saved profile counts: {sorted(unexpected)}")

        cycles = int(state_dict.get("cycles", 0))
        if cycles < 0:
            raise ValueError("saved sampler cycles cannot be negative")
        rng_state = state_dict.get("rng_state")
        if rng_state is None:
            raise ValueError("saved sampler state is missing rng_state")

        self._counts = loaded_counts
        self._cycles = cycles
        self._rng.setstate(rng_state)  # type: ignore[arg-type]

    def coverage_summary(self) -> dict[str, float | int]:
        counts = [self._counts[profile] for profile in self.profiles]
        return {
            "profiles": len(counts),
            "covered_profiles": sum(count > 0 for count in counts),
            "min_count": min(counts),
            "max_count": max(counts),
            "total_samples": sum(counts),
            "cycles": self._cycles,
        }


__all__ = ["Profile", "ProfileSampler"]
