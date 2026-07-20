import copy

import pytest

from training.profile_sampler import ProfileSampler


def _small_profile_space():
    return [(k0, k1) for k0 in (2, 4, 8) for k1 in (2, 4, 8)]


def test_sandwich_profiles_are_atomic_and_boundaries_are_always_present():
    sampler = ProfileSampler(
        _small_profile_space(),
        supported_k=(2, 4, 8),
        min_profile=(2, 2),
        max_profile=(8, 8),
        num_random=2,
        seed=7,
    )

    for _ in range(10):
        sampled = sampler.sample_profiles()
        assert sampled[:2] == [(8, 8), (2, 2)]
        assert len(sampled) == 4
        assert len(set(sampled)) == len(sampled)
        assert all(isinstance(profile, tuple) and len(profile) == 2 for profile in sampled)


def test_least_sampled_policy_covers_every_intermediate_before_repeating():
    profiles = _small_profile_space()
    sampler = ProfileSampler(
        profiles,
        supported_k=(2, 4, 8),
        min_profile=(2, 2),
        max_profile=(8, 8),
        num_random=1,
        seed=11,
    )

    intermediate_count = len(profiles) - 2
    selected_intermediates = []
    for _ in range(intermediate_count):
        selected_intermediates.append(sampler.sample_profiles()[2])

    assert len(set(selected_intermediates)) == intermediate_count
    counts = sampler.counts
    assert all(counts[profile] == 1 for profile in profiles if profile not in {(2, 2), (8, 8)})
    assert counts[(2, 2)] == intermediate_count
    assert counts[(8, 8)] == intermediate_count


def test_state_dict_restores_counts_cycles_and_next_random_tie_break():
    kwargs = dict(
        profiles=_small_profile_space(),
        supported_k=(2, 4, 8),
        min_profile=(2, 2),
        max_profile=(8, 8),
        num_random=2,
        seed=123,
    )
    original = ProfileSampler(**kwargs)
    for _ in range(5):
        original.sample_profiles()
    state = copy.deepcopy(original.state_dict())

    restored = ProfileSampler(**kwargs)
    restored.load_state_dict(state)

    assert restored.counts == original.counts
    assert restored.cycles == original.cycles
    assert restored.sample_profiles() == original.sample_profiles()
    assert restored.counts == original.counts


def test_custom_profile_list_gets_mandatory_boundaries_without_duplicates():
    sampler = ProfileSampler(
        [(4, 4)],
        supported_k=(2, 4, 8),
        min_profile=(2, 2),
        max_profile=(8, 8),
        num_random=5,
    )
    assert sampler.profiles == ((4, 4), (8, 8), (2, 2))
    assert sampler.sample_profiles() == [(8, 8), (2, 2), (4, 4)]


def test_strict_restore_rejects_profile_space_drift():
    source = ProfileSampler(
        _small_profile_space(),
        supported_k=(2, 4, 8),
        min_profile=(2, 2),
        max_profile=(8, 8),
    )
    target = ProfileSampler(
        [(2, 2), (4, 4), (8, 8)],
        supported_k=(2, 4, 8),
        min_profile=(2, 2),
        max_profile=(8, 8),
    )
    with pytest.raises(ValueError, match="profiles do not match"):
        target.load_state_dict(source.state_dict())

