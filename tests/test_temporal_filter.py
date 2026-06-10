"""
Tests for temporal_filter.py: Bayesian state smoothing.

Characterization tests pinning the forward-filter behavior: cold start,
single-glitch rejection (the module's reason for existing), sustained-change
flips, posterior validity, per-GPU isolation, and the non-finite-confidence
guard.
"""

import math

import pytest

from theta.agent.metrics import GPUState
from theta.agent.temporal_filter import (
    TemporalStateFilter,
    _observation_likelihood,
    _STATES,
)


class TestObservationLikelihood:
    def test_high_confidence_concentrates_on_observed(self):
        lik = _observation_likelihood(GPUState.UNDER_LOAD, 0.95)
        idx = _STATES.index(GPUState.UNDER_LOAD)
        assert lik[idx] == pytest.approx(0.95)
        # Remaining mass spread uniformly over the other 4 states
        others = [l for i, l in enumerate(lik) if i != idx]
        assert all(o == pytest.approx(0.05 / 4) for o in others)

    def test_confidence_clamped_to_floor(self):
        """conf below 0.5 is clamped up — a sub-coin-flip observation is noise."""
        lik = _observation_likelihood(GPUState.CLEAN_IDLE, 0.10)
        idx = _STATES.index(GPUState.CLEAN_IDLE)
        assert lik[idx] == pytest.approx(0.50)

    def test_confidence_clamped_to_ceiling(self):
        """conf above 0.99 is clamped down — never fully trust one window."""
        lik = _observation_likelihood(GPUState.CLEAN_IDLE, 1.0)
        idx = _STATES.index(GPUState.CLEAN_IDLE)
        assert lik[idx] == pytest.approx(0.99)

    def test_nonfinite_confidence_guarded(self):
        """NaN/Inf confidence falls back to 0.5 instead of poisoning the posterior."""
        for bad in (float('nan'), float('inf'), float('-inf')):
            lik = _observation_likelihood(GPUState.UNDER_LOAD, bad)
            assert all(math.isfinite(l) for l in lik)
            assert sum(lik) == pytest.approx(1.0)

    def test_unmodeled_state_maps_to_unknown(self):
        """DRIFTING/CRITICAL aren't filter states — they observe as UNKNOWN."""
        lik = _observation_likelihood(GPUState.DRIFTING, 0.9)
        idx = _STATES.index(GPUState.UNKNOWN)
        assert lik[idx] == pytest.approx(0.9)


class TestTemporalStateFilter:
    def test_cold_start_follows_observation(self):
        """First high-confidence observation dominates the uniform prior."""
        f = TemporalStateFilter()
        r = f.observe(0, GPUState.UNDER_LOAD, 0.95)
        assert r.state == GPUState.UNDER_LOAD
        assert r.n_observations == 1
        assert r.raw_state == GPUState.UNDER_LOAD
        assert r.raw_confidence == 0.95

    def test_posterior_is_valid_distribution(self):
        f = TemporalStateFilter()
        r = f.observe(0, GPUState.CLEAN_IDLE, 0.9)
        total = sum(r.posterior.values())
        assert total == pytest.approx(1.0)
        assert all(0.0 <= p <= 1.0 for p in r.posterior.values())

    def test_single_glitch_rejected(self):
        """
        The module's core promise: after a sustained UNDER_LOAD run, ONE
        contradicting observation must not flip the smoothed state.
        """
        f = TemporalStateFilter()
        for _ in range(30):
            r = f.observe(0, GPUState.UNDER_LOAD, 0.95)
        assert r.state == GPUState.UNDER_LOAD

        glitch = f.observe(0, GPUState.ZOMBIE_RECOVERY, 0.90)
        assert glitch.state == GPUState.UNDER_LOAD  # smoothed holds
        assert glitch.raw_state == GPUState.ZOMBIE_RECOVERY  # raw preserved
        # but the posterior should register doubt
        assert glitch.confidence < r.confidence

    def test_sustained_change_flips_state(self):
        """Repeated contradicting evidence must flip the state within a few ticks."""
        f = TemporalStateFilter()
        for _ in range(30):
            f.observe(0, GPUState.UNDER_LOAD, 0.95)

        flipped_at = None
        for i in range(10):
            r = f.observe(0, GPUState.ZOMBIE_RECOVERY, 0.90)
            if r.state == GPUState.ZOMBIE_RECOVERY:
                flipped_at = i + 1
                break
        assert flipped_at is not None, "filter never accepted sustained evidence"
        # Smoothing should cost a couple ticks, but not be sluggish
        assert 2 <= flipped_at <= 6

    def test_confidence_grows_with_consistent_evidence(self):
        f = TemporalStateFilter()
        first = f.observe(0, GPUState.UNDER_LOAD, 0.9)
        for _ in range(20):
            last = f.observe(0, GPUState.UNDER_LOAD, 0.9)
        assert last.confidence > first.confidence

    def test_per_gpu_isolation(self):
        f = TemporalStateFilter()
        for _ in range(10):
            f.observe(0, GPUState.UNDER_LOAD, 0.95)
            f.observe(1, GPUState.CLEAN_IDLE, 0.95)
        assert f.current(0).state == GPUState.UNDER_LOAD
        assert f.current(1).state == GPUState.CLEAN_IDLE

    def test_reset_clears_history(self):
        f = TemporalStateFilter()
        for _ in range(10):
            f.observe(0, GPUState.UNDER_LOAD, 0.95)
        f.reset(0)
        assert f.current(0) is None
        r = f.observe(0, GPUState.CLEAN_IDLE, 0.9)
        assert r.n_observations == 1
        assert r.state == GPUState.CLEAN_IDLE

    def test_current_none_before_observations(self):
        f = TemporalStateFilter()
        assert f.current(7) is None

    def test_current_matches_last_observe(self):
        f = TemporalStateFilter()
        r = f.observe(0, GPUState.UNDER_LOAD, 0.9)
        c = f.current(0)
        assert c.state == r.state
        assert c.confidence == pytest.approx(r.confidence)
        assert c.n_observations == 1

    def test_states_under_consideration_empty_cold(self):
        f = TemporalStateFilter()
        assert f.states_under_consideration(0) == []

    def test_states_under_consideration_sorted_and_filtered(self):
        f = TemporalStateFilter()
        for _ in range(5):
            f.observe(0, GPUState.UNDER_LOAD, 0.85)
        pairs = f.states_under_consideration(0, min_prob=0.01)
        # Sorted descending by probability
        probs = [p for _, p in pairs]
        assert probs == sorted(probs, reverse=True)
        # All above the floor
        assert all(p >= 0.01 for p in probs)
        # Argmax leads
        assert pairs[0][0] == GPUState.UNDER_LOAD

    def test_states_under_consideration_high_floor(self):
        """A dominant posterior should leave a single candidate above 0.5."""
        f = TemporalStateFilter()
        for _ in range(30):
            f.observe(0, GPUState.UNDER_LOAD, 0.95)
        pairs = f.states_under_consideration(0, min_prob=0.5)
        assert len(pairs) == 1
        assert pairs[0][0] == GPUState.UNDER_LOAD

    def test_nonfinite_confidence_does_not_poison_filter(self):
        """A NaN confidence tick must not corrupt subsequent posteriors."""
        f = TemporalStateFilter()
        for _ in range(5):
            f.observe(0, GPUState.UNDER_LOAD, 0.95)
        r = f.observe(0, GPUState.UNDER_LOAD, float('nan'))
        assert math.isfinite(r.confidence)
        assert sum(r.posterior.values()) == pytest.approx(1.0)
        # Filter keeps working afterwards
        r2 = f.observe(0, GPUState.UNDER_LOAD, 0.95)
        assert r2.state == GPUState.UNDER_LOAD
        assert math.isfinite(r2.confidence)

    def test_drifting_observation_does_not_crash(self):
        """Daemon may pass DRIFTING/CRITICAL — filter treats them as UNKNOWN evidence."""
        f = TemporalStateFilter()
        r = f.observe(0, GPUState.DRIFTING, 0.9)
        assert r.state in _STATES
        assert sum(r.posterior.values()) == pytest.approx(1.0)
