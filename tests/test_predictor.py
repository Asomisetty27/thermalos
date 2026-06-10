"""
Tests for predictor.py: degradation risk scoring.

Pins the Phase-1 rule-based scorer: activation threshold, per-signal
contributions (ECC dbit/sbit, R_theta slope, drift sigma, clock efficiency),
the diminishing-returns score combination, and maybe_alert gating/cooldown.
"""

import pytest

from theta.agent.metrics import GPUState
from theta.agent.detector import DriftResult
from theta.agent.predictor import (
    FailurePredictor,
    MIN_WINDOWS,
    ALERT_THRESHOLD,
    ALERT_COOLDOWN_S,
    W_ECC_DBIT,
    W_RTHETA_SLOPE,
)


def _drift(sigma=0.0):
    return DriftResult(
        gpu_index=0, timestamp=0.0, rtheta=0.72,
        baseline_mean=0.72, baseline_std=0.01, sigma_score=sigma,
        is_drifting=False, is_critical=False, confidence=0.0,
    )


def feed(pred, n, *, t0=0.0, dt=15.0, rtheta=0.72, rtheta_slope=0.0,
         sigma=0.0, sigma_slope=0.0, sbit=0, sbit_per_window=0,
         dbit_at=None, clock_eff=0.95, clock_slope=0.0, gpu=0):
    """Feed n stable windows with optional linear trends."""
    for i in range(n):
        ts = t0 + i * dt
        pred.update(
            gpu_index=gpu,
            ts=ts,
            rtheta=rtheta + rtheta_slope * (i * dt),
            drift=_drift(sigma + sigma_slope * (i * dt)),
            ecc_sbit=sbit + sbit_per_window * i,
            ecc_dbit=1 if (dbit_at is not None and i >= dbit_at) else 0,
            clock_eff=clock_eff + clock_slope * (i * dt),
            gpu_name="T4",
        )
    return t0 + (n - 1) * dt


class TestActivation:
    def test_no_data_returns_none(self):
        assert FailurePredictor().score(0) is None

    def test_below_min_windows_returns_none(self):
        p = FailurePredictor()
        feed(p, MIN_WINDOWS - 1)
        assert p.score(0) is None

    def test_at_min_windows_activates(self):
        p = FailurePredictor()
        feed(p, MIN_WINDOWS)
        assert p.score(0) is not None

    def test_get_score_zero_when_inactive(self):
        assert FailurePredictor().get_score(0) == 0.0


class TestHealthyBaseline:
    def test_healthy_records_score_zero(self):
        p = FailurePredictor()
        feed(p, 30)
        risk = p.score(0)
        assert risk.score == 0.0
        assert not risk.alert_worthy
        assert "no degradation signals" in risk.explanation

    def test_flat_rtheta_no_contribution(self):
        """Flat R_theta (healthy under load) must not contribute risk."""
        p = FailurePredictor()
        feed(p, 30, rtheta=0.72, rtheta_slope=0.0)
        assert p.score(0).score == 0.0


class TestSignals:
    def test_dbit_drives_near_certain_score(self):
        """Any uncorrectable ECC in the last 5 windows ≈ certain failure."""
        p = FailurePredictor()
        feed(p, 30, dbit_at=27)  # dbit appears in the final windows
        risk = p.score(0)
        assert risk.score >= 0.99
        assert risk.alert_worthy
        assert "uncorrectable" in risk.explanation

    def test_old_dbit_outside_window_ignored(self):
        """dbit that stopped appearing >5 windows ago no longer dominates."""
        p = FailurePredictor()
        # dbit only in early windows; ecc_dbit returns to 0 after window 10
        for i in range(30):
            p.update(
                gpu_index=0, ts=i * 15.0, rtheta=0.72, drift=_drift(),
                ecc_sbit=0, ecc_dbit=1 if i < 10 else 0,
                clock_eff=0.95,
            )
        risk = p.score(0)
        assert risk.score < W_ECC_DBIT  # not the near-certain dbit path

    def test_rising_rtheta_contributes(self):
        p = FailurePredictor()
        # +0.0005 C/W per second — half the "very concerning" normalizer
        feed(p, 30, rtheta_slope=0.0005)
        risk = p.score(0)
        assert risk.score > 0.0
        assert "R_θ rising" in risk.explanation
        # R_theta alone, norm=0.5 → 0.5 * 0.5 = 0.25; below alert threshold
        assert risk.score == pytest.approx(W_RTHETA_SLOPE * 0.5, abs=0.05)
        assert not risk.alert_worthy

    def test_sbit_cumulative_count_capped_not_summed(self):
        """
        Regression for the unit-mixing fix: a large but STATIC cumulative
        sbit count must score via count_norm (capped at 1.0), not blow up
        by summing count into an hourly rate.
        """
        p = FailurePredictor()
        feed(p, 30, sbit=150, sbit_per_window=0)  # 150 historical, zero new
        risk = p.score(0)
        # contribution = W_ECC_SBIT_RATE * max(rate_norm≈0, count_norm=1.0) = 0.6
        assert 0.55 <= risk.score <= 0.65
        assert "sbit" in risk.explanation

    def test_sbit_rising_rate_contributes(self):
        p = FailurePredictor()
        feed(p, 30, sbit=0, sbit_per_window=1)  # 1 new error per 15s window
        risk = p.score(0)
        assert risk.score > 0.0
        assert "sbit" in risk.explanation

    def test_declining_clock_efficiency_contributes(self):
        p = FailurePredictor()
        feed(p, 30, clock_slope=-0.0005)
        risk = p.score(0)
        assert risk.score > 0.0
        assert "clock efficiency declining" in risk.explanation

    def test_rising_sigma_contributes(self):
        p = FailurePredictor()
        feed(p, 30, sigma=1.2, sigma_slope=0.003)
        risk = p.score(0)
        assert risk.score > 0.0
        assert "drift sigma" in risk.explanation


class TestScoreCombination:
    def test_strongest_signal_sets_floor(self):
        """Combined score is never below the strongest single signal."""
        p_single = FailurePredictor()
        feed(p_single, 30, rtheta_slope=0.001)  # saturated R_theta signal
        single = p_single.score(0).score

        p_multi = FailurePredictor()
        feed(p_multi, 30, rtheta_slope=0.001, clock_slope=-0.001)
        multi = p_multi.score(0).score

        assert multi >= single

    def test_correlated_signals_do_not_compound_like_noisy_or(self):
        """
        Diminishing-returns combination: w1 + w2·(1−w1)·0.5 — strictly less
        than the old noisy-OR 1−(1−w1)(1−w2) for any two positive weights.
        """
        p = FailurePredictor()
        feed(p, 30, rtheta_slope=0.001, clock_slope=-0.001)
        score = p.score(0).score
        # saturated signals: w = [0.5, 0.4] → 0.5 + 0.4*0.5*0.5 = 0.60
        assert score == pytest.approx(0.60, abs=0.03)
        # noisy-OR would have been 1 − 0.5·0.6 = 0.70 — confirm we're below it
        assert score < 0.70

    def test_score_capped_at_one(self):
        p = FailurePredictor()
        feed(p, 30, dbit_at=25, sbit=200, sbit_per_window=3,
             rtheta_slope=0.002, sigma=3.0, sigma_slope=0.01, clock_slope=-0.002)
        assert p.score(0).score <= 1.0


class TestMaybeAlert:
    def test_no_alert_below_threshold(self):
        p = FailurePredictor()
        last_ts = feed(p, 30, rtheta_slope=0.0005)  # ~0.25 score
        assert p.score(0).score < ALERT_THRESHOLD
        assert p.maybe_alert(0, last_ts, GPUState.UNDER_LOAD) is None

    def test_alert_fires_above_threshold(self):
        p = FailurePredictor()
        # t0 past the initial cooldown horizon (last_alert defaults to 0.0)
        last_ts = feed(p, 30, dbit_at=27, t0=10_000.0)
        alert = p.maybe_alert(0, last_ts, GPUState.UNDER_LOAD)
        assert alert is not None
        assert alert.context["predictive"] is True
        assert alert.context["degradation_risk"] >= ALERT_THRESHOLD
        # score ≥ 0.90 → tightest horizon
        assert alert.context["horizon"] == "~1hr"

    def test_alert_suppressed_in_critical_state(self):
        """Don't pile predictive alerts onto a GPU already in incident states."""
        p = FailurePredictor()
        last_ts = feed(p, 30, dbit_at=27, t0=10_000.0)
        assert p.maybe_alert(0, last_ts, GPUState.CRITICAL) is None
        assert p.maybe_alert(0, last_ts, GPUState.ZOMBIE_RECOVERY) is None

    def test_alert_cooldown(self):
        p = FailurePredictor()
        last_ts = feed(p, 30, dbit_at=27, t0=10_000.0)
        first = p.maybe_alert(0, last_ts, GPUState.UNDER_LOAD)
        assert first is not None
        # Within cooldown → suppressed
        assert p.maybe_alert(0, last_ts + 60, GPUState.UNDER_LOAD) is None
        # After cooldown → fires again
        again = p.maybe_alert(0, last_ts + ALERT_COOLDOWN_S + 1, GPUState.UNDER_LOAD)
        assert again is not None

    def test_per_gpu_isolation(self):
        p = FailurePredictor()
        feed(p, 30, dbit_at=27, gpu=0)
        feed(p, 30, gpu=1)  # healthy
        assert p.score(0).alert_worthy
        assert not p.score(1).alert_worthy
