"""
Tests for fault_classifier.py: R_theta(P) curve-shape fault diagnosis.

Covers all six fault modes + NOMINAL + INSUFFICIENT_DATA, curve statistics,
the snapshot time-axis regression (drift rates were computed on a collapsed
x-axis), report cadence, and session handling.
"""

import pytest

from theta.agent.fault_classifier import (
    FaultCurveClassifier,
    FaultCause,
    _CurveTracker,
    _median,
    _linslope,
    MIN_BUCKET_SAMPLES,
    MIN_SNAPSHOTS,
    SNAPSHOT_INTERVAL,
    REPORT_INTERVAL_S,
    FAULT_REPORT_INTERVAL,
)

T0 = 1_700_000_000.0  # unix-like epoch base


def fill_buckets(tracker, *, low_r=1.28, high_r=0.72, low_p=15.0, high_p=68.0,
                 n=MIN_BUCKET_SAMPLES, ts=T0, mem=50.0):
    """Fill both power tiers with steady samples (1 s apart, no snapshots)."""
    for i in range(n):
        tracker.ingest(ts + i, low_r, low_p, mem, None)
        tracker.ingest(ts + i, high_r, high_p, mem, None)


def feed_hourly(tracker, hours, *, low_fn, high_fn, ts0=T0, fan_fn=None):
    """
    Simulate `hours` hours: each hour feeds a full refresh of both tiers
    (so bucket medians track the hour's values) then crosses the snapshot
    interval. low_fn/high_fn map hour-index → R_theta for that tier.
    """
    for h in range(hours):
        base = ts0 + h * SNAPSHOT_INTERVAL
        for i in range(MIN_BUCKET_SAMPLES):
            fan = fan_fn(h) if fan_fn else None
            tracker.ingest(base + i, low_fn(h), 15.0, 50.0, fan)
            tracker.ingest(base + i, high_fn(h), 68.0, 50.0, fan)


class TestHelpers:
    def test_median_odd_even(self):
        assert _median([3.0, 1.0, 2.0]) == 2.0
        assert _median([4.0, 1.0, 2.0, 3.0]) == 2.5

    def test_linslope_known_line(self):
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [1.0, 3.0, 5.0, 7.0]
        assert _linslope(xs, ys) == pytest.approx(2.0)

    def test_linslope_underdetermined(self):
        assert _linslope([1.0], [5.0]) == 0.0
        assert _linslope([2.0, 2.0], [1.0, 9.0]) == 0.0  # zero x-variance


class TestInsufficientData:
    def test_empty_tracker_insufficient(self):
        t = _CurveTracker()
        d = t.diagnose(0, T0)
        assert d.cause == FaultCause.INSUFFICIENT_DATA
        assert d.intercept is None and d.gap is None

    def test_one_tier_only_insufficient(self):
        t = _CurveTracker()
        for i in range(100):
            t.ingest(T0 + i, 0.72, 68.0, 50.0, None)  # high tier only
        assert t.diagnose(0, T0 + 200).cause == FaultCause.INSUFFICIENT_DATA

    def test_classifier_emits_insufficient_exactly_once(self):
        clf = FaultCurveClassifier()
        d1 = clf.update(0, T0, 0.72, 68.0, 50.0)
        assert d1 is not None and d1.cause == FaultCause.INSUFFICIENT_DATA
        d2 = clf.update(0, T0 + 1, 0.72, 68.0, 50.0)
        assert d2 is None  # not re-emitted


class TestCurveStats:
    def test_nominal_with_healthy_tiers(self):
        t = _CurveTracker()
        fill_buckets(t)
        d = t.diagnose(0, T0 + 100)
        assert d.cause == FaultCause.NOMINAL
        # Stage 1 T4 anchor points: intercept ≈ idle R_theta, gap = low − high
        assert d.intercept == pytest.approx(1.28, abs=0.01)
        assert d.gap == pytest.approx(0.56, abs=0.01)

    def test_curve_slope_negative(self):
        """R_theta falls with power on healthy hardware → negative physical slope."""
        t = _CurveTracker()
        fill_buckets(t)
        d = t.diagnose(0, T0 + 100)
        assert d.curve_slope is not None
        assert d.curve_slope == pytest.approx((0.72 - 1.28) / (68.0 - 15.0), abs=1e-4)


class TestSnapshotTimeAxis:
    def test_trend_rate_matches_known_drift(self):
        """
        Regression: snapshot x-coordinates were anchored to the FIRST
        snapshot's x (`snapshots[0][0] + interval`), collapsing snapshots
        2..N onto one point and inflating the fitted drift rate ~13×.
        A known parallel drift of 0.024 C/W per day must fit back out
        as ≈0.024, not ≈0.3.
        """
        t = _CurveTracker()
        rate_per_day = 0.024
        per_hour = rate_per_day / 24.0
        feed_hourly(
            t, MIN_SNAPSHOTS + 4,
            low_fn=lambda h: 1.28 + per_hour * h,
            high_fn=lambda h: 0.72 + per_hour * h,  # parallel — gap constant
        )
        drift_rate, gap_trend = t._trend_rates()
        assert drift_rate is not None
        assert drift_rate == pytest.approx(rate_per_day, rel=0.3)
        assert abs(gap_trend) < 0.005  # parallel shift → gap flat


class TestFaultModes:
    def test_dust_accumulation(self):
        """Slow parallel intercept drift with stable gap → dust."""
        t = _CurveTracker()
        per_hour = 0.005 / 24.0  # 0.005 C/W per day, 5× the dust threshold
        feed_hourly(
            t, MIN_SNAPSHOTS + 4,
            low_fn=lambda h: 1.28 + per_hour * h,
            high_fn=lambda h: 0.72 + per_hour * h,
        )
        d = t.diagnose(0, T0 + (MIN_SNAPSHOTS + 4) * SNAPSHOT_INTERVAL)
        assert d.cause == FaultCause.DUST_ACCUMULATION
        assert d.confidence > 0.5
        assert "Clean heatsink" in d.remediation

    def test_tim_degradation(self):
        """Gap narrowing (high-P R_theta rising toward low-P), no fan signal → TIM."""
        t = _CurveTracker()
        narrow_per_hour = 0.06 / 24.0  # gap shrinks 0.06 C/W per day
        feed_hourly(
            t, MIN_SNAPSHOTS + 4,
            low_fn=lambda h: 1.28,
            high_fn=lambda h: 0.72 + narrow_per_hour * h,
        )
        d = t.diagnose(0, T0 + (MIN_SNAPSHOTS + 4) * SNAPSHOT_INTERVAL)
        assert d.cause == FaultCause.TIM_DEGRADATION
        assert "repaste" in d.remediation

    def test_fan_bearing_wear(self):
        """Gap narrowing PLUS declining fan RPM → fan bearing, not TIM."""
        t = _CurveTracker()
        narrow_per_hour = 0.06 / 24.0
        feed_hourly(
            t, MIN_SNAPSHOTS + 4,
            low_fn=lambda h: 1.28,
            high_fn=lambda h: 0.72 + narrow_per_hour * h,
        )
        # Fill the entire fan buffer with a steep decline: 90% → 20% in 120 s
        end = T0 + (MIN_SNAPSHOTS + 4) * SNAPSHOT_INTERVAL
        for i in range(120):
            t.ingest(end + i, 0.0, 1.0, 0.0, 90.0 - (70.0 / 120.0) * i)
        d = t.diagnose(0, end + 200)
        assert d.cause == FaultCause.FAN_BEARING_WEAR
        assert "fan" in d.remediation.lower()

    def test_airflow_blockage(self):
        """Intra-session intercept step (gap trend quiet) → blockage."""
        t = _CurveTracker()
        fill_buckets(t, low_r=1.28)
        t._sess_start_rtheta = 1.18  # session started 0.10 lower
        t._sess_warmup_done = True
        d = t.diagnose(0, T0 + 100)
        assert d.cause == FaultCause.AIRFLOW_BLOCKAGE
        assert d.evidence["intra_delta"] == pytest.approx(0.10, abs=0.01)

    def test_mounting_event(self):
        """Inter-session step > 0.08 C/W → mounting event (highest priority)."""
        t = _CurveTracker()
        fill_buckets(t)
        t._prev_sess_rtheta = 1.16
        t._sess_start_rtheta = 1.28  # +0.12 across a restart
        d = t.diagnose(0, T0 + 100)
        assert d.cause == FaultCause.MOUNTING_EVENT
        assert d.session_delta == pytest.approx(0.12, abs=0.01)
        assert d.confidence == pytest.approx(0.75, abs=0.01)

    def test_hbm_thermal(self):
        """R_theta elevated only under high memory utilization → HBM."""
        t = _CurveTracker()
        # Low tier baseline
        for i in range(MIN_BUCKET_SAMPLES):
            t.ingest(T0 + i, 1.28, 15.0, 50.0, None)
        # High tier: alternating memory pressure, +0.08 lift when mem-hot
        for i in range(40):
            t.ingest(T0 + 100 + i, 0.80, 68.0, 85.0, None)  # hot HBM
            t.ingest(T0 + 100 + i, 0.72, 68.0, 20.0, None)  # cool HBM
        d = t.diagnose(0, T0 + 200)
        assert d.cause == FaultCause.HBM_THERMAL
        assert d.evidence["hbm_lift"] == pytest.approx(0.08, abs=0.01)


class TestReportCadence:
    def _ready_clf(self):
        """Classifier with GPU 0 buckets pre-filled (healthy)."""
        clf = FaultCurveClassifier()
        t = clf._tracker(0)
        fill_buckets(t)
        t._emitted_insufficient = True  # past the first-emit
        return clf

    def test_nominal_respects_report_interval(self):
        clf = self._ready_clf()
        ts = T0 + 1000
        clf._tracker(0)._last_diag_ts = ts
        # Within the 10-min nominal interval → suppressed
        assert clf.update(0, ts + REPORT_INTERVAL_S - 5, 0.72, 68.0, 50.0) is None
        # Past it → emits
        d = clf.update(0, ts + REPORT_INTERVAL_S + 5, 0.72, 68.0, 50.0)
        assert d is not None and d.cause == FaultCause.NOMINAL

    def test_fault_reports_faster_than_nominal(self):
        clf = self._ready_clf()
        t = clf._tracker(0)
        t._prev_sess_rtheta = 1.16
        t._sess_start_rtheta = 1.28  # mounting fault active
        ts = T0 + 1000
        t._last_diag_ts = ts
        # Faults use the 60 s interval, not the 600 s one
        d = clf.update(0, ts + FAULT_REPORT_INTERVAL + 5, 0.72, 68.0, 50.0)
        assert d is not None and d.cause == FaultCause.MOUNTING_EVENT

    def test_get_current_unknown_gpu(self):
        assert FaultCurveClassifier().get_current(99) is None

    def test_get_current_does_not_advance_timer(self):
        clf = self._ready_clf()
        before = clf._tracker(0)._last_diag_ts
        assert clf.get_current(0) is not None
        assert clf._tracker(0)._last_diag_ts == before


class TestSessions:
    def test_new_session_carries_bookmark(self):
        t = _CurveTracker()
        t._sess_start_rtheta = 1.20
        t._sess_warmup_done = True
        t.new_session()
        assert t._prev_sess_rtheta == 1.20
        assert t._sess_start_rtheta is None
        assert not t._sess_warmup_done

    def test_notify_new_session_via_classifier(self):
        clf = FaultCurveClassifier()
        clf.notify_new_session(0)  # must not raise on unseen GPU
        assert 0 in clf._trackers
