"""
Unit tests for the Theta agent core pipeline.
Tests run without a GPU (demo mode).
"""

import math
import time
import pytest

from theta.agent.metrics import (
    compute_rtheta, enrich, RawSample, GPUState, CLASS_INDEX_TO_STATE,
    MIN_POWER_W
)
from theta.agent.baseline import BaselineManager
from theta.agent.window   import SteadyStateWindow
from theta.agent.classifier import StateClassifier, _rule_classify
from theta.agent.detector import DriftDetector
from theta.agent.state    import GPUStateMachine
from theta.agent.metrics  import ClassifiedSample, enrich


# ── compute_rtheta ────────────────────────────────────────────────────────────

def test_rtheta_under_load():
    r, valid = compute_rtheta(temp_junction=70.0, t_ref=25.0, power_w=68.0)
    assert valid
    assert abs(r - 45.0 / 68.0) < 1e-6

def test_rtheta_clean_idle():
    r, valid = compute_rtheta(temp_junction=42.0, t_ref=25.0, power_w=11.4)
    assert valid
    assert r == pytest.approx(17.0 / 11.4, rel=1e-4)

def test_rtheta_low_power_invalid():
    r, valid = compute_rtheta(temp_junction=42.0, t_ref=25.0, power_w=2.0)
    assert not valid
    assert r is None

def test_rtheta_low_delta_invalid():
    r, valid = compute_rtheta(temp_junction=25.2, t_ref=25.0, power_w=15.0)
    assert not valid

def test_rtheta_zombie_range():
    # zombie: ~30W, T_j elevated, yields R_theta ~1.5 C/W
    r, valid = compute_rtheta(temp_junction=67.0, t_ref=25.0, power_w=31.2)
    assert valid
    assert 1.2 < r < 1.8


# ── baseline ──────────────────────────────────────────────────────────────────

def test_baseline_default_before_lock(tmp_path):
    bm = BaselineManager(_file=tmp_path / "b.json")
    assert bm.get_t_ref(0) == 25.0
    assert not bm.has_baseline(0)

def test_baseline_manual_set(tmp_path):
    bm = BaselineManager(_file=tmp_path / "b.json")
    bm.set_manual(0, 41.2)
    assert bm.get_t_ref(0) == 41.2
    assert bm.has_baseline(0)

def test_baseline_locks_on_stable_idle(tmp_path):
    bm = BaselineManager(window_sec=5.0, _file=tmp_path / "b.json")
    ts = time.time()
    for i in range(6):
        bm.update(gpu_index=0, temp=41.0 + i * 0.01, util=0.0, pstate=8, ts=ts + i)
    assert bm.has_baseline(0)
    b = bm.get_baseline(0)
    assert abs(b.t_ref - 41.0) < 0.5

def test_baseline_does_not_lock_under_load(tmp_path):
    bm = BaselineManager(window_sec=5.0, _file=tmp_path / "b.json")
    ts = time.time()
    for i in range(10):
        bm.update(gpu_index=0, temp=70.0, util=97.0, pstate=0, ts=ts + i)
    assert not bm.has_baseline(0)


# ── steady-state window ────────────────────────────────────────────────────────

def test_window_stable():
    win = SteadyStateWindow(window_sec=5.0, sigma_threshold=0.05, min_samples=3)
    ts  = time.time()
    for i in range(6):
        r = win.update(0, ts + i, rtheta=0.72 + i * 0.001, power=68.0, util=97.0, pstate=0)
    assert r.is_stable

def test_window_noisy_not_stable():
    win = SteadyStateWindow(window_sec=10.0, sigma_threshold=0.05, min_samples=3)
    ts  = time.time()
    import random
    random.seed(42)
    for i in range(12):
        r = win.update(0, ts + i, rtheta=0.72 + random.uniform(-0.5, 0.5),
                       power=68.0, util=97.0, pstate=0)
    assert not r.is_stable

def test_window_coverage():
    win = SteadyStateWindow(window_sec=15.0)
    ts  = time.time()
    r   = win.update(0, ts, 0.72, 68.0, 97.0, 0)
    assert win.coverage(0, ts) < 1.0


# ── classifier ────────────────────────────────────────────────────────────────

def test_rule_classify_under_load():
    state, conf = _rule_classify(rtheta=0.72, power=68.0, pstate=0)
    assert state == GPUState.UNDER_LOAD
    assert conf > 0.9

def test_rule_classify_zombie():
    state, conf = _rule_classify(rtheta=1.54, power=31.2, pstate=0)
    assert state == GPUState.ZOMBIE_RECOVERY
    assert conf == 1.0

def test_rule_classify_child_exit():
    state, conf = _rule_classify(rtheta=2.10, power=12.0, pstate=8)
    assert state == GPUState.CHILD_EXIT_RECOVERY
    assert conf > 0.9

def test_rule_classify_clean_idle():
    state, conf = _rule_classify(rtheta=1.28, power=9.5, pstate=8)
    assert state == GPUState.CLEAN_IDLE

def test_sklearn_classifier_loads():
    clf = StateClassifier(prefer_interpretable=True)
    assert clf.mode in ("ensemble", "dt", "nb", "rules")

def test_sklearn_classifier_under_load():
    clf = StateClassifier()
    from theta.agent.window import WindowResult
    w = WindowResult(
        gpu_index=0, timestamp=time.time(),
        rtheta_mean=0.72, rtheta_std=0.01,
        n_samples=15, is_stable=True,
        last_power=68.0, last_util=97.0, last_pstate=0
    )
    state, conf = clf.classify(w)
    assert state == GPUState.UNDER_LOAD
    assert conf > 0.9

def test_sklearn_classifier_zombie():
    clf = StateClassifier()
    from theta.agent.window import WindowResult
    w = WindowResult(
        gpu_index=0, timestamp=time.time(),
        rtheta_mean=1.54, rtheta_std=0.01,
        n_samples=15, is_stable=True,
        last_power=31.2, last_util=0.0, last_pstate=0
    )
    state, conf = clf.classify(w)
    assert state == GPUState.ZOMBIE_RECOVERY
    assert conf > 0.9


# ── drift detector ────────────────────────────────────────────────────────────

def _feed_baseline(det: DriftDetector, gpu: int, n: int = 30, rtheta: float = 0.72):
    ts = time.time()
    for i in range(n):
        det.update(gpu, ts + i, rtheta + i * 0.001, GPUState.UNDER_LOAD)

def test_drift_no_baseline():
    det = DriftDetector()
    r = det.update(0, time.time(), 0.72, GPUState.UNDER_LOAD)
    assert not r.is_drifting
    assert r.baseline_mean is None

def test_drift_no_alert_when_healthy():
    det = DriftDetector()
    _feed_baseline(det, 0)
    r = det.update(0, time.time() + 31, 0.75, GPUState.UNDER_LOAD)
    assert not r.is_drifting

def test_drift_detects_anomaly():
    det = DriftDetector(k_warn=2.0, sustained=3)
    _feed_baseline(det, 0, n=30, rtheta=0.72)
    ts = time.time() + 31
    for i in range(4):
        r = det.update(0, ts + i, 1.85, GPUState.CHILD_EXIT_RECOVERY)
    assert r.is_drifting

def test_drift_critical():
    det = DriftDetector(k_warn=2.0, k_critical=3.5, sustained=3)
    _feed_baseline(det, 0, n=30, rtheta=0.72)
    ts = time.time() + 31
    for i in range(4):
        r = det.update(0, ts + i, 2.5, GPUState.CHILD_EXIT_RECOVERY)
    assert r.is_critical


# ── state machine ─────────────────────────────────────────────────────────────

def _make_classified(gpu, state, rtheta=0.72, conf=0.99):
    raw = RawSample(
        gpu_index=gpu, timestamp=time.time(),
        temp_junction=70.0, power_w=68.0, util_pct=97.0,
        mem_util_pct=60.0, perf_state=0,
        clock_sm_mhz=1600, clock_mem_mhz=8000,
    )
    from theta.agent.metrics import EnrichedSample
    enriched = EnrichedSample(raw=raw, t_ref=25.0, rtheta=rtheta, rtheta_valid=True)
    return ClassifiedSample(enriched=enriched, state=state, confidence=conf, rtheta_mean=rtheta)

def _make_drift(gpu, drifting=False, critical=False, sigma=None, baseline=None):
    from theta.agent.detector import DriftResult
    return DriftResult(
        gpu_index=gpu, timestamp=time.time(), rtheta=0.72,
        baseline_mean=baseline, baseline_std=0.05,
        sigma_score=sigma, is_drifting=drifting,
        is_critical=critical, confidence=1.0 if drifting else 0.0,
    )

def test_state_machine_first_transition():
    sm = GPUStateMachine()
    c  = _make_classified(0, GPUState.UNDER_LOAD)
    d  = _make_drift(0)
    # UNKNOWN → UNDER_LOAD is not alert-worthy
    alert = sm.transition(c, d)
    assert alert is None

def test_state_machine_zombie_alert():
    sm = GPUStateMachine()
    # Establish healthy state first
    sm.transition(_make_classified(0, GPUState.UNDER_LOAD), _make_drift(0))
    # Transition to zombie
    c2    = _make_classified(0, GPUState.ZOMBIE_RECOVERY, rtheta=1.54)
    alert = sm.transition(c2, _make_drift(0))
    assert alert is not None
    assert alert.state == GPUState.ZOMBIE_RECOVERY
    assert "zombie" in alert.message.lower() or "CUDA" in alert.message

def test_state_machine_recovery_alert():
    sm = GPUStateMachine()
    sm.transition(_make_classified(0, GPUState.UNDER_LOAD), _make_drift(0))
    sm.transition(_make_classified(0, GPUState.ZOMBIE_RECOVERY, rtheta=1.54), _make_drift(0))
    # Recovery
    alert = sm.transition(_make_classified(0, GPUState.CLEAN_IDLE, rtheta=1.28), _make_drift(0))
    assert alert is not None
    assert alert.state == GPUState.CLEAN_IDLE

def test_state_machine_no_alert_same_state():
    sm = GPUStateMachine()
    sm.transition(_make_classified(0, GPUState.UNDER_LOAD), _make_drift(0))
    alert = sm.transition(_make_classified(0, GPUState.UNDER_LOAD), _make_drift(0))
    assert alert is None
