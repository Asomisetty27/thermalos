"""
Unit tests for `theta calibrate` (theta.agent.calibrate).

Covers the pure-logic pieces that decide what gets written to
~/.theta/calibration.json and how it overrides T4 defaults — the parts
a regression here would silently miscalibrate every non-T4 deployment.
Async phase-runners (run_idle_phase / run_load_phase) are exercised with
a fake collector so no GPU/NVML is required.
"""

import asyncio
import json

import pytest

from theta.agent.baseline import BaselineManager
from theta.agent.calibrate import (
    CalibrationManager, CalibrationResult, _PhaseAccumulator,
    derive_thresholds, run_idle_phase, run_load_phase,
    _T4_RTHETA_IDLE, _T4_LOAD_THRESHOLD, _T4_IDLE_THRESHOLD,
)


# ── derive_thresholds ─────────────────────────────────────────────────────────

def test_derive_thresholds_both_phases_splits_gap_35_20():
    # idle=1.0, load=0.6 -> gap=0.4 -> load_threshold = 0.6 + 0.4*0.35 = 0.74
    #                                  idle_threshold = 1.0 - 0.4*0.20 = 0.92
    load_t, idle_t = derive_thresholds(rtheta_idle=1.0, rtheta_load=0.6)
    assert load_t == pytest.approx(0.74, abs=1e-6)
    assert idle_t == pytest.approx(0.92, abs=1e-6)
    # dead-zone: load_threshold < idle_threshold, with daylight between them
    assert load_t < idle_t

def test_derive_thresholds_idle_only_scales_t4_ratio():
    # idle-only path scales the T4 idle/load and idle/idle-threshold ratios
    # onto the new floor — verify it actually uses T4's calibrated ratios,
    # not some independent constant.
    rtheta_idle = 2.0
    load_t, idle_t = derive_thresholds(rtheta_idle=rtheta_idle, rtheta_load=None)
    expected_load = round(rtheta_idle * (_T4_LOAD_THRESHOLD / _T4_RTHETA_IDLE), 3)
    expected_idle = round(rtheta_idle * (_T4_IDLE_THRESHOLD / _T4_RTHETA_IDLE), 3)
    assert load_t == expected_load
    assert idle_t == expected_idle
    assert load_t < rtheta_idle < idle_t

def test_derive_thresholds_scales_with_floor():
    # A GPU with double the T4's idle R_theta should get roughly double the
    # thresholds too (idle-only path) — this is the whole point of calibration:
    # a higher/lower floor shifts the decision boundary with it.
    lo_load, lo_idle = derive_thresholds(rtheta_idle=1.0, rtheta_load=None)
    hi_load, hi_idle = derive_thresholds(rtheta_idle=2.0, rtheta_load=None)
    # rel tolerance (not abs) to absorb the round(..., 3) quantisation on each side
    assert hi_load == pytest.approx(lo_load * 2, rel=2e-3)
    assert hi_idle == pytest.approx(lo_idle * 2, rel=2e-3)


# ── CalibrationManager ────────────────────────────────────────────────────────

def test_calibration_manager_defaults_to_t4_when_absent(tmp_path):
    mgr = CalibrationManager(_file=tmp_path / "calibration.json")
    assert mgr.get(0) is None
    assert mgr.load_threshold(0) == _T4_LOAD_THRESHOLD
    assert mgr.idle_threshold(0) == _T4_IDLE_THRESHOLD

def test_calibration_manager_set_overrides_thresholds(tmp_path):
    mgr = CalibrationManager(_file=tmp_path / "calibration.json")
    result = CalibrationResult(
        gpu_index=0, gpu_name="A100", rtheta_idle=0.30, rtheta_load=0.18,
        load_threshold=0.21, idle_threshold=0.27,
        calibrated_at=1_000_000.0, source="observed_both",
    )
    mgr.set(result)
    assert mgr.get(0) is result
    assert mgr.load_threshold(0) == 0.21
    assert mgr.idle_threshold(0) == 0.27
    # other GPU indices are untouched and still fall back to T4 defaults
    assert mgr.load_threshold(1) == _T4_LOAD_THRESHOLD

def test_calibration_manager_persists_and_reloads(tmp_path):
    cal_file = tmp_path / "calibration.json"
    mgr = CalibrationManager(_file=cal_file)
    result = CalibrationResult(
        gpu_index=2, gpu_name="H100", rtheta_idle=0.20, rtheta_load=0.12,
        load_threshold=0.15, idle_threshold=0.18,
        calibrated_at=2_000_000.0, source="idle_only",
    )
    mgr.set(result)
    assert cal_file.exists()

    # fresh manager loads from disk
    reloaded = CalibrationManager(_file=cal_file)
    c = reloaded.get(2)
    assert c is not None
    assert c.gpu_name == "H100"
    assert c.load_threshold == 0.15
    assert c.idle_threshold == 0.18
    assert c.source == "idle_only"

def test_calibration_manager_survives_corrupt_file(tmp_path):
    cal_file = tmp_path / "calibration.json"
    cal_file.write_text("{not valid json")
    mgr = CalibrationManager(_file=cal_file)
    # corrupt file -> empty store, falls back to T4 defaults rather than raising
    assert mgr.get(0) is None
    assert mgr.load_threshold(0) == _T4_LOAD_THRESHOLD

def test_calibration_result_age_hours(tmp_path):
    import time
    result = CalibrationResult(
        gpu_index=0, gpu_name="B200", rtheta_idle=0.10, rtheta_load=0.06,
        load_threshold=0.07, idle_threshold=0.09,
        calibrated_at=time.time() - 7200.0, source="observed_both",
    )
    assert result.age_hours() == pytest.approx(2.0, abs=0.05)


# ── _PhaseAccumulator ─────────────────────────────────────────────────────────

def test_phase_accumulator_returns_none_until_window_full():
    acc = _PhaseAccumulator(window_sec=20.0, sigma_max=0.06)
    for i in range(5):
        assert acc.push(timestamp=float(i), rtheta=1.0) is None

def test_phase_accumulator_locks_stable_mean():
    acc = _PhaseAccumulator(window_sec=20.0, sigma_max=0.06)
    result = None
    # 21 samples spanning >20s with near-identical R_theta -> stable
    for i in range(22):
        result = acc.push(timestamp=float(i), rtheta=1.20 + (0.001 if i % 2 else -0.001))
    assert result is not None
    assert result == pytest.approx(1.20, abs=0.01)

def test_phase_accumulator_rejects_noisy_window():
    acc = _PhaseAccumulator(window_sec=20.0, sigma_max=0.06)
    result = None
    # large oscillation -> sigma exceeds sigma_max -> never locks
    for i in range(22):
        result = acc.push(timestamp=float(i), rtheta=1.0 + (0.5 if i % 2 else -0.5))
    assert result is None

def test_phase_accumulator_reset_clears_buffer():
    acc = _PhaseAccumulator(window_sec=20.0, sigma_max=0.06)
    for i in range(10):
        acc.push(timestamp=float(i), rtheta=1.0)
    assert acc.progress(9.0) > 0.0
    acc.reset()
    assert acc.progress(9.0) == 0.0


# ── async phase runners (fake collector, no NVML) ─────────────────────────────

class _FakeSample:
    def __init__(self, gpu_index, timestamp, temp_junction, power_w, util_pct, perf_state=0):
        self.gpu_index = gpu_index
        self.timestamp = timestamp
        self.temp_junction = temp_junction
        self.power_w = power_w
        self.util_pct = util_pct
        self.perf_state = perf_state


class _FakeCollector:
    """Yields a fixed sequence of pre-built samples, one every `dt` virtual seconds."""
    def __init__(self, samples, dt=2.0):
        self._samples = samples
        self._dt = dt

    async def stream(self):
        for s in self._samples:
            await asyncio.sleep(0)  # yield control, keep it fast
            yield s


def _make_idle_samples(n=15, gpu_index=0, t0=0.0, dt=2.0,
                        temp=30.0, t_ref=25.0, power=15.0, util=2.0):
    # R_theta = (temp - t_ref) / power, held constant -> stable window quickly
    return [
        _FakeSample(gpu_index, t0 + i * dt, temp, power, util)
        for i in range(n)
    ]


def test_run_idle_phase_locks_on_stable_window(tmp_path):
    bm = BaselineManager(_file=tmp_path / "baseline.json")
    bm.set_manual(0, 25.0)  # pin T_ref so R_theta is deterministic
    samples = _make_idle_samples(n=15, temp=30.0, t_ref=25.0, power=15.0)
    collector = _FakeCollector(samples)

    rtheta = asyncio.run(run_idle_phase(collector, bm, max_wait_sec=120.0))
    assert rtheta is not None
    assert rtheta == pytest.approx((30.0 - 25.0) / 15.0, abs=1e-3)

def test_run_idle_phase_resets_on_high_util(tmp_path):
    bm = BaselineManager(_file=tmp_path / "baseline.json")
    bm.set_manual(0, 25.0)
    # first burst looks like load (util > 5%) then settles into idle —
    # the accumulator must discard the load samples, not blend them in
    samples = (
        [_FakeSample(0, i * 2.0, 50.0, 60.0, 80.0) for i in range(5)]
        + _make_idle_samples(n=15, t0=10.0, temp=30.0, t_ref=25.0, power=15.0)
    )
    collector = _FakeCollector(samples)

    rtheta = asyncio.run(run_idle_phase(collector, bm, max_wait_sec=120.0))
    assert rtheta is not None
    assert rtheta == pytest.approx((30.0 - 25.0) / 15.0, abs=1e-3)

def test_run_idle_phase_times_out_when_never_idle(tmp_path):
    bm = BaselineManager(_file=tmp_path / "baseline.json")
    bm.set_manual(0, 25.0)
    samples = [_FakeSample(0, i * 2.0, 70.0, 65.0, 90.0) for i in range(200)]
    collector = _FakeCollector(samples)

    rtheta = asyncio.run(run_idle_phase(collector, bm, max_wait_sec=0.0))
    assert rtheta is None

def test_run_load_phase_locks_on_stable_window(tmp_path):
    bm = BaselineManager(_file=tmp_path / "baseline.json")
    bm.set_manual(0, 25.0)
    samples = [
        _FakeSample(0, i * 2.0, 80.0, 65.0, 90.0)
        for i in range(15)
    ]
    collector = _FakeCollector(samples)

    rtheta = asyncio.run(run_load_phase(collector, bm, max_wait_sec=120.0))
    assert rtheta is not None
    assert rtheta == pytest.approx((80.0 - 25.0) / 65.0, abs=1e-3)
