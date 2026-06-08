"""
Per-hardware calibration for theta classifier thresholds.

The bundled models are trained on Tesla T4 Stage 1 data. On other hardware
(A100, H100, B200, etc.) R_theta ranges differ — T4 rules will misclassify.

`theta calibrate` runs a two-phase measurement:
  1. Idle phase   — wait for stable idle window, record R_theta_idle
  2. Load phase   — user starts a workload, record R_theta_load

Derived thresholds are saved to ~/.theta/calibration.json.
StateClassifier loads this at init and substitutes calibrated values.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .safeio import atomic_write_text

def _default_calibration_file() -> Path:
    """Resolve calibration file path.

    Priority:
      1. $THETA_CONFIG_DIR/calibration.json  — set by the systemd service unit
         so the 'theta' service user finds the same file the operator calibrated
         with (which may live in /etc/theta/).
      2. ~/.theta/calibration.json           — single-user default.
    """
    import os
    cfg_dir = os.environ.get("THETA_CONFIG_DIR")
    if cfg_dir:
        return Path(cfg_dir) / "calibration.json"
    return Path.home() / ".theta" / "calibration.json"


CALIBRATION_FILE = _default_calibration_file()

# T4 Stage 1 reference points
_T4_RTHETA_IDLE      = 1.28
_T4_RTHETA_LOAD      = 0.72
_T4_LOAD_THRESHOLD   = 0.87   # rtheta <= → under_load
_T4_IDLE_THRESHOLD   = 1.50   # rtheta >= → idle territory

# Phase acceptance criteria
_IDLE_UTIL_MAX  = 5.0    # % — GPU must be below this to accumulate idle samples
_LOAD_UTIL_MIN  = 70.0   # % — GPU must be above this to accumulate load samples
_WINDOW_SEC     = 20.0   # seconds of history needed for a valid window
_SIGMA_MAX      = 0.06   # max R_theta σ to accept the window as stable


@dataclass
class CalibrationResult:
    gpu_index:      int
    gpu_name:       str
    rtheta_idle:    float
    rtheta_load:    Optional[float]   # None if load phase was skipped
    load_threshold: float
    idle_threshold: float
    calibrated_at:  float
    source:         str   # "observed_both" | "idle_only"

    def age_hours(self) -> float:
        return (time.time() - self.calibrated_at) / 3600.0


class CalibrationManager:
    """Load and persist per-GPU calibration. Provides thresholds to StateClassifier."""

    def __init__(self, _file: Optional[Path] = None):
        self._file = _file or CALIBRATION_FILE
        self._cals: dict[int, CalibrationResult] = {}
        self._load()

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            for entry in json.loads(self._file.read_text()):
                c = CalibrationResult(**entry)
                self._cals[c.gpu_index] = c
        except Exception:
            pass

    def save(self) -> None:
        payload = json.dumps([asdict(c) for c in self._cals.values()], indent=2)
        atomic_write_text(self._file, payload)

    def get(self, gpu_index: int) -> Optional[CalibrationResult]:
        return self._cals.get(gpu_index)

    def set(self, result: CalibrationResult) -> None:
        self._cals[result.gpu_index] = result
        self.save()

    def load_threshold(self, gpu_index: int) -> float:
        c = self._cals.get(gpu_index)
        return c.load_threshold if c else _T4_LOAD_THRESHOLD

    def idle_threshold(self, gpu_index: int) -> float:
        c = self._cals.get(gpu_index)
        return c.idle_threshold if c else _T4_IDLE_THRESHOLD


def derive_thresholds(
    rtheta_idle: float,
    rtheta_load: Optional[float],
) -> tuple[float, float]:
    """
    Return (load_threshold, idle_threshold) from measured R_theta values.

    When both phases are observed the gap between them is split 35/20 —
    creating a narrow dead-zone that prevents flip-flopping near the boundary.
    When only idle is observed the T4 idle/load ratio is scaled to the new floor.
    """
    if rtheta_load is not None:
        gap = rtheta_idle - rtheta_load
        return (
            round(rtheta_load + gap * 0.35, 3),
            round(rtheta_idle - gap * 0.20, 3),
        )
    ratio_load = _T4_LOAD_THRESHOLD / _T4_RTHETA_IDLE
    ratio_idle = _T4_IDLE_THRESHOLD / _T4_RTHETA_IDLE
    return (
        round(rtheta_idle * ratio_load, 3),
        round(rtheta_idle * ratio_idle, 3),
    )


class _PhaseAccumulator:
    """Maintains a rolling R_theta window and reports when it stabilises."""

    def __init__(self, window_sec: float = _WINDOW_SEC, sigma_max: float = _SIGMA_MAX):
        self._window_sec = window_sec
        self._sigma_max  = sigma_max
        self._buf: deque = deque()

    def push(self, timestamp: float, rtheta: float) -> Optional[float]:
        """
        Feed a sample. Returns the stable mean R_theta when the window is ready,
        None otherwise.
        """
        self._buf.append((timestamp, rtheta))
        cutoff = timestamp - self._window_sec
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

        if not self._buf:
            return None

        span = self._buf[-1][0] - self._buf[0][0]
        if span < self._window_sec * 0.85:
            return None

        vals = [r for _, r in self._buf]
        mean = sum(vals) / len(vals)
        std  = math.sqrt(sum((r - mean) ** 2 for r in vals) / len(vals))
        return round(mean, 4) if std <= self._sigma_max else None

    def reset(self) -> None:
        self._buf.clear()

    def progress(self, timestamp: float) -> float:
        """Fraction of window filled (0–1). Used for progress display."""
        if not self._buf:
            return 0.0
        span = self._buf[-1][0] - self._buf[0][0]
        return min(1.0, span / self._window_sec)


async def run_idle_phase(
    collector,
    baseline_manager,
    max_wait_sec: float = 120.0,
) -> Optional[float]:
    """
    Stream samples from an open NVMLCollector and return stable R_theta_idle.

    Returns None on timeout. The BaselineManager is fed samples as a side-effect
    so T_ref is available for the subsequent load phase.
    """
    import asyncio
    from .metrics import compute_rtheta

    acc      = _PhaseAccumulator()
    deadline = asyncio.get_event_loop().time() + max_wait_sec

    async for sample in collector.stream():
        # Always feed baseline regardless of idle state
        baseline_manager.update(
            sample.gpu_index, sample.temp_junction,
            sample.util_pct, sample.perf_state, sample.timestamp,
        )
        if asyncio.get_event_loop().time() > deadline:
            return None

        if sample.util_pct > _IDLE_UTIL_MAX:
            acc.reset()
            continue

        t_ref = baseline_manager.get_t_ref(sample.gpu_index)
        rtheta, valid = compute_rtheta(sample.temp_junction, t_ref, sample.power_w)
        if not valid:
            continue

        result = acc.push(sample.timestamp, rtheta)
        if result is not None:
            return result

    return None


async def run_load_phase(
    collector,
    baseline_manager,
    max_wait_sec: float = 120.0,
) -> Optional[float]:
    """
    Stream samples and return stable R_theta_load once GPU util exceeds threshold.
    Returns None on timeout.
    """
    import asyncio
    from .metrics import compute_rtheta

    acc      = _PhaseAccumulator()
    deadline = asyncio.get_event_loop().time() + max_wait_sec

    async for sample in collector.stream():
        if asyncio.get_event_loop().time() > deadline:
            return None

        if sample.util_pct < _LOAD_UTIL_MIN:
            acc.reset()
            continue

        t_ref = baseline_manager.get_t_ref(sample.gpu_index)
        rtheta, valid = compute_rtheta(sample.temp_junction, t_ref, sample.power_w)
        if not valid:
            continue

        result = acc.push(sample.timestamp, rtheta)
        if result is not None:
            return result

    return None
