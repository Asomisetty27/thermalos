"""
Core metric types and R_theta computation.

R_theta_eff(t) = (T_junction(t) - T_ref) / P_GPU(t)

This is the foundational ThermalOS metric. No other tool computes it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class GPUState(Enum):
    UNKNOWN             = auto()
    CLEAN_IDLE          = auto()   # low load, healthy thermals
    UNDER_LOAD          = auto()   # high load, thermal equilibrium
    ZOMBIE_RECOVERY     = auto()   # stuck P0, CUDA context retained
    CHILD_EXIT_RECOVERY = auto()   # post child-exit, thermal lag
    DRIFTING            = auto()   # R_theta rising above k·σ threshold
    CRITICAL            = auto()   # R_theta exceeds critical threshold


STATE_LABELS = {
    GPUState.UNKNOWN:             "unknown",
    GPUState.CLEAN_IDLE:          "clean_idle",
    GPUState.UNDER_LOAD:          "under_load",
    GPUState.ZOMBIE_RECOVERY:     "zombie_recovery",
    GPUState.CHILD_EXIT_RECOVERY: "child_exit_recovery",
    GPUState.DRIFTING:            "drifting",
    GPUState.CRITICAL:            "critical",
}

# Classifier class index → GPUState (matches Stage 1 training order)
CLASS_INDEX_TO_STATE = {
    0: GPUState.CHILD_EXIT_RECOVERY,
    1: GPUState.CLEAN_IDLE,
    2: GPUState.UNDER_LOAD,
    3: GPUState.ZOMBIE_RECOVERY,
}


@dataclass(slots=True)
class RawSample:
    """Single telemetry snapshot from pynvml."""
    gpu_index:     int
    timestamp:     float          # unix seconds
    temp_junction: float          # °C  (T_junction from nvmlDeviceGetTemperature)
    power_w:       float          # W   (nvmlDeviceGetPowerUsage / 1000)
    util_pct:      float          # 0–100
    mem_util_pct:  float          # 0–100
    perf_state:    int            # 0–15 (P0 = max performance)
    clock_sm_mhz:  int
    clock_mem_mhz: int
    fan_speed_pct: Optional[float] = None


@dataclass(slots=True)
class EnrichedSample:
    """RawSample + derived R_theta."""
    raw:           RawSample
    t_ref:         float          # virtual ambient (°C)
    rtheta:        Optional[float]  # C/W — None when P_GPU < MIN_POWER
    rtheta_valid:  bool           # False when denominator too small

    @property
    def gpu_index(self) -> int:
        return self.raw.gpu_index

    @property
    def timestamp(self) -> float:
        return self.raw.timestamp


@dataclass
class ClassifiedSample:
    """EnrichedSample + classifier output."""
    enriched:    EnrichedSample
    state:       GPUState
    confidence:  float          # 0–1, max class probability
    rtheta_mean: Optional[float]  # mean R_theta over steady-state window

    @property
    def gpu_index(self) -> int:
        return self.enriched.gpu_index

    @property
    def timestamp(self) -> float:
        return self.enriched.timestamp


@dataclass
class AlertEvent:
    """Emitted when a GPU transitions to an anomalous state."""
    gpu_index:    int
    timestamp:    float
    state:        GPUState
    prev_state:   GPUState
    rtheta:       Optional[float]
    rtheta_baseline: Optional[float]
    drift_sigma:  Optional[float]   # how many σ above baseline
    confidence:   float
    message:      str
    context:      dict = field(default_factory=dict)  # last N samples for explainability


# ── R_theta computation ──────────────────────────────────────────────────────

MIN_POWER_W = 5.0    # below this, R_theta is numerically unstable
MIN_DELTA_T = 0.5    # below this ΔT, skip (noise floor)


def compute_rtheta(
    temp_junction: float,
    t_ref: float,
    power_w: float,
) -> tuple[Optional[float], bool]:
    """
    Compute R_theta_eff = (T_junction - T_ref) / P_GPU.

    Returns (rtheta, valid). valid=False when power is too low for a
    reliable estimate — caller should skip classification but still record.
    """
    if power_w < MIN_POWER_W:
        return None, False
    delta_t = temp_junction - t_ref
    if delta_t < MIN_DELTA_T:
        return None, False
    return delta_t / power_w, True


def enrich(sample: RawSample, t_ref: float) -> EnrichedSample:
    rtheta, valid = compute_rtheta(sample.temp_junction, t_ref, sample.power_w)
    return EnrichedSample(raw=sample, t_ref=t_ref, rtheta=rtheta, rtheta_valid=valid)
