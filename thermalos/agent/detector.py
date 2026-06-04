"""
Drift detector: R_theta baseline + k·σ alert threshold.

Per GPU, tracks the rolling R_theta baseline from healthy (under_load or
clean_idle) windows. Emits a DRIFTING event when:
    current R_theta > baseline_mean + k * baseline_sigma

for a sustained number of consecutive stable windows (not a single spike).
This is the "drift detection, not thresholds" capability (bento card 01).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from .metrics import GPUState

BASELINE_HEALTHY_STATES = {GPUState.UNDER_LOAD, GPUState.CLEAN_IDLE}

K_SIGMA_WARN     = 2.0   # σ above baseline → WARNING
K_SIGMA_CRITICAL = 3.5   # σ above baseline → CRITICAL
BASELINE_WINDOW  = 60    # number of stable samples for baseline rolling mean
MIN_BASELINE_SAMPLES = 20  # minimum before we trust the baseline
SUSTAINED_WINDOWS    = 3   # consecutive anomalous windows before alerting


@dataclass
class DriftResult:
    gpu_index:    int
    timestamp:    float
    rtheta:       float
    baseline_mean: Optional[float]
    baseline_std:  Optional[float]
    sigma_score:   Optional[float]   # how many σ above baseline
    is_drifting:  bool
    is_critical:  bool
    confidence:   float             # 0–1 based on sustained window count


class DriftDetector:
    """
    Per-GPU drift detector.

    Maintains a rolling baseline from healthy states and flags when
    R_theta deviates significantly.
    """

    def __init__(
        self,
        k_warn:     float = K_SIGMA_WARN,
        k_critical: float = K_SIGMA_CRITICAL,
        baseline_n: int   = BASELINE_WINDOW,
        sustained:  int   = SUSTAINED_WINDOWS,
    ):
        self._k_warn     = k_warn
        self._k_critical = k_critical
        self._baseline_n = baseline_n
        self._sustained  = sustained

        self._baselines:      dict[int, deque]         = {}
        self._anomaly_counts: dict[int, int]           = {}

    def update(
        self,
        gpu_index: int,
        timestamp: float,
        rtheta:    float,
        state:     GPUState,
    ) -> DriftResult:
        # Update baseline from healthy windows only
        if state in BASELINE_HEALTHY_STATES:
            if gpu_index not in self._baselines:
                self._baselines[gpu_index] = deque(maxlen=self._baseline_n)
            self._baselines[gpu_index].append(rtheta)

        buf = self._baselines.get(gpu_index)
        if not buf or len(buf) < MIN_BASELINE_SAMPLES:
            return DriftResult(
                gpu_index     = gpu_index,
                timestamp     = timestamp,
                rtheta        = rtheta,
                baseline_mean = None,
                baseline_std  = None,
                sigma_score   = None,
                is_drifting   = False,
                is_critical   = False,
                confidence    = 0.0,
            )

        vals = list(buf)
        mean = sum(vals) / len(vals)
        std  = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))

        # Guard against near-zero std (perfectly stable baseline)
        std = max(std, 0.01)

        sigma_score = (rtheta - mean) / std

        is_above_warn     = sigma_score > self._k_warn
        is_above_critical = sigma_score > self._k_critical

        count = self._anomaly_counts.get(gpu_index, 0)
        if is_above_warn:
            count += 1
        else:
            count = max(0, count - 1)  # decay slowly (don't snap back on single good reading)
        self._anomaly_counts[gpu_index] = count

        is_drifting  = is_above_warn     and count >= self._sustained
        is_critical  = is_above_critical and count >= self._sustained

        confidence = min(1.0, count / self._sustained) if is_above_warn else 0.0

        return DriftResult(
            gpu_index     = gpu_index,
            timestamp     = timestamp,
            rtheta        = rtheta,
            baseline_mean = round(mean, 4),
            baseline_std  = round(std, 4),
            sigma_score   = round(sigma_score, 2),
            is_drifting   = is_drifting,
            is_critical   = is_critical,
            confidence    = round(confidence, 2),
        )

    def reset_baseline(self, gpu_index: int) -> None:
        self._baselines.pop(gpu_index, None)
        self._anomaly_counts.pop(gpu_index, None)

    def get_baseline(self, gpu_index: int) -> Optional[tuple[float, float]]:
        """Returns (mean, std) or None if insufficient data."""
        buf = self._baselines.get(gpu_index)
        if not buf or len(buf) < MIN_BASELINE_SAMPLES:
            return None
        vals = list(buf)
        mean = sum(vals) / len(vals)
        std  = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
        return round(mean, 4), round(std, 4)
