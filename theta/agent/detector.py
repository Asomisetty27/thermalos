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
import statistics
from collections import deque
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

from .metrics import GPUState

BASELINE_HEALTHY_STATES = {GPUState.UNDER_LOAD, GPUState.CLEAN_IDLE}

K_SIGMA_WARN     = 2.0   # σ above baseline → WARNING
K_SIGMA_CRITICAL = 3.5   # σ above baseline → CRITICAL
BASELINE_WINDOW  = 60    # number of stable samples for baseline rolling mean
MIN_BASELINE_SAMPLES = 20  # minimum before we trust the baseline
SUSTAINED_WINDOWS    = 3   # consecutive anomalous windows before alerting

TREND_WINDOW         = 30   # samples for linear regression
PREDICT_HORIZON_S    = 300  # 5 min — warn if threshold crossing within this window
PREDICT_COOLDOWN_S   = 120  # seconds between repeated predictive warnings per GPU


def _theil_sen_slope(xs: list[float], ys: list[float]) -> Optional[float]:
    """
    Robust slope estimate: median of all pairwise slopes between points.

    A single transient spike (a brief power excursion, a noisy NVML read)
    can swing an OLS fit (np.polyfit) enough to trigger or suppress a
    predictive alert. The median of pairwise slopes is unmoved unless a
    majority of points shift — exactly the "sustained trend, not a blip"
    distinction this detector is meant to make.

    Returns None if every pair has dx ~ 0 (e.g. duplicate timestamps).
    """
    slopes = [
        (y2 - y1) / (x2 - x1)
        for (x1, y1), (x2, y2) in combinations(zip(xs, ys), 2)
        if abs(x2 - x1) > 1e-9
    ]
    return statistics.median(slopes) if slopes else None


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
    confidence:   float              # 0–1 based on sustained window count
    # Predictive fields
    trend_slope:      Optional[float] = None  # R_theta per second (positive = worsening)
    eta_to_drift_s:   Optional[float] = None  # seconds until warn threshold crossed
    is_predictive:    bool            = False  # ETA within PREDICT_HORIZON_S, GPU currently healthy


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
        min_std:    float = 0.010,
    ):
        self._k_warn     = k_warn
        self._k_critical = k_critical
        self._baseline_n = baseline_n
        self._sustained  = sustained
        self._min_std    = min_std   # per-detector noise floor; override for liquid-cooled HW

        self._baselines:      dict[int, deque]         = {}
        self._anomaly_counts: dict[int, int]           = {}
        self._trend_buffers:  dict[int, deque]         = {}
        self._predict_alerted: dict[int, float]        = {}   # gpu → last predictive alert ts

    def update(
        self,
        gpu_index: int,
        timestamp: float,
        rtheta:    float,
        state:     GPUState,
    ) -> DriftResult:
        # Trend buffer: ALL readings (healthy or not) for regression
        if gpu_index not in self._trend_buffers:
            self._trend_buffers[gpu_index] = deque(maxlen=TREND_WINDOW)
        self._trend_buffers[gpu_index].append((timestamp, rtheta))

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

        # Guard against near-zero std (perfectly stable baseline).
        # Floor scales with hardware class — liquid-cooled GPUs have compressed
        # R_theta range (~0.06 C/W) so the T4-derived 0.01 floor would swamp
        # the degradation signal (~0.014 C/W for +23% TIM degradation).
        std = max(std, self._min_std)

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

        # ── Predictive trend regression ──────────────────────────────────────
        trend_slope = eta_to_drift_s = None
        is_predictive = False

        tbuf = self._trend_buffers[gpu_index]
        if len(tbuf) >= 10 and not is_drifting and not is_critical:
            xs = [t for t, _ in tbuf]
            ys = [r for _, r in tbuf]
            x0 = xs[0]
            xs_norm = [x - x0 for x in xs]   # normalize to zero-start

            slope = _theil_sen_slope(xs_norm, ys)
            trend_slope = slope

            if slope is not None and slope > 0:
                # Robust intercept (Theil-Sen): line of this slope through
                # the median point, rather than the OLS intercept.
                intercept = statistics.median(ys) - slope * statistics.median(xs_norm)
                warn_threshold = mean + self._k_warn * std
                t_elapsed = xs_norm[-1]
                predicted_now = slope * t_elapsed + intercept
                if predicted_now < warn_threshold:
                    eta = (warn_threshold - predicted_now) / slope
                    if 0 < eta < PREDICT_HORIZON_S:
                        last_pred = self._predict_alerted.get(gpu_index, 0.0)
                        if timestamp - last_pred >= PREDICT_COOLDOWN_S:
                            eta_to_drift_s = float(eta)
                            is_predictive  = True
                            self._predict_alerted[gpu_index] = timestamp

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
            trend_slope   = round(trend_slope, 6) if trend_slope is not None else None,
            eta_to_drift_s= round(eta_to_drift_s, 1) if eta_to_drift_s is not None else None,
            is_predictive = is_predictive,
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
