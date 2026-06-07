"""
Sensor model + R_theta anomaly detector for the E-LT simulation.

This is the SAME detection logic Theta ships in the OSS agent
(theta/agent/window.py + the baseline+k-sigma drift rule), reimplemented
here on plain arrays so the lab analysis and the product share one definition.
The protocol is explicit that lab and product must use one detector.

Pipeline (per the protocol "Anomaly threshold definition"):

  1. Sensor model    : true T_j -> add noise -> quantise to integer degrees;
                       true P   -> add noise. (The detector NEVER sees ground truth.)
  2. Reference temp  : true ambient (lab thermocouple)  OR
                       virtual ambient (idle-window estimate, what the product uses).
  3. R_theta         : (T_j_sensed - T_ref) / P_sensed.
  4. Steady-state    : compute R_theta only over power-stable windows
                       (std(power) over the window < threshold) — Kundu's guidance.
  5. Baseline        : mean & std of windowed R_theta over the healthy baseline phase.
  6. Anomaly         : R_theta > mean + k*std, SUSTAINED for `persist_s` seconds.
                       Sweep k in {2,3,4}; each yields a t_anomaly -> a lead time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import params as P
from .thermal_model import SimResult


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sensor model
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SensedTelemetry:
    t: np.ndarray
    temp_j: np.ndarray         # integer-quantised junction temp (C)
    power: np.ndarray          # noisy power (W)
    t_ref: np.ndarray          # reference temp used for R_theta (C)
    ambient_mode: str          # 'true' | 'virtual'


def apply_sensor_model(sim: SimResult, rng: np.random.Generator,
                       ambient_mode: str = "true",
                       ambient_drift_c_per_hr: float = 0.0) -> SensedTelemetry:
    """
    Convert ground-truth SimResult into what the agent would actually observe.

    ambient_mode:
      'true'    — lab thermocouple reads true ambient (with optional drift).
      'virtual' — product estimate: ambient locked from the pre-degradation idle
                  baseline, then held fixed (goes stale if ambient drifts).
    """
    n = sim.t.size

    # True ambient, optionally drifting during the run (diurnal lab swing)
    amb_true = P.T_AMBIENT_C + ambient_drift_c_per_hr * (sim.t / 3600.0)

    # Junction sensor: sub-degree Gaussian noise then integer quantisation
    tj_noisy = sim.tj_true + rng.normal(0.0, P.TEMP_NOISE_C, n)
    tj_quant = np.round(tj_noisy / P.TEMP_QUANT_C) * P.TEMP_QUANT_C

    # Power sensor: Gaussian noise (the column is float in Stage 1 data)
    power = sim.p_eff + rng.normal(0.0, P.POWER_NOISE_W, n)
    power = np.maximum(power, 1e-3)

    if ambient_mode == "true":
        t_ref = amb_true.copy()
    elif ambient_mode == "virtual":
        # locked at t=0 from the (healthy) baseline; never updates under load
        t_ref = np.full(n, amb_true[0])
    else:
        raise ValueError(f"ambient_mode must be 'true' or 'virtual', got {ambient_mode!r}")

    return SensedTelemetry(t=sim.t, temp_j=tj_quant, power=power,
                           t_ref=t_ref, ambient_mode=ambient_mode)


# ─────────────────────────────────────────────────────────────────────────────
# 2-4. Steady-state windowed R_theta
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DetectorConfig:
    window_s: float = 15.0          # steady-state window length (agent default)
    power_std_max_w: float = 2.0    # window is "stable" if power std below this
    persist_s: float = 10.0         # anomaly must persist this long to fire
    k_values: tuple = (2.0, 3.0, 4.0)   # sigma multipliers to sweep


def windowed_rtheta(tel: SensedTelemetry, cfg: DetectorConfig,
                    dt_s: float = P.SAMPLE_PERIOD_S) -> tuple[np.ndarray, np.ndarray]:
    """
    Rolling steady-state window. Returns (rtheta, stable_mask) per sample.
    R_theta at sample i = mean over the trailing window of (T_j - T_ref)/P,
    valid only where the window's power std is below threshold.
    """
    n = tel.t.size
    w = max(1, int(round(cfg.window_s / dt_s)))
    rtheta = np.full(n, np.nan)
    stable = np.zeros(n, dtype=bool)

    inst = (tel.temp_j - tel.t_ref) / tel.power   # instantaneous R_theta

    for i in range(n):
        lo = max(0, i - w + 1)
        p_win = tel.power[lo:i + 1]
        r_win = inst[lo:i + 1]
        if p_win.size < w:
            continue                       # window not yet full
        if np.std(p_win) <= cfg.power_std_max_w:
            stable[i] = True
            rtheta[i] = np.mean(r_win)
    return rtheta, stable


# ─────────────────────────────────────────────────────────────────────────────
# 5-6. Baseline statistics + anomaly detection
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Baseline:
    mean: float
    std: float
    n: int


def fit_baseline(rtheta: np.ndarray, stable: np.ndarray, t: np.ndarray,
                 baseline_s: float) -> Baseline:
    """Mean/std of windowed R_theta over the healthy baseline phase (t < baseline_s)."""
    mask = stable & (t < baseline_s) & np.isfinite(rtheta)
    vals = rtheta[mask]
    if vals.size < 5:
        raise ValueError(f"too few baseline samples ({vals.size}) to fit; "
                         f"increase baseline_s")
    # floor std so a perfectly flat baseline can't make k*std == 0
    return Baseline(mean=float(np.mean(vals)),
                    std=max(float(np.std(vals)), 1e-4),
                    n=int(vals.size))


@dataclass
class AnomalyResult:
    k: float
    threshold: float
    t_anomaly: Optional[float]      # first sustained crossing (s) or None


def detect_anomaly(rtheta: np.ndarray, stable: np.ndarray, t: np.ndarray,
                   base: Baseline, k: float, cfg: DetectorConfig,
                   dt_s: float = P.SAMPLE_PERIOD_S) -> AnomalyResult:
    """
    First time windowed R_theta stays above mean + k*std for `persist_s`.
    Only stable, finite samples count toward persistence.
    """
    thr = base.mean + k * base.std
    persist_n = max(1, int(round(cfg.persist_s / dt_s)))

    run = 0
    for i in range(t.size):
        if stable[i] and np.isfinite(rtheta[i]) and rtheta[i] > thr:
            run += 1
            if run >= persist_n:
                # anomaly first declared persist_s ago (start of the run)
                return AnomalyResult(k=k, threshold=thr,
                                     t_anomaly=float(t[i - persist_n + 1]))
        else:
            run = 0
    return AnomalyResult(k=k, threshold=thr, t_anomaly=None)
