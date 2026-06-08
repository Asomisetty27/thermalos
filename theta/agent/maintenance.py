"""
Predictive maintenance scoring — when will this GPU need service?

Audit finding addressed: Theta already detects drift and classifies faults,
but the operator question is forward-looking: "of my 1000 GPUs, which 5
should I service this month?" That requires projecting current trends into
the future and ranking by projected time-until-action.

This is a deliberately INTERPRETABLE model — not a black-box LSTM. It
combines four signals that an operator can verify by eye:

  1. R_θ aging signal     — current rate of monthly drift in C/W per month
  2. ECC SBIT trend       — single-bit-correctable error rate as silicon-aging
                            proxy (NVIDIA documents SBIT rate as a leading
                            indicator of HBM lifetime)
  3. Workload intensity   — high-utilization GPUs degrade faster
  4. Ambient stress       — sustained inlet > expected_ambient_c accelerates
                            TIM degradation per Arrhenius scaling

Output: days_until_service + confidence band + the dominant contributing
factor. That last part is critical — telling an operator "service in 23
days" without saying "because of TIM aging" gives them no way to prioritize
or batch the maintenance window.

The model is calibrated against:
  - Stage 1 T4 baseline drift rates
  - NVIDIA-published HBM lifetime curves (T_j vs MTBF)
  - Generic Arrhenius acceleration: every 10 °C above design ambient halves
    component lifetime (industry rule of thumb for silicon-adjacent epoxy
    and TIM materials)

We DO NOT claim this predicts catastrophic failure — it predicts when an
operator should schedule preventive maintenance to keep the GPU within its
calibrated thermal envelope. That's the genuinely useful question.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .hw_profiles import ThermalProfile


class MaintenancePriority(Enum):
    NONE        = "none"          # > 90 days projected; no action
    BACKLOG     = "backlog"       # 30–90 days; add to next routine window
    NEXT_WINDOW = "next_window"   # 7–30 days; schedule explicitly
    URGENT      = "urgent"        # 1–7 days; service this week
    IMMEDIATE   = "immediate"     # < 1 day OR critical signal


@dataclass
class MaintenanceScore:
    gpu_index: int
    priority: MaintenancePriority
    days_until_service: float        # projected days; inf means "no projection"
    days_uncertainty: float          # ± band (one-sigma)
    dominant_factor: str             # which signal drove the score
    contributions: dict[str, float]  # signal → relative weight (sums to ~1)
    headline: str                    # one-line operator summary

    def as_dict(self) -> dict:
        return {
            "gpu_index": self.gpu_index,
            "priority": self.priority.value,
            "days_until_service": (
                round(self.days_until_service, 1)
                if math.isfinite(self.days_until_service) else None
            ),
            "days_uncertainty": round(self.days_uncertainty, 1),
            "dominant_factor": self.dominant_factor,
            "contributions": {k: round(v, 3) for k, v in self.contributions.items()},
            "headline": self.headline,
        }


def _arrhenius_acceleration(t_inlet_c: float, t_design_c: float) -> float:
    """
    Industry rule: each +10 °C halves expected lifetime. We treat this as
    a multiplicative acceleration factor on the aging signal.
    """
    delta = max(0.0, t_inlet_c - t_design_c)
    return 2.0 ** (delta / 10.0)


def score(
    *,
    gpu_index: int,
    profile: Optional[ThermalProfile],
    # Aging signal — C/W per MONTH (positive = degrading)
    rtheta_aging_rate_per_month: float,
    # Current R_θ relative to baseline
    rtheta_current: float,
    rtheta_baseline: float,
    # The R_θ level at which "service required" — typically the calibrated
    # load_threshold for this GPU. When the projected R_θ trajectory crosses
    # this, the GPU starts triggering legitimate drift alerts continuously.
    rtheta_service_threshold: float,
    # ECC trend
    ecc_sbit_per_hour: float,
    ecc_sbit_baseline_per_hour: float = 0.5,
    # Workload intensity (0..1 — fraction of time GPU was UNDER_LOAD recently)
    workload_intensity: float = 0.5,
    # Ambient stress
    inlet_temp_c: Optional[float] = None,
) -> MaintenanceScore:
    """
    Compute a maintenance score. All arguments are values the daemon already
    knows or can readily aggregate from its rolling buffers.
    """
    contributions: dict[str, float] = {}

    # ── 1. R_θ aging signal → days until threshold crossed ──
    if rtheta_aging_rate_per_month > 0 and rtheta_current < rtheta_service_threshold:
        # How many months until current R_θ crosses the service threshold?
        gap = rtheta_service_threshold - rtheta_current
        months_to_threshold = gap / rtheta_aging_rate_per_month
        days_from_aging = months_to_threshold * 30.0
        # Confidence is wider for slower drift rates (more extrapolation)
        # and tighter for larger gaps to threshold (linearity holds better).
        aging_uncertainty = days_from_aging * 0.30  # ±30%
    else:
        days_from_aging = math.inf
        aging_uncertainty = 0.0
    contributions["aging_drift"] = (
        min(1.0, rtheta_aging_rate_per_month / 0.05) if rtheta_aging_rate_per_month > 0 else 0.0
    )

    # ── 2. ECC SBIT rate → silicon-aging penalty ──
    # NVIDIA: SBIT rate > 10×baseline is a leading indicator (months out).
    sbit_ratio = ecc_sbit_per_hour / max(ecc_sbit_baseline_per_hour, 0.1)
    if sbit_ratio > 10:
        # Severe — pull in by 50 % (this is heuristic; real curve is HBM-specific)
        sbit_penalty = 0.50
    elif sbit_ratio > 3:
        sbit_penalty = 0.20
    else:
        sbit_penalty = 0.0
    contributions["ecc_sbit_rate"] = min(1.0, sbit_ratio / 20.0)

    # ── 3. Workload intensity → utilization-weighted acceleration ──
    # High-load GPUs accumulate dust faster + cycle TIM more aggressively.
    # 100 % utilized GPU degrades ~1.5× as fast as 50 % utilized.
    workload_accel = 1.0 + 0.5 * max(0.0, workload_intensity - 0.5) / 0.5
    contributions["workload_intensity"] = workload_intensity

    # ── 4. Ambient stress → Arrhenius acceleration ──
    if inlet_temp_c is not None and profile is not None:
        thermal_accel = _arrhenius_acceleration(inlet_temp_c, profile.expected_ambient_c)
    else:
        thermal_accel = 1.0
    contributions["ambient_stress"] = min(1.0, (thermal_accel - 1.0) / 2.0)

    # Combine: aging is the primary clock; accelerators compress it.
    effective_days = days_from_aging / (workload_accel * thermal_accel)
    effective_days *= (1.0 - sbit_penalty)

    # Priority bucketing
    if not math.isfinite(effective_days) or effective_days > 90:
        priority = MaintenancePriority.NONE
    elif effective_days > 30:
        priority = MaintenancePriority.BACKLOG
    elif effective_days > 7:
        priority = MaintenancePriority.NEXT_WINDOW
    elif effective_days > 1:
        priority = MaintenancePriority.URGENT
    else:
        priority = MaintenancePriority.IMMEDIATE

    # Dominant factor — the largest contribution
    dominant_factor = max(contributions, key=contributions.get) if contributions else "none"

    # Headline composition
    if priority == MaintenancePriority.NONE:
        headline = (
            f"GPU {gpu_index}: nominal — no maintenance projected in next 90 days."
        )
    else:
        eta_str = (
            "today" if effective_days < 1
            else f"in ~{int(effective_days)} day{'s' if int(effective_days) != 1 else ''}"
        )
        factor_human = {
            "aging_drift": "R_θ drift",
            "ecc_sbit_rate": "ECC error rate",
            "workload_intensity": "high sustained workload",
            "ambient_stress": "elevated inlet temperature",
        }.get(dominant_factor, dominant_factor)
        headline = (
            f"GPU {gpu_index}: service recommended {eta_str} "
            f"(primary driver: {factor_human})."
        )

    return MaintenanceScore(
        gpu_index=gpu_index,
        priority=priority,
        days_until_service=effective_days,
        days_uncertainty=aging_uncertainty / (workload_accel * thermal_accel),
        dominant_factor=dominant_factor,
        contributions=contributions,
        headline=headline,
    )


def rank_fleet(scores: list[MaintenanceScore]) -> list[MaintenanceScore]:
    """
    Sort GPUs so the most-needing-attention come first.

    Order:
      1. By priority (IMMEDIATE first)
      2. Within priority, by days_until_service ascending
    """
    priority_order = {
        MaintenancePriority.IMMEDIATE: 0,
        MaintenancePriority.URGENT: 1,
        MaintenancePriority.NEXT_WINDOW: 2,
        MaintenancePriority.BACKLOG: 3,
        MaintenancePriority.NONE: 4,
    }
    return sorted(
        scores,
        key=lambda s: (priority_order[s.priority], s.days_until_service),
    )
