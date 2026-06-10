"""
Degradation risk predictor — physics-informed scoring + online learning.

Phase 1 (current): Rule-based weighted scoring from telemetry trend features.
  Weights derived from hardware reliability research:
  - ECC double-bit = near-certain hardware failure (weight 1.0)
  - ECC single-bit rate climbing = memory degradation (weight 0.6)
  - R_theta slope positive + sigma rising = cooling path degrading (weight 0.5)
  - Clock efficiency declining under load = power/thermal stress (weight 0.4)

Phase 2 (once Cal Poly data accumulates): SGDClassifier trains online on
  (feature_vector, confirmed_outcome) pairs. Labeled by: ops team marks a GPU
  as "failed" → all feature vectors from the preceding 72hr get label=1.

Output: DegradationRisk(score_0_to_1, horizon_str, explanation)
Prometheus: theta_degradation_risk{gpu, horizon}

The score is intentionally conservative in Phase 1. A score of 0.80 should
mean "we are fairly confident something is wrong," not "we saw one bad reading."
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .metrics import GPUState, AlertEvent
from .detector import DriftResult

WINDOW_SIZE       = 60    # feature history (number of stable windows)
MIN_WINDOWS       = 20    # minimum before scoring activates
ALERT_THRESHOLD   = 0.65  # emit alert above this
ALERT_COOLDOWN_S  = 600   # 10-min minimum between risk alerts per GPU

# Phase 1 weights (physics-informed, tuned from Meta/Delta arxiv research)
W_ECC_DBIT        = 1.00  # any uncorrectable ECC = critical hardware damage
W_ECC_SBIT_RATE   = 0.60  # rising single-bit rate = early memory cell death
W_RTHETA_SLOPE    = 0.50  # positive R_theta trend = cooling path degrading
W_SIGMA_RISING    = 0.45  # drift sigma increasing = systematic degradation
W_CLOCK_EFF_SLOPE = 0.40  # declining clock efficiency = power/thermal stress


@dataclass
class WindowRecord:
    ts:          float
    rtheta:      Optional[float]
    sigma:       Optional[float]
    ecc_sbit:    int
    ecc_dbit:    int
    clock_eff:   Optional[float]


@dataclass
class DegradationRisk:
    gpu_index:   int
    score:       float          # 0–1
    confidence:  float          # 0–1, based on how much data we have
    explanation: str
    alert_worthy: bool


class FailurePredictor:
    """
    Per-GPU degradation risk scorer.

    Call update() on every stable window. When ready (min_windows reached),
    score() returns a DegradationRisk. The daemon calls maybe_alert() which
    wraps the risk in an AlertEvent if the score exceeds the threshold.
    """

    def __init__(self):
        self._histories:    dict[int, deque]  = {}
        self._last_alert:   dict[int, float]  = {}
        self._sgd_models:   dict[int, object] = {}   # Phase 2 — future
        self._gpu_names:    dict[int, str]    = {}

    def update(
        self,
        gpu_index:  int,
        ts:         float,
        rtheta:     Optional[float],
        drift:      DriftResult,
        ecc_sbit:   int,
        ecc_dbit:   int,
        clock_eff:  Optional[float],
        gpu_name:   str = "",
    ) -> None:
        if gpu_index not in self._histories:
            self._histories[gpu_index] = deque(maxlen=WINDOW_SIZE)
        self._gpu_names[gpu_index] = gpu_name
        self._histories[gpu_index].append(WindowRecord(
            ts        = ts,
            rtheta    = rtheta,
            sigma     = drift.sigma_score,
            ecc_sbit  = ecc_sbit,
            ecc_dbit  = ecc_dbit,
            clock_eff = clock_eff,
        ))

    def score(self, gpu_index: int) -> Optional[DegradationRisk]:
        hist = self._histories.get(gpu_index)
        if hist is None or len(hist) < MIN_WINDOWS:
            return None

        records = list(hist)
        n       = len(records)
        ts_arr  = np.array([r.ts for r in records])
        ts_norm = ts_arr - ts_arr[0]

        contributions: list[tuple[float, str]] = []

        # ── ECC double-bit (immediate, high weight) ───────────────────────────
        dbit_any = any(r.ecc_dbit > 0 for r in records[-5:])
        if dbit_any:
            contributions.append((W_ECC_DBIT, "uncorrectable ECC errors in last 5 windows"))

        # ── ECC single-bit rate ───────────────────────────────────────────────
        # ecc_sbit is a cumulative volatile counter (resets on driver reload),
        # not a per-window rate. sbit_slope (errors/sec, scaled to /hr below)
        # is the actual rate of new errors; sbit_count_now is the raw running
        # total. These are different quantities — adding them (as before)
        # let a large historical count permanently saturate the score even
        # after the error rate returned to zero. Score each on its own scale
        # and take the max.
        sbit_counts = np.array([r.ecc_sbit for r in records], dtype=float)
        sbit_slope  = float(np.polyfit(ts_norm, sbit_counts, 1)[0]) if n >= 5 else 0.0
        sbit_rate_per_hr = max(0.0, sbit_slope * 3600)
        sbit_count_now   = sbit_counts[-1]
        if sbit_rate_per_hr > 0.05 or sbit_count_now > 5:
            rate_norm  = min(1.0, sbit_rate_per_hr / 20.0)   # 20 errors/hr = severe
            count_norm = min(1.0, sbit_count_now / 100.0)    # 100 cumulative = severe
            norm = max(rate_norm, count_norm)
            contributions.append((W_ECC_SBIT_RATE * norm, f"sbit cumulative={sbit_count_now:.0f}, rate={sbit_rate_per_hr:.2f}/hr"))

        # ── R_theta slope ─────────────────────────────────────────────────────
        rthetas = [r.rtheta for r in records if r.rtheta is not None]
        if len(rthetas) >= 10:
            rt_arr  = np.array(rthetas, dtype=float)
            rt_ts   = ts_norm[-len(rthetas):]
            slope   = float(np.polyfit(rt_ts, rt_arr, 1)[0])
            if slope > 0:
                # Normalize: slope of 0.001 C/W per second = very concerning
                norm = min(1.0, slope / 0.001)
                contributions.append((W_RTHETA_SLOPE * norm, f"R_θ rising {slope*3600:.4f} C/W/hr"))

        # ── Drift sigma trend ─────────────────────────────────────────────────
        sigmas = [r.sigma for r in records if r.sigma is not None]
        if len(sigmas) >= 10:
            sig_arr = np.array(sigmas, dtype=float)
            sig_ts  = ts_norm[-len(sigmas):]
            sig_slope = float(np.polyfit(sig_ts, sig_arr, 1)[0])
            if sig_slope > 0 and sig_arr[-1] > 1.0:
                norm = min(1.0, sig_arr[-1] / 3.5)
                contributions.append((W_SIGMA_RISING * norm, f"drift sigma={sig_arr[-1]:.2f}σ and rising"))

        # ── Clock efficiency decline ──────────────────────────────────────────
        clocks = [r.clock_eff for r in records if r.clock_eff is not None]
        if len(clocks) >= 10:
            ck_arr  = np.array(clocks, dtype=float)
            ck_ts   = ts_norm[-len(clocks):]
            ck_slope = float(np.polyfit(ck_ts, ck_arr, 1)[0])
            if ck_slope < -0.0001:   # efficiency declining
                norm = min(1.0, abs(ck_slope) / 0.001)
                contributions.append((W_CLOCK_EFF_SLOPE * norm, f"clock efficiency declining {abs(ck_slope)*3600*100:.2f}%/hr"))

        if not contributions:
            return DegradationRisk(
                gpu_index=gpu_index, score=0.0, confidence=n / WINDOW_SIZE,
                explanation="no degradation signals", alert_worthy=False
            )

        # Combine signals. Noisy-OR (1 - prod(1-w_i)) treats every signal as
        # independent evidence — but R_θ slope, drift sigma, and clock
        # efficiency are largely downstream of the same physical cause
        # (cooling-path degradation), so noisy-OR double-counts correlated
        # evidence and inflates the score. Instead: the strongest signal sets
        # the floor, and each additional signal closes a diminishing fraction
        # of the remaining headroom — corroborating evidence raises
        # confidence without compounding as if each signal were independent.
        raw_weights = sorted((min(w, 0.999) for w, _ in contributions), reverse=True)
        score = raw_weights[0]
        for w in raw_weights[1:]:
            score += w * (1.0 - score) * 0.5
        score = min(1.0, score)

        explanation_parts = [desc for _, desc in sorted(contributions, key=lambda x: -x[0])]
        explanation = " | ".join(explanation_parts[:3])  # top 3

        return DegradationRisk(
            gpu_index    = gpu_index,
            score        = round(score, 3),
            confidence   = round(n / WINDOW_SIZE, 2),
            explanation  = explanation,
            alert_worthy = score >= ALERT_THRESHOLD,
        )

    def maybe_alert(
        self,
        gpu_index: int,
        timestamp: float,
        state:     GPUState,
    ) -> Optional[AlertEvent]:
        risk = self.score(gpu_index)
        if risk is None or not risk.alert_worthy:
            return None

        # Don't pile on if already in a critical/drifting state
        if state in (GPUState.CRITICAL, GPUState.ZOMBIE_RECOVERY):
            return None

        last = self._last_alert.get(gpu_index, 0.0)
        if timestamp - last < ALERT_COOLDOWN_S:
            return None

        self._last_alert[gpu_index] = timestamp

        horizon = (
            "~1hr"  if risk.score >= 0.90 else
            "~6hr"  if risk.score >= 0.80 else
            "~24hr" if risk.score >= 0.70 else
            "~72hr"
        )

        return AlertEvent(
            gpu_index       = gpu_index,
            timestamp       = timestamp,
            state           = state,
            prev_state      = state,
            rtheta          = None,
            rtheta_baseline = None,
            drift_sigma     = None,
            confidence      = risk.confidence,
            message         = (
                f"[WARNING] GPU {gpu_index} — degradation risk score {risk.score:.2f} "
                f"(estimated failure window {horizon}). "
                f"Signals: {risk.explanation}. "
                f"Consider scheduling maintenance before next long training job."
            ),
            context = {
                "severity":       "warning",
                "predictive":     True,
                "degradation_risk": risk.score,
                "horizon":        horizon,
                "signals":        risk.explanation,
                "confidence":     risk.confidence,
            },
        )

    def get_score(self, gpu_index: int) -> float:
        """Current risk score for Prometheus export. 0.0 if not yet active."""
        risk = self.score(gpu_index)
        return risk.score if risk else 0.0
