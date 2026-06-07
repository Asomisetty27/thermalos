"""
Fault cause classifier — distinguishes physical failure modes by R_theta curve shape.

Tracks R_theta at two power tiers per GPU:
  LOW_P  (5–25 W) : intercept region — dominated by conduction, not fan-dependent
  HIGH_P (55+ W)  : slope region — fan-dependent cooling

The R_theta(P) curve has two independently variable degrees of freedom:
  intercept  — R_theta at LOW_P (rises with dust, blockage, mounting events)
  gap        — R_theta_low minus R_theta_high (narrows when TIM or fan degrades)

Six fault causes and their curve signatures:
  DUST_ACCUMULATION : intercept drifts up slowly, gap stable (uniform parallel shift)
  TIM_DEGRADATION   : gap narrows over months (slope steepens from high-P end)
  FAN_BEARING_WEAR  : gap narrows above a power threshold, fan RPM declining
  AIRFLOW_BLOCKAGE  : sudden intra-session intercept step-change
  MOUNTING_EVENT    : intercept jumps between sessions (inter-session step)
  HBM_THERMAL       : R_theta elevated only under high memory-bandwidth load

Decision tree requires MIN_BUCKET_SAMPLES per tier and MIN_SNAPSHOTS (24h) for trends.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FaultCause(Enum):
    NOMINAL           = "nominal"
    DUST_ACCUMULATION = "dust_accumulation"
    TIM_DEGRADATION   = "tim_degradation"
    FAN_BEARING_WEAR  = "fan_bearing_wear"
    AIRFLOW_BLOCKAGE  = "airflow_blockage"
    MOUNTING_EVENT    = "mounting_event"
    HBM_THERMAL       = "hbm_thermal"
    INSUFFICIENT_DATA = "insufficient_data"


FAULT_REMEDIATION = {
    FaultCause.NOMINAL:           "No action required.",
    FaultCause.DUST_ACCUMULATION: "Clean heatsink fins and air filters. Schedule during next maintenance window.",
    FaultCause.TIM_DEGRADATION:   "Thermal interface material degrading. Schedule TIM replacement (repaste).",
    FaultCause.FAN_BEARING_WEAR:  "Fan bearing wearing. Replace cooling fan before failure.",
    FaultCause.AIRFLOW_BLOCKAGE:  "Airflow path obstructed. Check cable routing, rack clearance, HVAC.",
    FaultCause.MOUNTING_EVENT:    "Heatsink contact pressure may have changed. Verify mounting hardware after last maintenance.",
    FaultCause.HBM_THERMAL:       "HBM/VRAM thermal issue. Reduce memory-intensive workload frequency or check VRAM cooling.",
    FaultCause.INSUFFICIENT_DATA: "Still collecting samples across both power tiers — no diagnosis yet.",
}

# Power tier boundaries (watts)
LOW_P_MIN,  LOW_P_MAX  = 5.0,  25.0
HIGH_P_MIN             = 55.0

# Rolling windows
BUCKET_MAXLEN      = 120   # samples per tier
SNAPSHOT_INTERVAL  = 3600  # seconds between hourly snapshots
SNAPSHOT_MAXLEN    = 720   # 30 days of hourly snapshots
MIN_BUCKET_SAMPLES = 30    # both tiers before any diagnosis
MIN_SNAPSHOTS      = 24    # one full day before trend diagnosis

# Classification thresholds (calibrated against Stage 1 T4 data)
SESSION_DELTA_THRESH   = 0.08   # C/W — inter-session jump flags mounting event
BLOCKAGE_STEP_THRESH   = 0.07   # C/W — intra-session step flags airflow blockage
DUST_DRIFT_RATE        = 0.001  # C/W per day — slow monotonic drift
GAP_STABLE_BAND        = 0.015  # C/W per day — gap change within this = slope stable
TIM_GAP_RATE           = 0.025  # C/W per day — gap narrowing at this rate = TIM/fan
FAN_RPM_SLOPE_THRESH   = -0.005 # %/s — RPM declining at this rate = fan bearing
HBM_MEM_HIGH           = 70.0  # % mem util — "high memory" threshold
HBM_LIFT_THRESH        = 0.06   # C/W — R_theta lift under high mem vs low mem

# Reporting
REPORT_INTERVAL_S      = 600.0  # emit diagnosis at most every 10 min per GPU
FAULT_REPORT_INTERVAL  = 60.0   # anomalous faults report every 60 s


@dataclass
class FaultDiagnosis:
    gpu_index:    int
    timestamp:    float
    cause:        FaultCause
    confidence:   float          # 0–1
    remediation:  str
    intercept:    Optional[float]  # LOW_P R_theta median (C/W)
    gap:          Optional[float]  # intercept minus HIGH_P median (C/W)
    curve_slope:  Optional[float]  # (R_high - R_low) / (P_high - P_low) (C/W²)
    drift_rate:   Optional[float]  # intercept drift (C/W per day)
    gap_trend:    Optional[float]  # gap narrowing rate (C/W per day)
    session_delta: Optional[float] # this session minus last (C/W)
    evidence:     dict = field(default_factory=dict)


# ── Pure helpers ─────────────────────────────────────────────────────────────

def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _linslope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope. Returns 0 if underdetermined."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 1e-9 else 0.0


# ── Per-GPU state ─────────────────────────────────────────────────────────────

class _CurveTracker:
    """
    Per-GPU curve shape tracker.

    Maintains rolling R_theta samples per power tier, hourly snapshots for
    trend analysis, session bookmarks for inter-session comparison, and HBM
    split samples for workload-correlated diagnosis.
    """

    __slots__ = (
        "_low", "_low_p", "_high", "_high_p",
        "_hi_mem", "_lo_mem",
        "_fan_buf",
        "_snapshots", "_last_snap_ts",
        "_sess_start_ts", "_sess_warmup", "_sess_warmup_done",
        "_sess_start_rtheta", "_prev_sess_rtheta",
        "_recent_low_at_sess_start",
        "_last_diag_ts", "_emitted_insufficient",
    )

    def __init__(self) -> None:
        self._low:   deque[float] = deque(maxlen=BUCKET_MAXLEN)
        self._low_p: deque[float] = deque(maxlen=BUCKET_MAXLEN)
        self._high:  deque[float] = deque(maxlen=BUCKET_MAXLEN)
        self._high_p:deque[float] = deque(maxlen=BUCKET_MAXLEN)

        # HBM split — high vs low memory utilisation at HIGH_P
        self._hi_mem: deque[float] = deque(maxlen=60)
        self._lo_mem: deque[float] = deque(maxlen=60)

        # Fan speed % — (ts, pct) pairs for RPM trend
        self._fan_buf: deque[tuple[float, float]] = deque(maxlen=120)

        # Hourly snapshots: (ts_days_from_start, intercept, gap)
        self._snapshots:    deque[tuple[float, float, float]] = deque(maxlen=SNAPSHOT_MAXLEN)
        self._last_snap_ts: float = 0.0

        # Session tracking
        self._sess_start_ts:      float          = time.monotonic()
        self._sess_warmup:        list[float]    = []
        self._sess_warmup_done:   bool           = False
        self._sess_start_rtheta:  Optional[float] = None
        self._prev_sess_rtheta:   Optional[float] = None

        self._last_diag_ts:          float = 0.0
        self._emitted_insufficient:  bool  = False

    def ingest(self, ts: float, rtheta: float, power_w: float,
               mem_util: float, fan_pct: Optional[float]) -> None:
        in_low  = LOW_P_MIN <= power_w < LOW_P_MAX
        in_high = power_w >= HIGH_P_MIN

        if in_low:
            self._low.append(rtheta)
            self._low_p.append(power_w)
            # Session warmup bookmark: median of first 5 minutes of LOW_P samples
            if not self._sess_warmup_done:
                elapsed = time.monotonic() - self._sess_start_ts
                if elapsed < 300.0:
                    self._sess_warmup.append(rtheta)
                elif self._sess_warmup:
                    self._sess_start_rtheta = _median(self._sess_warmup)
                    self._sess_warmup_done  = True

        if in_high:
            self._high.append(rtheta)
            self._high_p.append(power_w)
            if mem_util >= HBM_MEM_HIGH:
                self._hi_mem.append(rtheta)
            elif mem_util < 30.0:
                self._lo_mem.append(rtheta)

        if fan_pct is not None:
            self._fan_buf.append((ts, fan_pct))

        # Hourly snapshot
        if ts - self._last_snap_ts >= SNAPSHOT_INTERVAL:
            ic, gap = self._curve_stats()
            if ic is not None and gap is not None:
                # Store time in days from first snapshot for slope computation
                origin = self._snapshots[0][0] if self._snapshots else 0.0
                self._snapshots.append((origin + (ts - self._last_snap_ts) / 86400.0,
                                        ic, gap))
                self._last_snap_ts = ts

    def new_session(self) -> None:
        if self._sess_warmup_done and self._sess_start_rtheta is not None:
            self._prev_sess_rtheta = self._sess_start_rtheta
        self._sess_start_ts     = time.monotonic()
        self._sess_warmup       = []
        self._sess_warmup_done  = False
        self._sess_start_rtheta = None

    def _curve_stats(self) -> tuple[Optional[float], Optional[float]]:
        """(intercept, gap) — None if insufficient data in either tier."""
        if len(self._low) < MIN_BUCKET_SAMPLES or len(self._high) < MIN_BUCKET_SAMPLES:
            return None, None
        ic  = _median(list(self._low))
        rhi = _median(list(self._high))
        return round(ic, 4), round(ic - rhi, 4)

    def _curve_slope(self) -> Optional[float]:
        """Physical slope (C/W per W): (R_high - R_low) / (P_high - P_low)."""
        if len(self._low_p) < MIN_BUCKET_SAMPLES or len(self._high_p) < MIN_BUCKET_SAMPLES:
            return None
        p_lo  = _median(list(self._low_p))
        p_hi  = _median(list(self._high_p))
        r_lo  = _median(list(self._low))
        r_hi  = _median(list(self._high))
        dp = p_hi - p_lo
        return round((r_hi - r_lo) / dp, 6) if dp > 5.0 else None

    def _trend_rates(self) -> tuple[Optional[float], Optional[float]]:
        """(intercept_drift_rate, gap_trend_rate) in C/W per day."""
        if len(self._snapshots) < MIN_SNAPSHOTS:
            return None, None
        snaps = list(self._snapshots)
        xs          = [s[0] for s in snaps]
        intercepts  = [s[1] for s in snaps]
        gaps        = [s[2] for s in snaps]
        # Normalise xs to [0, N_days]
        x0 = xs[0]
        xs = [x - x0 for x in xs]
        return (
            round(_linslope(xs, intercepts), 5),
            round(_linslope(xs, gaps), 5),
        )

    def _fan_declining(self) -> bool:
        if len(self._fan_buf) < 40:
            return False
        pairs = list(self._fan_buf)
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        return _linslope(xs, ys) < FAN_RPM_SLOPE_THRESH

    def _hbm_lift(self) -> Optional[float]:
        if len(self._hi_mem) < 15 or len(self._lo_mem) < 15:
            return None
        return round(_median(list(self._hi_mem)) - _median(list(self._lo_mem)), 4)

    def diagnose(self, gpu_index: int, ts: float) -> FaultDiagnosis:
        ic, gap = self._curve_stats()

        if ic is None:
            return FaultDiagnosis(
                gpu_index=gpu_index, timestamp=ts,
                cause=FaultCause.INSUFFICIENT_DATA, confidence=0.0,
                remediation=FAULT_REMEDIATION[FaultCause.INSUFFICIENT_DATA],
                intercept=None, gap=None, curve_slope=None,
                drift_rate=None, gap_trend=None, session_delta=None,
            )

        drift_rate, gap_trend = self._trend_rates()
        curve_slope           = self._curve_slope()
        hbm_lift              = self._hbm_lift()

        # Inter-session delta (mounting event detection)
        session_delta = None
        if self._prev_sess_rtheta and self._sess_start_rtheta:
            session_delta = round(self._sess_start_rtheta - self._prev_sess_rtheta, 4)

        # Intra-session delta (airflow blockage detection)
        intra_delta = None
        if self._sess_start_rtheta and len(self._low) >= MIN_BUCKET_SAMPLES:
            intra_delta = round(ic - self._sess_start_rtheta, 4)

        evidence = {
            "intercept_cwatt":  ic,
            "gap_cwatt":        gap,
            "curve_slope":      curve_slope,
            "drift_rate_per_day": drift_rate,
            "gap_trend_per_day":  gap_trend,
            "session_delta":    session_delta,
            "intra_delta":      intra_delta,
            "hbm_lift":         hbm_lift,
            "low_p_samples":    len(self._low),
            "high_p_samples":   len(self._high),
        }

        # ── Decision tree (priority ordered) ─────────────────────────────────
        cause, confidence = FaultCause.NOMINAL, 0.0

        if session_delta is not None and session_delta > SESSION_DELTA_THRESH:
            # Inter-session step: heatsink contact pressure changed
            cause      = FaultCause.MOUNTING_EVENT
            confidence = min(1.0, session_delta / (SESSION_DELTA_THRESH * 2))

        elif (intra_delta is not None and intra_delta > BLOCKAGE_STEP_THRESH
              and (gap_trend is None or abs(gap_trend) < GAP_STABLE_BAND)):
            # Intra-session step, gap stable: airflow obstruction
            cause      = FaultCause.AIRFLOW_BLOCKAGE
            confidence = min(1.0, intra_delta / (BLOCKAGE_STEP_THRESH * 2))

        elif (drift_rate is not None and drift_rate > DUST_DRIFT_RATE
              and gap_trend is not None and abs(gap_trend) < GAP_STABLE_BAND):
            # Slow uniform intercept drift, gap stable: dust accumulation
            cause      = FaultCause.DUST_ACCUMULATION
            confidence = min(1.0, drift_rate / (DUST_DRIFT_RATE * 5))

        elif gap_trend is not None and gap_trend < -TIM_GAP_RATE:
            # Gap narrowing: TIM or fan
            mag = abs(gap_trend) / TIM_GAP_RATE
            if self._fan_declining():
                cause      = FaultCause.FAN_BEARING_WEAR
                confidence = min(1.0, mag * 0.9)
            else:
                cause      = FaultCause.TIM_DEGRADATION
                confidence = min(1.0, mag * 0.8)

        elif hbm_lift is not None and hbm_lift > HBM_LIFT_THRESH:
            # Workload-correlated R_theta elevation: HBM thermal
            cause      = FaultCause.HBM_THERMAL
            confidence = min(1.0, hbm_lift / (HBM_LIFT_THRESH * 2))

        if cause != FaultCause.NOMINAL:
            evidence["decision"] = cause.value

        return FaultDiagnosis(
            gpu_index    = gpu_index,
            timestamp    = ts,
            cause        = cause,
            confidence   = round(confidence, 3),
            remediation  = FAULT_REMEDIATION[cause],
            intercept    = ic,
            gap          = gap,
            curve_slope  = curve_slope,
            drift_rate   = drift_rate,
            gap_trend    = gap_trend,
            session_delta = session_delta,
            evidence     = evidence,
        )


# ── Fleet interface ───────────────────────────────────────────────────────────

class FaultCurveClassifier:
    """
    Fleet-level fault curve classifier.

    Call update() on every stable steady-state window. Returns a FaultDiagnosis
    when the reporting interval has elapsed, or immediately on any non-NOMINAL
    cause at the faster FAULT_REPORT_INTERVAL rate.
    """

    def __init__(self) -> None:
        self._trackers: dict[int, _CurveTracker] = {}

    def _tracker(self, gpu: int) -> _CurveTracker:
        if gpu not in self._trackers:
            self._trackers[gpu] = _CurveTracker()
        return self._trackers[gpu]

    def update(
        self,
        gpu_index: int,
        ts:        float,
        rtheta:    float,
        power_w:   float,
        mem_util:  float,
        fan_pct:   Optional[float] = None,
    ) -> Optional[FaultDiagnosis]:
        t = self._tracker(gpu_index)
        t.ingest(ts, rtheta, power_w, mem_util, fan_pct)

        since = ts - t._last_diag_ts
        diagnosis = t.diagnose(gpu_index, ts)

        if diagnosis.cause == FaultCause.INSUFFICIENT_DATA:
            if t._emitted_insufficient:
                return None
            t._emitted_insufficient = True
            t._last_diag_ts = ts
            return diagnosis

        is_fault = diagnosis.cause != FaultCause.NOMINAL
        interval = FAULT_REPORT_INTERVAL if is_fault else REPORT_INTERVAL_S
        if since < interval:
            return None

        t._last_diag_ts = ts
        return diagnosis

    def notify_new_session(self, gpu_index: int) -> None:
        """Call when the agent restarts or a GPU driver reset is detected."""
        self._tracker(gpu_index).new_session()

    def get_current(self, gpu_index: int) -> Optional[FaultDiagnosis]:
        """Snapshot the current diagnosis without advancing the report timer."""
        if gpu_index not in self._trackers:
            return None
        t = self._trackers[gpu_index]
        return t.diagnose(gpu_index, time.time())
