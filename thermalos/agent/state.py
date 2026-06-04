"""
Per-GPU state machine.

Tracks state transitions and emits AlertEvents with full context
(previous state, duration, R_theta history, drift score).

State machine:
  UNKNOWN → CLEAN_IDLE | UNDER_LOAD   (on first classification)
  CLEAN_IDLE ↔ UNDER_LOAD            (normal work cycle)
  {any} → ZOMBIE_RECOVERY            (CUDA zombie detected)
  {any} → CHILD_EXIT_RECOVERY        (post child-exit thermal lag)
  {healthy} → DRIFTING               (R_theta k·σ above baseline)
  DRIFTING → CRITICAL                (R_theta 3.5σ above baseline)
  {anomalous} → CLEAN_IDLE           (recovery confirmed)

Transitions to anomalous states emit AlertEvents.
Transitions back to healthy states emit recovery AlertEvents.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .metrics import GPUState, AlertEvent, STATE_LABELS, ClassifiedSample
from .detector import DriftResult

ANOMALOUS_STATES = {
    GPUState.ZOMBIE_RECOVERY,
    GPUState.DRIFTING,
    GPUState.CRITICAL,
}

ALERT_WORTHY_TRANSITIONS = {
    # (from, to): severity
    (GPUState.UNDER_LOAD,  GPUState.ZOMBIE_RECOVERY):     "critical",
    (GPUState.CLEAN_IDLE,  GPUState.ZOMBIE_RECOVERY):     "critical",
    (GPUState.UNDER_LOAD,  GPUState.DRIFTING):            "warning",
    (GPUState.CLEAN_IDLE,  GPUState.DRIFTING):            "warning",
    (GPUState.DRIFTING,    GPUState.CRITICAL):            "critical",
    # Recovery events
    (GPUState.ZOMBIE_RECOVERY,     GPUState.CLEAN_IDLE):  "info",
    (GPUState.CHILD_EXIT_RECOVERY, GPUState.CLEAN_IDLE):  "info",
    (GPUState.DRIFTING,            GPUState.UNDER_LOAD):  "info",
    (GPUState.DRIFTING,            GPUState.CLEAN_IDLE):  "info",
    (GPUState.CRITICAL,            GPUState.UNDER_LOAD):  "info",
    (GPUState.CRITICAL,            GPUState.CLEAN_IDLE):  "info",
}

CONTEXT_HISTORY = 10   # samples to attach to alert for explainability


@dataclass
class GPUStateRecord:
    gpu_index:       int
    current_state:   GPUState = GPUState.UNKNOWN
    entered_at:      float    = field(default_factory=time.time)
    last_rtheta:     Optional[float] = None
    last_confidence: float    = 0.0
    history:         deque    = field(default_factory=lambda: deque(maxlen=CONTEXT_HISTORY))

    def duration_sec(self) -> float:
        return time.time() - self.entered_at


class GPUStateMachine:
    """
    Manages per-GPU state and emits AlertEvents on significant transitions.
    """

    def __init__(self):
        self._records: dict[int, GPUStateRecord] = {}

    def _get_or_create(self, gpu_index: int) -> GPUStateRecord:
        if gpu_index not in self._records:
            self._records[gpu_index] = GPUStateRecord(gpu_index=gpu_index)
        return self._records[gpu_index]

    def transition(
        self,
        classified:  ClassifiedSample,
        drift:       DriftResult,
    ) -> Optional[AlertEvent]:
        """
        Update state machine with new classification + drift result.
        Returns an AlertEvent if a notable transition occurred, else None.
        """
        gpu_index  = classified.gpu_index
        new_state  = classified.state
        timestamp  = classified.timestamp
        rtheta     = classified.rtheta_mean
        confidence = classified.confidence

        # Drift overrides classifier state for healthy GPUs
        if drift.is_critical and new_state not in ANOMALOUS_STATES:
            new_state = GPUState.CRITICAL
        elif drift.is_drifting and new_state not in ANOMALOUS_STATES:
            new_state = GPUState.DRIFTING

        rec = self._get_or_create(gpu_index)

        # Record to history always
        rec.history.append({
            "ts":    round(timestamp, 2),
            "state": STATE_LABELS.get(new_state, "unknown"),
            "r":     round(rtheta, 4) if rtheta else None,
            "conf":  round(confidence, 3),
            "sigma": drift.sigma_score,
        })

        prev_state = rec.current_state

        if new_state == prev_state:
            rec.last_rtheta     = rtheta
            rec.last_confidence = confidence
            return None

        # State changed
        rec.current_state   = new_state
        rec.entered_at      = timestamp
        rec.last_rtheta     = rtheta
        rec.last_confidence = confidence

        key = (prev_state, new_state)
        if key not in ALERT_WORTHY_TRANSITIONS:
            return None

        severity = ALERT_WORTHY_TRANSITIONS[key]
        message  = self._make_message(
            gpu_index, prev_state, new_state, rtheta,
            drift, confidence, severity
        )

        return AlertEvent(
            gpu_index        = gpu_index,
            timestamp        = timestamp,
            state            = new_state,
            prev_state       = prev_state,
            rtheta           = rtheta,
            rtheta_baseline  = drift.baseline_mean,
            drift_sigma      = drift.sigma_score,
            confidence       = confidence,
            message          = message,
            context          = {
                "severity":       severity,
                "duration_prev":  round(rec.duration_sec(), 1),
                "history":        list(rec.history),
            },
        )

    def _make_message(
        self,
        gpu_index: int,
        prev: GPUState,
        curr: GPUState,
        rtheta: Optional[float],
        drift: DriftResult,
        conf:  float,
        severity: str,
    ) -> str:
        r_str = f"R_θ={rtheta:.3f} C/W" if rtheta else "R_θ=n/a"

        if curr == GPUState.ZOMBIE_RECOVERY:
            return (
                f"[{severity.upper()}] GPU {gpu_index} — CUDA zombie detected. "
                f"{r_str} at 0% utilisation. CUDA context retained after process termination. "
                f"Action: identify and release stale CUDA context."
            )
        if curr == GPUState.DRIFTING:
            sigma = drift.sigma_score
            base  = drift.baseline_mean
            return (
                f"[{severity.upper()}] GPU {gpu_index} — R_θ drifting. "
                f"{r_str} ({sigma:.1f}σ above baseline {base:.3f} C/W). "
                f"Sustained over {drift.confidence * 3:.0f} windows. "
                f"Possible cooling path degradation."
            )
        if curr == GPUState.CRITICAL:
            sigma = drift.sigma_score
            return (
                f"[CRITICAL] GPU {gpu_index} — thermal anomaly. "
                f"{r_str} ({sigma:.1f}σ above baseline). "
                f"Throttling risk. Check cooling immediately."
            )
        if curr in (GPUState.CLEAN_IDLE, GPUState.UNDER_LOAD):
            return (
                f"[INFO] GPU {gpu_index} — recovered to {STATE_LABELS[curr]}. "
                f"{r_str} · prev state: {STATE_LABELS[prev]}."
            )
        return f"[INFO] GPU {gpu_index} → {STATE_LABELS[curr]}. {r_str}."

    def get_state(self, gpu_index: int) -> GPUState:
        rec = self._records.get(gpu_index)
        return rec.current_state if rec else GPUState.UNKNOWN

    def all_states(self) -> dict[int, GPUStateRecord]:
        return dict(self._records)
