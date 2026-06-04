"""
Fleet-wide event correlator.

Detects patterns across multiple GPUs simultaneously — shared cooling
degradation, power delivery issues, or thermal environment changes that
single-GPU monitoring cannot distinguish from individual failures.

When ≥2 GPUs enter anomalous states within the same polling cycle,
emits a fleet-level AlertEvent (gpu_index=-1) so operators can distinguish
"one GPU is acting up" from "something is wrong with the whole node."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .metrics import GPUState, AlertEvent

_ANOMALOUS = {
    GPUState.ZOMBIE_RECOVERY,
    GPUState.DRIFTING,
    GPUState.CRITICAL,
    GPUState.CHILD_EXIT_RECOVERY,
}

FLEET_COOLDOWN_S  = 120   # seconds between repeated fleet alerts
MIN_ANOMALOUS_GPUS = 2    # minimum GPUs in anomalous state to trigger


@dataclass
class FleetCorrelator:
    """
    Cross-GPU anomaly correlator. Stateless except for alert cooldown.

    Call check() once per polling tick, after all per-GPU samples are processed.
    Returns an AlertEvent (gpu_index=-1) if a fleet-level pattern is detected,
    else None.
    """
    min_gpus: int = MIN_ANOMALOUS_GPUS
    _last_alert_ts: Optional[float] = field(default=None, init=False, repr=False)

    def check(
        self,
        gpu_states: dict[int, GPUState],
        timestamp:  float,
    ) -> Optional[AlertEvent]:
        if len(gpu_states) < self.min_gpus:
            return None

        anomalous = {g: s for g, s in gpu_states.items() if s in _ANOMALOUS}

        if len(anomalous) < self.min_gpus:
            return None

        # Cooldown: don't repeat fleet alerts faster than FLEET_COOLDOWN_S
        if self._last_alert_ts and timestamp - self._last_alert_ts < FLEET_COOLDOWN_S:
            return None

        self._last_alert_ts = timestamp

        severity = (
            "critical" if GPUState.CRITICAL in anomalous.values() else "warning"
        )
        gpu_list = sorted(anomalous.keys())
        pct = round(100 * len(anomalous) / len(gpu_states))

        message = (
            f"[{severity.upper()}] Fleet event: {len(anomalous)}/{len(gpu_states)} GPUs "
            f"({pct}%) simultaneously anomalous {gpu_list}. "
            f"Possible shared cooling path or power delivery issue — "
            f"check node-level thermals, fans, and PDU."
        )

        return AlertEvent(
            gpu_index       = -1,    # sentinel: fleet-level event, not single GPU
            timestamp       = timestamp,
            state           = GPUState.CRITICAL if severity == "critical" else GPUState.DRIFTING,
            prev_state      = GPUState.UNKNOWN,
            rtheta          = None,
            rtheta_baseline = None,
            drift_sigma     = None,
            confidence      = 1.0,
            message         = message,
            context         = {
                "severity":     severity,
                "fleet_gpus":   gpu_list,
                "fleet_states": {g: s.name for g, s in anomalous.items()},
                "total_gpus":   len(gpu_states),
            },
        )
