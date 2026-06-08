"""
Fleet-wide event correlator.

Detects patterns across multiple GPUs simultaneously — shared cooling
degradation, power delivery issues, or thermal environment changes that
single-GPU monitoring cannot distinguish from individual failures.

When ≥2 GPUs enter anomalous states within the same polling cycle,
emits a fleet-level AlertEvent (gpu_index=-1) so operators can distinguish
"one GPU is acting up" from "something is wrong with the whole node."

DGX / NVLink note: on DGX nodes all GPUs share an NVSwitch fabric. Heavy
all-to-all NVLink traffic (e.g. tensor parallelism across 8 GPUs) causes
all GPUs to heat up simultaneously. This looks like a fleet cooling failure
but is actually correlated workload. When all anomalous GPUs are on the same
NVLink fabric, the alert message is downgraded and the root-cause hypothesis
shifts from "shared cooling" to "correlated NVLink load" so operators don't
drain nodes unnecessarily during legitimate large-scale training runs.
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

# GPU families with NVLink fabrics — correlated anomalies on these are more
# likely to be workload-driven than cooling failures.
_NVLINK_FAMILIES = {"hopper", "blackwell", "ampere"}


def _detect_nvlink_topology(gpu_names: dict[int, str]) -> bool:
    """Return True if the monitored GPUs appear to be on a shared NVLink fabric.

    Uses hw_profiles family names rather than live NVML NVLink queries so
    this works without extra permissions and doesn't add NVML call overhead
    per tick. Conservative: only flags families where NVLink is standard.
    """
    if not gpu_names:
        return False
    try:
        from .hw_profiles import resolve_or_default
        families = {
            resolve_or_default(name).family
            for name in gpu_names.values()
            if name
        }
        return bool(families & _NVLINK_FAMILIES)
    except Exception:
        return False


@dataclass
class FleetCorrelator:
    """
    Cross-GPU anomaly correlator. Stateless except for alert cooldown.

    Call check() once per polling tick, after all per-GPU samples are processed.
    Returns an AlertEvent (gpu_index=-1) if a fleet-level pattern is detected,
    else None.

    Pass gpu_names to enable NVLink-topology-aware message generation so DGX
    operators don't drain nodes during legitimate correlated training workloads.
    """
    min_gpus:  int = MIN_ANOMALOUS_GPUS
    gpu_names: dict[int, str] = field(default_factory=dict, repr=False)
    _last_alert_ts: Optional[float] = field(default=None, init=False, repr=False)
    _nvlink_node:   Optional[bool]  = field(default=None, init=False, repr=False)

    def register_gpu_names(self, names: dict[int, str]) -> None:
        """Call once after GPU discovery so topology detection works."""
        self.gpu_names = names
        self._nvlink_node = None   # reset cached detection

    def _is_nvlink_node(self) -> bool:
        if self._nvlink_node is None:
            self._nvlink_node = _detect_nvlink_topology(self.gpu_names)
        return self._nvlink_node

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

        # On NVLink nodes, all-GPU simultaneous anomalies are likely correlated
        # workload (tensor parallel, all-reduce) not shared cooling failure.
        # Distinguish so operators don't unnecessarily drain the node.
        all_anomalous = len(anomalous) == len(gpu_states)
        nvlink_correlated = self._is_nvlink_node() and all_anomalous

        if nvlink_correlated:
            message = (
                f"[{severity.upper()}] Fleet event: {len(anomalous)}/{len(gpu_states)} GPUs "
                f"({pct}%) simultaneously anomalous {gpu_list}. "
                f"NVLink fabric detected — all GPUs affected. "
                f"LIKELY CAUSE: correlated NVLink/all-reduce workload heat, not shared cooling failure. "
                f"Verify chassis inlet temperature and fan RPMs before draining. "
                f"If thermals are normal, this is workload-driven and resolves when the job finishes."
            )
            context_extra = {"nvlink_correlated": True, "drain_recommended": False}
        else:
            message = (
                f"[{severity.upper()}] Fleet event: {len(anomalous)}/{len(gpu_states)} GPUs "
                f"({pct}%) simultaneously anomalous {gpu_list}. "
                f"Possible shared cooling path or power delivery issue — "
                f"check node-level thermals, fans, and PDU."
            )
            context_extra = {"nvlink_correlated": False, "drain_recommended": severity == "critical"}

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
                **context_extra,
            },
        )
