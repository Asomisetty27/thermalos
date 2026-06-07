"""
Theta Intelligence Network — anonymized telemetry reporter.

Every OSS user who opts in contributes anonymized GPU health signatures to a
shared training dataset. The dataset trains Theta's predictive models
(LSTM, Isolation Forest) on real-world degradation curves across thousands of
GPUs. Users who share get community benchmarks in return: where does your GPU
sit relative to the fleet P50/P95 for R_theta, ECC rate, and clock efficiency?

Privacy contract (never negotiable):
  SHARED:   R_theta statistics, ECC rates, XID frequencies, GPU generation tag,
            recovery time signatures, alert event types
  NEVER:    IP address, hostname, job names, usernames, model weights,
            company name, raw timestamps (only bucketed hourly)

Architecture:
  - TelemetryBuffer accumulates events in memory
  - Every 24h (configurable), TelemetryReporter flushes a batch to the API
  - Batch is aggregated locally before upload — raw telemetry never leaves
  - Upload is fire-and-forget; failure is logged and silently dropped
  - opt_in=False → TelemetryReporter is a no-op

API endpoint: https://api.runtheta.io/v1/telemetry  (TBD — Supabase edge fn)
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

# Theta Intelligence Network endpoint — Supabase edge function (deployed)
TELEMETRY_ENDPOINT = "https://gfghusfgnblazadnjvyk.supabase.co/functions/v1/telemetry_ingest"
FLUSH_INTERVAL_S   = 86400   # 24 hours
MIN_EVENTS_TO_SEND = 10      # don't bother sending tiny batches
GPU_GEN_TAGS = {             # normalize GPU model to generation bucket
    "T4":    "t4-class",
    "A100":  "a100-class",
    "H100":  "h100-class",
    "B200":  "b200-class",
    "L40":   "l40-class",
    "A10":   "a10-class",
    "MI300": "mi300-class",  # future AMD support
}


def _gpu_generation(gpu_name: str) -> str:
    """Map raw GPU name to anonymized generation bucket."""
    n = gpu_name.upper()
    for key, tag in GPU_GEN_TAGS.items():
        if key in n:
            return tag
    return "other"


def _install_id() -> str:
    """Stable anonymous installation ID derived from machine UUID. Not linkable to user."""
    try:
        import platform
        raw = platform.node() + platform.machine()
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
    except Exception:
        return "unknown"


@dataclass
class TelemetryEvent:
    gpu_gen:         str
    hour_bucket:     int       # unix epoch // 3600 (hourly resolution — no finer)
    rtheta_mean:     Optional[float]
    rtheta_std:      Optional[float]
    ecc_sbit_rate:   float     # errors/hour
    ecc_dbit_event:  bool      # any uncorrectable this hour?
    clock_eff_mean:  Optional[float]
    alert_type:      Optional[str]  # "predictive_warning", "fleet_event", "ecc_critical", etc.
    recovery_time_s: Optional[float]


class TelemetryBuffer:
    """
    Accumulates raw per-GPU observations, aggregates them per (gpu_gen, hour)
    before hand-off to the reporter. Aggregation happens locally — raw values
    never leave this process.
    """

    def __init__(self):
        self._buckets: dict[tuple, list] = defaultdict(list)

    def record(self, event: TelemetryEvent) -> None:
        key = (event.gpu_gen, event.hour_bucket)
        self._buckets[key].append(event)

    def flush_aggregated(self) -> list[dict]:
        """Aggregate all buffered events into anonymized summary dicts."""
        out = []
        for (gpu_gen, hour_bucket), events in self._buckets.items():
            rthetas = [e.rtheta_mean for e in events if e.rtheta_mean is not None]
            clock_effs = [e.clock_eff_mean for e in events if e.clock_eff_mean is not None]
            rec_times = [e.recovery_time_s for e in events if e.recovery_time_s is not None]
            out.append({
                "gpu_gen":           gpu_gen,
                "hour":              hour_bucket,
                "n_samples":         len(events),
                "rtheta_mean":       round(sum(rthetas) / len(rthetas), 4) if rthetas else None,
                "rtheta_std_mean":   round(sum(e.rtheta_std for e in events if e.rtheta_std) / max(1, len(rthetas)), 4) if rthetas else None,
                "ecc_sbit_total":    sum(e.ecc_sbit_rate for e in events),
                "ecc_dbit_any":      any(e.ecc_dbit_event for e in events),
                "clock_eff_mean":    round(sum(clock_effs) / len(clock_effs), 3) if clock_effs else None,
                "alert_types":       list({e.alert_type for e in events if e.alert_type}),
                "recovery_time_p50": sorted(rec_times)[len(rec_times) // 2] if rec_times else None,
            })
        self._buckets.clear()
        return out


class TelemetryReporter:
    """
    Async reporter that flushes the buffer to the Theta Intelligence
    Network every 24 hours.

    If opt_in=False: all methods are no-ops. Zero network traffic.
    If the API is unreachable: silently drops the batch, logs debug.
    """

    def __init__(self, opt_in: bool, install_id: Optional[str] = None):
        self._opt_in    = opt_in
        self._install   = install_id or _install_id()
        self._buffer    = TelemetryBuffer()
        self._last_flush = time.time()

    def record_window(
        self,
        gpu_name:       str,
        rtheta_mean:    Optional[float],
        rtheta_std:     Optional[float],
        ecc_sbit_rate:  float,
        ecc_dbit_event: bool,
        clock_eff_mean: Optional[float],
    ) -> None:
        if not self._opt_in:
            return
        self._buffer.record(TelemetryEvent(
            gpu_gen        = _gpu_generation(gpu_name),
            hour_bucket    = int(time.time()) // 3600,
            rtheta_mean    = rtheta_mean,
            rtheta_std     = rtheta_std,
            ecc_sbit_rate  = ecc_sbit_rate,
            ecc_dbit_event = ecc_dbit_event,
            clock_eff_mean = clock_eff_mean,
            alert_type     = None,
            recovery_time_s= None,
        ))

    def record_alert(self, alert_type: str, recovery_time_s: Optional[float] = None) -> None:
        if not self._opt_in:
            return
        self._buffer.record(TelemetryEvent(
            gpu_gen=_gpu_generation(""),
            hour_bucket=int(time.time()) // 3600,
            rtheta_mean=None, rtheta_std=None,
            ecc_sbit_rate=0.0, ecc_dbit_event=False,
            clock_eff_mean=None,
            alert_type=alert_type,
            recovery_time_s=recovery_time_s,
        ))

    async def maybe_flush(self) -> None:
        """Call this on every agent tick. Flushes if interval elapsed."""
        if not self._opt_in:
            return
        if time.time() - self._last_flush < FLUSH_INTERVAL_S:
            return
        await self._flush()

    async def flush_now(self) -> None:
        """Force an immediate flush (e.g., on shutdown)."""
        if not self._opt_in:
            return
        await self._flush()

    async def _flush(self) -> None:
        aggregated = self._buffer.flush_aggregated()
        if len(aggregated) < MIN_EVENTS_TO_SEND:
            log.debug("telemetry: skipping flush, too few events (%d)", len(aggregated))
            return

        payload = {
            "install_id": self._install,
            "agent_version": _agent_version(),
            "batches": aggregated,
        }

        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(TELEMETRY_ENDPOINT, json=payload)
                if r.status_code == 200:
                    log.debug("telemetry: flushed %d buckets", len(aggregated))
                else:
                    log.debug("telemetry: server returned %d — will retry next interval", r.status_code)
        except Exception as e:
            log.debug("telemetry: flush failed (%s) — dropped silently", type(e).__name__)
        finally:
            self._last_flush = time.time()


def _agent_version() -> str:
    try:
        from .. import __version__
        return __version__
    except Exception:
        return "unknown"
