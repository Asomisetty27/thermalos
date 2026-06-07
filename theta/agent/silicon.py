"""
Silicon-level health monitors: ECC and micro-throttling.

These two detectors operate below the R_theta layer — they catch degradation
signals that temperature and power alone cannot see.

EccMonitor
  Double-bit (uncorrectable) ECC errors → CRITICAL immediately. Even one is
  a definitive sign of physical silicon damage. Single-bit rate tracking
  catches memory cell degradation before double-bit errors appear.

MicroThrottleDetector
  Compares actual SM clock to the GPU's max boost clock. When a GPU is under
  heavy load (util > 80%) but the SM clock is suppressed below 95% of boost,
  the driver is applying a throttle. If that persists for 5+ consecutive
  samples, it emits a WARNING with the NVML throttle reason decoded — telling
  the operator whether it's thermal, power cap, reliability voltage, or sync.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .metrics import RawSample, AlertEvent, GPUState

# ECC thresholds
SBIT_RATE_WARN_PER_HOUR = 10

# Micro-throttle thresholds
EFFICIENCY_THRESHOLD   = 0.95    # sm_clock / sm_clock_max below this = suppressed
LOAD_THRESHOLD         = 80.0    # only detect when GPU is genuinely loaded
SUSTAINED_SAMPLES      = 5       # consecutive samples before alerting
THROTTLE_COOLDOWN_S    = 300     # seconds between repeated micro-throttle alerts

# NVML throttle reason bitmask → human name
_THROTTLE_BITS: dict[int, str] = {
    0x0000000000000004: "sw_power_cap",
    0x0000000000000008: "hw_slowdown",
    0x0000000000000020: "sw_thermal_slowdown",
    0x0000000000000040: "hw_thermal_slowdown",
    0x0000000000000080: "hw_power_brake",
    0x0000000000000010: "sync_boost",
    0x0000000000000002: "app_clock_setting",
}


def decode_throttle_reasons(bitmask: int) -> list[str]:
    return [name for bit, name in _THROTTLE_BITS.items() if bitmask & bit]


class EccMonitor:
    """
    Tracks ECC error deltas per polling interval.

    Emits AlertEvent on:
    - Any double-bit (uncorrectable) error increase → CRITICAL
    - Single-bit rate >= SBIT_RATE_WARN_PER_HOUR → WARNING (once per hour)
    """

    def __init__(self, sbit_rate_warn: float = SBIT_RATE_WARN_PER_HOUR):
        self._sbit_rate_warn   = sbit_rate_warn
        self._last_sbit:       dict[int, int]   = {}
        self._last_dbit:       dict[int, int]   = {}
        self._sbit_history:    dict[int, deque] = {}
        self._sbit_alert_ts:   dict[int, float] = {}

    def update(self, sample: RawSample) -> Optional[AlertEvent]:
        gpu = sample.gpu_index
        ts  = sample.timestamp

        prev_sbit = self._last_sbit.get(gpu)
        prev_dbit = self._last_dbit.get(gpu)
        self._last_sbit[gpu] = sample.ecc_sbit
        self._last_dbit[gpu] = sample.ecc_dbit

        if prev_sbit is None:   # first sample — establish baseline
            return None

        sbit_delta = max(0, sample.ecc_sbit - prev_sbit)
        dbit_delta = max(0, sample.ecc_dbit - prev_dbit)

        # Double-bit uncorrectable: any increase is CRITICAL
        if dbit_delta > 0:
            return AlertEvent(
                gpu_index       = gpu,
                timestamp       = ts,
                state           = GPUState.CRITICAL,
                prev_state      = GPUState.UNKNOWN,
                rtheta          = None,
                rtheta_baseline = None,
                drift_sigma     = None,
                confidence      = 1.0,
                message         = (
                    f"[CRITICAL] GPU {gpu} — {dbit_delta} uncorrectable ECC error(s) "
                    f"this interval. Double-bit errors indicate physical silicon damage. "
                    f"Total volatile dbit: {sample.ecc_dbit}. "
                    f"Evacuate workloads immediately."
                ),
                context = {
                    "severity":   "critical",
                    "ecc_dbit":   dbit_delta,
                    "ecc_sbit":   sbit_delta,
                    "total_dbit": sample.ecc_dbit,
                    "total_sbit": sample.ecc_sbit,
                },
            )

        # Single-bit rate tracking — rolling 1h window
        if gpu not in self._sbit_history:
            self._sbit_history[gpu] = deque()
        if sbit_delta > 0:
            self._sbit_history[gpu].append((ts, sbit_delta))

        cutoff = ts - 3600
        while self._sbit_history[gpu] and self._sbit_history[gpu][0][0] < cutoff:
            self._sbit_history[gpu].popleft()

        rate = sum(c for _, c in self._sbit_history[gpu])
        last_alert = self._sbit_alert_ts.get(gpu, 0.0)
        if rate >= self._sbit_rate_warn and ts - last_alert >= 3600:
            self._sbit_alert_ts[gpu] = ts
            return AlertEvent(
                gpu_index       = gpu,
                timestamp       = ts,
                state           = GPUState.DRIFTING,
                prev_state      = GPUState.UNKNOWN,
                rtheta          = None,
                rtheta_baseline = None,
                drift_sigma     = None,
                confidence      = 0.8,
                message         = (
                    f"[WARNING] GPU {gpu} — elevated single-bit ECC rate: "
                    f"{rate:.0f} corrections/hour (threshold: {self._sbit_rate_warn}). "
                    f"Early memory cell degradation. Double-bit failure risk is rising."
                ),
                context = {
                    "severity":       "warning",
                    "ecc_sbit_rate":  rate,
                    "ecc_sbit_total": sample.ecc_sbit,
                },
            )

        return None


class XIDParser:
    """
    Parses the kernel ring buffer for NVIDIA XID error events and classifies
    them by severity and failure type.

    XID semantic categories (from NVIDIA XID Errors r590 docs):
      SDC_PRECURSOR   : XID 13 — graphics engine exception (known SDC precursor)
      MEMORY_ERROR    : XID 48, 63, 64, 92, 94, 95 — GPU memory faults
      GPU_RESET       : XID 79 — GPU needs reset
      FALLEN_OFF_BUS  : XID 31 — GPU unresponsive (PCI bus error)
      ECC_UNCORRECTED : XID 48 — uncorrectable ECC
      COMPUTE_ERROR   : XID 61, 62, 68 — CUDA/SM compute exceptions

    Reads dmesg output (no root required on most Linux systems).
    Polls once per POLL_INTERVAL_S; caches results between polls.
    """

    POLL_INTERVAL_S = 60

    _SDC_PRECURSOR  = {13}
    _MEMORY_ERRORS  = {48, 63, 64, 92, 94, 95}
    _GPU_RESET      = {79}
    _FALLEN_OFF_BUS = {31}
    _COMPUTE_ERRORS = {61, 62, 68}

    def __init__(self):
        self._last_poll:   float = 0.0
        self._last_counts: dict[int, dict[int, int]] = {}  # gpu → xid → count
        self._prev_counts: dict[int, dict[int, int]] = {}

    def _read_dmesg(self) -> dict[int, dict[int, int]]:
        """Parse dmesg for NVRM XID lines. Returns {gpu_idx: {xid: count}}."""
        import re
        import subprocess
        counts: dict[int, dict[int, int]] = {}
        try:
            result = subprocess.run(
                ["dmesg", "--since", "-10min"],
                capture_output=True, text=True, timeout=5.0
            )
            # Format: "NVRM: Xid (PCI:0000:01:00): 13, pid='<unknown>' ..."
            # Also:  "NVRM: Xid (GPU-00000000): 48, ..."
            pattern = re.compile(
                r"NVRM:.*Xid.*\(.*\):\s*(\d+)", re.IGNORECASE
            )
            for line in result.stdout.splitlines():
                m = pattern.search(line)
                if m:
                    xid = int(m.group(1))
                    # Best-effort GPU index from PCI bus — default to 0 if unparseable
                    gpu = 0
                    bus_m = re.search(r"PCI:[\da-fA-F:]+:(\d+):", line)
                    if bus_m:
                        gpu = int(bus_m.group(1)) % 8  # slot within host
                    counts.setdefault(gpu, {})
                    counts[gpu][xid] = counts[gpu].get(xid, 0) + 1
        except Exception:
            pass
        return counts

    def poll(self, timestamp: float) -> list[tuple[int, int, int]]:
        """
        Returns list of (gpu_index, xid, new_count) for XIDs that appeared
        since the last poll. Call once per monitoring tick.
        """
        if timestamp - self._last_poll < self.POLL_INTERVAL_S:
            return []
        self._last_poll  = timestamp
        self._prev_counts = dict(self._last_counts)
        self._last_counts = self._read_dmesg()

        events = []
        for gpu, xid_counts in self._last_counts.items():
            for xid, count in xid_counts.items():
                prev = self._prev_counts.get(gpu, {}).get(xid, 0)
                if count > prev:
                    events.append((gpu, xid, count - prev))
        return events

    def classify(self, xid: int) -> str:
        if xid in self._SDC_PRECURSOR:
            return "sdc_precursor"
        if xid in self._MEMORY_ERRORS:
            return "memory_error"
        if xid in self._GPU_RESET:
            return "gpu_reset_required"
        if xid in self._FALLEN_OFF_BUS:
            return "fallen_off_bus"
        if xid in self._COMPUTE_ERRORS:
            return "compute_error"
        return "other"

    def make_alert(self, gpu: int, xid: int, count: int, ts: float) -> Optional[AlertEvent]:
        category = self.classify(xid)
        severity = "critical" if category in ("memory_error", "fallen_off_bus", "gpu_reset_required") else \
                   "warning"  if category in ("sdc_precursor", "compute_error") else "info"
        if severity == "info":
            return None

        state = GPUState.CRITICAL if severity == "critical" else GPUState.DRIFTING
        return AlertEvent(
            gpu_index       = gpu,
            timestamp       = ts,
            state           = state,
            prev_state      = GPUState.UNKNOWN,
            rtheta          = None,
            rtheta_baseline = None,
            drift_sigma     = None,
            confidence      = 0.90,
            message         = (
                f"[{severity.upper()}] GPU {gpu} — XID {xid} ({category}) "
                f"appeared {count}× in the last 10 minutes. "
                f"{'SDC precursor — schedule idle validation window.' if category == 'sdc_precursor' else ''}"
                f"{'Memory hardware fault — consider draining.' if category == 'memory_error' else ''}"
                f"{'GPU fallen off PCIe bus — immediate attention required.' if category == 'fallen_off_bus' else ''}"
            ),
            context = {
                "severity": severity,
                "xid":      xid,
                "xid_category": category,
                "xid_count": count,
            },
        )


class MicroThrottleDetector:
    """
    Detects sustained SM clock suppression under active load.

    Ratio = sm_clock_mhz / sm_clock_max_mhz. If ratio < EFFICIENCY_THRESHOLD
    while util_pct >= LOAD_THRESHOLD for SUSTAINED_SAMPLES consecutive readings,
    a WARNING fires with decoded NVML throttle reasons so the operator knows
    whether it's thermal, power cap, reliability voltage, or sync boost.
    """

    def __init__(self):
        self._consecutive:    dict[int, int]   = {}
        self._last_alert_ts:  dict[int, float] = {}

    def update(self, sample: RawSample) -> Optional[AlertEvent]:
        gpu = sample.gpu_index
        ts  = sample.timestamp

        if sample.sm_clock_max_mhz <= 0:
            return None

        efficiency = sample.clock_sm_mhz / sample.sm_clock_max_mhz
        suppressed = (
            efficiency < EFFICIENCY_THRESHOLD
            and sample.util_pct >= LOAD_THRESHOLD
        )

        count = self._consecutive.get(gpu, 0)
        self._consecutive[gpu] = count + 1 if suppressed else 0

        if self._consecutive[gpu] < SUSTAINED_SAMPLES:
            return None

        last_alert = self._last_alert_ts.get(gpu, 0.0)
        if ts - last_alert < THROTTLE_COOLDOWN_S:
            return None

        self._last_alert_ts[gpu] = ts
        reasons = decode_throttle_reasons(sample.throttle_reasons)
        reason_str = f" Throttle causes: {', '.join(reasons)}." if reasons else ""

        return AlertEvent(
            gpu_index       = gpu,
            timestamp       = ts,
            state           = GPUState.DRIFTING,
            prev_state      = GPUState.UNDER_LOAD,
            rtheta          = None,
            rtheta_baseline = None,
            drift_sigma     = None,
            confidence      = 0.85,
            message         = (
                f"[WARNING] GPU {gpu} — micro-throttling detected. "
                f"SM clock {efficiency*100:.1f}% of boost "
                f"({sample.clock_sm_mhz}/{sample.sm_clock_max_mhz} MHz) "
                f"under {sample.util_pct:.0f}% load "
                f"for {self._consecutive[gpu]} consecutive samples.{reason_str} "
                f"Possible thermal paste degradation, PDU power cap, or voltage instability."
            ),
            context = {
                "severity":             "warning",
                "clock_efficiency_pct": round(efficiency * 100, 1),
                "sm_clock_mhz":         sample.clock_sm_mhz,
                "sm_clock_max_mhz":     sample.sm_clock_max_mhz,
                "throttle_reasons":     reasons,
                "consecutive_samples":  self._consecutive[gpu],
                "util_pct":             sample.util_pct,
            },
        )
