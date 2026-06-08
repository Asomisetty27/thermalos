"""
Virtual ambient (T_ref) estimation from GPU idle windows.

No thermocouple required. T_ref is derived from the GPU's own stable idle
periods: P-state ≥ P6, util ≈ 0%, stable temperature for ≥ 30 seconds.

Baseline is persisted to ~/.theta/baselines.json so the agent restores
the virtual ambient on restart without a cold-start idle window requirement.

COLD-START PRIORS: Until a real idle window is captured, T_ref is seeded
from the hardware-class profile (hw_profiles.py) — NOT a flat 25 °C default.
Downstream code can read `baseline.is_provisional` to know if the value is
a profile seed (uncertain) vs a measured lock (high confidence).

EXTERNAL OVERRIDE: When a BMC inlet-temperature reading is available
(Redfish collector), the cold-start prior can be replaced with the actual
chassis inlet — this is more accurate than any extrapolation. Use
`set_external_ambient(gpu_index, t_c, source)` to do so.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .hw_profiles import ThermalProfile, resolve_profile
from .safeio import atomic_write_text

def _default_baseline_file() -> Path:
    import os
    cfg_dir = os.environ.get("THETA_CONFIG_DIR")
    if cfg_dir:
        return Path(cfg_dir) / "baselines.json"
    return Path.home() / ".theta" / "baselines.json"


BASELINE_FILE = _default_baseline_file()

IDLE_UTIL_MAX   = 5.0    # % GPU utilization — below = idle candidate
IDLE_PSTATE_MIN = 4      # P-state ≥ P4 for idle candidacy
IDLE_WINDOW_SEC = 30.0   # must be stable for this long to lock baseline
TEMP_STABILITY  = 1.5    # °C max std dev in window to accept as stable


@dataclass
class Baseline:
    gpu_index:  int
    t_ref:      float   # virtual ambient °C
    sigma:      float   # std dev of the idle window
    n_samples:  int
    locked_at:  float   # unix timestamp
    # "idle_window"     — measured, hard-locked from stable idle
    # "longrun_update"  — exponentially-smoothed update during brief idle
    # "manual"          — operator set via theta calibrate / wizard
    # "profile_prior"   — cold-start seed from hw_profiles (provisional)
    # "external_bmc"    — Redfish chassis-inlet override (preferred prior)
    source:     str
    # Provisional baselines are NOT measured — downstream may want to widen
    # alert thresholds until a real idle window arrives.
    provisional: bool = False
    # Uncertainty band (°C) — only meaningful for provisional sources.
    # An external BMC reading has ~±1 °C; a profile prior has ~±3 °C.
    uncertainty_c: float = 0.0

    def age_hours(self) -> float:
        return (time.time() - self.locked_at) / 3600


class BaselineManager:
    """
    Tracks idle samples per GPU and locks a T_ref when a stable window is found.
    Falls back to a default of 25°C with a warning if no window found after timeout.
    """

    def __init__(self, window_sec: float = IDLE_WINDOW_SEC, _file: Path | None = None):
        self._window_sec   = window_sec
        self._file         = _file or BASELINE_FILE
        self._buffers:  dict[int, deque] = {}
        self._baselines: dict[int, Baseline] = {}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            raw = json.loads(self._file.read_text())
        except json.JSONDecodeError:
            # Corrupted file (probably a torn write from before atomic save
            # was introduced). Quarantine it so the operator can inspect.
            backup = self._file.with_suffix(".corrupt")
            try:
                self._file.rename(backup)
            except OSError:
                pass
            return
        except OSError:
            return
        # Tolerate forward-compatible fields (provisional, uncertainty_c)
        # being missing in older saved files.
        for entry in raw:
            try:
                b = Baseline(
                    gpu_index     = entry["gpu_index"],
                    t_ref         = entry["t_ref"],
                    sigma         = entry.get("sigma", 0.0),
                    n_samples     = entry.get("n_samples", 0),
                    locked_at     = entry.get("locked_at", time.time()),
                    source        = entry.get("source", "idle_window"),
                    provisional   = entry.get("provisional", False),
                    uncertainty_c = entry.get("uncertainty_c", 0.0),
                )
                self._baselines[b.gpu_index] = b
            except (KeyError, TypeError):
                continue  # skip malformed entry, keep the rest

    def save(self) -> None:
        """Atomic write — see safeio.atomic_write_text for the guarantee."""
        data = [asdict(b) for b in self._baselines.values()]
        atomic_write_text(self._file, json.dumps(data, indent=2))

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, gpu_index: int, temp: float, util: float, pstate: int, ts: float) -> None:
        """Feed a new sample. If idle window detected, lock baseline."""
        is_idle = util <= IDLE_UTIL_MAX and pstate >= IDLE_PSTATE_MIN

        if gpu_index not in self._buffers:
            self._buffers[gpu_index] = deque()

        buf = self._buffers[gpu_index]

        if not is_idle:
            buf.clear()
            return

        buf.append((ts, temp))

        # Evict samples older than window
        cutoff = ts - self._window_sec
        while buf and buf[0][0] < cutoff:
            buf.popleft()

        if not buf:
            return

        span = buf[-1][0] - buf[0][0]
        if span < self._window_sec * 0.9:
            return   # not enough time yet

        temps = [t for _, t in buf]
        mean_t = sum(temps) / len(temps)
        std_t  = math.sqrt(sum((t - mean_t) ** 2 for t in temps) / len(temps))

        if std_t > TEMP_STABILITY:
            return   # too noisy, wait

        # Lock baseline
        self._baselines[gpu_index] = Baseline(
            gpu_index  = gpu_index,
            t_ref      = round(mean_t, 2),
            sigma      = round(std_t, 3),
            n_samples  = len(temps),
            locked_at  = ts,
            source     = "idle_window",
        )
        buf.clear()
        self.save()

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_t_ref(self, gpu_index: int, gpu_name: str | None = None) -> float:
        """Return T_ref for this GPU.

        Priority (highest first):
          1. Locked baseline (measured idle window or manual calibration)
          2. Provisional baseline (BMC override or profile prior set earlier)
          3. Hardware-class prior (resolved live from gpu_name)
          4. 25 °C absolute fallback (logs a warning — should never hit this)

        Cold-start fix: previous behavior unconditionally returned 25 °C
        when no baseline was locked, which systematically biased R_θ in
        hot-aisle (38 °C ambient) or cold-aisle (18 °C ambient) deployments.
        Now we use the GPU-class profile's `expected_ambient_c` instead.
        """
        b = self._baselines.get(gpu_index)
        if b is not None:
            return b.t_ref
        # No baseline yet — try the hardware profile prior
        if gpu_name:
            prof = resolve_profile(gpu_name)
            if prof is not None:
                return prof.expected_ambient_c
        return 25.0   # absolute fallback for truly unknown hardware

    def seed_from_profile(self, gpu_index: int, gpu_name: str, ts: float | None = None) -> bool:
        """Install a provisional baseline from the hardware-class profile.

        Returns True if a profile was found and applied. Idempotent — won't
        overwrite a locked baseline. Use this on agent startup so downstream
        code has a sensible T_ref before the first idle window arrives.
        """
        if gpu_index in self._baselines:
            return False  # don't clobber an existing baseline
        prof = resolve_profile(gpu_name)
        if prof is None:
            return False
        self._baselines[gpu_index] = Baseline(
            gpu_index    = gpu_index,
            t_ref        = prof.expected_ambient_c,
            sigma        = 0.0,
            n_samples    = 0,
            locked_at    = ts if ts is not None else time.time(),
            source       = "profile_prior",
            provisional  = True,
            uncertainty_c = 3.0,   # hardware-class average has ~±3 °C spread
        )
        # Don't save() yet — let an idle window or BMC override upgrade this
        # before we commit. Saving provisional values would be churn.
        return True

    def set_external_ambient(
        self,
        gpu_index: int,
        t_c: float,
        source: str = "external_bmc",
        uncertainty_c: float = 1.0,
        ts: float | None = None,
    ) -> None:
        """Override T_ref with a measured external ambient (e.g., BMC inlet).

        This is the preferred cold-start prior when available — a BMC
        Redfish reading is dramatically more accurate than any profile
        extrapolation. Marked provisional so a real idle window can still
        upgrade it (and downstream may still want to widen alert bands).
        """
        existing = self._baselines.get(gpu_index)
        # Don't overwrite a hard-locked measurement with an external prior
        if existing is not None and not existing.provisional:
            return
        self._baselines[gpu_index] = Baseline(
            gpu_index    = gpu_index,
            t_ref        = round(t_c, 2),
            sigma        = 0.0,
            n_samples    = 0,
            locked_at    = ts if ts is not None else time.time(),
            source       = source,
            provisional  = True,
            uncertainty_c = uncertainty_c,
        )

    def get_baseline(self, gpu_index: int) -> Optional[Baseline]:
        return self._baselines.get(gpu_index)

    def has_baseline(self, gpu_index: int) -> bool:
        return gpu_index in self._baselines

    def has_locked_baseline(self, gpu_index: int) -> bool:
        """True only when baseline is a real measurement, not a provisional prior."""
        b = self._baselines.get(gpu_index)
        return b is not None and not b.provisional

    def maybe_update_longrun(
        self,
        gpu_index: int,
        temp: float,
        util: float,
        pstate: int,
        ts: float,
        alpha: float = 0.05,
    ) -> bool:
        """
        Soft-update T_ref during brief idle windows within a long-running job.

        Uses exponential smoothing (not a hard re-lock) so a 3°C diurnal ambient
        rise gradually shifts T_ref upward, while a rapid R_theta spike (actual
        degradation) is not absorbed. Returns True if T_ref was updated.

        Only applies when:
        - A baseline is already locked (don't create one from scratch here)
        - The GPU enters a transient idle window (util < threshold, pstate ≥ min)
        - The proposed new T_ref is within 5°C of the existing one (sanity gate)
        """
        if gpu_index not in self._baselines:
            return False
        if util > IDLE_UTIL_MAX or pstate < IDLE_PSTATE_MIN:
            return False

        existing = self._baselines[gpu_index]
        if abs(temp - existing.t_ref) > 5.0:
            # Difference too large — this is not ambient drift, don't absorb it
            return False

        new_tref = round(existing.t_ref * (1 - alpha) + temp * alpha, 2)
        if new_tref == existing.t_ref:
            return False

        self._baselines[gpu_index] = Baseline(
            gpu_index = gpu_index,
            t_ref     = new_tref,
            sigma     = existing.sigma,
            n_samples = existing.n_samples,
            locked_at = existing.locked_at,
            source    = "longrun_update",
        )
        return True

    def set_manual(self, gpu_index: int, t_ref: float) -> None:
        self._baselines[gpu_index] = Baseline(
            gpu_index = gpu_index,
            t_ref     = t_ref,
            sigma     = 0.0,
            n_samples = 0,
            locked_at = time.time(),
            source    = "manual",
        )
        self.save()

    def summary(self) -> list[dict]:
        return [
            {
                "gpu":     b.gpu_index,
                "t_ref":   b.t_ref,
                "sigma":   b.sigma,
                "age_h":   round(b.age_hours(), 1),
                "source":  b.source,
                "locked":  b.has_baseline if hasattr(b, "has_baseline") else True,
            }
            for b in self._baselines.values()
        ]
