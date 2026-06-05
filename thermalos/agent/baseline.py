"""
Virtual ambient (T_ref) estimation from GPU idle windows.

No thermocouple required. T_ref is derived from the GPU's own stable idle
periods: P-state ≥ P6, util ≈ 0%, stable temperature for ≥ 30 seconds.

Baseline is persisted to ~/.thermalos/baselines.json so the agent restores
the virtual ambient on restart without a cold-start idle window requirement.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

BASELINE_DIR    = Path.home() / ".thermalos"
BASELINE_FILE   = BASELINE_DIR / "baselines.json"

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
    source:     str     # "idle_window" | "manual" | "default"

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
            for entry in raw:
                b = Baseline(**entry)
                self._baselines[b.gpu_index] = b
        except Exception:
            pass

    def save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(b) for b in self._baselines.values()]
        self._file.write_text(json.dumps(data, indent=2))

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

    def get_t_ref(self, gpu_index: int) -> float:
        """Return T_ref for this GPU, falling back to 25°C if not yet locked."""
        b = self._baselines.get(gpu_index)
        if b is not None:
            return b.t_ref
        return 25.0   # conservative default — will be replaced when idle window found

    def get_baseline(self, gpu_index: int) -> Optional[Baseline]:
        return self._baselines.get(gpu_index)

    def has_baseline(self, gpu_index: int) -> bool:
        return gpu_index in self._baselines

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
