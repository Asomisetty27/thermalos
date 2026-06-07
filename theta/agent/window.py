"""
Steady-state rolling window filter.

Kundu direction (2026-06-03): only classify on stable rows.
A window is "stable" when σ(R_theta) < threshold over the last N seconds.
This takes NB accuracy from 84% → 99.8%.

Per GPU: maintains a deque of the last WINDOW_SEC seconds of R_theta values.
Emits a WindowResult when the window is full and stable.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


WINDOW_SEC_DEFAULT  = 15.0   # seconds of history required
SIGMA_STRICT        = 0.03   # C/W — strict threshold (publication grade)
SIGMA_RELAXED       = 0.10   # C/W — relaxed (production, more coverage)


@dataclass(slots=True)
class WindowResult:
    gpu_index:    int
    timestamp:    float
    rtheta_mean:  float
    rtheta_std:   float
    n_samples:    int
    is_stable:    bool
    last_power:   float
    last_util:    float
    last_pstate:  int


class SteadyStateWindow:
    """
    Per-GPU rolling window that decides when R_theta is stable enough to classify.

    The window stores (timestamp, rtheta, power, util, pstate) tuples.
    On each update, it returns a WindowResult indicating stability.
    """

    def __init__(
        self,
        window_sec: float = WINDOW_SEC_DEFAULT,
        sigma_threshold: float = SIGMA_STRICT,
        min_samples: int = 5,
    ):
        self._window_sec      = window_sec
        self._sigma_threshold = sigma_threshold
        self._min_samples     = min_samples
        self._buffers: dict[int, deque] = {}

    def update(
        self,
        gpu_index: int,
        timestamp: float,
        rtheta:    float,
        power:     float,
        util:      float,
        pstate:    int,
    ) -> WindowResult:
        if gpu_index not in self._buffers:
            self._buffers[gpu_index] = deque()

        buf = self._buffers[gpu_index]
        buf.append((timestamp, rtheta, power, util, pstate))

        # Evict old samples
        cutoff = timestamp - self._window_sec
        while buf and buf[0][0] < cutoff:
            buf.popleft()

        r_vals = [r for _, r, _, _, _ in buf]
        n = len(r_vals)

        if n < self._min_samples:
            return WindowResult(
                gpu_index   = gpu_index,
                timestamp   = timestamp,
                rtheta_mean = rtheta,
                rtheta_std  = 0.0,
                n_samples   = n,
                is_stable   = False,
                last_power  = power,
                last_util   = util,
                last_pstate = pstate,
            )

        mean = sum(r_vals) / n
        std  = math.sqrt(sum((r - mean) ** 2 for r in r_vals) / n)

        return WindowResult(
            gpu_index   = gpu_index,
            timestamp   = timestamp,
            rtheta_mean = round(mean, 4),
            rtheta_std  = round(std, 4),
            n_samples   = n,
            is_stable   = std < self._sigma_threshold,
            last_power  = power,
            last_util   = util,
            last_pstate = pstate,
        )

    def reset(self, gpu_index: int) -> None:
        self._buffers.pop(gpu_index, None)

    def coverage(self, gpu_index: int, timestamp: float) -> float:
        """Fraction of the window currently filled (0–1)."""
        buf = self._buffers.get(gpu_index)
        if not buf:
            return 0.0
        span = timestamp - buf[0][0]
        return min(1.0, span / self._window_sec)
