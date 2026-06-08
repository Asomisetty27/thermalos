"""
Async GPU telemetry collector via pynvml.

pynvml calls are synchronous C library wrappers — they block the event loop.
All NVML queries are offloaded to threads via asyncio.to_thread() per the
recommendation from monitoring agent best practices (2026).

One collector instance per process. GPU handles are cached after init.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

from .metrics import RawSample

log = logging.getLogger(__name__)


@dataclass
class CollectorConfig:
    interval_sec: float = 5.0        # sample every N seconds
    gpu_indices: Optional[list[int]] = None  # None = all GPUs


class NVMLCollector:
    """
    Async GPU telemetry collector.

    Usage:
        async with NVMLCollector(config) as collector:
            async for sample in collector.stream():
                process(sample)
    """

    # HAL protocol: vendor identity for downstream module routing
    vendor: str = "nvidia"

    def __init__(self, config: CollectorConfig):
        self.config  = config
        self._handles: list  = []
        self._n_gpus: int    = 0
        self._demo_mode: bool = not NVML_AVAILABLE
        self._gpu_names: list[str] = []  # populated in _init_nvml
        # Per-slot failure tracking for self-healing handle reinit
        self._failure_counts: dict[int, int] = {}
        self._failure_threshold: int = 3  # consecutive misses before reinit

    @property
    def gpu_count(self) -> int:
        """HAL protocol: number of GPUs this collector is monitoring."""
        return len(self._handles) if self._handles else self._n_gpus

    @property
    def gpu_names(self) -> list[str]:
        """HAL protocol: friendly model names, indexed by slot.

        In demo mode returns placeholder Tesla T4 names so hw_profiles
        resolution still works through to the measured profile.
        """
        if self._gpu_names:
            return list(self._gpu_names)
        if self._demo_mode:
            return ["Tesla T4"] * self._n_gpus
        return ["unknown"] * self._n_gpus

    async def __aenter__(self) -> "NVMLCollector":
        await asyncio.to_thread(self._init_nvml)
        return self

    async def __aexit__(self, *_) -> None:
        if not self._demo_mode:
            await asyncio.to_thread(self._shutdown_nvml)

    def _init_nvml(self) -> None:
        if self._demo_mode:
            log.warning("pynvml not available — running in demo mode with synthetic data")
            self._n_gpus = 4
            return
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError:
            # pynvml is installed but the NVIDIA driver / library is absent
            # (common on macOS or CPU-only Linux boxes). Fall back to demo mode.
            log.warning("NVML library not found — running in demo mode with synthetic data")
            self._demo_mode = True
            self._n_gpus = 4
            return
        self._n_gpus = pynvml.nvmlDeviceGetCount()
        indices = self.config.gpu_indices or list(range(self._n_gpus))
        self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in indices]
        # Populate GPU names for hw_profiles resolution downstream
        names: list[str] = []
        for h in self._handles:
            try:
                name = pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                names.append(name)
            except Exception:
                names.append("unknown")
        self._gpu_names = names
        log.info("NVML initialized", extra={"n_gpus": len(self._handles)})

    def _shutdown_nvml(self) -> None:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    def _collect_one(self, idx: int, handle) -> RawSample:
        """Synchronous — called via asyncio.to_thread()."""
        t0     = time.time()
        temp   = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power  = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
        util   = pynvml.nvmlDeviceGetUtilizationRates(handle)
        pstate = pynvml.nvmlDeviceGetPerformanceState(handle)

        try:
            sm_mhz  = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            mem_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        except Exception:
            sm_mhz = mem_mhz = 0

        try:
            fan = pynvml.nvmlDeviceGetFanSpeed(handle)
        except pynvml.NVMLError:
            fan = None

        # Silicon-level health metrics — each wrapped independently so a single
        # unsupported query on older drivers doesn't drop the whole sample
        try:
            ecc_sbit = pynvml.nvmlDeviceGetTotalEccErrors(
                handle, pynvml.NVML_SINGLE_BIT_ECC, pynvml.NVML_VOLATILE_ECC
            )
        except pynvml.NVMLError:
            ecc_sbit = 0

        try:
            ecc_dbit = pynvml.nvmlDeviceGetTotalEccErrors(
                handle, pynvml.NVML_DOUBLE_BIT_ECC, pynvml.NVML_VOLATILE_ECC
            )
        except pynvml.NVMLError:
            ecc_dbit = 0

        try:
            throttle_reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
        except pynvml.NVMLError:
            throttle_reasons = 0

        try:
            sm_clock_max_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
        except pynvml.NVMLError:
            sm_clock_max_mhz = 0

        return RawSample(
            gpu_index        = idx,
            timestamp        = time.time(),
            temp_junction    = float(temp),
            power_w          = float(power),
            util_pct         = float(util.gpu),
            mem_util_pct     = float(util.memory),
            perf_state       = int(str(pstate).replace("PerformanceState_", "").replace("P", "")),
            clock_sm_mhz     = sm_mhz,
            clock_mem_mhz    = mem_mhz,
            fan_speed_pct    = float(fan) if fan is not None else None,
            ecc_sbit         = int(ecc_sbit),
            ecc_dbit         = int(ecc_dbit),
            throttle_reasons = int(throttle_reasons),
            sm_clock_max_mhz = sm_clock_max_mhz,
            poll_latency_s   = time.time() - t0,
        )

    def _collect_demo(self, idx: int) -> RawSample:
        """Synthetic data for development / CI without a GPU."""
        import math
        t = time.time()
        phase = (t % 300) / 300   # 5 min cycle

        if phase < 0.2:            # idle
            temp, power, util, ps = 42.0, 11.4, 0.0, 8
        elif phase < 0.5:          # load
            temp, power, util, ps = 70.0, 68.0, 97.0, 0
        elif phase < 0.6:          # transition
            temp, power, util, ps = 80.0, 31.2, 0.0, 0  # zombie-like
        else:                      # recovery
            temp = 42.0 + 20.0 * math.exp(-(phase - 0.6) * 10)
            power, util, ps = 11.4, 0.0, 8

        noise = 0.5 * math.sin(t * 7.3 + idx)
        sm_max = 1980   # T4 boost clock
        sm_cur = 1600 if ps == 0 else 300
        return RawSample(
            gpu_index        = idx,
            timestamp        = t,
            temp_junction    = temp + noise,
            power_w          = power + abs(noise) * 0.3,
            util_pct         = util,
            mem_util_pct     = util * 0.6,
            perf_state       = ps,
            clock_sm_mhz     = sm_cur,
            clock_mem_mhz    = 8000 if ps == 0 else 405,
            fan_speed_pct    = 40.0 + temp * 0.3,
            ecc_sbit         = 0,
            ecc_dbit         = 0,
            throttle_reasons = 0,
            sm_clock_max_mhz = sm_max,
        )

    async def collect_all(self) -> list[RawSample]:
        """Collect one sample from all monitored GPUs concurrently.

        Resilience: per-GPU collection failures are isolated — one bad
        handle never drops the whole tick. After a configurable run of
        consecutive failures on a single GPU, the handle is re-initialized
        (NVMLError can be transient: driver reset, brief PCIe hang, etc.).
        Re-init failures are logged but never raise — the GPU simply
        remains absent from this tick's samples.
        """
        if self._demo_mode:
            n = self.config.gpu_indices or list(range(self._n_gpus))
            return [self._collect_demo(i) for i in (n if isinstance(n, list) else range(n))]

        tasks = [
            asyncio.to_thread(self._collect_one, idx, handle)
            for idx, handle in enumerate(self._handles)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        samples = []
        for slot, r in enumerate(results):
            if isinstance(r, Exception):
                # Per-GPU failure tracking — re-init the handle after N strikes
                self._failure_counts[slot] = self._failure_counts.get(slot, 0) + 1
                if self._failure_counts[slot] >= self._failure_threshold:
                    log.warning(
                        "collector reinit gpu=%d after %d consecutive failures: %s",
                        slot, self._failure_counts[slot], r,
                    )
                    self._try_reinit_handle(slot)
                else:
                    log.error("collection error gpu=%d (%d/%d): %s",
                              slot, self._failure_counts[slot],
                              self._failure_threshold, r)
            else:
                # Successful sample — reset the strike counter
                if slot in self._failure_counts:
                    del self._failure_counts[slot]
                samples.append(r)
        return samples

    def _try_reinit_handle(self, slot: int) -> None:
        """Attempt to re-acquire a single GPU's NVML handle. Best-effort."""
        try:
            indices = self.config.gpu_indices or list(range(self._n_gpus))
            if slot < len(indices):
                new_handle = pynvml.nvmlDeviceGetHandleByIndex(indices[slot])
                self._handles[slot] = new_handle
                log.info("collector reinit gpu=%d successful", slot)
                # Reset strike counter on successful reinit so we get another
                # full window of attempts before giving up again.
                self._failure_counts.pop(slot, None)
        except Exception as exc:
            log.error("collector reinit gpu=%d failed: %s", slot, exc)
            # Don't reset counter — leave it pinned so we don't reinit-loop
            # at 5s intervals. Operator restart will be needed if persistent.

    async def stream(self):
        """Yield batches of samples on every interval tick."""
        while True:
            t0 = asyncio.get_event_loop().time()
            samples = await self.collect_all()
            for s in samples:
                yield s
            elapsed = asyncio.get_event_loop().time() - t0
            await asyncio.sleep(max(0.0, self.config.interval_sec - elapsed))
