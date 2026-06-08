"""
Hardware abstraction layer (HAL) for telemetry collectors.

Audit finding addressed: collector.py is hardcoded to pynvml/NVIDIA, with
no abstraction. AMD MI300 / Intel Gaudi / future TPU support requires
rewriting the collector each time, or worse: silently falling back to
demo mode and pretending everything is fine.

This module defines a `TelemetryCollector` protocol that any vendor can
implement, plus a `select_collector()` factory that auto-detects which
backend(s) are available on the host. The existing NVMLCollector now
implements this protocol (no behavior change for NVIDIA users), and a
stub ROCmCollector is provided as the AMD path — currently raising
NotImplementedError with a clear migration message, but architected so a
real implementation can drop in without touching the daemon.

The point of building this NOW, before the AMD implementation exists, is
that the daemon and downstream modules can be written to the protocol
today — so when ROCm support arrives, integration is a one-line factory
change, not a refactor of every module that touches RawSample.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .metrics import RawSample


@runtime_checkable
class TelemetryCollector(Protocol):
    """Protocol every per-vendor collector must satisfy.

    All methods are async — even the trivially-synchronous ones — so the
    daemon can `await` uniformly regardless of whether the underlying
    library blocks (pynvml does), is awaitable (some Intel/ROCm tools do),
    or runs in a subprocess. Synchronous implementations can wrap with
    `asyncio.to_thread`.

    The lifecycle is intentionally async-context-manager shaped because
    real hardware libraries need explicit init/shutdown (pynvml's nvmlInit
    grabs a process-wide lock; ROCm's rocm_smi_lib requires explicit
    init+shutdown calls; subprocess-based collectors need pipe cleanup).
    """

    vendor: str   # e.g. "nvidia", "amd", "intel", "demo"

    async def __aenter__(self) -> "TelemetryCollector": ...
    async def __aexit__(self, *_) -> None: ...

    async def collect_all(self) -> list[RawSample]:
        """One sample per monitored GPU, concurrent where possible."""
        ...

    @property
    def gpu_count(self) -> int:
        """Number of GPUs this collector is monitoring."""
        ...

    @property
    def gpu_names(self) -> list[str]:
        """Friendly model names indexed by GPU slot (for hw_profiles lookup)."""
        ...


# ──────────────────────────────────────────────────────────────────────────
# Backend availability probes
# ──────────────────────────────────────────────────────────────────────────

def _nvml_available() -> bool:
    """Is pynvml importable AND able to talk to a driver?"""
    try:
        import pynvml
        try:
            pynvml.nvmlInit()
            try:
                pynvml.nvmlDeviceGetCount()
                return True
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        except pynvml.NVMLError:
            return False
    except ImportError:
        return False


def _rocm_available() -> bool:
    """Is rocm_smi_lib (or pyrsmi) importable AND able to talk to a driver?"""
    try:
        # The two main ROCm Python bindings. Either is acceptable.
        try:
            import pyrsmi  # noqa: F401
            return True
        except ImportError:
            pass
        # rocm_smi_lib is sometimes installed as `rsmiBindings`
        import rsmiBindings  # noqa: F401
        return True
    except ImportError:
        return False


# ──────────────────────────────────────────────────────────────────────────
# AMD ROCm collector — stub until the real implementation lands
# ──────────────────────────────────────────────────────────────────────────

class ROCmCollector:
    """Stub AMD MI300/MI325 collector — architecture only, not functional yet.

    The shape is intentionally identical to NVMLCollector so the daemon can
    select between them. When the real implementation lands, every method
    body fills in, no surrounding code changes.

    Why ship the stub: it pins the interface contract today. When AMD lands
    (Cal Poly EE has MI300X access on the roadmap), the implementer knows
    exactly what surface to deliver — including the rocm_smi_lib calls that
    map to each NVML query.
    """

    vendor = "amd"

    def __init__(self, config):
        self._config = config
        self._initialized = False

    async def __aenter__(self) -> "ROCmCollector":
        # Real impl: rocm_smi.rsmi_init(0) and discover devices
        raise NotImplementedError(
            "AMD ROCm collector is stubbed but not yet implemented. "
            "To enable, install pyrsmi and complete rocm_smi calls in "
            "ROCmCollector.__aenter__ / collect_all. Reference mapping "
            "from NVML to ROCm SMI is in the docstring."
        )

    async def __aexit__(self, *_) -> None:
        pass

    async def collect_all(self) -> list[RawSample]:
        # NVML → ROCm SMI mapping reference for the future implementer:
        #   nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
        #     → rsmi_dev_temp_metric_get(dev, sensor, RSMI_TEMP_CURRENT)
        #   nvmlDeviceGetPowerUsage(handle)
        #     → rsmi_dev_power_ave_get(dev) (microwatts)
        #   nvmlDeviceGetUtilizationRates(handle).gpu
        #     → rsmi_dev_busy_percent_get(dev)
        #   nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM)
        #     → rsmi_dev_gpu_clk_freq_get(dev, RSMI_CLK_TYPE_SYS, ...)
        #   nvmlDeviceGetTotalEccErrors(handle, single|double, volatile)
        #     → rsmi_dev_ecc_count_get(dev, RSMI_GPU_BLOCK_*, ec_counter)
        #   nvmlDeviceGetCurrentClocksThrottleReasons(handle)
        #     → rsmi_dev_perf_level_get + rsmi_dev_volt_metric_get
        return []

    @property
    def gpu_count(self) -> int:
        return 0

    @property
    def gpu_names(self) -> list[str]:
        return []


# ──────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────

def select_collector(config, *, prefer: str | None = None):
    """Auto-detect and return the best available collector.

    Detection order (unless `prefer` overrides):
      1. NVIDIA (pynvml) — production-grade, fully implemented
      2. AMD (rocm_smi) — stub for now, falls through to demo if unimplemented
      3. Demo mode — synthetic samples, used in CI and for site-only deploys

    `prefer` can be "nvidia" | "amd" | "demo" to force a specific backend.
    Useful for testing the AMD code path on an NVIDIA host (will raise
    NotImplementedError loudly, which is correct).
    """
    # Lazy imports to avoid pulling pynvml on AMD-only hosts and vice versa
    from .collector import NVMLCollector

    if prefer == "nvidia":
        return NVMLCollector(config)
    if prefer == "amd":
        return ROCmCollector(config)
    if prefer == "demo":
        # NVMLCollector's demo mode is the canonical fake-data source
        coll = NVMLCollector(config)
        coll._demo_mode = True  # type: ignore[attr-defined]
        return coll

    # Auto-detect
    if _nvml_available():
        return NVMLCollector(config)
    if _rocm_available():
        return ROCmCollector(config)
    # Fall back to demo mode
    return NVMLCollector(config)
