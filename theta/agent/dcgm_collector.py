"""
Optional DCGM (Data Center GPU Manager) telemetry enrichment.

pynvml polls ~1s intervals and exposes ~20 fields. DCGM provides:
  - Sub-100ms field update intervals
  - NVLink error tracking (NVLink is the GPU-to-GPU fabric on DGX/HGX nodes)
  - PCIe bandwidth saturation
  - SM and DRAM engine active fractions (more precise than util%)

This module tries to connect to the local nv-hostengine daemon. If it is
not running or pydcgm is not installed, all enrichment returns zeros silently.
The NVMLCollector continues to work normally — DCGM is additive, not required.

Usage (in AgentConfig):
    config = AgentConfig(use_dcgm=True)

Typical deployment: Linux data center host with:
    apt install datacenter-gpu-manager
    systemctl start nvidia-dcgm
    pip install nvidia-dcgm  # or pydcgm from DCGM install dir
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)

# DCGM field IDs (from dcgm_fields.py in the DCGM SDK)
_FI_DEV_NVLINK_CRC_DATA_ERR   = 409    # NVLink CRC data errors
_FI_DEV_NVLINK_RECOVERY_ERR   = 411    # NVLink recovery errors
_FI_DEV_PCIE_TX_THROUGHPUT    = 1009   # PCIe TX (KiB/s)
_FI_DEV_PCIE_RX_THROUGHPUT    = 1010   # PCIe RX (KiB/s)
_FI_PROF_GR_ENGINE_ACTIVE     = 1001   # SM engine active fraction (DCGM PROF group)
_FI_PROF_DRAM_ACTIVE          = 1005   # DRAM engine active fraction (DCGM PROF group)

_FIELD_IDS = [
    _FI_DEV_NVLINK_CRC_DATA_ERR,
    _FI_DEV_NVLINK_RECOVERY_ERR,
    _FI_DEV_PCIE_TX_THROUGHPUT,
    _FI_DEV_PCIE_RX_THROUGHPUT,
    _FI_PROF_GR_ENGINE_ACTIVE,
    _FI_PROF_DRAM_ACTIVE,
]


class DCGMEnricher:
    """
    Connects to nv-hostengine and fetches DCGM fields per GPU.

    Call enrich(gpu_index, raw_sample) to fill the DCGM fields in-place.
    If DCGM is unavailable, enrich() is a no-op and all fields remain 0.
    """

    def __init__(self, gpu_indices: Optional[list[int]] = None):
        self._available = False
        self._handle    = None
        self._group_id  = None
        self._gpu_indices: list[int] = gpu_indices or []
        self._try_init()

    def _try_init(self) -> None:
        try:
            import pydcgm
            import dcgm_structs  # noqa: F401  (availability probe)
            import dcgm_fields   # noqa: F401  (availability probe)
        except ImportError:
            log.info("pydcgm not installed — DCGM enrichment disabled")
            return

        try:
            self._handle = pydcgm.DcgmHandle(ipAddress="127.0.0.1", opMode=pydcgm.DCGM_OPERATION_MODE_AUTO)
            self._system = pydcgm.DcgmSystem(self._handle)

            # Create a GPU group (all monitored GPUs)
            self._group = self._system.GetDefaultGroup()

            # Field group for our target metrics
            self._field_group = pydcgm.DcgmFieldGroup(
                self._handle,
                "theta_fields",
                _FIELD_IDS,
            )

            # Watch fields at 100ms update interval, 30s keep-alive
            self._group.samples.WatchFields(self._field_group, 100000, 30.0, 0)

            self._available = True
            log.info("DCGM connected — sub-second telemetry enabled")
        except Exception as e:
            log.info(f"DCGM not available ({type(e).__name__}) — pynvml-only mode")

    @property
    def available(self) -> bool:
        return self._available

    def get_latest(self, gpu_index: int) -> dict:
        """Return latest DCGM field values for one GPU. Returns empty dict on failure."""
        if not self._available:
            return {}
        try:
            values = self._group.samples.GetLatest(self._field_group).values
            row = values.get(gpu_index, {})

            def _val(fid: int, default=0):
                entry = row.get(fid)
                if entry is None:
                    return default
                v = entry.value
                # DCGM returns DCGM_INT32_BLANK (0x7fffffff) or similar for unavailable
                if isinstance(v, int) and v > 2_000_000_000:
                    return default
                if isinstance(v, float) and (v != v or v > 1e10):
                    return default
                return v

            crc   = _val(_FI_DEV_NVLINK_CRC_DATA_ERR)
            recov = _val(_FI_DEV_NVLINK_RECOVERY_ERR)
            return {
                "nvlink_errors":    int(crc) + int(recov),
                "pcie_tx_kbps":     int(_val(_FI_DEV_PCIE_TX_THROUGHPUT)),
                "pcie_rx_kbps":     int(_val(_FI_DEV_PCIE_RX_THROUGHPUT)),
                "gr_engine_active": float(_val(_FI_PROF_GR_ENGINE_ACTIVE, 0.0)),
                "dram_active":      float(_val(_FI_PROF_DRAM_ACTIVE, 0.0)),
            }
        except Exception as e:
            log.debug(f"DCGM get_latest error gpu={gpu_index}: {e}")
            return {}

    def enrich(self, gpu_index: int, sample) -> None:
        """Fill DCGM fields on a RawSample in-place. No-op if unavailable."""
        if not self._available:
            return
        data = self.get_latest(gpu_index)
        if not data:
            return
        # RawSample uses slots=True — set via object.__setattr__ for frozen compat
        object.__setattr__(sample, "nvlink_errors",    data.get("nvlink_errors",    0))
        object.__setattr__(sample, "pcie_tx_kbps",     data.get("pcie_tx_kbps",     0))
        object.__setattr__(sample, "pcie_rx_kbps",     data.get("pcie_rx_kbps",     0))
        object.__setattr__(sample, "gr_engine_active", data.get("gr_engine_active", 0.0))
        object.__setattr__(sample, "dram_active",      data.get("dram_active",      0.0))

    def shutdown(self) -> None:
        if self._handle:
            try:
                self._handle.Disconnect()
            except Exception:
                pass
        self._available = False
