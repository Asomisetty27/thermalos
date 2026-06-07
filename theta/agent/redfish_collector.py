"""
Redfish / BMC out-of-band telemetry enrichment.

DGX B200 exposes Redfish by default on the BMC. This collector pulls chassis-
level metrics that are invisible to NVML/DCGM: inlet air temperature, fan RPM,
PSU health, NVLink fabric status. Cross-correlating these with in-band R_theta
enables root-cause attribution — "GPU 3 R_theta drifting + fan 2 at 60% RPM =
cooling path failure, not silicon degradation."

Falls back silently if:
  - Redfish endpoint is unreachable
  - Authentication fails
  - Running on a non-DGX host

Activate via AgentConfig(use_redfish=True, redfish_host="192.168.1.1",
                          redfish_user="admin", redfish_password="...").
Or read from ~/.theta/config.json (written by the setup wizard).

DGX B200 Redfish base URI: https://<BMC_IP>/redfish/v1/
Key endpoints:
  /Chassis/1/Thermal           — inlet temp, fan RPM, component temps
  /Chassis/1/Power             — PSU input/output, voltage rails
  /Systems/1/                  — system health rollup
  /NvidiaSystemComponents/1/   — NVLink fabric telemetry (HMC)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class RedfishSample:
    """Chassis-level snapshot from BMC Redfish."""
    timestamp:        float
    inlet_temp_c:     Optional[float] = None   # air temp entering the chassis
    fan_rpms:         list[int]       = field(default_factory=list)
    fan_health:       str             = "unknown"   # "OK", "Warning", "Critical"
    psu_input_w:      Optional[float] = None
    psu_health:       str             = "unknown"
    chassis_health:   str             = "unknown"
    nvlink_ok:        Optional[bool]  = None   # None = not available


class RedfishEnricher:
    """
    Pulls chassis Redfish metrics and makes them available per collection cycle.

    Usage:
        enricher = RedfishEnricher("192.168.1.1", "admin", "password")
        sample = await enricher.collect()   # None if unavailable
    """

    def __init__(
        self,
        host:     str,
        username: str,
        password: str,
        port:     int = 443,
        verify_ssl: bool = False,
    ):
        self._base   = f"https://{host}:{port}/redfish/v1"
        self._auth   = (username, password)
        self._verify = verify_ssl
        self._available: Optional[bool] = None

    @property
    def available(self) -> bool:
        return self._available is True

    async def probe(self) -> bool:
        """Test connectivity and auth. Call once at startup."""
        try:
            import httpx
            async with httpx.AsyncClient(verify=self._verify, timeout=5.0) as c:
                r = await c.get(f"{self._base}/", auth=self._auth)
                self._available = r.status_code == 200
        except Exception as e:
            log.info("Redfish unavailable (%s) — BMC metrics disabled", type(e).__name__)
            self._available = False
        return self._available

    async def collect(self) -> Optional[RedfishSample]:
        """Fetch a chassis snapshot. Returns None if Redfish is unavailable."""
        if self._available is False:
            return None
        if self._available is None:
            await self.probe()
        if not self._available:
            return None

        import time, httpx
        sample = RedfishSample(timestamp=time.time())

        async with httpx.AsyncClient(verify=self._verify, timeout=8.0) as c:
            # Thermal (fans + inlet temp)
            try:
                r = await c.get(f"{self._base}/Chassis/1/Thermal", auth=self._auth)
                if r.status_code == 200:
                    data = r.json()
                    temps = data.get("Temperatures", [])
                    for t in temps:
                        if "Inlet" in t.get("Name", "") or "Ambient" in t.get("Name", ""):
                            sample.inlet_temp_c = t.get("ReadingCelsius")
                    fans = data.get("Fans", [])
                    sample.fan_rpms = [
                        f.get("Reading", 0) for f in fans
                        if f.get("Reading") is not None
                    ]
                    statuses = [f.get("Status", {}).get("Health", "OK") for f in fans]
                    sample.fan_health = "Critical" if "Critical" in statuses else \
                                        "Warning"  if "Warning"  in statuses else "OK"
            except Exception as e:
                log.debug("Redfish thermal fetch failed: %s", e)

            # Power
            try:
                r = await c.get(f"{self._base}/Chassis/1/Power", auth=self._auth)
                if r.status_code == 200:
                    data = r.json()
                    psus = data.get("PowerSupplies", [])
                    if psus:
                        sample.psu_input_w = sum(
                            p.get("PowerInputWatts", 0) for p in psus
                            if p.get("PowerInputWatts") is not None
                        ) or None
                        statuses = [p.get("Status", {}).get("Health", "OK") for p in psus]
                        sample.psu_health = "Critical" if "Critical" in statuses else \
                                            "Warning"  if "Warning"  in statuses else "OK"
            except Exception as e:
                log.debug("Redfish power fetch failed: %s", e)

            # System health rollup
            try:
                r = await c.get(f"{self._base}/Systems/1/", auth=self._auth)
                if r.status_code == 200:
                    data = r.json()
                    sample.chassis_health = data.get("Status", {}).get("Health", "unknown")
            except Exception as e:
                log.debug("Redfish system health fetch failed: %s", e)

        return sample

    def correlate_alert(self, sample: RedfishSample, gpu_rtheta_drifting: bool) -> Optional[str]:
        """
        Cross-layer root-cause inference.

        If R_theta is drifting AND a chassis signal is degraded, the cause
        is environmental (cooling path) not silicon. Returns a human-readable
        root cause string, or None if no correlation.
        """
        if not gpu_rtheta_drifting:
            return None

        causes = []
        if sample.fan_health in ("Warning", "Critical"):
            low_fans = [rpm for rpm in sample.fan_rpms if rpm < 4000]
            causes.append(f"fan degradation ({len(low_fans)} fans below 4000 RPM)")
        if sample.psu_health in ("Warning", "Critical"):
            causes.append("PSU health warning — check power delivery")
        if sample.inlet_temp_c and sample.inlet_temp_c > 30:
            causes.append(f"high inlet air temp ({sample.inlet_temp_c:.1f}C) — check room cooling")

        if causes:
            return "Root cause likely environmental: " + "; ".join(causes)
        return None
