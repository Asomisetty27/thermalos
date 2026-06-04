"""
Prometheus metrics exporter.

Exposes a /metrics HTTP endpoint on the configured port (default 9101).
Follows OpenTelemetry + Prometheus naming conventions:
  thermalos_gpu_rtheta_cwatt            (gauge)
  thermalos_gpu_temperature_celsius     (gauge)
  thermalos_gpu_power_watts             (gauge)
  thermalos_gpu_utilization_ratio       (gauge)
  thermalos_gpu_state_info              (gauge, label=state)
  thermalos_gpu_drift_sigma             (gauge)
  thermalos_gpu_alerts_total            (counter, label=severity)
  thermalos_gpu_baseline_tref_celsius   (gauge)
  thermalos_build_info                  (gauge, static labels)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .metrics import AlertEvent, EnrichedSample, GPUState, STATE_LABELS
from .window import WindowResult
from .detector import DriftResult

log = logging.getLogger(__name__)

try:
    from prometheus_client import (
        Counter, Gauge, Info, start_http_server, REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    log.warning("prometheus_client not installed — metrics export disabled")


class PrometheusExporter:
    """
    Registers all ThermalOS metrics with the Prometheus default registry
    and starts the HTTP server.
    """

    def __init__(self, port: int = 9101):
        self._port    = port
        self._started = False

        if not PROMETHEUS_AVAILABLE:
            return

        # Gauges
        self.g_rtheta     = Gauge("thermalos_gpu_rtheta_cwatt",
                                   "Effective thermal resistance R_theta (C/W)",
                                   ["gpu_index"])
        self.g_temp       = Gauge("thermalos_gpu_temperature_celsius",
                                   "Junction temperature (°C)",
                                   ["gpu_index"])
        self.g_power      = Gauge("thermalos_gpu_power_watts",
                                   "GPU power consumption (W)",
                                   ["gpu_index"])
        self.g_util       = Gauge("thermalos_gpu_utilization_ratio",
                                   "GPU utilization 0–1",
                                   ["gpu_index"])
        self.g_pstate     = Gauge("thermalos_gpu_perf_state",
                                   "GPU performance state (0=max, 8=idle)",
                                   ["gpu_index"])
        self.g_state      = Gauge("thermalos_gpu_state_info",
                                   "Current classified GPU state (1=active)",
                                   ["gpu_index", "state"])
        self.g_drift      = Gauge("thermalos_gpu_drift_sigma",
                                   "R_theta deviation from baseline in σ units",
                                   ["gpu_index"])
        self.g_tref       = Gauge("thermalos_gpu_baseline_tref_celsius",
                                   "Virtual ambient temperature T_ref (°C)",
                                   ["gpu_index"])
        self.g_window_std = Gauge("thermalos_gpu_window_rtheta_std",
                                   "R_theta rolling window std dev (C/W)",
                                   ["gpu_index"])

        # Counters
        self.c_alerts     = Counter("thermalos_gpu_alerts_total",
                                    "Total alerts emitted",
                                    ["gpu_index", "severity", "state"])

        # Build info
        try:
            self.i_build = Info("thermalos_build", "ThermalOS agent build info")
            self.i_build.info({"version": "0.1.0", "stage1_rows": "4570"})
        except Exception:
            pass

    def start_server(self) -> None:
        if not PROMETHEUS_AVAILABLE or self._started:
            return
        start_http_server(self._port)
        self._started = True
        log.info(f"Prometheus metrics available at http://localhost:{self._port}/metrics")

    def update_sample(self, sample: EnrichedSample) -> None:
        if not PROMETHEUS_AVAILABLE:
            return
        idx = str(sample.gpu_index)
        self.g_temp.labels(idx).set(sample.raw.temp_junction)
        self.g_power.labels(idx).set(sample.raw.power_w)
        self.g_util.labels(idx).set(sample.raw.util_pct / 100.0)
        self.g_pstate.labels(idx).set(sample.raw.perf_state)
        self.g_tref.labels(idx).set(sample.t_ref)
        if sample.rtheta is not None:
            self.g_rtheta.labels(idx).set(sample.rtheta)

    def update_window(self, window: WindowResult) -> None:
        if not PROMETHEUS_AVAILABLE:
            return
        idx = str(window.gpu_index)
        self.g_window_std.labels(idx).set(window.rtheta_std)
        if window.is_stable:
            self.g_rtheta.labels(idx).set(window.rtheta_mean)

    def update_drift(self, drift: DriftResult) -> None:
        if not PROMETHEUS_AVAILABLE or drift.sigma_score is None:
            return
        self.g_drift.labels(str(drift.gpu_index)).set(drift.sigma_score)

    def update_state(self, gpu_index: int, state: GPUState) -> None:
        if not PROMETHEUS_AVAILABLE:
            return
        idx = str(gpu_index)
        for s in GPUState:
            label = STATE_LABELS.get(s, s.name)
            self.g_state.labels(idx, label).set(1 if s == state else 0)

    def record_alert(self, event: AlertEvent) -> None:
        if not PROMETHEUS_AVAILABLE:
            return
        ctx      = event.context
        severity = ctx.get("severity", "info") if isinstance(ctx, dict) else "info"
        state    = STATE_LABELS.get(event.state, event.state.name)
        self.c_alerts.labels(str(event.gpu_index), severity, state).inc()
