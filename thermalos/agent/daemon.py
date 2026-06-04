"""
Main async event loop — the ThermalOS agent daemon.

Pipeline per tick:
  Collector → EnrichedSample (R_theta) → BaselineManager.update()
                                       → SteadyStateWindow.update()
                                       → [if stable] StateClassifier.classify()
                                       → DriftDetector.update()
                                       → GPUStateMachine.transition()
                                       → [if AlertEvent] AlertRouter.route()
                                       → PrometheusExporter.update_*()

One pipeline runs for ALL GPUs concurrently (gather).
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from dataclasses import dataclass, field
from typing import Optional

import structlog

from .collector  import NVMLCollector, CollectorConfig
from .metrics    import EnrichedSample, GPUState, ClassifiedSample, AlertEvent, enrich
from .baseline   import BaselineManager
from .window     import SteadyStateWindow, SIGMA_STRICT
from .classifier import StateClassifier
from .detector   import DriftDetector, DriftResult
from .state      import GPUStateMachine
from .correlator import FleetCorrelator
from .silicon    import EccMonitor, MicroThrottleDetector
from .alerter    import AlertRouter, StdoutAlerter, WebhookAlerter, FileAlerter
from .exporter   import PrometheusExporter

log = structlog.get_logger(__name__)


@dataclass
class AgentConfig:
    # Collection
    interval_sec:       float = 5.0
    gpu_indices:        Optional[list[int]] = None

    # Steady-state window
    window_sec:         float = 15.0
    sigma_threshold:    float = SIGMA_STRICT

    # Drift detection
    k_warn:             float = 2.0
    k_critical:         float = 3.5

    # Classifier
    prefer_dt:          bool  = True   # Decision Tree = 100% acc, interpretable

    # Alerting
    webhook_url:        Optional[str]  = None
    alert_log_path:     Optional[str]  = None
    quiet:              bool  = False

    # Prometheus
    prometheus_port:    int   = 9101
    enable_prometheus:  bool  = True


class ThermalOSAgent:
    """
    The ThermalOS monitoring agent.

    Usage:
        config = AgentConfig(interval_sec=5, webhook_url="https://...")
        agent  = ThermalOSAgent(config)
        await  agent.run()   # blocks until SIGINT/SIGTERM
    """

    def __init__(self, config: AgentConfig):
        self.config     = config
        self._shutdown  = asyncio.Event()

        self._baseline     = BaselineManager()
        self._window       = SteadyStateWindow(config.window_sec, config.sigma_threshold)
        self._classifier   = StateClassifier(prefer_interpretable=config.prefer_dt)
        self._detector     = DriftDetector(config.k_warn, config.k_critical)
        self._statemachine   = GPUStateMachine()
        self._correlator     = FleetCorrelator()
        self._ecc_monitor    = EccMonitor()
        self._micro_throttle = MicroThrottleDetector()
        self._router         = self._build_router()
        self._exporter     = PrometheusExporter(config.prometheus_port)

        self._tick_count  = 0
        self._alert_count = 0

    def _build_router(self) -> AlertRouter:
        router = AlertRouter()
        if not self.config.quiet:
            router.add(StdoutAlerter())
        if self.config.webhook_url:
            router.add(WebhookAlerter(self.config.webhook_url))
        if self.config.alert_log_path:
            router.add(FileAlerter(self.config.alert_log_path))
        return router

    async def _process_sample(self, raw_sample) -> None:
        """Process one GPU sample through the full pipeline."""
        gpu = raw_sample.gpu_index
        ts  = raw_sample.timestamp

        # 0. Silicon-level checks run on every sample (before steady-state filter)
        for silicon_alert in (
            self._ecc_monitor.update(raw_sample),
            self._micro_throttle.update(raw_sample),
        ):
            if silicon_alert is not None:
                self._alert_count += 1
                self._exporter.record_alert(silicon_alert)
                await self._router.route(silicon_alert)

        # 1. Update virtual ambient from idle windows
        self._baseline.update(
            gpu, raw_sample.temp_junction,
            raw_sample.util_pct, raw_sample.perf_state, ts
        )
        t_ref = self._baseline.get_t_ref(gpu)

        # 2. Compute R_theta
        enriched = enrich(raw_sample, t_ref)
        self._exporter.update_sample(enriched)

        if not enriched.rtheta_valid or enriched.rtheta is None:
            return

        # 3. Update steady-state window
        window = self._window.update(
            gpu, ts, enriched.rtheta,
            raw_sample.power_w, raw_sample.util_pct, raw_sample.perf_state
        )
        self._exporter.update_window(window)

        if not window.is_stable:
            return

        # 4. Classify (only on stable windows)
        state, confidence = self._classifier.classify(window)

        classified = ClassifiedSample(
            enriched     = enriched,
            state        = state,
            confidence   = confidence,
            rtheta_mean  = window.rtheta_mean,
        )

        # 5. Drift detection
        drift = self._detector.update(gpu, ts, window.rtheta_mean, state)
        self._exporter.update_drift(drift)
        self._exporter.update_state(gpu, state)

        # 6. State machine → maybe alert
        alert = self._statemachine.transition(classified, drift)

        if alert is not None:
            self._alert_count += 1
            self._exporter.record_alert(alert)
            await self._router.route(alert)

            # Explainability: log the classifier's reasoning for every anomalous alert
            if alert.state not in (GPUState.CLEAN_IDLE, GPUState.UNDER_LOAD):
                explanation = self._classifier.explain(window)
                log.info("classification_reason", gpu=gpu, reason=explanation)

        # 7. Predictive alert — warn before the threshold is crossed
        if drift.is_predictive:
            eta_min = round(drift.eta_to_drift_s / 60, 1) if drift.eta_to_drift_s else "?"
            pred_alert = AlertEvent(
                gpu_index       = gpu,
                timestamp       = ts,
                state           = state,
                prev_state      = state,
                rtheta          = window.rtheta_mean,
                rtheta_baseline = drift.baseline_mean,
                drift_sigma     = drift.sigma_score,
                confidence      = 0.8,
                message         = (
                    f"[WARNING] GPU {gpu} — predictive thermal drift. "
                    f"R_θ trending at +{drift.trend_slope:.5f} C/W·s. "
                    f"Estimated {eta_min} min until drift threshold. "
                    f"No action required yet — monitor closely."
                ),
                context         = {
                    "severity":    "warning",
                    "predictive":  True,
                    "eta_minutes": eta_min,
                    "trend_slope": drift.trend_slope,
                },
            )
            self._alert_count += 1
            self._exporter.record_alert(pred_alert)
            await self._router.route(pred_alert)
            log.info("predictive_warning", gpu=gpu, eta_min=eta_min, slope=drift.trend_slope)

        # 8. Fleet correlation — detect cross-GPU anomalies after each sample
        fleet_alert = self._correlator.check(
            {g: r.current_state for g, r in self._statemachine.all_states().items()},
            ts,
        )
        if fleet_alert is not None:
            self._alert_count += 1
            self._exporter.record_alert(fleet_alert)
            await self._router.route(fleet_alert)
            log.warning("fleet_event", affected=fleet_alert.context.get("fleet_gpus"))

    async def run(self) -> None:
        """Main loop. Blocks until shutdown signal received."""
        loop = asyncio.get_running_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._shutdown.set)

        if self.config.enable_prometheus:
            self._exporter.start_server()

        collector_config = CollectorConfig(
            interval_sec = self.config.interval_sec,
            gpu_indices  = self.config.gpu_indices,
        )

        log.info(
            "agent_starting",
            interval=self.config.interval_sec,
            classifier=self._classifier.mode,
            prometheus_port=self.config.prometheus_port if self.config.enable_prometheus else None,
        )

        async with NVMLCollector(collector_config) as collector:
            async for raw_sample in collector.stream():
                if self._shutdown.is_set():
                    break
                try:
                    await self._process_sample(raw_sample)
                    self._tick_count += 1
                except Exception as e:
                    log.error("pipeline_error", exc_info=e)

        await self._router.close()
        log.info("agent_stopped", ticks=self._tick_count, alerts=self._alert_count)

    def status(self) -> dict:
        """Snapshot of current agent state — used by CLI `thermalos status`."""
        states = {}
        for gpu_idx, rec in self._statemachine.all_states().items():
            b = self._baseline.get_baseline(gpu_idx)
            states[gpu_idx] = {
                "state":       rec.current_state.name,
                "rtheta":      rec.last_rtheta,
                "confidence":  rec.last_confidence,
                "t_ref":       self._baseline.get_t_ref(gpu_idx),
                "baseline_locked": self._baseline.has_baseline(gpu_idx),
            }
        return {
            "uptime_ticks": self._tick_count,
            "alerts":       self._alert_count,
            "classifier":   self._classifier.mode,
            "gpus":         states,
        }
