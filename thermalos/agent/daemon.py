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
from .silicon         import EccMonitor, MicroThrottleDetector
from .unsupervised    import IsolationForestCritic
from .dcgm_collector  import DCGMEnricher
from .telemetry          import TelemetryReporter
from .predictor          import FailurePredictor
from .sdc_hunter         import SDCHunter
from .redfish_collector  import RedfishEnricher
from .alerter            import AlertRouter, StdoutAlerter, WebhookAlerter, FileAlerter
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

    # Optional DCGM enrichment (requires nv-hostengine running on the host)
    use_dcgm:           bool  = False

    # Optional Redfish/BMC out-of-band telemetry
    use_redfish:        bool  = False
    redfish_host:       Optional[str]  = None
    redfish_user:       Optional[str]  = None
    redfish_password:   Optional[str]  = None

    # ThermalOS Intelligence Network — anonymized telemetry opt-in
    data_sharing:       bool  = False


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
        self._critic         = IsolationForestCritic()
        self._dcgm           = DCGMEnricher() if config.use_dcgm else None
        self._predictor      = FailurePredictor()
        self._sdc_hunter     = SDCHunter(config.gpu_indices)
        self._redfish        = (
            RedfishEnricher(config.redfish_host, config.redfish_user, config.redfish_password)
            if config.use_redfish and config.redfish_host else None
        )
        self._telemetry      = TelemetryReporter(opt_in=config.data_sharing)
        self._router         = self._build_router()

        # Per-GPU live state for SDC hunter cross-GPU validation
        self._gpu_util:  dict[int, float] = {}
        self._gpu_power: dict[int, float] = {}
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

        # Track live GPU state for SDC hunter
        self._gpu_util[gpu]  = raw_sample.util_pct
        self._gpu_power[gpu] = raw_sample.power_w

        # 0a. DCGM enrichment — fills NVLink/PCIe/engine fields if nv-hostengine available
        if self._dcgm is not None:
            self._dcgm.enrich(gpu, raw_sample)

        # 0b. Silicon-level checks run on every sample (before steady-state filter)
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

        # Update silicon metrics in exporter
        sm_max = raw_sample.sm_clock_max_mhz
        clock_eff = (raw_sample.clock_sm_mhz / sm_max) if sm_max > 0 else None
        self._exporter.update_silicon(gpu, raw_sample.ecc_sbit, raw_sample.ecc_dbit, clock_eff)

        # 5. Drift detection + unsupervised critic
        drift = self._detector.update(gpu, ts, window.rtheta_mean, state)

        # Feed healthy windows to the Isolation Forest baseline
        healthy = state in (GPUState.CLEAN_IDLE, GPUState.UNDER_LOAD)
        if healthy:
            self._critic.update_healthy(gpu, window)

        # Score and check for critic/supervised disagreement
        critic_alert = self._critic.maybe_alert(gpu, window, state, ts)
        if critic_alert is not None:
            self._alert_count += 1
            self._exporter.record_alert(critic_alert)
            await self._router.route(critic_alert)
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

        # 8a. Failure predictor — update and check for degradation risk alert
        self._predictor.update(
            gpu_index  = gpu,
            ts         = ts,
            rtheta     = window.rtheta_mean if window.is_stable else None,
            drift      = drift,
            ecc_sbit   = raw_sample.ecc_sbit,
            ecc_dbit   = raw_sample.ecc_dbit,
            clock_eff  = clock_eff,
        )
        risk_alert = self._predictor.maybe_alert(gpu, ts, state)
        if risk_alert is not None:
            self._alert_count += 1
            self._exporter.record_alert(risk_alert)
            await self._router.route(risk_alert)
            log.info("degradation_risk_alert", gpu=gpu, score=risk_alert.context.get("degradation_risk"))
        self._exporter.update_risk(gpu, self._predictor.get_score(gpu))

        # 8b. Telemetry — record window for Intelligence Network (if opted in)
        gpu_name = getattr(raw_sample, 'gpu_name', '') if hasattr(raw_sample, 'gpu_name') else ''
        sm_max = getattr(raw_sample, 'sm_clock_max_mhz', 0)
        clock_eff = (raw_sample.clock_sm_mhz / sm_max) if sm_max > 0 else None
        self._telemetry.record_window(
            gpu_name       = gpu_name,
            rtheta_mean    = enriched.rtheta,
            rtheta_std     = window.rtheta_std if window.is_stable else None,
            ecc_sbit_rate  = float(raw_sample.ecc_sbit),
            ecc_dbit_event = raw_sample.ecc_dbit > 0,
            clock_eff_mean = clock_eff,
        )
        await self._telemetry.maybe_flush()

        # 9. Fleet correlation — detect cross-GPU anomalies after each sample
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

        # Probe Redfish BMC once at startup
        if self._redfish:
            await self._redfish.probe()
            if self._redfish.available:
                log.info("redfish_connected", host=self.config.redfish_host)

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

                    # SDC hunter — runs once all GPU states are up-to-date
                    # Only triggers on idle GPUs, rate-limited internally
                    if self._tick_count % 10 == 0:
                        gpu_states = {g: r.current_state for g, r in self._statemachine.all_states().items()}
                        sdc_alerts = await self._sdc_hunter.hunt(
                            gpu_states  = gpu_states,
                            gpu_util    = self._gpu_util,
                            gpu_power   = self._gpu_power,
                            timestamp   = raw_sample.timestamp,
                        )
                        for sdc_alert in sdc_alerts:
                            self._alert_count += 1
                            self._exporter.record_alert(sdc_alert)
                            await self._router.route(sdc_alert)

                    # Redfish chassis poll — every 60 ticks (~5 min)
                    if self._redfish and self._tick_count % 60 == 0:
                        chassis = await self._redfish.collect()
                        if chassis:
                            fan_min = min(chassis.fan_rpms) if chassis.fan_rpms else None
                            self._exporter.update_redfish(
                                inlet_temp = chassis.inlet_temp_c,
                                fan_rpm_min= fan_min,
                                psu_watts  = chassis.psu_input_w,
                            )
                            # Cross-layer correlation: is R_theta drift caused by cooling?
                            for g, rec in self._statemachine.all_states().items():
                                if rec.current_state in (GPUState.DRIFTING, GPUState.CRITICAL):
                                    root_cause = self._redfish.correlate_alert(chassis, True)
                                    if root_cause:
                                        log.warning("redfish_correlation gpu=%d cause=%s", g, root_cause)

                except Exception as e:
                    log.error("pipeline_error", exc_info=e)

        await self._router.close()
        if self._dcgm:
            self._dcgm.shutdown()
        if self._redfish:
            self._redfish._available = False
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
