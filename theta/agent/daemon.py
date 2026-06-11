"""
Main async event loop — the Theta agent daemon.

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
import json
import signal
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import structlog

from .collector  import NVMLCollector, CollectorConfig
from .metrics    import GPUState, ClassifiedSample, AlertEvent, enrich
from .baseline   import BaselineManager
from .window     import SteadyStateWindow, SIGMA_STRICT
from .classifier import StateClassifier
from .calibrate  import CalibrationManager
from .detector   import DriftDetector
from .state      import GPUStateMachine
from .correlator import FleetCorrelator
from .silicon         import EccMonitor, MicroThrottleDetector, XIDParser
from .unsupervised    import IsolationForestCritic
from .dcgm_collector  import DCGMEnricher
from .telemetry          import TelemetryReporter
from .predictor          import FailurePredictor
from .sdc_hunter         import SDCHunter
from .redfish_collector  import RedfishEnricher
from .alerter            import AlertRouter, StdoutAlerter, WebhookAlerter, FileAlerter
from .health_api         import HealthAPIServer
from .fault_classifier   import FaultCurveClassifier, FaultCause
from .profile_learner    import ProfileLearner
from ..                  import __version__
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

    # Health API server
    health_api_port:    int   = 9102   # 0 = disabled

    # Optional Redfish/BMC out-of-band telemetry
    use_redfish:        bool  = False
    redfish_host:       Optional[str]  = None
    redfish_user:       Optional[str]  = None
    redfish_password:   Optional[str]  = None

    # Theta Intelligence Network — anonymized telemetry opt-in
    data_sharing:       bool  = False

    # Auto-recalibrate after CRITICAL→RECOVERY transitions.
    # When True, spawns `theta calibrate --gpu N` subprocess after a
    # 5-min stable cooldown. When False (default), logs a suggestion instead.
    auto_recalibrate:   bool  = False


class ThetaAgent:
    """
    The Theta monitoring agent.

    Usage:
        config = AgentConfig(interval_sec=5, webhook_url="https://...")
        agent  = ThetaAgent(config)
        await  agent.run()   # blocks until SIGINT/SIGTERM
    """

    def __init__(self, config: AgentConfig):
        self.config        = config
        self._shutdown     = asyncio.Event()
        # Hot-reload signal — set by SIGHUP, applied between samples
        self._reload_event = asyncio.Event()
        # Path the wizard wrote — used to source the reloaded config from disk
        self._config_path  = Path.home() / ".theta" / "config.json"

        self._baseline     = BaselineManager()
        self._window       = SteadyStateWindow(config.window_sec, config.sigma_threshold)
        self._calibration  = CalibrationManager()
        self._classifier   = StateClassifier(
            prefer_interpretable=config.prefer_dt,
            calibration=self._calibration,
        )
        self._detector     = DriftDetector(config.k_warn, config.k_critical)
        self._statemachine   = GPUStateMachine()
        self._correlator     = FleetCorrelator()
        self._ecc_monitor    = EccMonitor()
        self._micro_throttle = MicroThrottleDetector()
        self._critic         = IsolationForestCritic()
        self._dcgm           = DCGMEnricher() if config.use_dcgm else None
        self._xid_parser     = XIDParser()
        self._predictor      = FailurePredictor()
        self._sdc_hunter     = SDCHunter(config.gpu_indices)
        self._fault_classifier = FaultCurveClassifier()

        # ── Upgrade modules (audit-driven additions) ──
        # Temporal Bayesian filter — smooths raw classifier output through a
        # discrete-state HMM so single-tick noise stops flipping the alert state.
        from .temporal_filter import TemporalStateFilter
        self._temporal_filter = TemporalStateFilter()
        # Cascading-1D-CNN predictor — augments the rule-based FailurePredictor
        # when trained weights are available. Architecture per Cai et al. 2026
        # (PRAUC 0.90). Silent no-op when torch/weights absent — daemon falls
        # back transparently to the rule-based predictor.
        from .predictor_cnn import CascadingCNNPredictor
        self._cnn_predictor = CascadingCNNPredictor()
        # Per-GPU rolling utilization fraction for workload-intensity scoring
        # (maintenance scorer reads it; populated in _process_sample).
        self._utilization_history: dict[int, "deque[float]"] = {}
        # Cache of last-known smoothed/causal/maintenance state per GPU for
        # the /api/v1/agent/gpu/{i}/details endpoint.
        self._agent_details_cache: dict[int, dict] = {}

        # Poll latency tracking — rolling mean per GPU (for health API + observability)
        self._poll_latency_ema: dict[int, float] = {}
        self._poll_latency_baseline: dict[int, float] = {}
        self._poll_latency_samples: dict[int, int] = {}
        self._poll_latency_alert_ts: dict[int, float] = {}
        self._redfish        = (
            RedfishEnricher(config.redfish_host, config.redfish_user, config.redfish_password)
            if config.use_redfish and config.redfish_host else None
        )
        self._telemetry      = TelemetryReporter(opt_in=config.data_sharing)
        self._router         = self._build_router()

        # Per-GPU live state for SDC hunter cross-GPU validation
        self._gpu_util:  dict[int, float] = {}
        self._gpu_power: dict[int, float] = {}

        # Health API — exposes /api/v1/health for SLURM prolog / MPI integration
        # and /api/v1/agent/* for the Agent Control Center site.
        self._health_api: Optional[HealthAPIServer] = None
        if config.health_api_port > 0:
            self._health_api = HealthAPIServer(
                port              = config.health_api_port,
                get_status        = self._health_status,
                get_poll_latency  = lambda: self._poll_latency_ema,
                get_agent_details = self._get_agent_details,
                auth_token        = getattr(config, "health_api_token", None),
            )
        self._exporter     = PrometheusExporter(config.prometheus_port)

        self._tick_count  = 0
        self._alert_count = 0
        # Per-stage error counters — populated by _stage() when an advisory
        # pipeline stage fails. Surfaced in status() so a silently-broken
        # enrichment stage is visible to `theta status` instead of only as
        # log noise.
        self._stage_errors: dict[str, int] = {}

        # ── Self-improvement modules ──────────────────────────────────────────
        # Profile learner: fires once per GPU when enough load samples
        # exist to upgrade hw_profiles.py confidence to 'measured'.
        self._profile_learner = ProfileLearner()
        # Auto-recalibrate queue: GPUs that recovered from anomaly and need
        # a recalibration pass before next alert cycle.
        self._recal_queue: set[int] = set()
        self._recal_cooldown_ts: dict[int, float] = {}
        # Model drift tracker: count critic/classifier disagreements per GPU.
        self._critic_disagree: dict[int, int] = {}
        self._critic_total: dict[int, int] = {}

    def _build_router(self) -> AlertRouter:
        router = AlertRouter()
        if not self.config.quiet:
            router.add(StdoutAlerter())
        if self.config.webhook_url:
            router.add(WebhookAlerter(self.config.webhook_url))
        if self.config.alert_log_path:
            router.add(FileAlerter(self.config.alert_log_path))
        return router

    @contextmanager
    def _stage(self, name: str, gpu: int):
        """
        Error-isolation boundary for ADVISORY pipeline stages.

        Before this existed, a deterministic exception in any enrichment
        stage (fault classifier, CNN feed, telemetry, DCGM…) aborted the
        remainder of _process_sample for that tick via the outer
        pipeline_error handler — including the state machine and alert
        routing. A bug in a nice-to-have stage could permanently kill core
        alerting while the daemon still looked alive.

        The critical path (baseline → R_θ → window → classify → temporal
        filter → state machine → alert routing) is deliberately NOT wrapped:
        if it breaks, the tick is genuinely unusable and the outer handler
        should own it.
        """
        try:
            yield
        except Exception as exc:
            self._stage_errors[name] = self._stage_errors.get(name, 0) + 1
            log.error("stage_error", stage=name, gpu=gpu,
                      count=self._stage_errors[name], exc_info=exc)

    async def _process_sample(self, raw_sample) -> None:
        """Process one GPU sample through the full pipeline."""
        gpu = raw_sample.gpu_index
        ts  = raw_sample.timestamp

        # Track live GPU state for SDC hunter
        self._gpu_util[gpu]  = raw_sample.util_pct
        self._gpu_power[gpu] = raw_sample.power_w

        # XID parsing runs once per minute (rate-limited internally)
        with self._stage("xid_parser", gpu):
            for xid_gpu, xid, xid_count in self._xid_parser.poll(ts):
                xid_alert = self._xid_parser.make_alert(xid_gpu, xid, xid_count, ts)
                if xid_alert:
                    self._alert_count += 1
                    self._exporter.record_alert(xid_alert)
                    await self._router.route(xid_alert)
                    log.warning("xid_event", gpu=xid_gpu, xid=xid, category=xid_alert.context.get("xid_category"))

        # ── Poll latency tracking (monitoring pipeline observability) ─────────
        with self._stage("poll_latency", gpu):
            lat = raw_sample.poll_latency_s
            alpha = 0.1
            ema = self._poll_latency_ema.get(gpu, lat)
            new_ema = ema * (1 - alpha) + lat * alpha
            self._poll_latency_ema[gpu] = new_ema

            n = self._poll_latency_samples.get(gpu, 0) + 1
            self._poll_latency_samples[gpu] = n
            if n == 20:  # establish baseline after warm-up
                self._poll_latency_baseline[gpu] = new_ema

            baseline_lat = self._poll_latency_baseline.get(gpu)
            if baseline_lat and new_ema > baseline_lat * 2.5 and n > 20:
                last_lat_alert = self._poll_latency_alert_ts.get(gpu, 0.0)
                if ts - last_lat_alert > 300:
                    self._poll_latency_alert_ts[gpu] = ts
                    lat_alert = AlertEvent(
                        gpu_index=gpu, timestamp=ts,
                        state=GPUState.UNKNOWN, prev_state=GPUState.UNKNOWN,
                        rtheta=None, rtheta_baseline=None, drift_sigma=None,
                        confidence=0.75,
                        message=(
                            f"[WARNING] GPU {gpu} — NVML poll latency {new_ema*1000:.1f}ms "
                            f"({new_ema/baseline_lat:.1f}× baseline {baseline_lat*1000:.1f}ms). "
                            f"GPU may be hanging or driver is unresponsive. "
                            f"Monitor closely — abrupt failure possible."
                        ),
                        context={"severity": "warning", "poll_latency_ms": round(new_ema*1000, 2),
                                 "baseline_ms": round(baseline_lat*1000, 2)},
                    )
                    self._alert_count += 1
                    self._exporter.record_alert(lat_alert)
                    await self._router.route(lat_alert)

        # 0a. DCGM enrichment — fills NVLink/PCIe/engine fields if nv-hostengine available
        with self._stage("dcgm_enrich", gpu):
            if self._dcgm is not None:
                self._dcgm.enrich(gpu, raw_sample)

        # 0b. Silicon-level checks: ECC, micro-throttle, XID semantic parsing
        with self._stage("silicon", gpu):
            for silicon_alert in (
                self._ecc_monitor.update(raw_sample),
                self._micro_throttle.update(raw_sample),
            ):
                if silicon_alert is not None:
                    self._alert_count += 1
                    self._exporter.record_alert(silicon_alert)
                    await self._router.route(silicon_alert)

        # 1. Update virtual ambient — hard lock on first idle window,
        #    soft exponential-smoothing update during long-run transient idles.
        #    gpu_name is passed so liquid-cooled profiles (t_ref_strategy='coolant_inlet')
        #    skip idle-window locking and use BMC inlet / expected_ambient_c instead.
        _gpu_name = self._collector_gpu_names.get(gpu) if hasattr(self, "_collector_gpu_names") else None
        self._baseline.update(
            gpu, raw_sample.temp_junction,
            raw_sample.util_pct, raw_sample.perf_state, ts,
            gpu_name=_gpu_name,
        )
        self._baseline.maybe_update_longrun(
            gpu, raw_sample.temp_junction,
            raw_sample.util_pct, raw_sample.perf_state, ts
        )
        t_ref = self._baseline.get_t_ref(gpu, _gpu_name)

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
        raw_state, raw_confidence = self._classifier.classify(window)

        # 4b. Smooth through the temporal Bayesian filter — turns a per-tick
        # observation into a posterior over states. We use the SMOOTHED state
        # for state machine + alert routing, but keep the raw available for
        # explainability ("classifier said X, filter says Y because...").
        filtered = self._temporal_filter.observe(gpu, raw_state, raw_confidence)
        state, confidence = filtered.state, filtered.confidence

        # Track per-GPU utilization for the maintenance scorer (rolling fraction
        # of time spent UNDER_LOAD over the last ~5 minutes of samples).
        util_buf = self._utilization_history.setdefault(gpu, deque(maxlen=60))
        util_buf.append(1.0 if state == GPUState.UNDER_LOAD else 0.0)

        # Feed CNN predictor buffer — silently ignored when no model loaded.
        # We push every stable window so the model has a continuous time series
        # to convolve over (not just alert-time spot checks).
        with self._stage("cnn_feed", gpu):
            sm_max_for_cnn = raw_sample.sm_clock_max_mhz
            clock_eff_for_cnn = (
                raw_sample.clock_sm_mhz / sm_max_for_cnn if sm_max_for_cnn else 0.0
            )
            self._cnn_predictor.update(gpu, ts, {
                "rtheta":    enriched.rtheta or 0.0,
                "power_w":   raw_sample.power_w,
                "temp_c":    raw_sample.temp_junction,
                "util_pct":  raw_sample.util_pct,
                "clock_eff": clock_eff_for_cnn,
            })

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
        # (detector itself is CRITICAL path — the state machine consumes drift)
        drift = self._detector.update(gpu, ts, window.rtheta_mean, state)

        with self._stage("critic", gpu):
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
                # Track disagreement for model drift monitoring
                self._critic_disagree[gpu] = self._critic_disagree.get(gpu, 0) + 1
            self._critic_total[gpu] = self._critic_total.get(gpu, 0) + 1

        self._exporter.update_drift(drift)
        self._exporter.update_state(gpu, state)

        # Profile learner — accumulate load R_θ samples toward upgrade signal
        with self._stage("profile_learner", gpu):
            _gpu_name_pl = self._collector_gpu_names.get(gpu, "unknown") if hasattr(self, "_collector_gpu_names") else "unknown"
            self._profile_learner.update(
                gpu_index  = gpu,
                gpu_name   = _gpu_name_pl,
                rtheta_mean= window.rtheta_mean,
                power_w    = raw_sample.power_w,
                is_stable  = window.is_stable,
            )

        # 5b. Fault curve classifier — R_theta curve shape analysis (dust/TIM/fan/blockage)
        with self._stage("fault_classifier", gpu):
            fault = self._fault_classifier.update(
                gpu_index = gpu,
                ts        = ts,
                rtheta    = window.rtheta_mean,
                power_w   = raw_sample.power_w,
                mem_util  = raw_sample.mem_util_pct,
                fan_pct   = raw_sample.fan_speed_pct,
            )
            if fault is not None:
                self._exporter.update_fault_diagnosis(fault)
                # Populate the agent-details cache so the /api/v1/agent/gpu/{i}/details
                # endpoint can serve rich state (causal explanation + maintenance score)
                # to the site's Agent Control Center on every poll.
                self._update_agent_details_cache(
                    gpu=gpu, ts=ts,
                    filtered=filtered, raw_state=raw_state, raw_confidence=raw_confidence,
                    fault=fault, window=window, drift=drift,
                    power_w=raw_sample.power_w, ecc_dbit=raw_sample.ecc_dbit,
                    inlet_temp_c=getattr(raw_sample, "inlet_temp_c", None),
                )
                if fault.cause not in (FaultCause.NOMINAL, FaultCause.INSUFFICIENT_DATA):
                    fault_alert = AlertEvent(
                        gpu_index       = gpu,
                        timestamp       = ts,
                        state           = state,
                        prev_state      = state,
                        rtheta          = window.rtheta_mean,
                        rtheta_baseline = drift.baseline_mean,
                        drift_sigma     = drift.sigma_score,
                        confidence      = fault.confidence,
                        message         = (
                            f"[FAULT] GPU {gpu} — {fault.cause.value.replace('_', ' ').upper()}. "
                            f"{fault.remediation} "
                            f"R_θ intercept={fault.intercept:.3f} C/W, gap={fault.gap:.3f} C/W "
                            f"(confidence {fault.confidence:.0%})"
                        ),
                        context         = {
                            "severity":    "warning",
                            "fault_cause": fault.cause.value,
                            "confidence":  fault.confidence,
                            "intercept":   fault.intercept,
                            "gap":         fault.gap,
                            "curve_slope": fault.curve_slope,
                            "drift_rate":  fault.drift_rate,
                            "gap_trend":   fault.gap_trend,
                            "remediation": fault.remediation,
                            **fault.evidence,
                        },
                    )
                    self._alert_count += 1
                    self._exporter.record_alert(fault_alert)
                    await self._router.route(fault_alert)
                    log.info("fault_classified",
                             gpu=gpu,
                             cause=fault.cause.value,
                             confidence=fault.confidence,
                             intercept=fault.intercept,
                             gap=fault.gap)

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

            # Auto-recalibrate: queue GPU when it recovers from an anomalous state.
            # A recalibration pass refreshes the baseline after a real incident
            # (TIM/cooling degradation changes the unit's R_theta permanently).
            _recovery_states = (GPUState.CLEAN_IDLE, GPUState.UNDER_LOAD)
            _anomalous_from = {GPUState.CRITICAL, GPUState.DRIFTING, GPUState.ZOMBIE_RECOVERY}
            if alert.state in _recovery_states and alert.prev_state in _anomalous_from:
                self._recal_queue.add(gpu)
                self._recal_cooldown_ts[gpu] = ts
                log.info(
                    "recalibration_queued",
                    gpu=gpu,
                    prev_state=alert.prev_state.name if hasattr(alert.prev_state, "name") else str(alert.prev_state),
                    note="GPU recovered — recalibration recommended within 5 min",
                )

        # 7. Predictive alert — warn before the threshold is crossed
        with self._stage("predictive_alert", gpu):
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
        with self._stage("failure_predictor", gpu):
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
        with self._stage("telemetry", gpu):
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
        with self._stage("fleet_correlator", gpu):
            fleet_alert = self._correlator.check(
                {g: r.current_state for g, r in self._statemachine.all_states().items()},
                ts,
            )
            if fleet_alert is not None:
                self._alert_count += 1
                self._exporter.record_alert(fleet_alert)
                await self._router.route(fleet_alert)
                log.warning("fleet_event", affected=fleet_alert.context.get("fleet_gpus"))

    def _update_agent_details_cache(
        self, *, gpu: int, ts: float, filtered, raw_state, raw_confidence,
        fault, window, drift, power_w: float, ecc_dbit: int,
        inlet_temp_c: Optional[float] = None,
    ) -> None:
        """Compose causal + maintenance state for one GPU and cache it.

        Cached values are read by `_get_agent_details()`, which the Health API
        serves at /api/v1/agent/gpu/{i}/details. Composition runs once per
        sample (not per request) so polling the API is essentially free.
        """
        from .causal import reason as causal_reason
        from .maintenance import score as maintenance_score
        from .hw_profiles import resolve_profile

        # Resolve hardware profile for this GPU (cached in classifier already)
        gpu_name = self._collector_gpu_names.get(gpu, "unknown") if hasattr(self, "_collector_gpu_names") else "unknown"
        profile = resolve_profile(gpu_name)

        # Top-K state hypotheses from the temporal filter
        alternatives = self._temporal_filter.states_under_consideration(gpu, min_prob=0.05)

        def _safe_float(x, fallback=0.0):
            """Coerce optional metric to float; preserves None-as-zero semantics."""
            try:
                return float(x) if x is not None else fallback
            except (TypeError, ValueError):
                return fallback

        # Causal explanation
        try:
            trend_slope = _safe_float(getattr(drift, "trend_slope", 0.0))
            causal = causal_reason(
                gpu_index=gpu,
                smoothed_state=filtered.state,
                state_confidence=filtered.confidence,
                alternative_states=alternatives,
                fault_cause=fault.cause,
                fault_confidence=_safe_float(fault.confidence),
                rtheta_current=_safe_float(window.rtheta_mean),
                rtheta_baseline=_safe_float(drift.baseline_mean),
                rtheta_k_sigma=_safe_float(drift.sigma_score),
                rtheta_trend_per_min=trend_slope * 60.0,
                eta_to_threshold_sec=getattr(drift, "eta_seconds", None),
                ecc_dbit_any=ecc_dbit > 0,
                micro_throttle=False,
                correlated_gpus=(),
            )
            causal_dict = causal.as_dict()
        except Exception as exc:
            log.error("causal_reason_failed", gpu=gpu, error=str(exc))
            causal_dict = None

        # Maintenance score
        try:
            util_history = self._utilization_history.get(gpu)
            workload_intensity = (
                sum(util_history) / len(util_history) if util_history else 0.0
            )
            aging_per_month = max(0.0, _safe_float(getattr(drift, "trend_slope", 0.0))) * 86400 * 30
            service_threshold = (
                profile.rtheta_load_threshold if profile else self.config.k_warn * 0.05
            )
            maint = maintenance_score(
                gpu_index=gpu,
                profile=profile,
                rtheta_aging_rate_per_month=aging_per_month,
                rtheta_current=_safe_float(window.rtheta_mean),
                rtheta_baseline=_safe_float(drift.baseline_mean),
                rtheta_service_threshold=service_threshold,
                ecc_sbit_per_hour=_safe_float(
                    getattr(self._ecc_monitor, "rate_per_hour", lambda g: 0.0)(gpu)
                    if hasattr(self._ecc_monitor, "rate_per_hour") else 0.0
                ),
                workload_intensity=workload_intensity,
                inlet_temp_c=inlet_temp_c,
            )
            maint_dict = maint.as_dict()
        except Exception as exc:
            log.error("maintenance_score_failed", gpu=gpu, error=str(exc))
            maint_dict = None

        def _state_name(s):
            """Serialize GPUState as a stable human-readable label.

            GPUState's value happens to be an int (sklearn class index), so
            we use the enum's `.name` for JSON — much more useful for an
            operator reading the response than a magic number.
            """
            return s.name.lower() if hasattr(s, "name") else str(s)

        # CNN prediction (None when no weights loaded — site can see this and
        # display a "trained model not deployed" badge in the Reasoning tab).
        cnn_pred_dict = None
        if self._cnn_predictor.is_ready:
            try:
                cnn = self._cnn_predictor.predict(gpu, ts)
                if cnn is not None:
                    cnn_pred_dict = {
                        "p_failure_by_horizon": {
                            str(h): round(p, 4)
                            for h, p in cnn.p_failure_by_horizon.items()
                        },
                        "model_confidence": round(cnn.model_confidence, 3),
                        "alert_level": cnn.horizon_alert_level(),
                    }
            except Exception as exc:
                log.warning("cnn_predict_failed", gpu=gpu, error=str(exc))

        self._agent_details_cache[gpu] = {
            "gpu_index": gpu,
            "timestamp": ts,
            "cnn_prediction": cnn_pred_dict,
            "smoothed_state": {
                "state": _state_name(filtered.state),
                "confidence": round(filtered.confidence, 4),
                "n_observations": filtered.n_observations,
                "posterior": {
                    _state_name(s): round(p, 4)
                    for s, p in filtered.posterior.items()
                },
            },
            "raw_classifier": {
                "state": _state_name(raw_state),
                "confidence": round(raw_confidence, 4),
            },
            "fault": {
                "cause": fault.cause.value,
                "confidence": round(fault.confidence, 3),
                "intercept": fault.intercept,
                "gap": fault.gap,
                "remediation": fault.remediation,
            },
            "causal_explanation": causal_dict,
            "maintenance": maint_dict,
            "hw_profile": {
                "canonical_name": profile.canonical_name,
                "vendor": profile.vendor,
                "cooling": profile.cooling,
                "confidence": profile.confidence,
            } if profile else None,
        }

    def _get_agent_details(self, gpu_index: int) -> Optional[dict]:
        """Health API callback: return cached rich state for one GPU."""
        return self._agent_details_cache.get(gpu_index)

    def _check_hardware_ready(self) -> bool:
        """Block daemon start on extrapolated-profile hardware without calibration.

        Returns True if safe to proceed. Prints actionable instructions and
        returns False if the operator must run `theta calibrate` first.

        The check is profile-confidence-gated: T4 profiles are measured, so they
        pass unconditionally. Extrapolated profiles (all other GPU classes) require
        a matching calibration entry so the classifier uses measured thresholds
        instead of physics-based guesses.
        """
        from .calibrate import CalibrationManager
        from .hw_profiles import resolve_or_default

        import sys
        from rich.console import Console as _Console
        from rich.panel import Panel as _Panel

        cal = CalibrationManager()
        bad: list[tuple[int, str, str]] = []  # (slot, name, canonical)

        for slot, name in (getattr(self, "_collector_gpu_names", None) or {}).items():
            profile = resolve_or_default(name)
            if profile.confidence == "extrapolated" and cal.get(slot) is None:
                bad.append((slot, name, profile.canonical_name))

        if not bad:
            return True

        _c = _Console(stderr=True)
        gpu_lines = "\n".join(
            f"  GPU {slot}: {name}  →  profile={canonical}  (extrapolated, uncalibrated)"
            for slot, name, canonical in bad
        )
        _c.print()
        _c.print(_Panel(
            f"  [bold red]Theta cannot start — hardware calibration required.[/]\n\n"
            f"  The following GPU(s) use extrapolated R_θ thresholds, not measured ones.\n"
            f"  Running with T4 defaults on these GPUs will systematically misclassify\n"
            f"  healthy nodes as anomalous (and vice versa):\n\n"
            f"{gpu_lines}\n\n"
            f"  [bold yellow]Fix:[/] run calibration once, then restart the daemon:\n\n"
            f"  [bold green]theta calibrate --gpu 0[/]         "
            f"[dim]# repeat for each GPU index[/]\n"
            f"  [bold green]theta calibrate --ambient 22.0[/]  "
            f"[dim]# if GPU is too busy to idle (DGX, AI Factory)[/]\n\n"
            f"  [dim]Calibration takes ~60 seconds and writes to ~/.theta/calibration.json.\n"
            f"  Use --calibration-file /etc/theta/calibration.json for shared service installs.[/]",
            border_style="red",
            title="[red]Calibration required[/]",
            title_align="left",
            padding=(1, 2),
        ))
        _c.print()
        log.error(
            "startup_blocked_uncalibrated",
            gpus=[{"slot": s, "name": n, "profile": c} for s, n, c in bad],
        )
        return False

    def _reload_config(self) -> None:
        """Re-read ~/.theta/config.json and apply hot-reloadable fields.

        Hot-reloadable: alert thresholds (k_warn, k_critical), webhook URL,
        prometheus settings, telemetry opt-in. NOT hot-reloadable: GPU
        indices, interval_sec, prometheus_port, classifier mode — these
        require module re-init and would lose in-memory state, so changes
        only take effect on full agent restart.
        """
        try:
            if not self._config_path.exists():
                log.warning("config_reload_skipped", reason="config_file_missing")
                return
            raw = json.loads(self._config_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            log.error("config_reload_failed", error=str(exc))
            return

        applied: list[str] = []

        # Drift thresholds — pushed into the detector at runtime
        k_warn = raw.get("k_warn")
        if isinstance(k_warn, (int, float)) and k_warn != self.config.k_warn:
            self.config.k_warn = float(k_warn)
            if hasattr(self._detector, "k_warn"):
                self._detector.k_warn = float(k_warn)
            applied.append(f"k_warn={k_warn}")

        k_crit = raw.get("k_critical")
        if isinstance(k_crit, (int, float)) and k_crit != self.config.k_critical:
            self.config.k_critical = float(k_crit)
            if hasattr(self._detector, "k_critical"):
                self._detector.k_critical = float(k_crit)
            applied.append(f"k_critical={k_crit}")

        # Webhook URL — re-create the router so new URL takes effect on next alert
        new_url = raw.get("webhook_url")
        if new_url != self.config.webhook_url:
            self.config.webhook_url = new_url
            try:
                # Lazy import here to avoid circular at module load
                from .alerter import AlertRouter
                self._router = AlertRouter(
                    webhook_url   = new_url,
                    alert_log_path = self.config.alert_log_path,
                )
                applied.append("webhook_url updated")
            except Exception as exc:
                log.error("webhook_reload_failed", error=str(exc))

        # Telemetry opt-in — only honors flipping ON; flipping OFF would
        # require flushing in-flight batches, which isn't safe mid-run.
        new_opt_in = raw.get("data_sharing")
        if new_opt_in is True and not self.config.data_sharing:
            self.config.data_sharing = True
            if hasattr(self, "_telemetry"):
                try:
                    self._telemetry.enable()
                    applied.append("telemetry_enabled")
                except Exception as exc:
                    log.error("telemetry_enable_failed", error=str(exc))

        if applied:
            log.info("config_reloaded", changes=applied)
        else:
            log.info("config_reloaded", changes="no_op_no_changes_detected")

    async def run(self) -> None:
        """Main loop. Blocks until shutdown signal received."""
        loop = asyncio.get_running_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._shutdown.set)

        # SIGHUP triggers config hot-reload — operators can re-tune thresholds
        # (k_warn, k_critical, webhook URL) without restarting the daemon and
        # losing all in-memory baselines, drift buffers, and IsolationForest
        # models. The handler sets a flag; the main loop applies it between
        # samples so we never reload mid-classification.
        try:
            loop.add_signal_handler(signal.SIGHUP, self._reload_event.set)
        except (AttributeError, NotImplementedError):
            # SIGHUP unavailable (Windows, or signal not implemented on this
            # event loop) — silently skip; hot-reload simply unsupported.
            pass

        if self.config.enable_prometheus:
            self._exporter.start_server()

        if self._health_api:
            self._health_api.start()

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

        # Select the appropriate collector for this host's hardware (HAL).
        # Defaults to NVIDIA via NVMLCollector when pynvml works; falls back
        # to demo mode on hosts without a GPU driver. AMD ROCm path is
        # stubbed — will activate once pyrsmi is installed AND the
        # ROCmCollector implementation lands.
        from .hal import select_collector
        async with select_collector(collector_config) as collector:
            # Cache GPU names on `self` so _update_agent_details_cache can
            # resolve hw_profiles per GPU.
            try:
                self._collector_gpu_names = {
                    i: name for i, name in enumerate(collector.gpu_names)
                }
            except Exception:
                self._collector_gpu_names = {}

            # Register each GPU with the classifier so it picks the right
            # threshold tier from hw_profiles instead of falling back to T4
            # defaults — fixes silent misclassification on H100/B200/MI300X.
            # Also seed baseline with the profile's expected ambient so the
            # first R_θ computations aren't biased by a flat 25°C assumption.
            _min_std_values: list[float] = []
            for slot, name in self._collector_gpu_names.items():
                profile = self._classifier.register_gpu(slot, name)
                if profile is not None:
                    self._baseline.seed_from_profile(slot, name)
                    _min_std_values.append(getattr(profile, "drift_min_std", 0.010))
                    log.info(
                        "gpu_registered",
                        slot=slot, name=name,
                        family=profile.family, vendor=profile.vendor,
                        load_threshold=profile.rtheta_load_threshold,
                        cooling=profile.cooling,
                        t_ref_strategy=getattr(profile, "t_ref_strategy", "idle_window"),
                        drift_min_std=getattr(profile, "drift_min_std", 0.010),
                    )
                else:
                    log.warning(
                        "gpu_unprofiled",
                        slot=slot, name=name,
                        note="no hardware profile matched — using T4 defaults",
                    )
            # Use the smallest min_std across all profiled GPUs so the fleet
            # detector sensitivity honours the hardware with the tightest range.
            if _min_std_values:
                _effective_min_std = min(_min_std_values)
                if _effective_min_std != 0.010:
                    # Rebuild detector with correct noise floor for this hardware
                    self._detector = DriftDetector(
                        self.config.k_warn,
                        self.config.k_critical,
                        min_std=_effective_min_std,
                    )
                    log.info(
                        "drift_detector_reconfigured",
                        min_std=_effective_min_std,
                        reason="hardware profile drift_min_std override",
                    )

            # Wire GPU names into the fleet correlator for NVLink topology detection
            if self._collector_gpu_names:
                self._correlator.register_gpu_names(self._collector_gpu_names)

            # ── Pre-flight: block on un-calibrated extrapolated hardware ─────
            # T4 profiles are measured (Stage 1 data). All other profiles are
            # extrapolated — their thresholds are physics-based estimates, not
            # measurements. Running on extrapolated thresholds without a prior
            # `theta calibrate` run will produce systematic misclassification:
            # a healthy B200 under full load reads R_θ ≈ 0.27 C/W but the T4
            # fallback threshold is 0.87 — so it looks permanently IDLE to the
            # classifier. Hard-block here rather than emit silently-wrong output.
            if not self._check_hardware_ready():
                return

            async for raw_sample in collector.stream():
                if self._shutdown.is_set():
                    break
                if self._reload_event.is_set():
                    # Apply pending config reload between samples — never
                    # mid-classification, so in-memory state stays coherent.
                    self._reload_event.clear()
                    self._reload_config()
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

                    # ── Self-improvement periodic tasks ───────────────────
                    # Profile upgrade check — every 100 ticks (~8 min)
                    if self._tick_count % 100 == 0:
                        for _gpu_idx in list(self._collector_gpu_names or {}).copy():
                            _upgrade = self._profile_learner.ready_to_upgrade(_gpu_idx)
                            if _upgrade is not None:
                                log.info(
                                    "profile_upgrade_ready",
                                    action="update hw_profiles.py confidence to 'measured'",
                                    **_upgrade.as_log_dict(),
                                )
                                # Also emit as a structured alert so operators see it
                                _upgrade_alert = AlertEvent(
                                    gpu_index       = _gpu_idx,
                                    timestamp       = raw_sample.timestamp,
                                    state           = self._statemachine.get_state(_gpu_idx),
                                    prev_state      = self._statemachine.get_state(_gpu_idx),
                                    rtheta          = _upgrade.rtheta_mean,
                                    rtheta_baseline = _upgrade.rtheta_mean,
                                    drift_sigma     = 0.0,
                                    confidence      = 0.99,
                                    message         = (
                                        f"[INFO] GPU {_gpu_idx} ({_upgrade.gpu_name}) — "
                                        f"profile upgrade ready. {_upgrade.n_samples} load samples: "
                                        f"R_θ={_upgrade.rtheta_mean:.4f} ± {_upgrade.rtheta_std:.5f} C/W. "
                                        f"Run: theta calibrate --gpu {_gpu_idx}. "
                                        f"Then update hw_profiles.py confidence to 'measured'."
                                    ),
                                    context         = {
                                        "severity":         "info",
                                        "profile_upgrade":  True,
                                        "rtheta_mean":      _upgrade.rtheta_mean,
                                        "rtheta_std":       _upgrade.rtheta_std,
                                        "warn_threshold":   _upgrade.warn_threshold,
                                        "crit_threshold":   _upgrade.crit_threshold,
                                        "n_samples":        _upgrade.n_samples,
                                    },
                                )
                                await self._router.route(_upgrade_alert)

                    # Auto-recalibrate — check queue every 60 ticks, 5-min cooldown
                    if self._tick_count % 60 == 0 and self._recal_queue:
                        _now = raw_sample.timestamp
                        for _recal_gpu in list(self._recal_queue):
                            _queued_ts = self._recal_cooldown_ts.get(_recal_gpu, 0.0)
                            if _now - _queued_ts < 300:  # 5-min stable cooldown
                                continue
                            self._recal_queue.discard(_recal_gpu)
                            if getattr(self.config, "auto_recalibrate", False):
                                import asyncio as _asyncio
                                import sys as _sys
                                log.info("auto_recalibrate_start", gpu=_recal_gpu)
                                try:
                                    await _asyncio.create_subprocess_exec(
                                        _sys.executable, "-m", "theta.cli",
                                        "calibrate", "--gpu", str(_recal_gpu),
                                        "--non-interactive",
                                    )
                                except Exception as _e:
                                    log.error("auto_recalibrate_failed", gpu=_recal_gpu, error=str(_e))
                            else:
                                log.info(
                                    "recalibration_recommended",
                                    gpu=_recal_gpu,
                                    action=f"theta calibrate --gpu {_recal_gpu}",
                                    note="GPU has been stable 5+ min post-recovery",
                                )

                    # Model drift monitor — every 1000 ticks (~83 min)
                    if self._tick_count % 1000 == 0 and self._critic_total:
                        for _gpu_idx, _total in self._critic_total.items():
                            if _total < 100:
                                continue
                            _disagree = self._critic_disagree.get(_gpu_idx, 0)
                            _rate = _disagree / _total
                            if _rate > 0.15:
                                log.warning(
                                    "model_drift_detected",
                                    gpu=_gpu_idx,
                                    disagreement_rate=round(_rate, 3),
                                    total_windows=_total,
                                    action="theta train <path/to/data.csv> to retrain",
                                    note=(
                                        "Isolation Forest and DT classifier disagree on >15% of "
                                        "windows — operating conditions may have drifted from "
                                        "Stage 1 training data. Consider retraining."
                                    ),
                                )
                        # Reset counters after check
                        self._critic_disagree.clear()
                        self._critic_total.clear()

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

    def _health_status(self) -> dict:
        """Snapshot for the health API — includes degradation_risk from predictor."""
        base = self.status()
        for idx_str, gpu in base.get("gpus", {}).items():
            idx = int(idx_str)
            gpu["degradation_risk"] = self._predictor.get_score(idx)
        base["agent_version"] = __version__
        return base

    def status(self) -> dict:
        """Snapshot of current agent state — used by CLI `theta status`."""
        states = {}
        for gpu_idx, rec in self._statemachine.all_states().items():
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
            # Advisory-stage failures (see _stage) — nonzero counts mean an
            # enrichment stage is broken but core alerting is still running.
            "stage_errors": dict(self._stage_errors),
        }
