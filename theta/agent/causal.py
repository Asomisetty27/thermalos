"""
Causal reasoning engine — turns Theta's signals into a plain-English diagnosis.

Audit finding addressed: alerts ship as opaque enums and threshold numbers,
which forces operators to guess "WHY is GPU 3 drifting?" Theta's agent
already KNOWS — between fault_classifier (curve shape), detector (rate +
ETA), silicon (ECC + throttle), state machine (transitions), correlator
(fleet patterns), and now temporal_filter (posterior over states) — it has
six independent pieces of evidence per alert. This module synthesizes them.

The output is structured:

    CausalExplanation:
      headline:      one-sentence summary an operator can read at a glance
      hypothesis:    the most likely physical cause + confidence
      alternatives:  competing hypotheses with confidence (not just argmax)
      evidence:      the specific signals that drove the conclusion
      actions:       ranked remediations with effort + expected impact
      timeline:      when did this start, and when do we project resolution

This is the equivalent of a doctor reading a chart and writing a SOAP note
(Subjective / Objective / Assessment / Plan) instead of just shouting
"ALERT: ANOMALY." It's the layer that makes the agent feel intelligent
rather than just instrumented.

Design principle: this module is PURE — given the inputs, the output is
deterministic. No I/O, no global state. That makes it trivially testable
and lets the same engine drive (a) Slack/email alerts, (b) the site's
Agent Control Center, (c) the maintenance scoring module, (d) the LLM
context for future agent-to-agent communication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .fault_classifier import FaultCause, FAULT_REMEDIATION
from .metrics import GPUState


class Urgency(Enum):
    """How quickly an operator should act. Drives alert routing priority."""
    INFO        = "info"        # observe; no action expected this shift
    WATCH       = "watch"       # monitor; review at next routine inspection
    ACT_SOON    = "act_soon"    # action within current work week
    ACT_NOW     = "act_now"     # action within current shift
    EMERGENCY   = "emergency"   # immediate action; risk of data loss / fire


@dataclass
class Action:
    """One ranked remediation step with metadata for operator triage."""
    title: str
    detail: str
    effort: str               # "30s", "5m", "30m maintenance window", etc.
    expected_impact: str      # "restores R_θ to baseline", "buys 30 days runway"
    blocks_workload: bool     # does this require draining/throttling the GPU?
    integration: Optional[str] = None  # "slurm:drain", "k8s:cordon", etc.


@dataclass
class Hypothesis:
    cause: FaultCause
    confidence: float         # 0..1 — calibrated where possible
    one_line: str             # "Cooling path is degrading — likely dust or TIM."


@dataclass
class Evidence:
    """One specific signal the reasoning engine cites as supporting a hypothesis."""
    name: str                 # "rtheta_drift_2.1_sigma"
    value: str                # "R_θ rose to 0.84 C/W, 2.1σ above baseline"
    weight: float             # 0..1 — how much this drove the conclusion


@dataclass
class CausalExplanation:
    headline: str
    urgency: Urgency
    hypothesis: Hypothesis
    alternatives: list[Hypothesis]
    evidence: list[Evidence]
    actions: list[Action]
    # Timeline fields are optional — we may not always have them
    when_started: Optional[str] = None     # human-readable, e.g. "12 minutes ago"
    eta_to_threshold: Optional[str] = None # "8 minutes" or None
    eta_to_recovery: Optional[str] = None  # "4 minutes" or None

    def as_dict(self) -> dict:
        """Serialize for JSON APIs / Slack blocks / Prometheus annotations."""
        return {
            "headline": self.headline,
            "urgency": self.urgency.value,
            "hypothesis": {
                "cause": self.hypothesis.cause.value,
                "confidence": round(self.hypothesis.confidence, 3),
                "one_line": self.hypothesis.one_line,
            },
            "alternatives": [
                {
                    "cause": h.cause.value,
                    "confidence": round(h.confidence, 3),
                    "one_line": h.one_line,
                }
                for h in self.alternatives
            ],
            "evidence": [
                {"name": e.name, "value": e.value, "weight": round(e.weight, 3)}
                for e in self.evidence
            ],
            "actions": [
                {
                    "title": a.title,
                    "detail": a.detail,
                    "effort": a.effort,
                    "expected_impact": a.expected_impact,
                    "blocks_workload": a.blocks_workload,
                    "integration": a.integration,
                }
                for a in self.actions
            ],
            "when_started": self.when_started,
            "eta_to_threshold": self.eta_to_threshold,
            "eta_to_recovery": self.eta_to_recovery,
        }


# ──────────────────────────────────────────────────────────────────────────
# Hypothesis one-liners — what does each cause look like physically?
# ──────────────────────────────────────────────────────────────────────────

_HYPOTHESIS_LINES: dict[FaultCause, str] = {
    FaultCause.NOMINAL:           "GPU is operating within its normal envelope.",
    FaultCause.DUST_ACCUMULATION: "Heatsink fins are loading with dust — uniform R_θ rise across the power curve.",
    FaultCause.TIM_DEGRADATION:   "Thermal interface material is aging — slope of R_θ(P) curve is steepening.",
    FaultCause.FAN_BEARING_WEAR:  "Cooling fan bearing is wearing — RPM dropping under sustained load.",
    FaultCause.AIRFLOW_BLOCKAGE:  "Airflow path is obstructed — sudden R_θ step within a single session.",
    FaultCause.MOUNTING_EVENT:    "Heatsink mounting pressure shifted between sessions — likely after a service event.",
    FaultCause.HBM_THERMAL:       "HBM stacks are running hot under memory-bandwidth load — package thermal asymmetry.",
    FaultCause.INSUFFICIENT_DATA: "Not enough samples across power tiers to diagnose yet.",
}


# ──────────────────────────────────────────────────────────────────────────
# Action library — every remediation we know how to recommend.
# ──────────────────────────────────────────────────────────────────────────

def _slurm_drain_action(gpu_index: int) -> Action:
    return Action(
        title=f"Drain GPU {gpu_index} from the SLURM queue",
        detail=(
            f"Run `scontrol update nodename=$(hostname) state=drain reason='theta:r_theta_drift'`. "
            f"In-flight jobs complete; new jobs route around this GPU."
        ),
        effort="30s — automatic if SLURM webhook is wired",
        expected_impact="Prevents new workload from compounding the thermal issue.",
        blocks_workload=True,
        integration="slurm:drain",
    )


def _k8s_cordon_action(gpu_index: int) -> Action:
    return Action(
        title=f"Cordon node hosting GPU {gpu_index}",
        detail=(
            f"`kubectl cordon $(hostname)` — marks the node unschedulable. "
            f"Existing pods continue; new pod scheduling skips this node."
        ),
        effort="30s — automatic if cluster admission webhook is wired",
        expected_impact="Prevents new pods from landing on the degrading hardware.",
        blocks_workload=False,
        integration="k8s:cordon",
    )


def _clean_dust_action() -> Action:
    return Action(
        title="Clean heatsink fins and air filters",
        detail=(
            "Compressed-air blowout of the heatsink, replacement of any chassis air "
            "filters that have been in service > 6 months. No GPU removal required."
        ),
        effort="20m maintenance window",
        expected_impact=(
            "Restores R_θ to within ~5% of baseline. Validate by re-running "
            "`theta calibrate --gpu {idx}` after cleaning."
        ),
        blocks_workload=True,
    )


def _repaste_action() -> Action:
    return Action(
        title="Replace thermal interface material (TIM)",
        detail=(
            "Remove heatsink, clean old TIM from die and cold-plate, apply fresh "
            "thermal paste or pad. Reseat with calibrated torque pattern."
        ),
        effort="45m maintenance window per GPU",
        expected_impact="Restores R_θ to within ~3% of original measured baseline. Buys 12–18 months of runtime.",
        blocks_workload=True,
    )


def _replace_fan_action() -> Action:
    return Action(
        title="Replace cooling fan assembly",
        detail=(
            "Fan bearing wear is progressive — once detected, RPM will continue "
            "declining and eventually a thermal trip will hard-stop the GPU. "
            "Replace the fan now, before the trip."
        ),
        effort="30m maintenance window per GPU",
        expected_impact="Eliminates the failure mode entirely. Restores nominal R_θ envelope.",
        blocks_workload=True,
    )


def _check_airflow_action() -> Action:
    return Action(
        title="Inspect airflow path",
        detail=(
            "Check for cable routing blocking GPU intake, rack-level baffles "
            "misaligned, hot-aisle/cold-aisle containment breach, or new equipment "
            "installed upstream that is preheating intake air."
        ),
        effort="10m walk-the-rack inspection",
        expected_impact="Step-change R_θ recovery within minutes of unblocking the airflow path.",
        blocks_workload=False,
    )


def _recalibrate_action(gpu_index: int) -> Action:
    return Action(
        title=f"Recalibrate thresholds for GPU {gpu_index}",
        detail=(
            f"Run `theta calibrate --gpu {gpu_index}` to re-derive load/idle "
            f"R_θ thresholds against the current healthy steady state. Use this "
            f"AFTER physical remediation, not before, so the new baseline reflects "
            f"the post-fix hardware."
        ),
        effort="5m — agent runs the calibration sweep automatically",
        expected_impact="Per-unit thresholds replace hardware-class defaults, reducing false-positive alerts.",
        blocks_workload=False,
    )


def _verify_mounting_action() -> Action:
    return Action(
        title="Verify heatsink mounting pressure",
        detail=(
            "Inspect the captive screws / cam-lock mechanism on the heatsink. "
            "Re-torque to spec if any are loose. If recent maintenance occurred, "
            "the most likely failure is uneven pressure causing a TIM air gap."
        ),
        effort="15m maintenance window",
        expected_impact="Step-change R_θ recovery to baseline within one work session after correction.",
        blocks_workload=True,
    )


def _hbm_workload_action() -> Action:
    return Action(
        title="Reduce memory-bandwidth-heavy workload concentration",
        detail=(
            "Stagger LLM inference batches or memory-stressing tests across this "
            "GPU so HBM stacks have time to dissipate. If pattern persists, this "
            "GPU may have a degraded HBM thermal interface."
        ),
        effort="Scheduling change — no maintenance window required",
        expected_impact="Reduces peak HBM temperature ~5–8 °C without losing throughput.",
        blocks_workload=False,
    )


# ──────────────────────────────────────────────────────────────────────────
# Reasoning entry point
# ──────────────────────────────────────────────────────────────────────────

def reason(
    *,
    gpu_index: int,
    smoothed_state: GPUState,
    state_confidence: float,
    alternative_states: list[tuple[GPUState, float]],
    fault_cause: FaultCause,
    fault_confidence: float,
    rtheta_current: float,
    rtheta_baseline: float,
    rtheta_k_sigma: float,
    rtheta_trend_per_min: float = 0.0,
    eta_to_threshold_sec: Optional[float] = None,
    ecc_dbit_any: bool = False,
    micro_throttle: bool = False,
    correlated_gpus: tuple[int, ...] = (),
    fleet_cause_hint: Optional[str] = None,
) -> CausalExplanation:
    """
    Synthesize a CausalExplanation from the agent's signal bundle.

    All inputs are values the daemon already computes — this function ONLY
    composes them into a coherent narrative. No new measurements, no I/O.

    The output urgency level is the highest of:
      - severity implied by R_θ k_sigma magnitude
      - DBIT ECC → always ACT_NOW (uncorrectable error)
      - micro-throttle present → at least WATCH
      - fleet correlation present → escalate one level
      - smoothed_state is ZOMBIE_RECOVERY → ACT_SOON minimum
    """
    # ── Urgency calculation ──
    urgency = _compute_urgency(
        rtheta_k_sigma=rtheta_k_sigma,
        ecc_dbit_any=ecc_dbit_any,
        micro_throttle=micro_throttle,
        smoothed_state=smoothed_state,
        fleet_correlation=bool(correlated_gpus),
    )

    # ── Headline ──
    headline = _compose_headline(
        gpu_index=gpu_index,
        fault_cause=fault_cause,
        rtheta_k_sigma=rtheta_k_sigma,
        smoothed_state=smoothed_state,
        ecc_dbit_any=ecc_dbit_any,
        urgency=urgency,
    )

    # ── Hypothesis + alternatives ──
    primary = Hypothesis(
        cause=fault_cause,
        confidence=fault_confidence,
        one_line=_HYPOTHESIS_LINES.get(fault_cause, "Cause undetermined."),
    )
    alternatives: list[Hypothesis] = []
    # If fleet correlation suggests a chassis/rack-level cause, surface that
    # as an alternative even when the fault_classifier hasn't escalated.
    if correlated_gpus and fault_cause not in (FaultCause.AIRFLOW_BLOCKAGE,):
        alternatives.append(
            Hypothesis(
                cause=FaultCause.AIRFLOW_BLOCKAGE,
                confidence=0.35 + 0.10 * min(len(correlated_gpus), 4),
                one_line=(
                    f"Chassis-level airflow issue — {len(correlated_gpus) + 1} GPUs "
                    f"in the same chassis show correlated R_θ drift "
                    f"({fleet_cause_hint or 'pattern unclear'})."
                ),
            )
        )
    # If state suggests stuck CUDA context, surface that even when fault classifier
    # diagnosis is INSUFFICIENT_DATA (zombie can occur before fault curve fills).
    if smoothed_state == GPUState.ZOMBIE_RECOVERY and primary.cause != FaultCause.NOMINAL:
        alternatives.append(
            Hypothesis(
                cause=FaultCause.NOMINAL,  # zombie isn't a physical fault, just stuck context
                confidence=state_confidence * 0.6,
                one_line=(
                    "GPU may be holding a stale CUDA context — power retained "
                    "despite no work scheduled. Killing the parent process usually "
                    "releases it without hardware intervention."
                ),
            )
        )

    # ── Evidence list ──
    evidence = _collect_evidence(
        rtheta_current=rtheta_current,
        rtheta_baseline=rtheta_baseline,
        rtheta_k_sigma=rtheta_k_sigma,
        rtheta_trend_per_min=rtheta_trend_per_min,
        smoothed_state=smoothed_state,
        state_confidence=state_confidence,
        alternative_states=alternative_states,
        ecc_dbit_any=ecc_dbit_any,
        micro_throttle=micro_throttle,
        correlated_gpus=correlated_gpus,
    )

    # ── Action ranking ──
    actions = _rank_actions(
        gpu_index=gpu_index,
        fault_cause=fault_cause,
        urgency=urgency,
        ecc_dbit_any=ecc_dbit_any,
        smoothed_state=smoothed_state,
    )

    # ── Timeline ──
    eta_thr = _format_seconds(eta_to_threshold_sec) if eta_to_threshold_sec else None

    return CausalExplanation(
        headline=headline,
        urgency=urgency,
        hypothesis=primary,
        alternatives=alternatives,
        evidence=evidence,
        actions=actions,
        when_started=None,  # caller can fill from state machine if desired
        eta_to_threshold=eta_thr,
        eta_to_recovery=None,
    )


# ── Internal helpers ─────────────────────────────────────────────────────

def _compute_urgency(
    *,
    rtheta_k_sigma: float,
    ecc_dbit_any: bool,
    micro_throttle: bool,
    smoothed_state: GPUState,
    fleet_correlation: bool,
) -> Urgency:
    # DBIT is always serious — uncorrectable bit flip means user data at risk.
    if ecc_dbit_any:
        return Urgency.ACT_NOW

    # Base level from R_θ deviation in σ-units (calibrated against baseline noise).
    if rtheta_k_sigma >= 4.0:
        base = Urgency.ACT_NOW
    elif rtheta_k_sigma >= 2.5:
        base = Urgency.ACT_SOON
    elif rtheta_k_sigma >= 1.5:
        base = Urgency.WATCH
    else:
        base = Urgency.INFO

    # Stuck CUDA context bumps to ACT_SOON minimum (it's burning power for nothing).
    if smoothed_state == GPUState.ZOMBIE_RECOVERY and base.value in ("info", "watch"):
        base = Urgency.ACT_SOON

    # Micro-throttle (sustained clock suppression under load) is a stronger
    # signal than R_θ alone — bump by one level if present.
    if micro_throttle:
        base = _escalate(base, 1)

    # Fleet-correlated drift suggests a shared upstream cause, which has
    # broader blast radius — bump by one level if present.
    if fleet_correlation:
        base = _escalate(base, 1)

    return base


def _escalate(u: Urgency, steps: int) -> Urgency:
    order = [Urgency.INFO, Urgency.WATCH, Urgency.ACT_SOON, Urgency.ACT_NOW, Urgency.EMERGENCY]
    idx = order.index(u)
    return order[min(idx + steps, len(order) - 1)]


def _compose_headline(
    *,
    gpu_index: int,
    fault_cause: FaultCause,
    rtheta_k_sigma: float,
    smoothed_state: GPUState,
    ecc_dbit_any: bool,
    urgency: Urgency,
) -> str:
    if ecc_dbit_any:
        return f"GPU {gpu_index}: uncorrectable ECC error — investigate immediately."
    if smoothed_state == GPUState.ZOMBIE_RECOVERY:
        return f"GPU {gpu_index}: stuck CUDA context retaining power — release recommended."
    if fault_cause == FaultCause.DUST_ACCUMULATION:
        return f"GPU {gpu_index}: heatsink loading with dust — R_θ drifted {rtheta_k_sigma:.1f}σ above baseline."
    if fault_cause == FaultCause.TIM_DEGRADATION:
        return f"GPU {gpu_index}: thermal interface material aging — slope of R_θ(P) is steepening."
    if fault_cause == FaultCause.FAN_BEARING_WEAR:
        return f"GPU {gpu_index}: cooling fan bearing wearing — RPM falling under load."
    if fault_cause == FaultCause.AIRFLOW_BLOCKAGE:
        return f"GPU {gpu_index}: airflow obstruction detected — sudden R_θ step-change."
    if fault_cause == FaultCause.MOUNTING_EVENT:
        return f"GPU {gpu_index}: heatsink mounting pressure shifted — likely post-service event."
    if fault_cause == FaultCause.HBM_THERMAL:
        return f"GPU {gpu_index}: HBM running hot under memory-bandwidth load."
    # Fallback — describe what we see even when no specific cause assigned
    return f"GPU {gpu_index}: R_θ {rtheta_k_sigma:.1f}σ above baseline (no specific cause assigned yet)."


def _collect_evidence(
    *,
    rtheta_current: float,
    rtheta_baseline: float,
    rtheta_k_sigma: float,
    rtheta_trend_per_min: float,
    smoothed_state: GPUState,
    state_confidence: float,
    alternative_states: list[tuple[GPUState, float]],
    ecc_dbit_any: bool,
    micro_throttle: bool,
    correlated_gpus: tuple[int, ...],
) -> list[Evidence]:
    out: list[Evidence] = []

    # R_θ deviation — always cited
    out.append(Evidence(
        name="rtheta_deviation",
        value=(
            f"R_θ is {rtheta_current:.3f} C/W vs baseline {rtheta_baseline:.3f} C/W "
            f"({rtheta_k_sigma:+.1f}σ)"
        ),
        weight=min(1.0, abs(rtheta_k_sigma) / 5.0),
    ))

    # Trend — cite if non-trivial
    if abs(rtheta_trend_per_min) > 1e-5:
        direction = "rising" if rtheta_trend_per_min > 0 else "falling"
        out.append(Evidence(
            name="rtheta_trend",
            value=f"R_θ is {direction} at {abs(rtheta_trend_per_min)*60:.3f} C/W per hour",
            weight=min(1.0, abs(rtheta_trend_per_min) * 600),
        ))

    # State machine
    out.append(Evidence(
        name="smoothed_state",
        value=f"Filtered state: {smoothed_state.name} (posterior {state_confidence:.0%})",
        weight=state_confidence,
    ))
    if len(alternative_states) > 1:
        alt_str = ", ".join(f"{s.name} {p:.0%}" for s, p in alternative_states[:3])
        out.append(Evidence(
            name="state_alternatives",
            value=f"Other states under consideration: {alt_str}",
            weight=0.5,
        ))

    # Silicon-level signals
    if ecc_dbit_any:
        out.append(Evidence(
            name="ecc_dbit",
            value="Uncorrectable (double-bit) ECC error reported by NVML",
            weight=1.0,
        ))
    if micro_throttle:
        out.append(Evidence(
            name="micro_throttle",
            value="SM clock sustained below 95% of boost under > 80% load",
            weight=0.8,
        ))

    # Fleet correlation
    if correlated_gpus:
        out.append(Evidence(
            name="fleet_correlation",
            value=f"Correlated R_θ drift on GPUs {sorted(correlated_gpus)} — shared upstream cause likely",
            weight=0.7,
        ))

    return out


def _rank_actions(
    *,
    gpu_index: int,
    fault_cause: FaultCause,
    urgency: Urgency,
    ecc_dbit_any: bool,
    smoothed_state: GPUState,
) -> list[Action]:
    actions: list[Action] = []

    # EMERGENCY / ACT_NOW → drain first to limit blast radius
    if urgency in (Urgency.EMERGENCY, Urgency.ACT_NOW):
        actions.append(_slurm_drain_action(gpu_index))
        actions.append(_k8s_cordon_action(gpu_index))

    # Cause-specific remediation
    if fault_cause == FaultCause.DUST_ACCUMULATION:
        actions.append(_clean_dust_action())
        actions.append(_recalibrate_action(gpu_index))
    elif fault_cause == FaultCause.TIM_DEGRADATION:
        actions.append(_repaste_action())
        actions.append(_recalibrate_action(gpu_index))
    elif fault_cause == FaultCause.FAN_BEARING_WEAR:
        actions.append(_replace_fan_action())
        actions.append(_recalibrate_action(gpu_index))
    elif fault_cause == FaultCause.AIRFLOW_BLOCKAGE:
        actions.append(_check_airflow_action())
    elif fault_cause == FaultCause.MOUNTING_EVENT:
        actions.append(_verify_mounting_action())
        actions.append(_recalibrate_action(gpu_index))
    elif fault_cause == FaultCause.HBM_THERMAL:
        actions.append(_hbm_workload_action())

    # Stuck CUDA context — try killing the offending process before anything physical
    if smoothed_state == GPUState.ZOMBIE_RECOVERY:
        actions.insert(0, Action(
            title=f"Release stuck CUDA context on GPU {gpu_index}",
            detail=(
                f"Find processes still holding the context with `nvidia-smi -i {gpu_index} "
                f"--query-compute-apps=pid,used_memory --format=csv` and kill any that "
                f"are not the live workload. Often the GPU recovers to CLEAN_IDLE within "
                f"a few seconds of release."
            ),
            effort="30s",
            expected_impact="Returns the GPU to CLEAN_IDLE without a maintenance window.",
            blocks_workload=False,
        ))

    # ECC DBIT — always escalate to an admin
    if ecc_dbit_any:
        actions.append(Action(
            title="Open a hardware ticket with the vendor",
            detail=(
                "Uncorrectable ECC errors are tracked by NVIDIA / AMD as silicon "
                "faults eligible for RMA. Attach `nvidia-smi -q` output and the "
                "Theta JSONL alert log for the affected window."
            ),
            effort="15m to file the ticket",
            expected_impact="Replacement GPU under warranty if the rate exceeds vendor SLA.",
            blocks_workload=False,
        ))

    # Always offer recalibrate as a safety net if no other action surfaced
    if not actions:
        actions.append(_recalibrate_action(gpu_index))

    return actions


def _format_seconds(s: Optional[float]) -> Optional[str]:
    if s is None or s <= 0:
        return None
    if s < 90:
        return f"{int(s)} seconds"
    if s < 5400:
        return f"{int(s / 60)} minutes"
    if s < 172800:
        return f"{int(s / 3600)} hours"
    return f"{int(s / 86400)} days"
