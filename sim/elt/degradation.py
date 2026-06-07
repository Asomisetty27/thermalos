"""
The three E-LT degradation modes, each as a time-varying parameter trajectory.

Per the protocol (raw/strategy/01_leadtime_testbed_protocol.md) each mode runs in
two variants:
  * gradual ramp  — the realistic, product-relevant case
  * step          — abrupt worst case

Mode -> physical parameter:
  TIM degradation     -> R_ct multiplier rises   (case-to-sink conduction)
  airflow restriction -> airflow multiplier falls (intake occlusion)
  fan/pump reduction  -> fan duty cap falls        (cooling actuator step-down)

Each builder returns the callables a Scenario consumes. A healthy baseline window
of `baseline_s` keeps parameters nominal so the detector can learn a clean baseline
before degradation begins — exactly the lab procedure.

Timescales (defaults, overridable):
  TIM dry-out      hours   (slowest, most valuable to predict)
  airflow occlusion tens of minutes
  fan step-down    minutes (fastest, most controllable)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .thermal_model import Scenario
from . import params as P


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory primitives
# ─────────────────────────────────────────────────────────────────────────────
def _linear_ramp(baseline_s: float, end_s: float,
                 start_val: float, end_val: float) -> Callable[[float], float]:
    """Hold start_val through baseline, then ramp linearly to end_val by end_s."""
    def f(t: float) -> float:
        if t <= baseline_s:
            return start_val
        if t >= end_s:
            return end_val
        frac = (t - baseline_s) / (end_s - baseline_s)
        return start_val + (end_val - start_val) * frac
    return f


def _exp_dryout(baseline_s: float, tau_s: float,
                start_val: float, end_val: float) -> Callable[[float], float]:
    """
    Exponential approach start_val -> end_val with time constant tau_s after
    baseline. Physically realistic for TIM pump-out / dry-out (diffusion-like).
    """
    def f(t: float) -> float:
        if t <= baseline_s:
            return start_val
        prog = 1.0 - np.exp(-(t - baseline_s) / tau_s)
        return start_val + (end_val - start_val) * prog
    return f


def _step(baseline_s: float, start_val: float, end_val: float) -> Callable[[float], float]:
    """Abrupt step from start_val to end_val at t = baseline_s (worst case)."""
    def f(t: float) -> float:
        return start_val if t < baseline_s else end_val
    return f


# ─────────────────────────────────────────────────────────────────────────────
# Mode builders
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DegradationSpec:
    """Human-readable description of an instantiated degradation arm."""
    mode: str          # 'tim' | 'airflow' | 'fan'
    variant: str       # 'gradual' | 'step'
    severity: float    # final multiplier/cap reached
    horizon_s: float   # when degradation completes (or steps)


def tim_degradation(duration_s: float, baseline_s: float = 600.0,
                    severity: float = 2.4, variant: str = "gradual",
                    workload_w: float = P.LOAD_POWER_W) -> tuple[Scenario, DegradationSpec]:
    """
    TIM dry-out: R_ct rises by `severity`x. severity 2.4x drives the junction
    well past the 93 C throttle at load (validated: ~1.8x already throttles).
    """
    horizon = duration_s
    if variant == "gradual":
        # exponential dry-out over ~half the post-baseline window
        tau = (duration_s - baseline_s) / 3.0
        fn = _exp_dryout(baseline_s, tau, 1.0, severity)
    elif variant == "step":
        fn = _step(baseline_s, 1.0, severity)
        horizon = baseline_s
    else:
        raise ValueError(f"unknown variant {variant!r}")

    scn = Scenario(
        duration_s=duration_s, workload_power_w=workload_w,
        rct_mult_fn=fn, baseline_s=baseline_s,
        label=f"TIM-{variant}-x{severity}",
    )
    return scn, DegradationSpec("tim", variant, severity, horizon)


def tim_degradation_arrhenius(duration_s: float, baseline_s: float = 0.0,
                              severity: float = 2.4, variant: str = "emergent",
                              workload_w: float = P.LOAD_POWER_W
                              ) -> tuple[Scenario, DegradationSpec]:
    """
    EMERGENT TIM dry-out: R_ct is not handed an externally-imposed schedule.
    Instead a degradation-progress state x in [0,1] evolves under temperature-
    activated Arrhenius kinetics (dx/dt = k(T_c)*(1-x), see
    params.tim_arrhenius_rate / thermal_model._rct_kinetic), and
    R_ct = R_ct0*(1 + (severity-1)*x).

    This is the complementary counterpart to tim_degradation()'s controlled
    ramp (the protocol's "begin controlled degradation ramp" procedure, which
    is what the physical testbed will actually run). Running both and
    comparing answers a genuinely open question: does the externally-imposed
    ramp the protocol assumes resemble what real thermally-activated material
    kinetics produce on their own — including the positive feedback (hotter
    TIM -> faster degradation -> hotter still) a prescribed trajectory cannot
    represent?

    `baseline_s` and `variant` are accepted for interface uniformity with the
    other MODE_BUILDERS entries (the Monte Carlo / trial harness calls every
    builder the same way) but are NOT used to gate the physical process: the
    kinetic dry-out is continuous from t=0 by construction — it is not
    "switched on" by an operator, that is precisely what makes it emergent
    rather than imposed. The detector's baseline-fitting window can still use
    `baseline_s` (early in the run x~0, R_theta is near-healthy).
    """
    if variant != "emergent":
        raise ValueError(f"tim_degradation_arrhenius has only one mode: 'emergent' (got {variant!r})")
    scn = Scenario(
        duration_s=duration_s, workload_power_w=workload_w,
        tim_kinetic_severity=severity, baseline_s=baseline_s,
        label=f"TIM-arrhenius-x{severity}",
    )
    return scn, DegradationSpec("tim_arrhenius", "emergent", severity, duration_s)


def airflow_restriction(duration_s: float, baseline_s: float = 600.0,
                        severity: float = 0.45, variant: str = "gradual",
                        workload_w: float = P.LOAD_POWER_W) -> tuple[Scenario, DegradationSpec]:
    """
    Airflow restriction: intake occlusion reduces airflow to `severity` fraction.
    0.45 (55% occlusion) raises R_sa enough to cross throttle at load.
    """
    horizon = duration_s
    if variant == "gradual":
        fn = _linear_ramp(baseline_s, duration_s, 1.0, severity)
    elif variant == "step":
        fn = _step(baseline_s, 1.0, severity)
        horizon = baseline_s
    else:
        raise ValueError(f"unknown variant {variant!r}")

    scn = Scenario(
        duration_s=duration_s, workload_power_w=workload_w,
        airflow_mult_fn=fn, baseline_s=baseline_s,
        label=f"airflow-{variant}-{severity}",
    )
    return scn, DegradationSpec("airflow", variant, severity, horizon)


def fan_reduction(duration_s: float, baseline_s: float = 600.0,
                  severity: float = 0.40, variant: str = "step",
                  workload_w: float = P.LOAD_POWER_W) -> tuple[Scenario, DegradationSpec]:
    """
    Fan/pump reduction: cap fan duty to `severity` fraction. Fastest mode;
    default variant is step (a partial cooling-actuator failure).
    """
    horizon = duration_s
    if variant == "gradual":
        fn = _linear_ramp(baseline_s, duration_s, 1.0, severity)
    elif variant == "step":
        fn = _step(baseline_s, 1.0, severity)
        horizon = baseline_s
    else:
        raise ValueError(f"unknown variant {variant!r}")

    scn = Scenario(
        duration_s=duration_s, workload_power_w=workload_w,
        fan_cap_fn=fn, baseline_s=baseline_s,
        label=f"fan-{variant}-{severity}",
    )
    return scn, DegradationSpec("fan", variant, severity, horizon)


# Registry for the CLI / Monte Carlo
MODE_BUILDERS = {
    "tim": tim_degradation,
    "tim_arrhenius": tim_degradation_arrhenius,
    "airflow": airflow_restriction,
    "fan": fan_reduction,
}

# Default timescales (s) — realistic onset per mode. Used by the CLI defaults.
DEFAULT_HORIZON_S = {
    "tim": 6 * 3600.0,            # 6 h dry-out
    "tim_arrhenius": 6 * 3600.0,  # same horizon — emergent counterpart to "tim"
    "airflow": 45 * 60.0,         # 45 min occlusion
    "fan": 10 * 60.0,             # 10 min (step engages at baseline)
}
