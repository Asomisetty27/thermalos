"""
Validation: prove the calibrated model reproduces Stage 1 ground truth and that
the numerics are sound. Run before trusting any lead-time number.

Checks:
  1. Steady-state operating points match Stage 1 (idle, load) within tolerance.
  2. Throttle point: power required to throttle a healthy GPU is physical.
  3. Energy balance at steady state: heat in == heat out across every node.
  4. Thermal time constants are in the expected range (fast junction, slow sink).
  5. Detector recovers the known baseline R_theta on a healthy (no-degradation) run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import params as P
from .thermal_model import simulate, steady_state, Scenario
from .detector import apply_sensor_model, windowed_rtheta, fit_baseline, DetectorConfig


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def _steady_check() -> list[Check]:
    checks = []
    prm = P.DEFAULT
    healthy = Scenario(duration_s=1.0)   # no degradation

    tj_idle = steady_state(P.IDLE_POWER_W, healthy, prm)[0]
    tj_load = steady_state(P.LOAD_POWER_W, healthy, prm)[0]

    # Load is calibrated exactly; idle has a documented residual.
    checks.append(Check(
        "steady load T_j == Stage 1 (81 C)",
        abs(tj_load - P.LOAD_TEMP_C) < 0.5,
        f"sim={tj_load:.2f}C measured={P.LOAD_TEMP_C}C resid={tj_load-P.LOAD_TEMP_C:+.2f}C",
    ))
    checks.append(Check(
        "steady idle T_j ~ Stage 1 (39 C, +/-3C slack)",
        abs(tj_idle - P.IDLE_TEMP_C) < 3.0,
        f"sim={tj_idle:.2f}C measured={P.IDLE_TEMP_C}C resid={tj_idle-P.IDLE_TEMP_C:+.2f}C "
        f"(within sensor 1C quantisation + assumed-ambient slack)",
    ))
    return checks


def _energy_balance_check() -> list[Check]:
    """
    At steady state, heat into each node equals heat out (within tolerance).
    The sink node now splits its outflow into convective + radiative branches
    (parallel heat-loss paths), so Q_ct must equal their SUM, not Q_conv alone.
    """
    prm = P.DEFAULT
    healthy = Scenario(duration_s=1.0)
    y = steady_state(P.LOAD_POWER_W, healthy, prm)
    tj, tc, ts, airflow, _x = y
    rct = prm.r_ct0
    rsa = P.r_sa(airflow, prm.r_sa_ref, prm.r_natural)

    q_in = P.LOAD_POWER_W
    q_jc = (tj - tc) / prm.r_jc
    q_ct = (tc - ts) / rct
    q_conv = (ts - prm.t_amb_c) / rsa
    q_rad = P.q_radiative(ts, prm.t_amb_c, prm.rad_emissivity, prm.rad_area_m2)
    q_sa = q_conv + q_rad
    max_err = max(abs(q_in - q_jc), abs(q_jc - q_ct), abs(q_ct - q_sa))
    return [Check(
        "steady-state energy balance (Q_in == Q_out each node, conv+rad)",
        max_err < 1e-3,
        f"max node imbalance = {max_err:.2e} W (Q_in={q_in:.2f} "
        f"Q_jc={q_jc:.2f} Q_ct={q_ct:.2f} Q_conv={q_conv:.2f} "
        f"Q_rad={q_rad:.2f} Q_sa={q_sa:.2f})",
    )]


def _time_constant_check() -> list[Check]:
    """Junction time constant fast (<2s), heatsink slow (20-150s)."""
    prm = P.DEFAULT
    tau_jc = prm.r_jc * prm.c_j
    rsa_at_load = P.r_sa(P.fan_duty(P.LOAD_TEMP_C), prm.r_sa_ref, prm.r_natural)
    tau_sa = rsa_at_load * prm.c_s
    return [
        Check("junction time constant < 2 s", tau_jc < 2.0,
              f"tau_jc = Rjc*Cj = {tau_jc:.2f} s"),
        Check("heatsink time constant in 20-150 s", 20.0 < tau_sa < 150.0,
              f"tau_sa = Rsa(load)*Cs = {tau_sa:.1f} s"),
    ]


def _radiation_floor_check() -> list[Check]:
    """
    Radiative loss should be a small-but-meaningful fraction of total dissipation
    at the load operating point — large enough to matter (it's a real parallel
    heat-loss path), small enough that it isn't secretly dominating the balance
    the convective calibration was tuned against.
    """
    prm = P.DEFAULT
    healthy = Scenario(duration_s=1.0)
    tj, tc, ts, airflow, _x = steady_state(P.LOAD_POWER_W, healthy, prm)
    q_rad = P.q_radiative(ts, prm.t_amb_c, prm.rad_emissivity, prm.rad_area_m2)
    frac = q_rad / P.LOAD_POWER_W
    return [Check(
        "radiative loss is a meaningful minority of total dissipation (1-15%)",
        0.01 < frac < 0.15,
        f"Q_rad={q_rad:.2f} W of {P.LOAD_POWER_W:.0f} W total "
        f"({100*frac:.1f}%) at T_s={ts:.1f}C",
    )]


def _natural_convection_floor_check() -> list[Check]:
    """
    R_sa must stay FINITE as commanded airflow -> 0 (a stalled fan still cools
    by buoyancy-driven natural convection) — the physical replacement for the
    old numerical max(airflow, eps) guard. Also: forced convection must
    dominate at full fan speed (R_sa(1.0) << R_sa(0)).
    """
    prm = P.DEFAULT
    r_stall = P.r_sa(0.0, prm.r_sa_ref, prm.r_natural)
    r_full = P.r_sa(1.0, prm.r_sa_ref, prm.r_natural)
    finite = np.isfinite(r_stall) and r_stall < 1e6
    return [Check(
        "R_sa stays finite at zero airflow (natural convection, not a numerical floor)",
        finite and r_stall == prm.r_natural,
        f"R_sa(airflow=0)={r_stall:.3f} C/W == R_natural={prm.r_natural:.3f} C/W",
    ), Check(
        "forced convection dominates over natural at full fan speed",
        r_full < r_stall,
        f"R_sa(full)={r_full:.3f} C/W < R_sa(stall)={r_stall:.3f} C/W "
        f"(ratio {r_stall/r_full:.1f}x)",
    )]


def _capacitance_plausibility_check() -> list[Check]:
    """Hand-tuned C_* values must fall inside geometry+material derived bounds."""
    bounds = P.capacitance_plausibility_range()
    checks = []
    for key, c in [("C_j", P.C_J_JK), ("C_c", P.C_C_JK), ("C_s", P.C_S_JK)]:
        lo, hi = bounds[key]
        checks.append(Check(
            f"{key} within geometry-derived plausibility bounds",
            lo <= c <= hi,
            f"{key}={c:.2f} J/K, plausible range [{lo:.2f}, {hi:.2f}] J/K "
            f"(mass x specific-heat for the dominant material)",
        ))
    return checks


def _fan_lag_check() -> list[Check]:
    """
    Airflow must respond to a duty-target step with first-order lag — not
    instantaneously. After one time constant (fan_tau_s) it should have closed
    ~63% of the gap to the new target (the textbook first-order-lag fingerprint).
    """
    prm = P.DEFAULT
    # A step-fan-reduction scenario forces a sharp target change at baseline_s.
    from .degradation import fan_reduction
    scn, _spec = fan_reduction(duration_s=400.0, baseline_s=60.0,
                               severity=0.40, variant="step")
    sim = simulate(scn, prm, dt_s=1.0)
    i0 = int(scn.baseline_s)
    target_before = sim.airflow_target[i0 - 1]
    target_after = sim.airflow_target[i0 + 1]
    af0 = sim.airflow_true[i0]
    i_tau = i0 + int(round(prm.fan_tau_s))
    af_tau = sim.airflow_true[min(i_tau, len(sim.airflow_true) - 1)]
    expected_progress = 1.0 - np.exp(-1.0)   # ~63%
    gap0 = target_after - af0
    actual_progress = (af_tau - af0) / gap0 if abs(gap0) > 1e-9 else 0.0
    close_enough = abs(actual_progress - expected_progress) < 0.15
    not_instant = abs(sim.airflow_true[i0 + 1] - target_after) > 0.5 * abs(gap0)
    return [Check(
        "airflow responds to a duty step with first-order lag (not instantaneously)",
        not_instant,
        f"one sample after the step: airflow={sim.airflow_true[i0+1]:.3f} "
        f"target={target_after:.3f} (still {abs(sim.airflow_true[i0+1]-target_after):.3f} "
        f"from target — a real rotor has inertia)",
    ), Check(
        "airflow closes ~63% of the gap after one fan_tau_s (first-order fingerprint)",
        close_enough,
        f"progress after tau={prm.fan_tau_s:.1f}s: {100*actual_progress:.0f}% "
        f"(textbook 1-e^-1 = {100*expected_progress:.0f}%)",
    )]


def _arrhenius_feedback_check() -> list[Check]:
    """
    Emergent kinetic TIM mode must show genuine positive feedback: the
    degradation rate at the (hotter) end-state temperature must exceed the
    rate at the start temperature by more than the temperature rise alone
    would explain via a LINEAR model — i.e. it is exponential/Arrhenius, and
    it must throttle SLOWER than nothing (it starts healthy) but the rate
    must accelerate as T_c climbs (k(T) increasing in T).
    """
    prm = P.DEFAULT
    k_cool = P.tim_arrhenius_rate(P.LOAD_TEMP_C - 1.0)
    k_hot = P.tim_arrhenius_rate(P.LOAD_TEMP_C - 1.0 + 16.0)
    ratio = k_hot / k_cool
    return [Check(
        "Arrhenius rate accelerates with temperature (runaway feedback mechanism)",
        ratio > 1.5,
        f"k({P.LOAD_TEMP_C-1:.0f}C)={k_cool:.2e}/s, "
        f"k({P.LOAD_TEMP_C-1+16:.0f}C)={k_hot:.2e}/s, ratio={ratio:.2f}x "
        f"over a 16C rise (engineering accelerated-aging rule of thumb: ~2-4x)",
    )]


def _throttle_physics_check() -> list[Check]:
    """The power that throttles a HEALTHY GPU must exceed the load power cap."""
    prm = P.DEFAULT
    healthy = Scenario(duration_s=1.0)
    # bisect power that drives healthy steady T_j to throttle
    from scipy.optimize import brentq
    f = lambda pw: steady_state(pw, healthy, prm)[0] - prm.throttle_c
    p_throttle = brentq(f, 1.0, 300.0)
    return [Check(
        "healthy GPU throttles only above load power (78 W cap)",
        p_throttle > P.LOAD_POWER_W,
        f"healthy throttle power = {p_throttle:.1f} W (> load {P.LOAD_POWER_W} W: "
        f"a healthy GPU at load does NOT throttle, as observed)",
    )]


def _detector_baseline_check() -> list[Check]:
    """On a healthy run the detector must recover the known steady R_theta."""
    healthy = Scenario(duration_s=900.0)
    sim = simulate(healthy)
    rng = np.random.default_rng(7)
    tel = apply_sensor_model(sim, rng, ambient_mode="true")
    rtheta, stable = windowed_rtheta(tel, DetectorConfig())
    base = fit_baseline(rtheta, stable, tel.t, 900.0)
    expected = P.R_THETA_LOAD
    return [Check(
        "detector baseline R_theta == known load R_theta",
        abs(base.mean - expected) < 0.02,
        f"detector μ={base.mean:.4f} expected={expected:.4f} "
        f"σ={base.std:.5f} n={base.n}",
    )]


def run_all() -> tuple[bool, list[Check]]:
    checks: list[Check] = []
    checks += _steady_check()
    checks += _energy_balance_check()
    checks += _time_constant_check()
    checks += _radiation_floor_check()
    checks += _natural_convection_floor_check()
    checks += _capacitance_plausibility_check()
    checks += _fan_lag_check()
    checks += _arrhenius_feedback_check()
    checks += _throttle_physics_check()
    checks += _detector_baseline_check()
    ok = all(c.passed for c in checks)
    return ok, checks


def format_report(checks: list[Check]) -> str:
    lines = ["E-LT model validation", "=" * 60]
    for c in checks:
        mark = "PASS" if c.passed else "FAIL"
        lines.append(f"[{mark}] {c.name}")
        lines.append(f"       {c.detail}")
    n_pass = sum(c.passed for c in checks)
    lines.append("-" * 60)
    lines.append(f"{n_pass}/{len(checks)} checks passed")
    return "\n".join(lines)


if __name__ == "__main__":
    ok, checks = run_all()
    print(format_report(checks))
    raise SystemExit(0 if ok else 1)
