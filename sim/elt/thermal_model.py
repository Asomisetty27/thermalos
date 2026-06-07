"""
Transient 4-node/state Cauer thermal network, integrated with a stiff ODE solver.

State vector  y = [T_j, T_c, T_s, airflow]  (junction, case, heatsink, actual
normalised airflow) in degrees C (first three) and dimensionless (last).

    C_j dT_j/dt = P_eff(t,T_j)            - (T_j - T_c)/R_jc
    C_c dT_c/dt = (T_j - T_c)/R_jc        - (T_c - T_s)/R_ct(t)
    C_s dT_s/dt = (T_c - T_s)/R_ct(t)     - [(T_s - T_amb)/R_sa(airflow) + Q_rad(T_s)]
       d(airflow)/dt = (airflow_target(t,T_j) - airflow) / fan_tau_s

Time-varying inputs:
  * R_ct(t)              : TIM resistance, raised by the TIM-degradation mode
  * airflow_target(t,Tj) : fan curve (auto-ramps with T_j) x airflow degradation
                           factor x fan-cap factor — the commanded operating point
  * airflow              : the ACTUAL normalised airflow, which lags the commanded
                           target by a first-order rotor-spin-up/down time constant
                           (fan_tau_s) — a real fan cannot change speed instantly
  * Q_rad(T_s)           : Stefan-Boltzmann radiative loss from the heatsink,
                           a parallel heat-loss path alongside convection
  * P_eff                : demanded workload power, reduced once thermal throttling engages

The solver is BDF (implicit, stiff-stable): the junction time constant (~0.3 s) and
heatsink time constant (~70 s) span >2 decades, so the system is stiff. A precise
throttle-crossing time is captured with a solve_ivp event (no grid-snapping error).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

from . import params as P


# ─────────────────────────────────────────────────────────────────────────────
# Scenario: everything that defines one run except the fixed physical params
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Scenario:
    """One simulated E-LT run."""
    duration_s: float                       # total wall-clock to integrate
    workload_power_w: float = P.LOAD_POWER_W # FIXED load power (the critical control)
    # degradation callables: t (s) -> multiplier. Default = no degradation.
    rct_mult_fn: Callable[[float], float] = lambda t: 1.0      # TIM dry-out (>=1)
    airflow_mult_fn: Callable[[float], float] = lambda t: 1.0  # airflow restriction (<=1)
    fan_cap_fn: Callable[[float], float] = lambda t: 1.0       # fan duty cap (<=1)
    baseline_s: float = 0.0                 # healthy window before degradation begins
    # Emergent (Arrhenius-kinetic) TIM dry-out: when set, R_ct is NOT driven by
    # rct_mult_fn(t) but by an internal degradation-progress state x in [0,1]
    # that evolves as dx/dt = k(T_c)*(1-x), k from temperature-activated reaction
    # kinetics (see params.tim_arrhenius_rate). R_ct = R_ct0*(1+(severity-1)*x).
    # This produces a positive feedback the externally-imposed ramp cannot:
    # hotter TIM degrades faster -> R_ct rises -> runs hotter -> degrades faster.
    tim_kinetic_severity: Optional[float] = None
    label: str = "scenario"


@dataclass
class SimResult:
    """Ground-truth + sensed telemetry on a uniform 1 Hz grid."""
    t: np.ndarray              # time (s)
    tj_true: np.ndarray        # true junction temp (C)
    tc_true: np.ndarray        # case temp (C)
    ts_true: np.ndarray        # heatsink temp (C)
    airflow_true: np.ndarray   # actual (lagged) normalised airflow [0,1]
    airflow_target: np.ndarray # commanded normalised airflow [0,1] (fan curve x degradation)
    p_demand: np.ndarray       # demanded workload power (W)
    p_eff: np.ndarray          # effective power after throttle (W)
    rct: np.ndarray            # instantaneous TIM resistance (C/W)
    rsa: np.ndarray            # instantaneous convective resistance (C/W)
    q_rad: np.ndarray          # instantaneous radiative loss from heatsink (W)
    tim_progress: np.ndarray   # Arrhenius degradation-progress state x in [0,1]
                               # (always 0 for externally-imposed-trajectory scenarios)
    rtheta_true: np.ndarray    # true (T_j - T_amb)/P_eff  (C/W)
    throttling: np.ndarray     # bool: thermal throttle active this sample
    t_throttle: Optional[float]  # exact first-throttle time (s) or None
    params: P.ThermalParams
    scenario: Scenario


# ─────────────────────────────────────────────────────────────────────────────
# Soft throttle: hold T_j near the limit by clock/power-limiting. Smooth (logistic)
# so the integrator stays stable; the *exact* crossing time comes from the event.
# ─────────────────────────────────────────────────────────────────────────────
def _throttle_factor(tj: float, prm: P.ThermalParams) -> float:
    """
    Fraction of demanded power delivered. Exactly 1.0 at or below the thermal
    limit (a real GPU runs at full clocks until it hits the limit); above the
    limit the clock/power governor reduces power toward the floor to hold the
    junction near the limit. One-sided so the first 93 C crossing — the
    ground-truth throttle event we measure lead time against — is unaffected.
    """
    over = tj - prm.throttle_c
    if over <= 0.0:
        return 1.0
    width = max(P.THROTTLE_HYSTERESIS_C, 0.5)
    # smooth above the limit: 0 at the limit, ->(1-floor) reduction well above
    s = 1.0 - np.exp(-over / width)
    return 1.0 - (1.0 - P.THROTTLE_POWER_FLOOR) * s


def _airflow_target(tj: float, t: float, scn: Scenario, prm: P.ThermalParams) -> float:
    """
    Commanded normalised airflow operating point: the auto fan curve (which
    follows T_j), capped and restricted by the active degradation mode. This is
    the SET POINT a real fan controller asks the rotor to reach — not the
    airflow the heatsink actually sees right now (that is the lagged ODE state;
    see _rhs / FAN_TAU_S). A stalled rotor still has a non-zero floor.
    """
    duty = P.fan_duty(tj, prm.fan_duty_min, prm.fan_duty_max,
                      prm.fan_knee_lo, prm.fan_knee_hi)
    duty *= scn.fan_cap_fn(t)                 # fan/pump reduction mode
    target = duty * scn.airflow_mult_fn(t)    # airflow restriction mode
    return max(target, 1e-3)


def _rct_kinetic(tc: float, x: float, scn: Scenario, prm: P.ThermalParams) -> tuple[float, float]:
    """
    Emergent (Arrhenius) TIM path: returns (R_ct, dx/dt). Degradation rate is
    set by the LOCAL case/TIM-interface temperature T_c, not by elapsed time —
    the mechanism that creates runaway feedback.
    """
    k = P.tim_arrhenius_rate(tc)
    dx = k * (1.0 - x)
    rct = prm.r_ct0 * (1.0 + (scn.tim_kinetic_severity - 1.0) * x)
    return rct, dx


def _rhs(t: float, y: np.ndarray, scn: Scenario, prm: P.ThermalParams) -> np.ndarray:
    """Cauer-network + lagged-airflow + (optional) kinetic-TIM right-hand side dy/dt."""
    tj, tc, ts, airflow, x = y

    if scn.tim_kinetic_severity is not None:
        rct, dx = _rct_kinetic(tc, x, scn, prm)
    else:
        rct = prm.r_ct0 * scn.rct_mult_fn(t)
        dx = 0.0   # externally-imposed trajectories don't use the kinetic state

    rsa = P.r_sa(airflow, prm.r_sa_ref, prm.r_natural)

    p_eff = scn.workload_power_w * _throttle_factor(tj, prm)

    q_jc = (tj - tc) / prm.r_jc
    q_ct = (tc - ts) / rct
    q_conv = (ts - prm.t_amb_c) / rsa
    q_rad = P.q_radiative(ts, prm.t_amb_c, prm.rad_emissivity, prm.rad_area_m2)
    q_sa = q_conv + q_rad

    af_target = _airflow_target(tj, t, scn, prm)

    dtj = (p_eff - q_jc) / prm.c_j
    dtc = (q_jc - q_ct) / prm.c_c
    dts = (q_ct - q_sa) / prm.c_s
    daf = (af_target - airflow) / prm.fan_tau_s
    return np.array([dtj, dtc, dts, daf, dx])


def steady_state(power_w: float, scn: Scenario, prm: P.ThermalParams,
                 at_t: float = 0.0) -> np.ndarray:
    """
    Self-consistent steady state at fixed degradation (used as initial
    condition). At steady state the airflow lag has settled, so the actual
    airflow equals the commanded target — but T_j (which the target depends on,
    via the auto fan curve) is itself the unknown, so we still need a coupled
    root-find: outer solves T_j, inner solves T_s against the parallel
    convective + radiative heat-loss balance at that operating point.
    """
    from scipy.optimize import brentq

    def inner_ts_residual(ts: float, rsa: float) -> float:
        q_conv = (ts - prm.t_amb_c) / rsa
        q_rad = P.q_radiative(ts, prm.t_amb_c, prm.rad_emissivity, prm.rad_area_m2)
        return (q_conv + q_rad) - power_w

    def solve_ts(rsa: float) -> float:
        # Wide fixed bracket — see params._steady_temp_full for why [T_amb, T_j]
        # is the wrong bracket during an outer search over inconsistent T_j trials.
        return brentq(lambda ts: inner_ts_residual(ts, rsa),
                      prm.t_amb_c, 600.0, xtol=1e-7)

    # Initial degradation state: kinetic scenarios start with healthy TIM
    # (x=0, the material has not yet begun to dry out); externally-imposed
    # scenarios use rct_mult_fn(at_t) and never touch x (held at 0).
    def rct_at(tj: float) -> float:
        if scn.tim_kinetic_severity is not None:
            return prm.r_ct0   # x=0 -> R_ct = R_ct0
        return prm.r_ct0 * scn.rct_mult_fn(at_t)

    def outer_tj_residual(tj: float) -> float:
        rct = rct_at(tj)
        af_target = _airflow_target(tj, at_t, scn, prm)
        rsa = P.r_sa(af_target, prm.r_sa_ref, prm.r_natural)
        ts = solve_ts(rsa)
        tc = ts + power_w * rct
        return tj - (tc + power_w * prm.r_jc)

    tj = brentq(outer_tj_residual, prm.t_amb_c, 600.0, xtol=1e-6)

    rct = rct_at(tj)
    af_target = _airflow_target(tj, at_t, scn, prm)
    rsa = P.r_sa(af_target, prm.r_sa_ref, prm.r_natural)
    ts = solve_ts(rsa)
    tc = ts + power_w * rct
    x0 = 0.0
    return np.array([tj, tc, ts, af_target, x0])


def simulate(scn: Scenario, prm: P.ThermalParams = P.DEFAULT,
             dt_s: float = P.SAMPLE_PERIOD_S) -> SimResult:
    """
    Integrate the scenario and return uniformly-sampled ground-truth telemetry.
    Start from the healthy steady state at the workload power.
    """
    y0 = steady_state(scn.workload_power_w, scn, prm, at_t=0.0)
    t_eval = np.arange(0.0, scn.duration_s + dt_s, dt_s)

    # Event: T_j crosses the throttle temperature upward -> exact t_throttle.
    def cross(t, y, *_):
        return y[0] - prm.throttle_c
    cross.direction = 1.0
    cross.terminal = False

    sol = solve_ivp(
        _rhs, (0.0, scn.duration_s), y0,
        method="BDF", t_eval=t_eval, events=cross,
        args=(scn, prm), rtol=1e-7, atol=1e-9, max_step=dt_s,
    )
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")

    tj, tc, ts, airflow, x = sol.y[0], sol.y[1], sol.y[2], sol.y[3], sol.y[4]

    # Re-derive the time-varying quantities on the grid (vectorised where cheap)
    if scn.tim_kinetic_severity is not None:
        rct = prm.r_ct0 * (1.0 + (scn.tim_kinetic_severity - 1.0) * x)
        tim_progress = x
    else:
        rct = np.array([prm.r_ct0 * scn.rct_mult_fn(t) for t in sol.t])
        tim_progress = np.zeros_like(sol.t)
    rsa = np.array([P.r_sa(af_i, prm.r_sa_ref, prm.r_natural)
                    for af_i in airflow])
    q_rad = np.array([P.q_radiative(ts_i, prm.t_amb_c, prm.rad_emissivity, prm.rad_area_m2)
                      for ts_i in ts])
    airflow_target = np.array([_airflow_target(tj_i, t, scn, prm)
                               for tj_i, t in zip(tj, sol.t)])
    thr_factor = np.array([_throttle_factor(tj_i, prm) for tj_i in tj])
    p_demand = np.full_like(sol.t, scn.workload_power_w)
    p_eff = p_demand * thr_factor
    rtheta_true = (tj - prm.t_amb_c) / np.maximum(p_eff, 1e-6)
    throttling = tj >= prm.throttle_c

    t_throttle = float(sol.t_events[0][0]) if sol.t_events[0].size else None

    return SimResult(
        t=sol.t, tj_true=tj, tc_true=tc, ts_true=ts,
        airflow_true=airflow, airflow_target=airflow_target,
        p_demand=p_demand, p_eff=p_eff, rct=rct, rsa=rsa, q_rad=q_rad,
        tim_progress=tim_progress,
        rtheta_true=rtheta_true, throttling=throttling,
        t_throttle=t_throttle, params=prm, scenario=scn,
    )
